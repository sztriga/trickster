"""Feature encoding for the Ulti auction (bidding) phase.

Produces a flat numpy vector for the AI to decide:
  - Pickup / Pass (during auction)
  - Bid selection + discard choice (during bid phase)

  Section                     Features   Offset
  ─────────────────────────────────────────────
  Raw hand                      32         0
  Talon seen (if picked up)     32        32
  Talon history flag             1        64
  Current bid rank               1        65
  Am I the holder                1        66
  Number of passes               1        67
  Seat relative to dealer        3        68
  Auction history (last 3)      21        71
  Suit strength                 16        92
  Marriage potential              4       108
  High-card counts               4       112
  ─────────────────────────────────────────────
  Total                        116

The hand bitmap uses 32 bits even though the player may hold
10 or 12 cards — only the held cards are lit.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

from trickster.games.ulti.cards import (
    ALL_RANKS,
    ALL_SUITS,
    Card,
    NUM_PLAYERS,
    Rank,
    Suit,
)

# ---------------------------------------------------------------------------
#  Card index look-up (shared with encoder.py)
# ---------------------------------------------------------------------------

_SUIT_IDX: dict[Suit, int] = {s: i for i, s in enumerate(ALL_SUITS)}

_CARD_IDX: dict[Card, int] = {
    Card(s, r): si * 8 + int(r)
    for si, s in enumerate(ALL_SUITS)
    for r in ALL_RANKS
}

NUM_CARDS = 32

# High-value ranks for suit-strength computation
_HIGH_RANKS: frozenset[Rank] = frozenset({Rank.ACE, Rank.TEN, Rank.KING})

# ---------------------------------------------------------------------------
#  Encoding dimensions
# ---------------------------------------------------------------------------

_HAND_OFF = 0                                # 32  raw hand bitmap
_TALON_OFF = _HAND_OFF + NUM_CARDS           # 32  talon bitmap (if seen)
_TALON_FLAG_OFF = _TALON_OFF + NUM_CARDS     #  1  "have I seen the talon?"
_BID_RANK_OFF = _TALON_FLAG_OFF + 1          #  1  current bid rank (normalised)
_HOLDER_OFF = _BID_RANK_OFF + 1              #  1  am I the current holder?
_PASSES_OFF = _HOLDER_OFF + 1                #  1  consecutive passes (normalised)
_SEAT_OFF = _PASSES_OFF + 1                  #  3  seat relative to dealer (one-hot)
_HIST_OFF = _SEAT_OFF + 3                    # 21  auction history (3 × 7)
_SUIT_STR_OFF = _HIST_OFF + 21               # 16  suit strength (4 × 4)
_MARRIAGE_OFF = _SUIT_STR_OFF + 16           #  4  marriage potential per suit
_HIGHCARD_OFF = _MARRIAGE_OFF + 4            #  4  high-card count per suit

AUCTION_STATE_DIM = _HIGHCARD_OFF + 4        # 116


# ---------------------------------------------------------------------------
#  Auction history action types
# ---------------------------------------------------------------------------

# One-hot indices for auction actions (5 types)
_ACT_BID = 0
_ACT_PICKUP = 1
_ACT_PASS = 2
_ACT_STAND = 3
_ACT_NONE = 4  # padding for missing history entries

_HIST_ENTRY_DIM = 7  # player_rel(1) + action_type(5) + bid_rank(1)
_HIST_ENTRIES = 3    # last 3 actions


# ---------------------------------------------------------------------------
#  Encoder
# ---------------------------------------------------------------------------


class AuctionEncoder:
    """Feature encoder for the Ulti auction phase.

    Encodes the bidding state for AI decision-making:
    pickup/pass decisions and bid selection.
    """

    state_dim: int = AUCTION_STATE_DIM

    def encode_state(
        self,
        hand: Sequence[Card],
        player: int,
        dealer: int,
        current_bid_rank: int | None,
        holder: int | None,
        consecutive_passes: int,
        history: list[tuple[int, str, object]],
        talon_cards: Sequence[Card] | None = None,
    ) -> np.ndarray:
        """Encode the auction observation for *player*.

        Parameters
        ----------
        hand : cards currently held (10 or 12)
        player : observer player index
        dealer : dealer index
        current_bid_rank : rank of the current highest bid (None = no bid)
        holder : player who holds the current bid (None = no bid)
        consecutive_passes : number of consecutive passes
        history : auction history — list of (player, action_str, bid_or_None)
        talon_cards : the talon cards if the player has seen them
        """
        x = np.zeros(AUCTION_STATE_DIM, dtype=np.float64)
        _ci = _CARD_IDX

        # ── Section 1: Raw hand (32 binary) ─────────────────────────
        for c in hand:
            x[_HAND_OFF + _ci[c]] = 1.0

        # ── Section 2: Talon bitmap (32 binary) ─────────────────────
        # Only populated if the player has seen the talon.
        if talon_cards is not None:
            for c in talon_cards:
                x[_TALON_OFF + _ci[c]] = 1.0
            x[_TALON_FLAG_OFF] = 1.0

        # ── Section 3: Current bid rank (normalised) ────────────────
        max_rank = 38.0
        if current_bid_rank is not None and current_bid_rank > 0:
            x[_BID_RANK_OFF] = current_bid_rank / max_rank

        # ── Section 4: Am I the holder? ─────────────────────────────
        x[_HOLDER_OFF] = 1.0 if holder == player else 0.0

        # ── Section 5: Consecutive passes (normalised) ──────────────
        x[_PASSES_OFF] = min(consecutive_passes, 3) / 3.0

        # ── Section 6: Seat relative to dealer (one-hot 3) ──────────
        seat = (player - dealer) % NUM_PLAYERS
        x[_SEAT_OFF + seat] = 1.0

        # ── Section 7: Auction history — last 3 actions (21) ────────
        # Each entry: player_rel(1) + action_type(5 one-hot) + bid_rank(1)
        recent = history[-_HIST_ENTRIES:] if len(history) >= _HIST_ENTRIES else history
        for i in range(_HIST_ENTRIES):
            off = _HIST_OFF + i * _HIST_ENTRY_DIM
            if i < len(recent):
                h_player, h_action, h_bid = recent[i]
                # Player relative to observer
                delta = (h_player - player) % NUM_PLAYERS
                x[off + 0] = delta / 2.0  # 0.0=me, 0.5=next, 1.0=prev

                # Action type one-hot
                act_idx = {
                    "bid": _ACT_BID,
                    "pickup": _ACT_PICKUP,
                    "pass": _ACT_PASS,
                    "stand": _ACT_STAND,
                }.get(h_action, _ACT_NONE)
                x[off + 1 + act_idx] = 1.0

                # Bid rank (normalised)
                if h_bid is not None and hasattr(h_bid, "rank"):
                    x[off + 6] = h_bid.rank / max_rank
            else:
                # Padding: set the "none" action type
                x[off + 1 + _ACT_NONE] = 1.0

        # ── Section 8: Suit strength (4 × 4) ────────────────────────
        # Per suit: [card_count, high_card_count, has_ace, has_ten]
        # normalised so the AI can evaluate potential trumps.
        hand_list = list(hand)
        suit_counts: Counter[Suit] = Counter(c.suit for c in hand_list)
        for si, suit in enumerate(ALL_SUITS):
            off = _SUIT_STR_OFF + si * 4
            cards_of_suit = [c for c in hand_list if c.suit == suit]
            x[off + 0] = suit_counts[suit] / 8.0  # max possible = 8
            high = sum(1 for c in cards_of_suit if c.rank in _HIGH_RANKS)
            x[off + 1] = high / 3.0  # max 3 high cards per suit
            x[off + 2] = 1.0 if any(c.rank == Rank.ACE for c in cards_of_suit) else 0.0
            x[off + 3] = 1.0 if any(c.rank == Rank.TEN for c in cards_of_suit) else 0.0

        # ── Section 9: Marriage potential (4 binary) ─────────────────
        # Does the player hold both K+Q in each suit?
        hand_set = set(hand)
        for si, suit in enumerate(ALL_SUITS):
            if Card(suit, Rank.KING) in hand_set and Card(suit, Rank.QUEEN) in hand_set:
                x[_MARRIAGE_OFF + si] = 1.0

        # ── Section 10: High-card counts per suit (4) ────────────────
        # Total A+10+K count per suit (normalised).
        for si, suit in enumerate(ALL_SUITS):
            high = sum(
                1 for c in hand_list
                if c.suit == suit and c.rank in _HIGH_RANKS
            )
            x[_HIGHCARD_OFF + si] = high / 3.0

        return x
