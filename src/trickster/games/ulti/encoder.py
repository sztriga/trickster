"""Feature encoding for Ulti — AlphaZero state representation (v2).

Produces a flat numpy vector of shape ``(STATE_DIM,)`` encoding
everything the current player can observe during the **play phase**.

v2 upgrades ("detective model"):
  - Per-player captured card bitmaps (card counting)
  - Contract DNA (component flags)
  - Trick momentum (leader/winner history)
  - Auction context (bid value, seat position, kontras)
  - Expanded marriage memory

  Section                       Features   Offset
  ───────────────────────────────────────────────
  My hand                         32         0
  Player 0 captured cards         32        32
  Player 1 captured cards         32        64
  Player 2 captured cards         32        96
  Trick card 0                    32       128
  Trick card 1                    32       160
  Trump suit                       4       192
  Scalars                          6       196
  Void flags                       8       202
  Marriage bitmask                 4       210
  Seven-in-hand                    1       214
  Contract DNA                     8       215
  Trick momentum (leaders)        10       223
  Trick momentum (winners)        10       233
  Auction context                 10       243
  Marriage memory                  6       253
  ───────────────────────────────────────────────
  Total                          259

Cards are indexed as ``suit_idx * 8 + rank_idx`` (0 .. 31).

Relative player encoding: positions are expressed relative to the
current player — 0.33 = me, 0.67 = left opponent, 1.0 = right opponent.
0.0 = no data (trick not yet played).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from trickster.games.ulti.cards import (
    ALL_RANKS,
    ALL_SUITS,
    Card,
    NUM_PLAYERS,
    Rank,
    Suit,
    TOTAL_POINTS,
    TRICKS_PER_GAME,
)

# ---------------------------------------------------------------------------
#  Card / suit index look-ups
# ---------------------------------------------------------------------------

_SUIT_IDX: dict[Suit, int] = {s: i for i, s in enumerate(ALL_SUITS)}
_RANK_IDX: dict[Rank, int] = {r: int(r) for r in ALL_RANKS}

_CARD_IDX: dict[Card, int] = {
    Card(s, r): si * 8 + int(r)
    for si, s in enumerate(ALL_SUITS)
    for r in ALL_RANKS
}

NUM_CARDS = 32

# ---------------------------------------------------------------------------
#  Encoding dimensions (v2 — 259 total)
# ---------------------------------------------------------------------------

_HAND_OFF = 0                                 # 32  my hand
_CAP0_OFF = _HAND_OFF + NUM_CARDS             # 32  player 0 captured
_CAP1_OFF = _CAP0_OFF + NUM_CARDS             # 32  player 1 captured
_CAP2_OFF = _CAP1_OFF + NUM_CARDS             # 32  player 2 captured
_TRICK0_OFF = _CAP2_OFF + NUM_CARDS           # 32  trick card 0
_TRICK1_OFF = _TRICK0_OFF + NUM_CARDS         # 32  trick card 1
_TRUMP_OFF = _TRICK1_OFF + NUM_CARDS          #  4  trump suit one-hot
_SCALAR_OFF = _TRUMP_OFF + 4                  #  6  scalars
_VOID_OFF = _SCALAR_OFF + 6                   #  8  void flags (2 opp × 4 suits)
_MARRIAGE_OFF = _VOID_OFF + 8                 #  4  my marriage bitmask
_SEVEN_OFF = _MARRIAGE_OFF + 4                #  1  seven-in-hand
_CONTRACT_OFF = _SEVEN_OFF + 1                #  8  contract DNA
_LEAD_HIST_OFF = _CONTRACT_OFF + 8            # 10  trick leaders
_WIN_HIST_OFF = _LEAD_HIST_OFF + TRICKS_PER_GAME  # 10  trick winners
_AUCTION_OFF = _WIN_HIST_OFF + TRICKS_PER_GAME    # 10  auction context
_MAR_MEM_OFF = _AUCTION_OFF + 10              #  6  marriage memory
STATE_DIM = _MAR_MEM_OFF + 6                  # 259

# Relative position encoding values
_REL_ME = 0.33
_REL_LEFT = 0.67
_REL_RIGHT = 1.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _relative_position(absolute: int, observer: int) -> float:
    """Encode an absolute player index relative to the observer.

    Returns _REL_ME / _REL_LEFT / _REL_RIGHT.
    Play order is counter-clockwise (0 → 1 → 2 → 0), so
    the "left" opponent is next in order and "right" is previous.
    """
    if absolute == observer:
        return _REL_ME
    delta = (absolute - observer) % NUM_PLAYERS
    return _REL_LEFT if delta == 1 else _REL_RIGHT


# ---------------------------------------------------------------------------
#  Encoder
# ---------------------------------------------------------------------------


class UltiEncoder:
    """Numpy-native feature encoder for the Ulti play phase (v2).

    Expanded from 153-dim (v1) to 259-dim with card counting,
    contract awareness, trick history, and auction context.
    """

    state_dim: int = STATE_DIM

    def encode_state(
        self,
        hand: Sequence[Card],
        captured: list[list[Card]],
        trick_cards: list[tuple[int, Card]],
        trump: Suit | None,
        betli: bool,
        soloist: int,
        player: int,
        trick_no: int,
        scores: list[int],
        known_voids: tuple[frozenset[Suit], ...],
        marriages: list[tuple[int, Suit, int]] | None = None,
        # --- v2 additions ---
        trick_history: list[tuple[int, int]] | None = None,
        contract_components: frozenset[str] | None = None,
        is_red: bool = False,
        is_open: bool = False,
        bid_rank: int = 0,
        soloist_saw_talon: bool = True,
        dealer: int = 0,
        component_kontras: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Encode observable state into a 1-D float64 array (259 dims)."""
        x = np.zeros(STATE_DIM, dtype=np.float64)
        _ci = _CARD_IDX

        # ── Section 1: My hand (32 binary) ───────────────────────────
        for c in hand:
            x[_HAND_OFF + _ci[c]] = 1.0

        # ── Section 2: Per-player captured cards (3 × 32 binary) ─────
        # Card-counting brain: who captured what.
        offsets = [_CAP0_OFF, _CAP1_OFF, _CAP2_OFF]
        for p in range(NUM_PLAYERS):
            off = offsets[p]
            for c in captured[p]:
                x[off + _ci[c]] = 1.0

        # ── Section 3: Trick card 0 (32 one-hot) ────────────────────
        if len(trick_cards) >= 1:
            x[_TRICK0_OFF + _ci[trick_cards[0][1]]] = 1.0

        # ── Section 4: Trick card 1 (32 one-hot) ────────────────────
        if len(trick_cards) >= 2:
            x[_TRICK1_OFF + _ci[trick_cards[1][1]]] = 1.0

        # ── Section 5: Trump suit (4 one-hot; zeros for Betli) ──────
        if trump is not None:
            x[_TRUMP_OFF + _SUIT_IDX[trump]] = 1.0

        # ── Section 6: Scalars (6) ──────────────────────────────────
        x[_SCALAR_OFF + 0] = 1.0 if betli else 0.0
        x[_SCALAR_OFF + 1] = 1.0 if player == soloist else 0.0
        x[_SCALAR_OFF + 2] = trick_no / float(TRICKS_PER_GAME)
        x[_SCALAR_OFF + 3] = scores[player] / float(TOTAL_POINTS)
        if player == soloist:
            enemy = sum(s for i, s in enumerate(scores) if i != soloist)
        else:
            enemy = scores[soloist]
        x[_SCALAR_OFF + 4] = enemy / float(TOTAL_POINTS)
        x[_SCALAR_OFF + 5] = len(trick_cards) / 2.0  # 0, 0.5, or 1.0

        # ── Section 7: Void flags (8) — 2 opponents × 4 suits ──────
        opps = sorted(i for i in range(NUM_PLAYERS) if i != player)
        for oi, opp in enumerate(opps):
            for si, suit in enumerate(ALL_SUITS):
                if suit in known_voids[opp]:
                    x[_VOID_OFF + oi * 4 + si] = 1.0

        # ── Section 8: Marriage bitmask (4) — K+Q pairs in my hand ──
        hand_set = set(hand)
        for si, suit in enumerate(ALL_SUITS):
            if Card(suit, Rank.KING) in hand_set and Card(suit, Rank.QUEEN) in hand_set:
                x[_MARRIAGE_OFF + si] = 1.0

        # ── Section 9: Seven-in-hand (1) ────────────────────────────
        if trump is not None and Card(trump, Rank.SEVEN) in hand_set:
            x[_SEVEN_OFF] = 1.0

        # ── Section 10: Contract DNA (8 bits) ───────────────────────
        # Light switch: tells the network WHAT the contract requires.
        if contract_components is not None:
            _DNA = {
                "parti":      0,
                "ulti":       1,
                "betli":      2,
                "durchmars":  3,
                "40":         4,
                "20":         5,
            }
            for comp, idx in _DNA.items():
                if comp in contract_components:
                    x[_CONTRACT_OFF + idx] = 1.0
            x[_CONTRACT_OFF + 6] = 1.0 if is_red else 0.0
            x[_CONTRACT_OFF + 7] = 1.0 if is_open else 0.0
        elif betli:
            # Legacy fallback: at least set the betli flag
            x[_CONTRACT_OFF + 2] = 1.0

        # ── Section 11: Trick momentum — leaders (10) ───────────────
        # Who led each completed trick (relative to me).
        # 0.0 = not played, 0.33 = me, 0.67 = left, 1.0 = right.
        if trick_history is not None:
            for i, (leader, _winner) in enumerate(trick_history):
                if i < TRICKS_PER_GAME:
                    x[_LEAD_HIST_OFF + i] = _relative_position(leader, player)

        # ── Section 12: Trick momentum — winners (10) ───────────────
        if trick_history is not None:
            for i, (_leader, winner) in enumerate(trick_history):
                if i < TRICKS_PER_GAME:
                    x[_WIN_HIST_OFF + i] = _relative_position(winner, player)

        # ── Section 13: Auction context (10) ─────────────────────────
        # [0]   final bid rank normalised (0–1)
        # [1]   soloist saw talon (bool)
        # [2-4] seat position relative to dealer (one-hot: 0/1/2 seats away)
        # [5-8] per-component kontra levels (parti/ulti/duri/betli,
        #       normalised 0=none, 0.5=kontra, 1.0=rekontra)
        # [9]   has_ulti flag (7esre tartás active)
        max_rank = 38.0
        x[_AUCTION_OFF + 0] = bid_rank / max_rank if bid_rank > 0 else 0.0
        x[_AUCTION_OFF + 1] = 1.0 if soloist_saw_talon and player == soloist else 0.0
        seat = (player - dealer) % NUM_PLAYERS
        for si in range(NUM_PLAYERS):
            x[_AUCTION_OFF + 2 + si] = 1.0 if si == seat else 0.0
        if component_kontras:
            _KONTRA_KEYS = ["parti", "ulti", "durchmars", "betli"]
            for ki, key in enumerate(_KONTRA_KEYS):
                level = component_kontras.get(key, 0)
                x[_AUCTION_OFF + 5 + ki] = level / 2.0  # 0, 0.5, or 1.0
        has_ulti = contract_components is not None and "ulti" in contract_components
        x[_AUCTION_OFF + 9] = 1.0 if has_ulti else 0.0

        # ── Section 14: Marriage memory (6) ──────────────────────────
        # Per-player declared marriage totals (normalised by 100)
        # + per-player "has declared any marriage" flag.
        # Players see values but not suits (tournament rules).
        if marriages:
            for p in range(NUM_PLAYERS):
                pts = sum(v for mp, _, v in marriages if mp == p)
                x[_MAR_MEM_OFF + p] = pts / 100.0
                x[_MAR_MEM_OFF + 3 + p] = 1.0 if pts > 0 else 0.0

        return x
