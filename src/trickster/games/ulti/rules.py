"""Trick-taking rules for 3-player Ulti.

Two rule sets depending on contract:

**Normal (Simple / Ulti / Durchmars / 40-100 / …):**
  - Must follow the led suit.
  - If following suit, must beat the highest same-suit card played so far
    (if possible).
  - If unable to follow suit, must play trump (if possible).
  - If trumping, must beat the highest trump played so far (if possible).

**Betli:**
  - Must follow the led suit.
  - No must-beat obligation.
  - No trump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from trickster.games.ulti.cards import BETLI_STRENGTH, Card, Suit


# ---------------------------------------------------------------------------
#  Trick result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrickResult:
    """Result of a completed 3-card trick."""
    cards: tuple[Card, ...]          # cards in play order (leader, 2nd, 3rd)
    players: tuple[int, ...]         # player indices in play order
    winner: int                       # player index who won the trick


# ---------------------------------------------------------------------------
#  Legal response
# ---------------------------------------------------------------------------


def legal_response(
    hand: Sequence[Card],
    led_suit: Suit,
    played_cards: Sequence[Card],
    *,
    trump: Optional[Suit],
    betli: bool = False,
) -> List[Card]:
    """Return the legal cards a player may play given the trick so far.

    Parameters
    ----------
    hand : cards in the player's hand
    led_suit : suit of the first card played this trick
    played_cards : cards already on the table (1 or 2)
    trump : trump suit (None for Betli)
    betli : if True, no must-beat and no trump obligations
    """
    same_suit = [c for c in hand if c.suit == led_suit]

    if same_suit:
        if betli:
            return same_suit  # must follow, no must-beat

        # Must beat: find highest same-suit card already played
        led_played = [c for c in played_cards if c.suit == led_suit]
        if led_played:
            max_str = max(c.strength() for c in led_played)
            higher = [c for c in same_suit if c.strength() > max_str]
            if higher:
                return higher
        return same_suit

    # --- Can't follow suit ---

    if betli or trump is None:
        return list(hand)  # no trump obligation

    trumps = [c for c in hand if c.suit == trump]
    if not trumps:
        return list(hand)  # no trump in hand, play anything

    # Must trump.  If trumps already played, must beat them if possible.
    trumps_played = [c for c in played_cards if c.suit == trump]
    if trumps_played:
        max_trump_str = max(c.strength() for c in trumps_played)
        higher = [c for c in trumps if c.strength() > max_trump_str]
        if higher:
            return higher
    return trumps


# ---------------------------------------------------------------------------
#  Trick resolution
# ---------------------------------------------------------------------------


def resolve_trick(
    plays: Sequence[tuple[int, Card]],
    *,
    trump: Optional[Suit],
    betli: bool = False,
) -> TrickResult:
    """Determine the winner of a completed 3-card trick.

    Parameters
    ----------
    plays : sequence of ``(player_index, card)`` in play order
    trump : trump suit (``None`` for Betli / no-trump contracts)
    betli : if True, use Betli card ordering (10 below J)
    """
    assert len(plays) == 3, f"A trick needs exactly 3 cards, got {len(plays)}"

    def _strength(card: Card) -> int:
        return BETLI_STRENGTH[card.rank] if betli else int(card.rank)

    led_suit = plays[0][1].suit

    best_player, best_card = plays[0]
    best_is_trump = trump is not None and best_card.suit == trump

    for player, card in plays[1:]:
        card_is_trump = trump is not None and card.suit == trump

        if card_is_trump and not best_is_trump:
            # Trump beats non-trump
            best_player, best_card, best_is_trump = player, card, True
        elif card_is_trump and best_is_trump:
            # Both trump: higher strength wins
            if _strength(card) > _strength(best_card):
                best_player, best_card = player, card
        elif not card_is_trump and not best_is_trump:
            # Neither is trump: only a card of the led suit can win
            if card.suit == led_suit and best_card.suit == led_suit:
                if _strength(card) > _strength(best_card):
                    best_player, best_card = player, card
            elif card.suit == led_suit:
                # card follows suit but current best doesn't (defensive)
                best_player, best_card = player, card
        # else: non-trump can't beat trump → no change

    return TrickResult(
        cards=tuple(c for _, c in plays),
        players=tuple(p for p, _ in plays),
        winner=best_player,
    )
