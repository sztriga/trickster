from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from trickster.games.snapszer.cards import Card, Color


@dataclass(frozen=True, slots=True)
class TrickResult:
    leader_card: Card
    responder_card: Card
    winner: int  # 0 or 1 (player index)


def can_follow_color(hand: Sequence[Card], color: Color) -> bool:
    return any(c.color == color for c in hand)


def legal_response_cards(hand: Sequence[Card], lead: Card, *, must_follow: bool, trump: Color) -> List[Card]:
    """
    Legal response cards to a lead.

    - If `must_follow` is False (open talon / draw pile not empty): you may play any card.
    - If `must_follow` is True  (closed phase / draw pile empty): you must follow color if possible,
      and if you can beat with that color you must play a higher one.
    """
    if not must_follow:
        return list(hand)

    same = [c for c in hand if c.color == lead.color]
    if not same:
        # Closed phase: if you cannot follow suit, you must play trump if you have any.
        trumps = [c for c in hand if c.color == trump]
        return trumps if trumps else list(hand)

    higher = [c for c in same if c.number > lead.number]
    if higher:
        # Obligation: if you can beat with same color, you must play a higher one.
        return higher

    # Obligation: if you can follow color but cannot beat, you must play same color (lower).
    return same


def resolve_trick(*, leader_idx: int, leader_card: Card, responder_card: Card, trump: Color) -> TrickResult:
    """
    Trick resolution with a fixed trump suit:
    - Trump beats any non-trump.
    - If both are trump, higher wins.
    - If neither is trump:
      - Different color => leader wins
      - Same color => higher wins
    """
    lead_is_trump = leader_card.color == trump
    resp_is_trump = responder_card.color == trump

    if resp_is_trump and not lead_is_trump:
        return TrickResult(leader_card=leader_card, responder_card=responder_card, winner=1 - leader_idx)
    if lead_is_trump and not resp_is_trump:
        return TrickResult(leader_card=leader_card, responder_card=responder_card, winner=leader_idx)

    # Both trump or both non-trump: if different color (only possible for non-trump), leader wins.
    if responder_card.color != leader_card.color:
        return TrickResult(leader_card=leader_card, responder_card=responder_card, winner=leader_idx)

    # Same color: higher number wins for responder, otherwise leader.
    if responder_card.number > leader_card.number:
        return TrickResult(leader_card=leader_card, responder_card=responder_card, winner=1 - leader_idx)
    return TrickResult(leader_card=leader_card, responder_card=responder_card, winner=leader_idx)


def hand_color_counts(hand: Iterable[Card]) -> dict[Color, int]:
    d: dict[Color, int] = defaultdict(int)
    for c in hand:
        d[c.color] += 1
    return dict(d)

