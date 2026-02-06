from __future__ import annotations

from trickster.games.snapszer.cards import Card, Color
from trickster.games.snapszer.rules import legal_response_cards, resolve_trick


def test_must_follow_color_if_possible_closed_phase() -> None:
    lead = Card(Color.HEARTS, 10)
    hand = [Card(Color.HEARTS, 2), Card(Color.BELLS, 11)]
    legal = legal_response_cards(hand, lead, must_follow=True, trump=Color.BELLS)
    assert legal == [Card(Color.HEARTS, 2)]


def test_must_play_higher_if_possible_closed_phase() -> None:
    lead = Card(Color.LEAVES, 3)
    hand = [Card(Color.LEAVES, 2), Card(Color.LEAVES, 4), Card(Color.BELLS, 11)]
    legal = legal_response_cards(hand, lead, must_follow=True, trump=Color.BELLS)
    assert legal == [Card(Color.LEAVES, 4)]


def test_can_play_any_if_cannot_follow_closed_phase() -> None:
    lead = Card(Color.ACORNS, 2)
    hand = [Card(Color.HEARTS, 11), Card(Color.BELLS, 10)]
    legal = legal_response_cards(hand, lead, must_follow=True, trump=Color.LEAVES)
    assert set(legal) == set(hand)


def test_open_phase_can_play_anything_even_if_can_follow() -> None:
    lead = Card(Color.HEARTS, 10)
    hand = [Card(Color.HEARTS, 2), Card(Color.BELLS, 11)]
    legal = legal_response_cards(hand, lead, must_follow=False, trump=Color.BELLS)
    assert set(legal) == set(hand)


def test_trick_resolution() -> None:
    # same color, responder higher -> responder wins
    res = resolve_trick(leader_idx=0, leader_card=Card(Color.HEARTS, 3), responder_card=Card(Color.HEARTS, 4), trump=Color.BELLS)
    assert res.winner == 1

    # different color -> leader wins
    res = resolve_trick(leader_idx=1, leader_card=Card(Color.BELLS, 10), responder_card=Card(Color.HEARTS, 2), trump=Color.LEAVES)
    assert res.winner == 1

    # trump beats non-trump
    res = resolve_trick(
        leader_idx=0,
        leader_card=Card(Color.HEARTS, 11),
        responder_card=Card(Color.BELLS, 2),
        trump=Color.BELLS,
    )
    assert res.winner == 1

