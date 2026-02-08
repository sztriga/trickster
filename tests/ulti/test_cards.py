"""Tests for the Ulti card module."""

from trickster.games.ulti.cards import (
    ALL_RANKS,
    ALL_SUITS,
    BETLI_STRENGTH,
    Card,
    LAST_TRICK_BONUS,
    Rank,
    Suit,
    TOTAL_CARD_POINTS,
    TOTAL_POINTS,
    make_deck,
)


def test_deck_has_32_cards():
    deck = make_deck()
    assert len(deck) == 32


def test_deck_has_4_suits_8_ranks():
    deck = make_deck()
    suits = {c.suit for c in deck}
    ranks = {c.rank for c in deck}
    assert len(suits) == 4
    assert len(ranks) == 8


def test_deck_all_unique():
    deck = make_deck()
    assert len(set(deck)) == 32


def test_card_points_ace_and_ten():
    for s in ALL_SUITS:
        assert Card(s, Rank.ACE).points() == 10
        assert Card(s, Rank.TEN).points() == 10


def test_card_points_others_zero():
    zero_ranks = [Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.JACK, Rank.QUEEN, Rank.KING]
    for s in ALL_SUITS:
        for r in zero_ranks:
            assert Card(s, r).points() == 0, f"{Card(s, r)} should be 0 points"


def test_total_card_points_is_80():
    deck = make_deck()
    total = sum(c.points() for c in deck)
    assert total == 80
    assert total == TOTAL_CARD_POINTS


def test_total_points_is_90():
    assert TOTAL_POINTS == TOTAL_CARD_POINTS + LAST_TRICK_BONUS
    assert TOTAL_POINTS == 90


def test_normal_strength_order():
    """Normal order: 7 < 8 < 9 < J < Q < K < 10 < A."""
    expected = [
        Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.JACK,
        Rank.QUEEN, Rank.KING, Rank.TEN, Rank.ACE,
    ]
    for i in range(len(expected) - 1):
        a = Card(Suit.HEARTS, expected[i])
        b = Card(Suit.HEARTS, expected[i + 1])
        assert a.strength() < b.strength(), f"{a} should be weaker than {b}"


def test_betli_strength_order():
    """Betli order: 7 < 8 < 9 < 10 < J < Q < K < A."""
    expected = [
        Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
        Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
    ]
    for i in range(len(expected) - 1):
        a = Card(Suit.HEARTS, expected[i])
        b = Card(Suit.HEARTS, expected[i + 1])
        assert a.strength(betli=True) < b.strength(betli=True), (
            f"Betli: {a} should be weaker than {b}"
        )


def test_ten_stronger_than_king_in_normal():
    ten = Card(Suit.BELLS, Rank.TEN)
    king = Card(Suit.BELLS, Rank.KING)
    assert ten.strength() > king.strength()


def test_ten_weaker_than_jack_in_betli():
    ten = Card(Suit.BELLS, Rank.TEN)
    jack = Card(Suit.BELLS, Rank.JACK)
    assert ten.strength(betli=True) < jack.strength(betli=True)


def test_short_labels():
    assert Card(Suit.HEARTS, Rank.ACE).short() == "HA"
    assert Card(Suit.BELLS, Rank.TEN).short() == "B10"
    assert Card(Suit.LEAVES, Rank.JACK).short() == "LJ"
    assert Card(Suit.ACORNS, Rank.SEVEN).short() == "A7"
