from __future__ import annotations

from trickster.games.snapszer.cards import Card, Color
from trickster.games.snapszer.game import can_exchange_trump_jack, deal, exchange_trump_jack, talon_size


def test_trump_exchange_swaps_upcard_with_trump_jack() -> None:
    st = deal(seed=0, starting_leader=0)
    # Force a known trump upcard and trump color.
    st.trump_card = Card(Color.BELLS, 11)
    st.trump_color = Color.BELLS

    # Ensure talon size condition (>=2) holds.
    assert talon_size(st) >= 2

    trump_jack = Card(Color.BELLS, 2)
    # Put trump jack into leader's hand.
    st.hands[0][0] = trump_jack
    assert trump_jack in st.hands[0]

    assert can_exchange_trump_jack(st, 0)
    ok = exchange_trump_jack(st, 0)
    assert ok

    # Upcard becomes the jack, and player receives the previous upcard.
    assert st.trump_card == trump_jack
    assert Card(Color.BELLS, 11) in st.hands[0]
    assert trump_jack not in st.hands[0]


def test_trump_exchange_not_allowed_when_talon_too_small() -> None:
    st = deal(seed=1, starting_leader=0)
    st.trump_card = Card(Color.HEARTS, 10)
    st.trump_color = Color.HEARTS
    # Make talon size 1 (only the upcard).
    st.draw_pile.clear()
    assert talon_size(st) == 1

    st.hands[0][0] = Card(Color.HEARTS, 2)
    assert not can_exchange_trump_jack(st, 0)

