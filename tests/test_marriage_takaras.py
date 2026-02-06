from __future__ import annotations

from collections import deque

import pytest

from trickster.games.snapszer.cards import Card, Color, TRICKS_PER_GAME
from trickster.games.snapszer.game import (
    can_close_talon,
    close_talon,
    deal,
    deal_awarded_game_points,
    deal_winner,
    declare_marriage,
    legal_actions,
    play_trick,
)
from trickster.games.snapszer.rules import TrickResult


def test_marriage_adds_points_and_forces_lead_card() -> None:
    st = deal(seed=0, starting_leader=0)
    st.trump_color = Color.HEARTS

    # Ensure we hold the King+Queen of hearts.
    k = Card(Color.HEARTS, 4)
    q = Card(Color.HEARTS, 3)
    st.hands[0][0] = k
    st.hands[0][1] = q

    pts = declare_marriage(st, 0, Color.HEARTS)
    assert pts == 40
    assert st.scores[0] == 40
    assert st.pending_marriage is not None
    assert st.declared_marriages[-1] == (0, Color.HEARTS, 40)

    # If you declared, you must lead K or Q of that suit.
    st.hands[0][2] = Card(Color.BELLS, 11)
    st.hands[1][0] = Card(Color.ACORNS, 2)
    with pytest.raises(ValueError):
        play_trick(st, Card(Color.BELLS, 11), Card(Color.ACORNS, 2))

    # Leading the King is allowed, and pending marriage clears after the trick.
    st.hands[1][0] = Card(Color.ACORNS, 2)
    st, _ = play_trick(st, k, Card(Color.ACORNS, 2))
    assert st.pending_marriage is None


def test_takaras_legality_requires_at_least_4_talon_cards() -> None:
    st = deal(seed=1, starting_leader=0)
    assert can_close_talon(st, 0)

    # Reduce talon to 3 total cards (2 face-down + upcard) => not allowed.
    st.draw_pile = deque(list(st.draw_pile)[:2])
    assert can_close_talon(st, 0) is False


def test_no_draw_after_takaras_and_strict_follow_applies() -> None:
    st = deal(seed=2, starting_leader=0)
    assert close_talon(st, 0)
    assert st.talon_closed is True
    assert st.trump_upcard_visible is False

    # Strict follow now applies even though the stock still exists.
    st.trump_color = Color.BELLS
    st.hands[0] = [Card(Color.HEARTS, 2), Card(Color.BELLS, 11), Card(Color.ACORNS, 7), Card(Color.LEAVES, 8), Card(Color.LEAVES, 9)]
    lead = Card(Color.HEARTS, 10)
    legal = legal_actions(st, 0, lead)
    assert legal == [Card(Color.HEARTS, 2)]

    # No drawing occurs after tricks: hand sizes drop.
    st.hands[0][0] = Card(Color.HEARTS, 11)
    st.hands[1][0] = Card(Color.HEARTS, 2)
    draw_before = len(st.draw_pile)
    st, _ = play_trick(st, Card(Color.HEARTS, 11), Card(Color.HEARTS, 2))
    assert len(st.hands[0]) == 4
    assert len(st.hands[1]) == 4
    assert len(st.draw_pile) == draw_before


def test_last_trick_wins_when_no_one_reaches_66() -> None:
    st = deal(seed=3, starting_leader=0)
    st.scores = [60, 60]
    st.trick_no = TRICKS_PER_GAME
    st.last_trick = TrickResult(
        winner=1,
        leader_card=Card(Color.HEARTS, 2),
        responder_card=Card(Color.BELLS, 11),
    )
    assert deal_winner(st) == 1


def test_awarded_game_points_normal_and_takaras_failure() -> None:
    # Schwarz (loser took no tricks) => 3.
    st = deal(seed=4, starting_leader=0)
    st.scores = [66, 0]
    st.captured[1] = []
    w, pts, reason = deal_awarded_game_points(st)
    assert (w, pts, reason) == (0, 3, "schwarz")

    # Loser under 33 but took a trick => 2.
    st = deal(seed=5, starting_leader=0)
    st.scores = [66, 10]
    st.captured[1] = [Card(Color.HEARTS, 2), Card(Color.ACORNS, 2)]
    w, pts, reason = deal_awarded_game_points(st)
    assert (w, pts, reason) == (0, 2, "under_33")

    # Loser at least 33 => 1.
    st = deal(seed=6, starting_leader=0)
    st.scores = [66, 33]
    st.captured[1] = [Card(Color.HEARTS, 2), Card(Color.ACORNS, 2)]
    w, pts, reason = deal_awarded_game_points(st)
    assert (w, pts, reason) == (0, 1, "at_least_33")

    # Takarás failure: winner (non-closer) gets 2 or 3 depending on snapshot.
    st = deal(seed=7, starting_leader=0)
    st.talon_closed = True
    st.talon_closed_by = 0
    st.talon_close_any_zero_tricks = True
    st.scores = [10, 66]  # AI reaches 66; closer (human) failed
    w, pts, reason = deal_awarded_game_points(st)
    assert (w, pts, reason) == (1, 3, "takaras_failed")

