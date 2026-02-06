from __future__ import annotations

from trickster.games.snapszer.game import deal, is_terminal


def test_game_ends_at_66_points() -> None:
    st = deal(seed=0, starting_leader=0)
    st.scores[0] = 66
    st.trick_no = 0
    assert is_terminal(st)

