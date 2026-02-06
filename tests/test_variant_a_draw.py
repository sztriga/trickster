from __future__ import annotations

from trickster.games.snapszer.cards import Card, Color
from trickster.games.snapszer.game import deal, play_trick


def test_variant_a_draws_to_five_winner_first() -> None:
    st = deal(seed=123, starting_leader=0)
    assert len(st.hands[0]) == 5
    assert len(st.hands[1]) == 5
    assert len(st.draw_pile) == 9
    assert st.trump_card is not None

    # Force a known trick: leader plays a low card, responder beats in same color
    lead = Card(Color.HEARTS, 2)
    resp = Card(Color.HEARTS, 11)
    # Put these into hands deterministically
    st.hands[0][0] = lead
    st.hands[1][0] = resp

    # Make top of draw pile identifiable
    top1 = Card(Color.BELLS, 2)
    top2 = Card(Color.BELLS, 3)
    st.draw_pile[0] = top1
    st.draw_pile[1] = top2

    st, result = play_trick(st, lead, resp)
    assert result.winner == 1  # responder wins

    # both should be back at 5 (pile still non-empty)
    assert len(st.hands[0]) == 5
    assert len(st.hands[1]) == 5

    # Winner draws first -> winner should contain top1, loser gets top2
    assert top1 in st.hands[1]
    assert top2 in st.hands[0]

