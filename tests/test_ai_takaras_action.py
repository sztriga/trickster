from __future__ import annotations

import random

from trickster.games.snapszer.agent import LearnedAgent
from trickster.models.linear import LinearBinaryModel
from trickster.games.snapszer.cards import Card, Color
from trickster.games.snapszer.features import lead_action_features, lead_close_talon_features


def test_lead_model_can_score_close_talon_action_higher_than_cards() -> None:
    model = LinearBinaryModel()
    # Bias the model strongly toward "close talon".
    model.weights["__bias__"] = 0.0
    model.weights["a.kind.close_talon"] = 10.0
    model.weights["a.kind.card"] = -10.0

    agent = LearnedAgent(lead_model=model, follow_model=model, rng=random.Random(0), epsilon=0.0)

    hand = [Card(Color.HEARTS, 11), Card(Color.HEARTS, 10), Card(Color.BELLS, 4), Card(Color.ACORNS, 3), Card(Color.LEAVES, 2)]
    legal = list(hand)

    do_close, chosen = agent.choose_lead_or_close_talon(
        hand,
        legal,
        can_close_talon=True,
        draw_pile_size=9,
        captured_self=(),
        captured_opp=(),
        trump_color=Color.HEARTS,
        trump_upcard=Card(Color.HEARTS, 2),
    )
    assert do_close is True
    assert chosen in legal


def test_close_talon_features_do_not_depend_on_hidden_upcard_when_none() -> None:
    hand = [Card(Color.HEARTS, 11)]
    feats_none = lead_close_talon_features(
        hand,
        draw_pile_size=5,
        captured_self=(),
        captured_opp=(),
        trump_color=Color.BELLS,
        trump_upcard=None,
    )
    feats_some = lead_action_features(
        hand,
        Card(Color.HEARTS, 11),
        draw_pile_size=5,
        captured_self=(),
        captured_opp=(),
        trump_color=Color.BELLS,
        trump_upcard=None,
        exchanged_trump=False,
    )
    # Ensure "upcard present" feature is 0 when upcard is hidden (None) for both action kinds.
    assert feats_none["pub.trump_up.present"] == 0.0
    assert feats_some["pub.trump_up.present"] == 0.0

