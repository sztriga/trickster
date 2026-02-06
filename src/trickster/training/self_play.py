from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from trickster.games.snapszer.agent import LearnedAgent
from trickster.games.snapszer.features import follow_action_features, lead_action_features, lead_close_talon_features
from trickster.training.policy import TrainedPolicy, create_policy
from trickster.games.snapszer.game import can_close_talon, close_talon, deal, deal_winner, is_terminal, legal_actions, play_trick
from trickster.training.model_spec import ModelSpec


@dataclass(slots=True)
class TrainStats:
    episodes: int = 0
    p0_wins: int = 0
    p1_wins: int = 0
    draws: int = 0
    last_score0: int = 0
    last_score1: int = 0

    @property
    def winrate_p0(self) -> float:
        if self.episodes <= 0:
            return 0.0
        return self.p0_wins / self.episodes


def save_policy(policy: TrainedPolicy, path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(policy, f)


def load_policy(path: str | os.PathLike[str]) -> TrainedPolicy:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def train_self_play(
    *,
    spec: ModelSpec,
    episodes: int,
    seed: int = 0,
    lr: float = 0.05,
    l2: float = 1e-6,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.02,
    initial_policy: Optional[TrainedPolicy] = None,
    on_progress: Optional[Callable[[int, TrainStats], None]] = None,
) -> Tuple[TrainedPolicy, TrainStats]:
    """
    Trains two lightweight models:
    - lead_model: choose a lead card
    - follow_model: choose a response card

    Learning target is the final *game* outcome (win/lose/draw) for the player who
    took the action (simple Monte Carlo credit assignment).
    """

    rng = random.Random(seed)
    policy = initial_policy or create_policy(spec)
    lead_model = policy.lead_model
    follow_model = policy.follow_model
    stats = TrainStats()

    for ep in range(episodes):
        # linear epsilon schedule
        t = 0.0 if episodes <= 1 else ep / (episodes - 1)
        eps = epsilon_start + (epsilon_end - epsilon_start) * t

        st = deal(seed=seed + ep, starting_leader=ep % 2)
        agents = [
            LearnedAgent(lead_model, follow_model, random.Random(rng.randrange(1 << 30)), epsilon=eps),
            LearnedAgent(lead_model, follow_model, random.Random(rng.randrange(1 << 30)), epsilon=eps),
        ]

        # Collect (state, action) pairs and assign credit after the game ends.
        #
        # IMPORTANT: features must only use information available at decision time, so we
        # snapshot public context (captures, talon size, visible trump upcard).
        lead_traces: list[tuple[int, tuple, int, tuple, tuple, Card | None, object, str]] = []
        follow_traces: list[tuple[int, tuple, int, tuple, tuple, Card | None, object, object]] = []

        while not is_terminal(st):
            leader = st.leader
            responder = 1 - leader

            lead_hand = st.hands[leader]
            # Optional learned takarás decision before selecting the lead card.
            lead_hand_before_close = tuple(lead_hand)
            draw_before_close = int(len(st.draw_pile))
            cap0_before_close = tuple(st.captured[0])
            cap1_before_close = tuple(st.captured[1])
            trump_up_before_close = st.trump_card if getattr(st, "trump_upcard_visible", True) else None

            can_close = can_close_talon(st, leader)
            lead_legal_pre = legal_actions(st, leader, lead_card=None)
            do_close, _tmp_card = agents[leader].choose_lead_or_close_talon(
                lead_hand,
                lead_legal_pre,
                can_close_talon=can_close,
                draw_pile_size=len(st.draw_pile),
                captured_self=st.captured[leader],
                captured_opp=st.captured[responder],
                trump_color=st.trump_color,
                trump_upcard=trump_up_before_close,
            )
            if do_close and can_close:
                # Record a lead-trace for the close action itself.
                lead_traces.append(
                    (
                        leader,
                        lead_hand_before_close,
                        draw_before_close,
                        cap0_before_close,
                        cap1_before_close,
                        trump_up_before_close,
                        None,
                        "close_talon",
                    )
                )
                close_talon(st, leader)

            # Now select the actual lead card (possibly after closing).
            lead_legal = legal_actions(st, leader, lead_card=None)
            trump_up_for_lead = st.trump_card if getattr(st, "trump_upcard_visible", True) else None
            lead_card = agents[leader].choose_lead(
                lead_hand,
                lead_legal,
                draw_pile_size=len(st.draw_pile),
                captured_self=st.captured[leader],
                captured_opp=st.captured[responder],
                trump_color=st.trump_color,
                trump_upcard=trump_up_for_lead,
            )

            resp_hand = st.hands[responder]
            resp_legal = legal_actions(st, responder, lead_card=lead_card)
            resp_card = agents[responder].choose_follow(
                resp_hand,
                lead_card,
                resp_legal,
                draw_pile_size=len(st.draw_pile),
                captured_self=st.captured[responder],
                captured_opp=st.captured[leader],
                trump_color=st.trump_color,
                trump_upcard=st.trump_card,
            )

            # IMPORTANT: `play_trick` mutates `st` in-place (removes cards and draws).
            # Snapshot hands *before* applying the action so the features match the
            # action that was actually chosen.
            lead_hand_before = tuple(lead_hand)
            resp_hand_before = tuple(resp_hand)

            # Store traces for game-level credit assignment.
            draw_before = int(len(st.draw_pile))
            cap0_before = tuple(st.captured[0])
            cap1_before = tuple(st.captured[1])
            trump_up_before = st.trump_card if getattr(st, "trump_upcard_visible", True) else None
            lead_traces.append((leader, lead_hand_before, draw_before, cap0_before, cap1_before, trump_up_before, lead_card, "card"))
            follow_traces.append(
                (responder, resp_hand_before, draw_before, cap0_before, cap1_before, trump_up_before, lead_card, resp_card)
            )

            st, _result = play_trick(st, lead_card, resp_card)

        # Deal winner under the current rules (includes 66, takarás, last-trick win).
        winner = deal_winner(st)

        # Apply Monte Carlo updates (final game outcome).
        if winner is None:
            y0 = 0.5
            y1 = 0.5
        else:
            y0 = 1.0 if winner == 0 else 0.0
            y1 = 1.0 if winner == 1 else 0.0

        for player, hand_before, draw_before, cap0_before, cap1_before, trump_up_before, card, kind in lead_traces:
            y = y0 if player == 0 else y1
            cap_self = cap0_before if player == 0 else cap1_before
            cap_opp = cap1_before if player == 0 else cap0_before
            if kind == "close_talon":
                lead_model.update(
                    lead_close_talon_features(
                        hand_before,
                        draw_pile_size=draw_before,
                        captured_self=cap_self,
                        captured_opp=cap_opp,
                        trump_color=st.trump_color,
                        trump_upcard=trump_up_before,
                    ),
                    y,
                    lr=lr,
                    l2=l2,
                )
            else:
                assert card is not None
                lead_model.update(
                    lead_action_features(
                        hand_before,
                        card,
                        draw_pile_size=draw_before,
                        captured_self=cap_self,
                        captured_opp=cap_opp,
                        trump_color=st.trump_color,
                        trump_upcard=trump_up_before,
                        exchanged_trump=False,
                    ),
                    y,
                    lr=lr,
                    l2=l2,
                )
        for player, hand_before, draw_before, cap0_before, cap1_before, trump_up_before, lead_card, card in follow_traces:
            y = y0 if player == 0 else y1
            cap_self = cap0_before if player == 0 else cap1_before
            cap_opp = cap1_before if player == 0 else cap0_before
            follow_model.update(
                follow_action_features(
                    hand_before,
                    lead_card,
                    card,
                    draw_pile_size=draw_before,
                    captured_self=cap_self,
                    captured_opp=cap_opp,
                    trump_color=st.trump_color,
                    trump_upcard=trump_up_before,
                ),
                y,
                lr=lr,
                l2=l2,
            )

        stats.episodes += 1
        s0, s1 = st.scores
        stats.last_score0 = s0
        stats.last_score1 = s1
        if winner == 0:
            stats.p0_wins += 1
        elif winner == 1:
            stats.p1_wins += 1
        else:
            stats.draws += 1

        if on_progress is not None:
            on_progress(ep + 1, stats)

    return policy, stats



