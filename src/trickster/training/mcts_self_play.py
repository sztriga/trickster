"""MCTS-guided self-play training with experience replay and optional parallelism."""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional, Tuple

from trickster.games.snapszer.features import (
    follow_action_features,
    lead_action_features,
    lead_close_talon_features,
)
from trickster.games.snapszer.game import (
    close_talon,
    deal,
    deal_winner,
    is_terminal,
    play_trick,
)
from trickster.games.snapszer.mcts_agent import MCTSConfig, mcts_choose
from trickster.training.model_spec import ModelSpec
from trickster.training.policy import TrainedPolicy, create_policy
from trickster.training.replay_buffer import ReplayBuffer
from trickster.training.self_play import TrainStats

# Type aliases
Experience = Tuple[Dict[str, float], float]  # (features, target_y)
GameResult = Tuple[Optional[int], Tuple[int, int]]  # (winner, (score0, score1))


# ---------------------------------------------------------------------------
#  Single-game logic (used by both sequential and parallel paths)
# ---------------------------------------------------------------------------

def _play_single_game(
    policy: TrainedPolicy,
    mcts_config: MCTSConfig,
    game_seed: int,
    ep_idx: int,
) -> Tuple[List[Experience], List[Experience], GameResult]:
    """Play one full game with MCTS; return (lead_exps, follow_exps, result).

    This function is self-contained and safe to call in a worker process.
    """
    rng = random.Random(game_seed)
    st = deal(seed=game_seed, starting_leader=ep_idx % 2)
    agent_rngs = [
        random.Random(rng.randrange(1 << 30)),
        random.Random(rng.randrange(1 << 30)),
    ]

    lead_traces: list[tuple] = []
    follow_traces: list[tuple] = []

    while not is_terminal(st):
        leader = st.leader
        responder = 1 - leader

        hand_bc = tuple(st.hands[leader])
        draw_bc = len(st.draw_pile)
        cap0_bc = tuple(st.captured[0])
        cap1_bc = tuple(st.captured[1])
        trump_up_bc = st.trump_card if st.trump_upcard_visible else None

        action = mcts_choose(st, None, leader, mcts_config, agent_rngs[leader])

        if action == "close_talon":
            lead_traces.append(
                (leader, hand_bc, draw_bc, cap0_bc, cap1_bc, trump_up_bc, None, "close_talon")
            )
            close_talon(st, leader)
            action = mcts_choose(st, None, leader, mcts_config, agent_rngs[leader])

        lead_card = action

        hand_bl = tuple(st.hands[leader])
        draw_bl = len(st.draw_pile)
        cap0_bl = tuple(st.captured[0])
        cap1_bl = tuple(st.captured[1])
        trump_up_bl = st.trump_card if st.trump_upcard_visible else None

        resp_action = mcts_choose(st, lead_card, responder, mcts_config, agent_rngs[responder])
        resp_card = resp_action

        resp_hand_before = tuple(st.hands[responder])

        lead_traces.append(
            (leader, hand_bl, draw_bl, cap0_bl, cap1_bl, trump_up_bl, lead_card, "card")
        )
        follow_traces.append(
            (responder, resp_hand_before, draw_bl, cap0_bl, cap1_bl, trump_up_bl, lead_card, resp_card)
        )

        st, _ = play_trick(st, lead_card, resp_card)

    # -- Compute outcome --
    winner = deal_winner(st)
    if winner is None:
        y0, y1 = 0.5, 0.5
    else:
        y0 = 1.0 if winner == 0 else 0.0
        y1 = 1.0 if winner == 1 else 0.0

    # -- Convert traces to experiences --
    lead_exps: list[Experience] = []
    for player, hand_before, draw_before, cap0_before, cap1_before, trump_up_before, card, kind in lead_traces:
        y = y0 if player == 0 else y1
        cap_self = cap0_before if player == 0 else cap1_before
        cap_opp = cap1_before if player == 0 else cap0_before
        if kind == "close_talon":
            features = lead_close_talon_features(
                hand_before,
                draw_pile_size=draw_before,
                captured_self=cap_self,
                captured_opp=cap_opp,
                trump_color=st.trump_color,
                trump_upcard=trump_up_before,
            )
        else:
            assert card is not None
            features = lead_action_features(
                hand_before,
                card,
                draw_pile_size=draw_before,
                captured_self=cap_self,
                captured_opp=cap_opp,
                trump_color=st.trump_color,
                trump_upcard=trump_up_before,
                exchanged_trump=False,
            )
        lead_exps.append((features, y))

    follow_exps: list[Experience] = []
    for player, hand_before, draw_before, cap0_before, cap1_before, trump_up_before, lc, card in follow_traces:
        y = y0 if player == 0 else y1
        cap_self = cap0_before if player == 0 else cap1_before
        cap_opp = cap1_before if player == 0 else cap0_before
        features = follow_action_features(
            hand_before,
            lc,
            card,
            draw_pile_size=draw_before,
            captured_self=cap_self,
            captured_opp=cap_opp,
            trump_color=st.trump_color,
            trump_upcard=trump_up_before,
        )
        follow_exps.append((features, y))

    return lead_exps, follow_exps, (winner, (st.scores[0], st.scores[1]))


# ---------------------------------------------------------------------------
#  Worker function for multiprocessing (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _worker_batch(args: tuple) -> Tuple[List[Experience], List[Experience], List[GameResult]]:
    """Play a batch of games in a worker process."""
    policy, mcts_config, game_specs = args
    all_lead: list[Experience] = []
    all_follow: list[Experience] = []
    results: list[GameResult] = []
    for game_seed, ep_idx in game_specs:
        le, fe, res = _play_single_game(policy, mcts_config, game_seed, ep_idx)
        all_lead.extend(le)
        all_follow.extend(fe)
        results.append(res)
    return all_lead, all_follow, results


# ---------------------------------------------------------------------------
#  Replay training helper
# ---------------------------------------------------------------------------

def _replay_train(
    model,
    buf: ReplayBuffer,
    batch_size: int,
    updates: int,
    rng: random.Random,
    lr: float,
    l2: float,
) -> None:
    """Sample mini-batches from *buf* and run vectorised SGD updates on *model*."""
    if len(buf) < batch_size:
        return
    for _ in range(updates):
        batch = buf.sample(batch_size, rng)
        features_list = [f for f, _ in batch]
        ys = [y for _, y in batch]
        model.batch_update(features_list, ys, lr=lr, l2=l2)


# ---------------------------------------------------------------------------
#  Stats helper
# ---------------------------------------------------------------------------

def _update_stats(stats: TrainStats, results: List[GameResult]) -> None:
    for winner, (s0, s1) in results:
        stats.episodes += 1
        stats.last_score0, stats.last_score1 = s0, s1
        if winner == 0:
            stats.p0_wins += 1
        elif winner == 1:
            stats.p1_wins += 1
        else:
            stats.draws += 1


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def train_mcts_self_play(
    *,
    spec: ModelSpec,
    episodes: int,
    mcts_config: MCTSConfig,
    seed: int = 0,
    lr: float = 0.05,
    l2: float = 1e-6,
    buffer_capacity: int = 50_000,
    batch_size: int = 128,
    updates_per_game: int = 4,
    workers: int = 1,
    initial_policy: Optional[TrainedPolicy] = None,
    on_progress: Optional[Callable[[int, TrainStats], None]] = None,
) -> Tuple[TrainedPolicy, TrainStats]:
    """
    Train via MCTS-guided self-play with experience replay.

    When ``workers > 1``, self-play games are distributed across processes.
    Each round, every worker plays a batch of games with the *current* model
    weights, sends experiences back, and the main process trains on the
    combined replay buffer before syncing fresh weights to the next round.

    Parameters
    ----------
    workers : int
        Number of parallel self-play processes.  ``1`` = sequential (no
        multiprocessing overhead).  On a multi-core machine, set to the
        number of performance cores for near-linear speedup.
    """
    rng = random.Random(seed)
    policy = initial_policy or create_policy(spec)
    lead_model = policy.lead_model
    follow_model = policy.follow_model
    stats = TrainStats()

    lead_buf = ReplayBuffer(capacity=buffer_capacity)
    follow_buf = ReplayBuffer(capacity=buffer_capacity)

    if workers <= 1:
        # ---- Sequential path (no multiprocessing overhead) ----
        for ep in range(episodes):
            le, fe, res = _play_single_game(policy, mcts_config, seed + ep, ep)

            for feat, y in le:
                lead_buf.add(feat, y)
            for feat, y in fe:
                follow_buf.add(feat, y)

            _update_stats(stats, [res])

            _replay_train(lead_model, lead_buf, batch_size, updates_per_game, rng, lr, l2)
            _replay_train(follow_model, follow_buf, batch_size, updates_per_game, rng, lr, l2)

            if on_progress is not None:
                on_progress(ep + 1, stats)
    else:
        # ---- Parallel path ----
        # Games per worker per round: balance freshness vs overhead.
        games_per_worker = max(1, min(10, episodes // max(1, workers * 10)))
        games_per_round = games_per_worker * workers
        total_played = 0

        with mp.Pool(workers) as pool:
            while total_played < episodes:
                remaining = episodes - total_played
                this_round = min(games_per_round, remaining)

                # Distribute games across workers
                specs_per_worker: list[list[tuple[int, int]]] = [[] for _ in range(workers)]
                for i in range(this_round):
                    ep_idx = total_played + i
                    game_seed = seed + ep_idx
                    specs_per_worker[i % workers].append((game_seed, ep_idx))

                tasks = [
                    (policy, mcts_config, specs)
                    for specs in specs_per_worker
                    if specs
                ]
                batch_results = pool.map(_worker_batch, tasks)

                # Collect experiences and stats
                round_results: list[GameResult] = []
                for lead_exps, follow_exps, game_results in batch_results:
                    for feat, y in lead_exps:
                        lead_buf.add(feat, y)
                    for feat, y in follow_exps:
                        follow_buf.add(feat, y)
                    round_results.extend(game_results)

                _update_stats(stats, round_results)
                total_played += this_round

                # Train on replay buffer (proportional to games played)
                n_updates = updates_per_game * this_round
                _replay_train(lead_model, lead_buf, batch_size, n_updates, rng, lr, l2)
                _replay_train(follow_model, follow_buf, batch_size, n_updates, rng, lr, l2)

                if on_progress is not None:
                    on_progress(total_played, stats)

    return policy, stats
