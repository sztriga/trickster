from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import Optional

from trickster.games.snapszer.agent import LearnedAgent
from trickster.games.snapszer.game import (
    deal, deal_awarded_game_points, deal_winner, is_terminal, legal_actions, play_trick,
)
from trickster.training.policy import TrainedPolicy


@dataclass(slots=True)
class EvalStats:
    """Tracks game-point totals over a set of deals."""

    deals: int = 0
    a_points: int = 0
    b_points: int = 0

    @property
    def a_ppd(self) -> float:
        """A's average points per deal."""
        return 0.0 if self.deals == 0 else self.a_points / self.deals

    @property
    def b_ppd(self) -> float:
        """B's average points per deal."""
        return 0.0 if self.deals == 0 else self.b_points / self.deals


def _agent_from_policy(policy: TrainedPolicy, seed: int) -> LearnedAgent:
    return LearnedAgent(policy.lead_model, policy.follow_model, random.Random(seed), epsilon=0.0)


def _random_action(rng: random.Random, legal) -> object:
    lst = list(legal)
    return rng.choice(lst)


def _mix_seed(seed: int, game_index: int, tag: int) -> int:
    x = (seed & 0xFFFFFFFF) ^ ((game_index * 0x9E3779B1) & 0xFFFFFFFF) ^ ((tag * 0x85EBCA6B) & 0xFFFFFFFF)
    x ^= (x >> 16) & 0xFFFFFFFF
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15) & 0xFFFFFFFF
    return x & 0x7FFFFFFF


def _score_deal(st, a_idx: int, stats: EvalStats) -> None:
    """Score a finished deal and update stats in-place."""
    winner, pts, _reason = deal_awarded_game_points(st)
    stats.deals += 1
    if winner == a_idx:
        stats.a_points += pts
    else:
        stats.b_points += pts


def _merge_stats(target: EvalStats, part: EvalStats) -> None:
    target.deals += part.deals
    target.a_points += part.a_points
    target.b_points += part.b_points


# ---------------------------------------------------------------------------
# Policy A vs Policy B
# ---------------------------------------------------------------------------

def _eval_range(policy_a: TrainedPolicy, policy_b: TrainedPolicy, *, start_g: int, games: int, seed: int) -> EvalStats:
    stats = EvalStats()
    for g in range(start_g, start_g + games):
        if g % 2 == 0:
            policies = (policy_a, policy_b)
            a_idx = 0
        else:
            policies = (policy_b, policy_a)
            a_idx = 1

        agents = [
            _agent_from_policy(policies[0], _mix_seed(seed, g, 0)),
            _agent_from_policy(policies[1], _mix_seed(seed, g, 1)),
        ]

        base_seed = seed + (g // 2)
        st = deal(seed=base_seed, starting_leader=0)
        while not is_terminal(st):
            leader = st.leader
            responder = 1 - leader
            lead_card = agents[leader].choose_lead(
                st.hands[leader], legal_actions(st, leader, None),
                draw_pile_size=len(st.draw_pile),
                captured_self=st.captured[leader], captured_opp=st.captured[responder],
                trump_color=st.trump_color, trump_upcard=st.trump_card,
            )
            resp_card = agents[responder].choose_follow(
                st.hands[responder], lead_card, legal_actions(st, responder, lead_card),
                draw_pile_size=len(st.draw_pile),
                captured_self=st.captured[responder], captured_opp=st.captured[leader],
                trump_color=st.trump_color, trump_upcard=st.trump_card,
            )
            st, _ = play_trick(st, lead_card, resp_card)

        _score_deal(st, a_idx, stats)
    return stats


_WORKER_POLICY_A: Optional[TrainedPolicy] = None
_WORKER_POLICY_B: Optional[TrainedPolicy] = None
_WORKER_SEED: int = 0


def _worker_init(policy_a: TrainedPolicy, policy_b: TrainedPolicy, seed: int) -> None:
    global _WORKER_POLICY_A, _WORKER_POLICY_B, _WORKER_SEED
    _WORKER_POLICY_A = policy_a
    _WORKER_POLICY_B = policy_b
    _WORKER_SEED = seed


def _worker_eval(args: tuple[int, int]) -> EvalStats:
    start_g, n_games = args
    assert _WORKER_POLICY_A is not None and _WORKER_POLICY_B is not None
    return _eval_range(_WORKER_POLICY_A, _WORKER_POLICY_B, start_g=start_g, games=n_games, seed=_WORKER_SEED)


def evaluate_policies_parallel(
    policy_a: TrainedPolicy, policy_b: TrainedPolicy, *,
    games: int = 1000, seed: int = 0, workers: int = 2, chunk_games: int = 250,
    on_progress: Optional[callable] = None,
) -> EvalStats:
    if workers <= 1 or games <= 0:
        return evaluate_policies(policy_a, policy_b, games=games, seed=seed, on_progress=on_progress)
    workers = max(1, int(workers))
    chunk_games = max(1, int(chunk_games))
    tasks = _make_tasks(games, chunk_games)
    ctx = mp.get_context("spawn")
    stats = EvalStats()
    with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(policy_a, policy_b, seed)) as pool:
        for part in pool.imap_unordered(_worker_eval, tasks):
            _merge_stats(stats, part)
            if on_progress is not None:
                on_progress(stats)
    return stats


def evaluate_policies(
    policy_a: TrainedPolicy, policy_b: TrainedPolicy, *,
    games: int = 1000, seed: int = 0, on_progress: Optional[callable] = None,
) -> EvalStats:
    stats = _eval_range(policy_a, policy_b, start_g=0, games=games, seed=seed)
    if on_progress is not None:
        on_progress(stats)
    return stats


# ---------------------------------------------------------------------------
# Policy A vs random
# ---------------------------------------------------------------------------

def _eval_range_vs_random(policy: TrainedPolicy, *, start_g: int, games: int, seed: int) -> EvalStats:
    stats = EvalStats()
    for g in range(start_g, start_g + games):
        if g % 2 == 0:
            a_idx, policy_idx = 0, 0
        else:
            a_idx, policy_idx = 1, 1

        pol_agent = _agent_from_policy(policy, _mix_seed(seed, g, 10))
        rnd_rng = random.Random(_mix_seed(seed, g, 11))
        base_seed = seed + (g // 2)
        st = deal(seed=base_seed, starting_leader=0)

        while not is_terminal(st):
            leader = st.leader
            responder = 1 - leader
            if leader == policy_idx:
                lead_card = pol_agent.choose_lead(
                    st.hands[leader], legal_actions(st, leader, None),
                    draw_pile_size=len(st.draw_pile),
                    captured_self=st.captured[leader], captured_opp=st.captured[responder],
                    trump_color=st.trump_color, trump_upcard=st.trump_card,
                )
            else:
                lead_card = _random_action(rnd_rng, legal_actions(st, leader, None))
            if responder == policy_idx:
                resp_card = pol_agent.choose_follow(
                    st.hands[responder], lead_card, legal_actions(st, responder, lead_card),
                    draw_pile_size=len(st.draw_pile),
                    captured_self=st.captured[responder], captured_opp=st.captured[leader],
                    trump_color=st.trump_color, trump_upcard=st.trump_card,
                )
            else:
                resp_card = _random_action(rnd_rng, legal_actions(st, responder, lead_card))
            st, _ = play_trick(st, lead_card, resp_card)

        _score_deal(st, a_idx, stats)
    return stats


_WORKER_POLICY: Optional[TrainedPolicy] = None
_WORKER_SEED_VR: int = 0


def _worker_init_vs_random(policy: TrainedPolicy, seed: int) -> None:
    global _WORKER_POLICY, _WORKER_SEED_VR
    _WORKER_POLICY = policy
    _WORKER_SEED_VR = seed


def _worker_eval_vs_random(args: tuple[int, int]) -> EvalStats:
    start_g, n_games = args
    assert _WORKER_POLICY is not None
    return _eval_range_vs_random(_WORKER_POLICY, start_g=start_g, games=n_games, seed=_WORKER_SEED_VR)


def evaluate_policy_vs_random_parallel(
    policy: TrainedPolicy, *, games: int = 1000, seed: int = 0,
    workers: int = 2, chunk_games: int = 250, on_progress: Optional[callable] = None,
) -> EvalStats:
    if workers <= 1 or games <= 0:
        return evaluate_policy_vs_random(policy, games=games, seed=seed, on_progress=on_progress)
    workers = max(1, int(workers))
    chunk_games = max(1, int(chunk_games))
    tasks = _make_tasks(games, chunk_games)
    ctx = mp.get_context("spawn")
    stats = EvalStats()
    with ctx.Pool(processes=workers, initializer=_worker_init_vs_random, initargs=(policy, seed)) as pool:
        for part in pool.imap_unordered(_worker_eval_vs_random, tasks):
            _merge_stats(stats, part)
            if on_progress is not None:
                on_progress(stats)
    return stats


def evaluate_policy_vs_random(
    policy: TrainedPolicy, *, games: int = 1000, seed: int = 0,
    on_progress: Optional[callable] = None,
) -> EvalStats:
    stats = _eval_range_vs_random(policy, start_g=0, games=games, seed=seed)
    if on_progress is not None:
        on_progress(stats)
    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tasks(games: int, chunk_games: int) -> list[tuple[int, int]]:
    tasks: list[tuple[int, int]] = []
    start = 0
    while start < games:
        n = min(chunk_games, games - start)
        tasks.append((start, n))
        start += n
    return tasks
