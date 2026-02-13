"""Hybrid MCTS + alpha-beta training engine for Ulti.

Shared training loop used by all Ulti training scripts.  Follows the
same pattern as ``trickster.training.alpha_zero`` for Snapszer: the
engine owns the full training loop (self-play → buffer → SGD) with
parallel workers, and callers pass configuration.

Features consolidated from train_baseline.py and train_ulti.py:
  - Parallel self-play via ProcessPoolExecutor (``num_workers``)
  - Cosine LR annealing (``lr_start`` / ``lr_end``)
  - Decoupled SGD steps (fixed ``train_steps`` per iteration)
  - Deal enrichment via value head (``enrich_*`` params)
  - Per-role diagnostics (soloist vs defender value loss)
  - Policy accuracy tracking
  - Opponent pool support (``opponent_state_dict``)
  - Checkpoint load / resume (``initial_net``)
  - Progress callback (``on_progress``)

Usage (from a tier script)::

    from trickster.training.ulti_hybrid import train_ulti_hybrid, UltiTrainConfig

    cfg = UltiTrainConfig(steps=500, games_per_step=8, ...)
    net, stats = train_ulti_hybrid(cfg)
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, UltiNetWrapper
from trickster.train_utils import ReplayBuffer, simple_outcome


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class UltiTrainConfig:
    """All hyperparameters for one training run.

    Tier scripts build this from their tier definitions and pass it
    to ``train_ulti_hybrid()``.
    """

    # -- Self-play budget --
    steps: int = 500
    games_per_step: int = 8

    # -- SGD --
    train_steps: int = 50          # SGD steps per iteration
    batch_size: int = 64
    buffer_size: int = 50_000
    lr_start: float = 1e-3
    lr_end: float = 2e-4           # cosine decay target (set = lr_start to disable)

    # -- MCTS (soloist / defender) --
    sol_sims: int = 20
    sol_dets: int = 1
    def_sims: int = 8
    def_dets: int = 1

    # -- Solver / hybrid --
    endgame_tricks: int = 6
    pimc_dets: int = 20
    solver_temp: float = 0.5

    # -- Network architecture --
    body_units: int = 256
    body_layers: int = 4

    # -- Parallelism --
    num_workers: int = 1

    # -- Deal enrichment --
    enrichment: bool = False
    enrich_warmup: int = 20        # skip enrichment for first N steps
    enrich_random_frac: float = 0.3  # fraction of games always random

    # -- General --
    seed: int = 42
    device: str = "cpu"
    training_mode: str = "simple"  # "simple" | "betli" | "ulti" | etc.

    @property
    def total_games(self) -> int:
        return self.steps * self.games_per_step


# ---------------------------------------------------------------------------
#  Training stats (returned to caller + passed to on_progress)
# ---------------------------------------------------------------------------


@dataclass
class UltiTrainStats:
    """Cumulative training diagnostics."""

    step: int = 0
    total_steps: int = 0
    total_games: int = 0
    total_samples: int = 0
    total_sgd_steps: int = 0

    # Current step losses
    vloss: float = 0.0
    ploss: float = 0.0
    pacc: float = 0.0
    sol_vloss: float = 0.0
    def_vloss: float = 0.0
    lr: float = 0.0

    # Rolling history (for trend detection)
    history_vloss: list = field(default_factory=list)
    history_ploss: list = field(default_factory=list)
    history_pacc: list = field(default_factory=list)

    # Self-play win rate (current step)
    sp_sol_wins: int = 0
    sp_sol_total: int = 0

    train_time_s: float = 0.0


# ---------------------------------------------------------------------------
#  Multiprocessing helpers (module-level for pickling)
# ---------------------------------------------------------------------------

_WORKER_GAME: UltiGame | None = None
_WORKER_NET: UltiNet | None = None
_WORKER_WRAPPER: UltiNetWrapper | None = None
_WORKER_OPP_NET: UltiNet | None = None
_WORKER_OPP_WRAPPER: UltiNetWrapper | None = None


def _init_worker(net_kwargs: dict, device: str) -> None:
    """Called once per worker process to create game + networks."""
    global _WORKER_GAME, _WORKER_NET, _WORKER_WRAPPER
    global _WORKER_OPP_NET, _WORKER_OPP_WRAPPER
    _WORKER_GAME = UltiGame()
    _WORKER_NET = UltiNet(**net_kwargs)
    _WORKER_WRAPPER = UltiNetWrapper(_WORKER_NET, device=device)
    _WORKER_OPP_NET = UltiNet(**net_kwargs)
    _WORKER_OPP_WRAPPER = UltiNetWrapper(_WORKER_OPP_NET, device=device)


def _play_game_in_worker(
    args: tuple,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Worker entry-point for parallel self-play."""
    (state_dict, opp_state_dict, sol_mcts_cfg, def_mcts_cfg,
     seed, endgame_tricks, pimc_dets, solver_temp,
     training_mode, enrich_thresh) = args
    _WORKER_NET.load_state_dict(state_dict)
    opp_wrapper = None
    if opp_state_dict is not None:
        _WORKER_OPP_NET.load_state_dict(opp_state_dict)
        opp_wrapper = _WORKER_OPP_WRAPPER
    init_state = None
    if enrich_thresh > -999.0:
        init_state, _ = _value_enriched_new_game(
            _WORKER_GAME, _WORKER_WRAPPER, seed,
            min_value=enrich_thresh, training_mode=training_mode,
        )
    return _play_one_game(
        _WORKER_GAME, _WORKER_WRAPPER, sol_mcts_cfg, def_mcts_cfg, seed,
        endgame_tricks=endgame_tricks,
        pimc_dets=pimc_dets,
        solver_temp=solver_temp,
        training_mode=training_mode,
        opponent_wrapper=opp_wrapper,
        initial_state=init_state,
    )


# ---------------------------------------------------------------------------
#  Deal enrichment via value head
# ---------------------------------------------------------------------------


def _value_enriched_new_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    seed: int,
    min_value: float,
    training_mode: str = "simple",
    max_attempts: int = 20,
) -> tuple[UltiNode, float]:
    """Deal games until the value head rates the soloist above *min_value*.

    Returns (state, soloist_value) — the best deal found, or the last
    attempt if none pass the threshold.
    """
    best_state = None
    best_val = -1.0
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt * 100_000
        state = game.new_game(
            seed=attempt_seed,
            training_mode=training_mode,
            starting_leader=seed % 3,
        )
        sol = state.gs.soloist
        feats = game.encode_state(state, sol)
        val = wrapper.predict_value(feats)
        if val > best_val:
            best_state, best_val = state, val
        if val >= min_value:
            return state, val
    return best_state, best_val


def _enrichment_threshold(step: int, total_steps: int) -> float:
    """Anneal the minimum soloist value from 0.0 -> -1.0 over training.

    Phase 1 (first 25%):   v >= 0.0   (only "even-ish or better" deals)
    Phase 2 (25%-60%):     v >= -0.25  (slightly losing included)
    Phase 3 (60%+):        disabled    (all deals accepted)

    Returns the threshold, or -999 to signal "accept everything".
    """
    frac = step / max(1, total_steps)
    if frac < 0.25:
        return 0.0
    if frac < 0.60:
        return -0.25
    return -999.0


# ---------------------------------------------------------------------------
#  Self-play: one full game -> training samples
# ---------------------------------------------------------------------------


def _play_one_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    sol_mcts_config: MCTSConfig,
    def_mcts_config: MCTSConfig,
    seed: int,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
    solver_temp: float = 1.0,
    training_mode: str = "simple",
    opponent_wrapper: UltiNetWrapper | None = None,
    initial_state: UltiNode | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Play one hybrid self-play game.

    Uses HybridPlayer for all seats: neural MCTS for opening tricks,
    PIMC + exact alpha-beta for the endgame.

    If *initial_state* is provided (e.g. from deal enrichment),
    it is used directly instead of dealing a new game.
    """
    rng = random.Random(seed)

    if initial_state is not None:
        state = initial_state
    else:
        state = game.new_game(
            seed=seed,
            training_mode=training_mode,
            starting_leader=seed % 3,
        )
    soloist_idx = state.gs.soloist

    sol_hybrid = HybridPlayer(
        game, wrapper,
        mcts_config=sol_mcts_config,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )
    def_wrapper = opponent_wrapper if opponent_wrapper is not None else wrapper
    def_hybrid = HybridPlayer(
        game, def_wrapper,
        mcts_config=def_mcts_config,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )

    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        if player == soloist_idx:
            pi, action = sol_hybrid.choose_action_with_policy(state, player, rng)
        else:
            pi, action = def_hybrid.choose_action_with_policy(state, player, rng)

        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append((
            state_feats.copy(),
            mask.copy(),
            np.asarray(pi, dtype=np.float32).copy(),
            player,
        ))

        state = game.apply(state, action)

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for state_feats, mask, pi, player in trajectory:
        reward = simple_outcome(state, player)
        is_sol = (player == soloist_idx)
        samples.append((state_feats, mask, pi, reward, is_sol))

    return samples


# ---------------------------------------------------------------------------
#  Cosine LR schedule
# ---------------------------------------------------------------------------


def _cosine_lr(step: int, total_steps: int, lr_start: float, lr_end: float) -> float:
    if total_steps <= 1:
        return lr_start
    frac = step / total_steps
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * frac))


# ---------------------------------------------------------------------------
#  Main training entry point
# ---------------------------------------------------------------------------


def train_ulti_hybrid(
    cfg: UltiTrainConfig,
    *,
    initial_net: UltiNet | None = None,
    opponent_state_dict: dict | None = None,
    on_progress: Callable[[UltiTrainStats], None] | None = None,
) -> tuple[UltiNet, UltiTrainStats]:
    """Run hybrid self-play training for Ulti.

    Parameters
    ----------
    cfg : UltiTrainConfig
        All hyperparameters for the run.
    initial_net : UltiNet, optional
        Pass an existing net to resume training (checkpoint load).
        If None, a fresh network is created.
    opponent_state_dict : dict, optional
        If provided, defenders use a different network (opponent pool).
    on_progress : callable, optional
        Called after every step with an ``UltiTrainStats`` snapshot.

    Returns
    -------
    (UltiNet, UltiTrainStats)
    """
    device = cfg.device
    game = UltiGame()

    # -- Network --
    if initial_net is not None:
        net = initial_net
    else:
        net = UltiNet(
            input_dim=game.state_dim,
            body_units=cfg.body_units,
            body_layers=cfg.body_layers,
            action_dim=game.action_space_size,
        )
    wrapper = UltiNetWrapper(net, device=device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg.lr_start, weight_decay=1e-4,
    )
    buffer = ReplayBuffer(capacity=cfg.buffer_size, seed=cfg.seed + 1)
    np_rng = np.random.default_rng(cfg.seed)

    # -- MCTS configs --
    sol_cfg = MCTSConfig(
        simulations=cfg.sol_sims,
        determinizations=cfg.sol_dets,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=1.0,
    )
    def_cfg = MCTSConfig(
        simulations=cfg.def_sims,
        determinizations=cfg.def_dets,
        c_puct=1.5,
        dirichlet_alpha=0.1,
        dirichlet_weight=0.15,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.5,
    )

    # -- Parallel self-play pool --
    executor = None
    net_kwargs = {
        "input_dim": game.state_dim,
        "body_units": cfg.body_units,
        "body_layers": cfg.body_layers,
        "action_dim": game.action_space_size,
    }
    if cfg.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            initializer=_init_worker,
            initargs=(net_kwargs, "cpu"),
        )

    # -- Stats --
    stats = UltiTrainStats(total_steps=cfg.steps)
    t0 = time.perf_counter()

    try:
        for step in range(1, cfg.steps + 1):
            # ---- 0. Cosine LR schedule ----
            lr = _cosine_lr(step, cfg.steps, cfg.lr_start, cfg.lr_end)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ---- 1. Self-play ----
            step_samples = 0
            sp_wins = 0
            sp_total = 0

            # Enrichment threshold
            past_warmup = cfg.enrichment and step > cfg.enrich_warmup
            thresh = (
                _enrichment_threshold(step, cfg.steps)
                if past_warmup else -999.0
            )

            if executor is not None:
                # --- Parallel self-play ---
                state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
                tasks = []
                for g in range(cfg.games_per_step):
                    use_enrich = (
                        past_warmup
                        and thresh > -999.0
                        and (g / cfg.games_per_step) >= cfg.enrich_random_frac
                    )
                    tasks.append((
                        state_dict,
                        opponent_state_dict,
                        sol_cfg,
                        def_cfg,
                        cfg.seed + step * 1000 + g,
                        cfg.endgame_tricks,
                        cfg.pimc_dets,
                        cfg.solver_temp,
                        cfg.training_mode,
                        thresh if use_enrich else -999.0,
                    ))
                for samples in executor.map(_play_game_in_worker, tasks):
                    # Tally soloist self-play win rate
                    sol_r = [r for _, _, _, r, is_sol in samples if is_sol]
                    if sol_r:
                        sp_total += 1
                        if sol_r[0] > 0:
                            sp_wins += 1
                    for s, m, p, r, is_sol in samples:
                        buffer.push(s, m, p, r, is_soloist=is_sol)
                    step_samples += len(samples)
                    stats.total_games += 1
            else:
                # --- Sequential self-play ---
                for g in range(cfg.games_per_step):
                    game_seed = cfg.seed + step * 1000 + g

                    use_enrich = (
                        past_warmup
                        and thresh > -999.0
                        and (g / cfg.games_per_step) >= cfg.enrich_random_frac
                    )
                    if use_enrich:
                        init_state, _ = _value_enriched_new_game(
                            game, wrapper, game_seed,
                            min_value=thresh,
                            training_mode=cfg.training_mode,
                        )
                    else:
                        init_state = None

                    samples = _play_one_game(
                        game, wrapper, sol_cfg, def_cfg, game_seed,
                        endgame_tricks=cfg.endgame_tricks,
                        pimc_dets=cfg.pimc_dets,
                        solver_temp=cfg.solver_temp,
                        training_mode=cfg.training_mode,
                        opponent_wrapper=None,
                        initial_state=init_state,
                    )

                    sol_r = [r for _, _, _, r, is_sol in samples if is_sol]
                    if sol_r:
                        sp_total += 1
                        if sol_r[0] > 0:
                            sp_wins += 1
                    for s, m, p, r, is_sol in samples:
                        buffer.push(s, m, p, r, is_soloist=is_sol)
                    step_samples += len(samples)
                    stats.total_games += 1

            stats.total_samples += step_samples

            # ---- 2. Train (decoupled SGD steps) ----
            avg_vloss = 0.0
            avg_ploss = 0.0
            avg_pacc = 0.0
            avg_sol_vloss = 0.0
            avg_def_vloss = 0.0

            if len(buffer) >= cfg.batch_size:
                net.train()
                n_train = cfg.train_steps
                total_vloss = 0.0
                total_ploss = 0.0
                total_pacc = 0.0
                total_sol_vloss = 0.0
                total_def_vloss = 0.0
                sol_count = 0
                def_count = 0

                for _ in range(n_train):
                    states, masks, policies, rewards, is_sol = buffer.sample(
                        cfg.batch_size, np_rng,
                    )

                    s_t = torch.from_numpy(states).float().to(device)
                    m_t = torch.from_numpy(masks).bool().to(device)
                    pi_t = torch.from_numpy(policies).float().to(device)
                    z_t = torch.from_numpy(rewards).float().to(device)
                    is_sol_t = torch.from_numpy(is_sol).bool().to(device)

                    log_probs, values = net.forward_dual(s_t, m_t, is_sol_t)

                    value_loss = F.mse_loss(values, z_t)
                    policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
                    loss = value_loss + policy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

                    total_vloss += value_loss.item()
                    total_ploss += policy_loss.item()

                    with torch.no_grad():
                        pred_top1 = log_probs.argmax(dim=-1)
                        target_top1 = pi_t.argmax(dim=-1)
                        total_pacc += (pred_top1 == target_top1).float().mean().item()

                        sol_mask = is_sol_t
                        def_mask = ~is_sol_t
                        if sol_mask.any():
                            sv = F.mse_loss(values[sol_mask], z_t[sol_mask]).item()
                            total_sol_vloss += sv
                            sol_count += 1
                        if def_mask.any():
                            dv = F.mse_loss(values[def_mask], z_t[def_mask]).item()
                            total_def_vloss += dv
                            def_count += 1

                stats.total_sgd_steps += n_train
                avg_vloss = total_vloss / n_train
                avg_ploss = total_ploss / n_train
                avg_pacc = total_pacc / n_train
                avg_sol_vloss = total_sol_vloss / max(1, sol_count)
                avg_def_vloss = total_def_vloss / max(1, def_count)

            # ---- 3. Update stats ----
            stats.step = step
            stats.vloss = avg_vloss
            stats.ploss = avg_ploss
            stats.pacc = avg_pacc
            stats.sol_vloss = avg_sol_vloss
            stats.def_vloss = avg_def_vloss
            stats.lr = lr
            stats.sp_sol_wins = sp_wins
            stats.sp_sol_total = sp_total
            stats.train_time_s = time.perf_counter() - t0

            stats.history_vloss.append(avg_vloss)
            stats.history_ploss.append(avg_ploss)
            stats.history_pacc.append(avg_pacc)

            # ---- 4. Callback ----
            if on_progress is not None:
                on_progress(stats)

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    stats.train_time_s = time.perf_counter() - t0
    return net, stats
