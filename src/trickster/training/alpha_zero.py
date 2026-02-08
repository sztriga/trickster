"""AlphaZero-style training: SharedAlphaNet learns from MCTS self-play.

Training loop:
1. **Self-play** — MCTS (with policy priors + value evaluation) plays
   games against itself, collecting *(state, mask, π, z)* tuples.
2. **Train** — the SharedAlphaNet is updated on mini-batches from a
   replay buffer using the combined AlphaZero loss:

       L = (z − v)² − π·log(p) + c·‖θ‖²

Exploration is ensured by:
- **Dirichlet noise** on root priors (``MCTSConfig.dirichlet_alpha``).
- **Temperature** on visit counts (``MCTSConfig.visit_temp``), so
  the policy target is a soft distribution even when MCTS is strong.

Parallel self-play is supported via ``num_workers`` — each iteration's
games are distributed across worker processes using
:class:`~concurrent.futures.ProcessPoolExecutor`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from trickster.games.interface import GameInterface
from trickster.mcts import MCTSConfig, alpha_mcts_policy
from trickster.models.alpha_net import SharedAlphaNet, create_shared_alpha_net


# ---------------------------------------------------------------------------
#  Multiprocessing helpers (must be at module level for pickling)
# ---------------------------------------------------------------------------

_WORKER_GAME: GameInterface | None = None


def _init_self_play_worker(game: GameInterface) -> None:
    """Called once per worker process to set the game instance."""
    global _WORKER_GAME
    _WORKER_GAME = game


def _play_game_in_worker(
    args: tuple[SharedAlphaNet, MCTSConfig, int, int],
) -> "list[AlphaZeroSample]":
    """Worker entry-point — uses the pre-initialized game."""
    net, config, game_seed, ep_idx = args
    assert _WORKER_GAME is not None
    return _play_one_game(_WORKER_GAME, net, config, game_seed, ep_idx)


# ---------------------------------------------------------------------------
#  Training sample
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AlphaZeroSample:
    """One decision point from self-play."""

    state_feats: np.ndarray  # (state_dim,)
    mask: np.ndarray         # (action_space,) bool
    pi: np.ndarray           # (action_space,) MCTS visit distribution
    player: int              # who made this decision
    z: float = 0.0           # game outcome for *player* in [-1,+1]


# ---------------------------------------------------------------------------
#  Self-play data collection
# ---------------------------------------------------------------------------


def _play_one_game(
    game: GameInterface,
    net: SharedAlphaNet,
    config: MCTSConfig,
    game_seed: int,
    ep_idx: int,
) -> list[AlphaZeroSample]:
    """Play one game with MCTS, return training samples."""
    rng = random.Random(game_seed)
    state = game.new_game(seed=game_seed, starting_leader=ep_idx % 2)
    trajectory: list[AlphaZeroSample] = []

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        # Forced move — nothing to learn
        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        # MCTS search → visit distribution + sampled action
        pi, action = alpha_mcts_policy(
            state, game, net, player, config, rng,
        )

        # Record training sample
        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append(AlphaZeroSample(
            state_feats=state_feats.copy(),
            mask=mask.copy(),
            pi=pi.copy(),
            player=player,
        ))

        state = game.apply(state, action)

    # Fill in outcomes now that the game is over
    for s in trajectory:
        s.z = game.outcome(state, s.player)

    return trajectory


# ---------------------------------------------------------------------------
#  SGD training on replay buffer
# ---------------------------------------------------------------------------


def _train_on_buffer(
    net: SharedAlphaNet,
    buffer: list[AlphaZeroSample],
    rng: random.Random,
    lr: float,
    l2: float,
    steps: int,
    batch_size: int,
) -> tuple[float, float]:
    """Fixed-count SGD steps sampling mini-batches from *buffer*.

    Returns *(avg_value_mse, avg_policy_ce)*.
    """
    if not buffer:
        return 0.0, 0.0

    total_vmse = 0.0
    total_pce = 0.0
    buf_len = len(buffer)

    for _ in range(steps):
        B = min(batch_size, buf_len)
        batch = [buffer[rng.randrange(buf_len)] for _ in range(B)]
        states = np.vstack([s.state_feats for s in batch])
        masks = np.vstack([s.mask for s in batch])
        pis = np.vstack([s.pi for s in batch])
        zs = np.array([s.z for s in batch], dtype=np.float64)

        vmse, pce = net.train_batch(states, masks, pis, zs, lr, l2)
        total_vmse += vmse
        total_pce += pce

    return total_vmse / max(1, steps), total_pce / max(1, steps)


# ---------------------------------------------------------------------------
#  Stats
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AlphaZeroStats:
    iterations: int = 0
    total_games: int = 0
    total_samples: int = 0
    last_value_mse: float = 0.0
    last_policy_ce: float = 0.0
    buffer_size: int = 0
    phase: str = "bootstrap"  # "bootstrap" or "alphazero"


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------


def train_alpha_zero(
    *,
    game: GameInterface,
    iterations: int = 100,
    games_per_iter: int = 100,
    train_steps: int = 200,
    mcts_config: MCTSConfig | None = None,
    body_units: int = 128,
    body_layers: int = 2,
    head_units: int = 64,
    lr: float = 0.01,
    l2: float = 1e-4,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    bootstrap_games: int = 0,
    seed: int = 0,
    net: SharedAlphaNet | None = None,
    num_workers: int = 1,
    on_progress: Optional[Callable[[AlphaZeroStats], None]] = None,
) -> tuple[SharedAlphaNet, AlphaZeroStats]:
    """Run AlphaZero training: MCTS self-play → train SharedAlphaNet.

    Parameters
    ----------
    game : GameInterface
        The game to train on (e.g. ``SnapszerGame()``).
    iterations : int
        Number of collect-then-train iterations.
    games_per_iter : int
        Self-play games per iteration.
    train_steps : int
        SGD mini-batch steps per iteration.
    mcts_config : MCTSConfig, optional
        MCTS settings.  Defaults enable **value head** evaluation,
        **policy priors**, **Dirichlet noise**, and **τ = 1** temperature
        for diverse exploration.
    bootstrap_games : int
        If > 0, the first *bootstrap_games* self-play games use
        **random rollouts** (``use_value_head=False``) instead of the
        value head.  Both heads are still trained on the resulting data.
        This "hybrid bootstrap" gives the network reliable search
        signal before switching to pure AlphaZero (value-head) mode.
    net : SharedAlphaNet, optional
        Pass an existing net to continue training.
    num_workers : int
        Number of parallel processes for self-play.  ``1`` (default)
        runs everything sequentially in the main process.  Values > 1
        distribute games across worker processes — each worker receives
        a snapshot of the latest network weights per iteration.
    on_progress : callable, optional
        Called after every iteration with an :class:`AlphaZeroStats`.

    Returns
    -------
    (SharedAlphaNet, AlphaZeroStats)
    """
    from dataclasses import replace as _replace

    if mcts_config is None:
        mcts_config = MCTSConfig(
            simulations=100,
            determinizations=6,
            use_value_head=True,
            use_policy_priors=True,
            dirichlet_alpha=0.3,
            dirichlet_weight=0.25,
            visit_temp=1.0,
        )

    if net is None:
        net = create_shared_alpha_net(
            state_dim=game.state_dim,
            action_space_size=game.action_space_size,
            body_units=body_units,
            body_layers=body_layers,
            head_units=head_units,
            seed=seed,
        )

    # Bootstrap config: same MCTS but with rollout evaluation
    bootstrap_config = _replace(
        mcts_config,
        use_value_head=False,
        use_policy_priors=True,
    )

    rng = random.Random(seed)
    stats = AlphaZeroStats()
    replay_buffer: list[AlphaZeroSample] = []

    # Set up process pool (created once, reused across iterations).
    # The game object is set in each worker via the initializer so it
    # is pickled only num_workers times.  The network weights (≈400 KB)
    # are sent per-task so workers always use the latest parameters.
    executor = None
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_self_play_worker,
            initargs=(game,),
        )

    try:
        for it in range(iterations):
            games_so_far = it * games_per_iter
            # Choose config: bootstrap (rollouts) vs full AlphaZero (value head)
            if bootstrap_games > 0 and games_so_far < bootstrap_games:
                iter_config = bootstrap_config
                phase = "bootstrap"
            else:
                iter_config = mcts_config
                phase = "alphazero"

            # 1. Self-play with MCTS expert
            new_samples: list[AlphaZeroSample] = []

            if executor is not None:
                # --- parallel self-play ---
                tasks = [
                    (net, iter_config,
                     seed + it * games_per_iter + g,
                     it * games_per_iter + g)
                    for g in range(games_per_iter)
                ]
                for samples in executor.map(_play_game_in_worker, tasks):
                    new_samples.extend(samples)
            else:
                # --- sequential self-play ---
                for g in range(games_per_iter):
                    game_seed = seed + it * games_per_iter + g
                    samples = _play_one_game(
                        game, net, iter_config, game_seed,
                        it * games_per_iter + g,
                    )
                    new_samples.extend(samples)

            # 2. Add to replay buffer (FIFO eviction)
            replay_buffer.extend(new_samples)
            if len(replay_buffer) > buffer_capacity:
                replay_buffer = replay_buffer[-buffer_capacity:]

            # 3. Train on buffer
            vmse, pce = _train_on_buffer(
                net, replay_buffer, rng, lr, l2, train_steps, batch_size,
            )

            # 4. Stats
            stats.iterations = it + 1
            stats.total_games += games_per_iter
            stats.total_samples += len(new_samples)
            stats.last_value_mse = vmse
            stats.last_policy_ce = pce
            stats.buffer_size = len(replay_buffer)
            stats.phase = phase  # type: ignore[attr-defined]

            if on_progress is not None:
                on_progress(stats)

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    return net, stats
