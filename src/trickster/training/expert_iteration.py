"""Expert Iteration: MCTS plays games, MLPBinaryModel learns from outcomes.

The MCTS (rollout evaluation, uniform priors) plays games against itself.
For each decision point we record the chosen (state+action) features and
the game outcome from that player's perspective.  The existing
MLPBinaryModel learns "P(win | state, action)" — exactly like Direct
self-play but with a vastly stronger player generating the data.

This preserves the gradient structure that makes Direct training work:
- Each sample has a unique (state, action) pair
- The target (game outcome) varies with the state → state features learn
- The signal is naturally ~50/50 balanced (both players contribute)
"""

from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from trickster.games.snapszer.adapter import SnapszerGame, SnapszerNode
from trickster.games.snapszer.features import get_fast_encoder
from trickster.mcts import MCTSConfig, alpha_mcts_choose
from trickster.models.alpha_net import create_alpha_net
from trickster.training.policy import TrainedPolicy, create_policy
from trickster.training.model_spec import ModelSpec


# ---------------------------------------------------------------------------
#  Experience
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExpertSample:
    """One decision point: (state+action) features + eventual game outcome."""
    features: np.ndarray   # (feature_dim,)
    player: int            # who made this decision
    decision_type: str     # "lead" or "follow"
    outcome: float = 0.0   # filled after game: scaled by points (1.0 = 3pt win, 0.667 = 1pt win, 0.333 = 1pt loss, 0.0 = 3pt loss)


# ---------------------------------------------------------------------------
#  Self-play data collection
# ---------------------------------------------------------------------------

def _play_one_game(
    game: SnapszerGame,
    net,
    enc,
    mcts_config: MCTSConfig,
    game_seed: int,
    ep_idx: int,
) -> list[ExpertSample]:
    """Play one game with MCTS, collect (state+action, player) samples."""
    rng = random.Random(game_seed)
    node = game.new_game(seed=game_seed, starting_leader=ep_idx % 2)
    samples: list[ExpertSample] = []

    while not game.is_terminal(node):
        player = game.current_player(node)
        actions = game.legal_actions(node)

        if len(actions) <= 1:
            node = game.apply(node, actions[0])
            continue

        # Run MCTS to choose the best action
        chosen = alpha_mcts_choose(
            node, game, net, player, mcts_config, rng,
        )

        # Handle close_talon: apply it and re-search for a card
        if chosen == "close_talon":
            node = game.apply(node, chosen)
            continue

        gs = node.gs
        dt = game.decision_type(node)

        if dt == "lead":
            card_actions = [a for a in actions if a != "close_talon"]
            if not card_actions:
                node = game.apply(node, chosen)
                continue

            # Encode only the chosen action
            features = enc.encode_lead_batch(
                hand=gs.hands[player],
                actions=[chosen],
                draw_pile_size=len(gs.draw_pile),
                captured_self=gs.captured[player],
                captured_opp=gs.captured[1 - player],
                trump_color=gs.trump_color,
                trump_upcard=gs.trump_card if gs.trump_upcard_visible else None,
            )
            samples.append(ExpertSample(
                features=features[0].copy(),
                player=player,
                decision_type="lead",
            ))

        elif dt == "follow":
            lead_card = node.pending_lead
            features = enc.encode_follow_batch(
                hand=gs.hands[player],
                lead_card=lead_card,
                actions=[chosen],
                draw_pile_size=len(gs.draw_pile),
                captured_self=gs.captured[player],
                captured_opp=gs.captured[1 - player],
                trump_color=gs.trump_color,
                trump_upcard=gs.trump_card if gs.trump_upcard_visible else None,
            )
            samples.append(ExpertSample(
                features=features[0].copy(),
                player=player,
                decision_type="follow",
            ))

        node = game.apply(node, chosen)

    # Fill in outcomes: who won the game?
    if game.is_terminal(node):
        for s in samples:
            raw = game.outcome(node, s.player)  # in [-1, +1]
            # Map to [0, 1] for binary cross-entropy
            s.outcome = (raw + 1.0) / 2.0

    return samples


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def _train_on_buffer(
    policy: TrainedPolicy,
    buffer: list[ExpertSample],
    rng: random.Random,
    lr: float,
    l2: float,
    steps: int = 200,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Fixed number of SGD steps sampling mini-batches from the buffer."""
    if not buffer:
        return 0.0, 0.0

    lead_samples = [s for s in buffer if s.decision_type == "lead"]
    follow_samples = [s for s in buffer if s.decision_type == "follow"]

    lead_loss = 0.0
    follow_loss = 0.0
    n_lead = 0
    n_follow = 0

    for _ in range(steps):
        if lead_samples:
            B = min(batch_size, len(lead_samples))
            batch = [lead_samples[rng.randrange(len(lead_samples))] for _ in range(B)]
            X = np.vstack([s.features for s in batch])
            Y = np.array([s.outcome for s in batch], dtype=np.float64)
            loss = policy.lead_model.batch_update_raw(X, Y, lr=lr, l2=l2)
            lead_loss += loss
            n_lead += 1

        if follow_samples:
            B = min(batch_size, len(follow_samples))
            batch = [follow_samples[rng.randrange(len(follow_samples))] for _ in range(B)]
            X = np.vstack([s.features for s in batch])
            Y = np.array([s.outcome for s in batch], dtype=np.float64)
            loss = policy.follow_model.batch_update_raw(X, Y, lr=lr, l2=l2)
            follow_loss += loss
            n_follow += 1

    return (
        lead_loss / max(1, n_lead),
        follow_loss / max(1, n_follow),
    )


# ---------------------------------------------------------------------------
#  Stats
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExpertTrainStats:
    iterations: int = 0
    total_games: int = 0
    total_samples: int = 0
    last_lead_loss: float = 0.0
    last_follow_loss: float = 0.0


# ---------------------------------------------------------------------------
#  Multiprocessing worker for self-play
# ---------------------------------------------------------------------------

# Per-worker globals (initialised once per process via Pool initializer)
_W_GAME: SnapszerGame | None = None
_W_ENC = None
_W_NET = None
_W_CFG: MCTSConfig | None = None


def _worker_init(mcts_config: MCTSConfig, state_dim: int, action_space_size: int, seed: int) -> None:
    global _W_GAME, _W_ENC, _W_NET, _W_CFG
    _W_GAME = SnapszerGame()
    _W_ENC = get_fast_encoder()
    _W_NET = create_alpha_net(state_dim=state_dim, action_space_size=action_space_size, seed=seed)
    _W_CFG = mcts_config


def _worker_play(args: tuple[int, int]) -> list[ExpertSample]:
    """Play one game in a worker process."""
    game_seed, ep_idx = args
    assert _W_GAME is not None and _W_ENC is not None and _W_NET is not None and _W_CFG is not None
    return _play_one_game(_W_GAME, _W_NET, _W_ENC, _W_CFG, game_seed, ep_idx)


# ---------------------------------------------------------------------------
#  Main entry
# ---------------------------------------------------------------------------

def train_expert_iteration(
    *,
    spec: ModelSpec,
    iterations: int = 100,
    games_per_iter: int = 50,
    train_steps: int = 200,
    mcts_config: MCTSConfig,
    seed: int = 0,
    lr: float = 0.01,
    l2: float = 1e-6,
    buffer_capacity: int = 20_000,
    workers: int = 1,
    on_progress: Optional[Callable[[ExpertTrainStats], None]] = None,
) -> tuple[TrainedPolicy, ExpertTrainStats]:
    """Run Expert Iteration training.

    Returns a standard TrainedPolicy compatible with all existing eval.

    Set ``workers > 1`` to parallelise self-play across CPU cores.
    Training (SGD) always runs single-threaded on the main process.
    """
    game = SnapszerGame()
    enc = get_fast_encoder()
    policy = create_policy(spec)

    # Dummy AlphaNet — never actually used (uniform priors + rollouts)
    dummy_net = create_alpha_net(
        state_dim=game.state_dim,
        action_space_size=game.action_space_size,
        seed=seed,
    )

    rng = random.Random(seed)
    stats = ExpertTrainStats()
    replay_buffer: list[ExpertSample] = []

    workers = max(1, int(workers))
    use_mp = workers > 1

    # Set up multiprocessing pool (reused across iterations)
    pool = None
    if use_mp:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(mcts_config, game.state_dim, game.action_space_size, seed),
        )

    try:
        for it in range(iterations):
            # 1. Self-play with MCTS expert
            tasks = [
                (seed + it * games_per_iter + g, it * games_per_iter + g)
                for g in range(games_per_iter)
            ]

            new_samples: list[ExpertSample] = []
            if use_mp and pool is not None:
                for batch in pool.imap_unordered(_worker_play, tasks, chunksize=max(1, games_per_iter // (workers * 4))):
                    new_samples.extend(batch)
            else:
                for game_seed, ep_idx in tasks:
                    samples = _play_one_game(
                        game, dummy_net, enc, mcts_config, game_seed, ep_idx,
                    )
                    new_samples.extend(samples)

            # 2. Add to replay buffer
            replay_buffer.extend(new_samples)
            if len(replay_buffer) > buffer_capacity:
                replay_buffer = replay_buffer[-buffer_capacity:]

            # 3. Train
            lead_loss, follow_loss = _train_on_buffer(
                policy, replay_buffer, rng, lr, l2,
                steps=train_steps, batch_size=64,
            )

            # 4. Stats
            stats.iterations = it + 1
            stats.total_games += games_per_iter
            stats.total_samples += len(new_samples)
            stats.last_lead_loss = lead_loss
            stats.last_follow_loss = follow_loss

            if on_progress is not None:
                on_progress(stats)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    return policy, stats
