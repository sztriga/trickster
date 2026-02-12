#!/usr/bin/env python3
"""Train Snapszer agents using Hybrid Self-Play and evaluate against old models.

Hybrid self-play: MCTS (neural-guided) for opening + Minimax for endgame.
During training:
- Phase 1 Early: MCTS with policy priors → visit distribution as policy target
- Phase 1 Late:  PIMC Minimax → one-hot best action as policy target
- Phase 2:       Pure Minimax → one-hot best action as policy target
- Value target:  actual game outcome (as always)

This produces better training data because endgame positions are solved
exactly, giving the neural net "perfect endgame knowledge" to learn from.

Usage:
    python3 scripts/train_hybrid.py                       # default Bishop-level
    python3 scripts/train_hybrid.py --tier rook            # Rook-level budget
    python3 scripts/train_hybrid.py --tier captain         # Captain-level budget
    python3 scripts/train_hybrid.py --eval-only            # skip training, eval only
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from trickster.games.snapszer.adapter import SnapszerGame, SnapszerNode
from trickster.games.snapszer.constants import (
    DEFAULT_EVAL_DEALS,
    DEFAULT_EVAL_DETS,
    DEFAULT_EVAL_SIMS,
    DEFAULT_LATE_THRESHOLD,
    DEFAULT_PIMC_SAMPLES,
)
from trickster.games.snapszer.game import deal_awarded_game_points
from trickster.games.snapszer.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig, alpha_mcts_choose
from trickster.models.alpha_net import SharedAlphaNet, create_shared_alpha_net
from trickster.training.alpha_zero import AlphaZeroSample, AlphaZeroStats


# ---------------------------------------------------------------------------
#  Tier definitions (mirrors train_ladder.py but with hybrid suffix)
# ---------------------------------------------------------------------------

@dataclass
class HybridTier:
    name: str
    body_units: int
    body_layers: int
    head_units: int
    iters: int
    gpi: int
    sims: int
    dets: int
    bootstrap: int
    pimc_samples: int = DEFAULT_PIMC_SAMPLES
    late_threshold: int = DEFAULT_LATE_THRESHOLD
    train_steps: int = 100
    batch_size: int = 64
    buffer_cap: int = 30000
    lr: float = 0.01
    l2: float = 1e-4
    seed: int = 42
    workers: int = 4
    description: str = ""

    @property
    def total_games(self) -> int:
        return self.iters * self.gpi


TIERS: dict[str, HybridTier] = {
    "scout": HybridTier(
        name="H1-Scout",
        body_units=64, body_layers=2, head_units=32,
        iters=150, gpi=30, sims=40, dets=3,
        bootstrap=2000, train_steps=60, batch_size=32,
        buffer_cap=15000,
        description="Hybrid Scout (64x2/32, 4.5k games) — matches T2-Scout budget",
    ),
    "knight": HybridTier(
        name="H2-Knight",
        body_units=64, body_layers=2, head_units=32,
        iters=200, gpi=40, sims=50, dets=4,
        bootstrap=3000, train_steps=80, batch_size=32,
        buffer_cap=20000,
        description="Hybrid Knight (64x2/32, 8k games) — matches T3-Knight budget",
    ),
    "bishop": HybridTier(
        name="H3-Bishop",
        body_units=128, body_layers=2, head_units=64,
        iters=300, gpi=50, sims=60, dets=5,
        bootstrap=6000, train_steps=100, batch_size=64,
        buffer_cap=30000,
        description="Hybrid Bishop (128x2/64, 15k games) — matches T4-Bishop budget",
    ),
    "rook": HybridTier(
        name="H4-Rook",
        body_units=128, body_layers=2, head_units=64,
        iters=400, gpi=50, sims=60, dets=5,
        bootstrap=8000, train_steps=100, batch_size=64,
        buffer_cap=40000,
        description="Hybrid Rook (128x2/64, 20k games) — matches T5-Rook budget",
    ),
    "captain": HybridTier(
        name="H5-Captain",
        body_units=128, body_layers=3, head_units=64,
        iters=600, gpi=80, sims=80, dets=6,
        bootstrap=12000, train_steps=150, batch_size=64,
        buffer_cap=60000, lr=0.005,
        description="Hybrid Captain (128x3/64, 48k games) — matches T6-Captain budget",
    ),
}


# ---------------------------------------------------------------------------
#  Multiprocessing helpers (must be at module level for pickling)
# ---------------------------------------------------------------------------

_WORKER_GAME: SnapszerGame | None = None


def _init_hybrid_worker(game: SnapszerGame) -> None:
    """Called once per worker process to set the game instance."""
    global _WORKER_GAME
    _WORKER_GAME = game


def _hybrid_play_in_worker(
    args: tuple[SharedAlphaNet, MCTSConfig, MCTSConfig, int, int, bool, int, int],
) -> list[AlphaZeroSample]:
    """Worker entry-point — uses the pre-initialized game."""
    net, mcts_cfg, boot_cfg, game_seed, ep_idx, is_boot, pimc_n, late_thr = args
    assert _WORKER_GAME is not None
    return hybrid_play_one_game(
        _WORKER_GAME, net, mcts_cfg, boot_cfg,
        game_seed, ep_idx, is_boot, pimc_n, late_thr,
    )


# ---------------------------------------------------------------------------
#  Hybrid self-play
# ---------------------------------------------------------------------------


def hybrid_play_one_game(
    game: SnapszerGame,
    net: SharedAlphaNet,
    mcts_config: MCTSConfig,
    bootstrap_config: MCTSConfig,
    game_seed: int,
    ep_idx: int,
    is_bootstrap: bool,
    pimc_samples: int = DEFAULT_PIMC_SAMPLES,
    late_threshold: int = DEFAULT_LATE_THRESHOLD,
) -> list[AlphaZeroSample]:
    """Play one game with hybrid search, return training samples.

    Early game: MCTS produces visit distributions (for policy training).
    Late/endgame: Minimax produces one-hot targets (perfect play).
    """
    rng = random.Random(game_seed)
    state = game.new_game(seed=game_seed, starting_leader=ep_idx % 2)
    trajectory: list[AlphaZeroSample] = []

    config = bootstrap_config if is_bootstrap else mcts_config
    hybrid = HybridPlayer(
        net=net,
        mcts_config=config,
        game=game,
        pimc_samples=pimc_samples,
        late_threshold=late_threshold,
    )

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        # Forced move — skip
        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        # Get policy target + action from hybrid player
        pi, action = hybrid.choose_action_for_training(state, player, rng)

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

    # Fill in outcomes
    for s in trajectory:
        s.z = game.outcome(state, s.player)

    return trajectory


# ---------------------------------------------------------------------------
#  SGD training (same as alpha_zero.py)
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
    if not buffer:
        return 0.0, 0.0
    total_vmse, total_pce = 0.0, 0.0
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
#  Main training loop
# ---------------------------------------------------------------------------


def train_hybrid(tier: HybridTier, game: SnapszerGame) -> tuple[SharedAlphaNet, AlphaZeroStats, Path]:
    model_dir = Path("models") / tier.name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_config = MCTSConfig(
        simulations=tier.sims,
        determinizations=tier.dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        visit_temp=1.0,
    )
    bootstrap_config = MCTSConfig(
        simulations=tier.sims,
        determinizations=tier.dets,
        use_value_head=False,
        use_policy_priors=True,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        visit_temp=1.0,
    )

    net = create_shared_alpha_net(
        state_dim=game.state_dim,
        action_space_size=game.action_space_size,
        body_units=tier.body_units,
        body_layers=tier.body_layers,
        head_units=tier.head_units,
        seed=tier.seed,
    )

    rng = random.Random(tier.seed)
    stats = AlphaZeroStats()
    replay_buffer: list[AlphaZeroSample] = []

    # Set up process pool (same pattern as alpha_zero.py)
    executor = None
    num_workers = tier.workers
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_hybrid_worker,
            initargs=(game,),
        )

    t0 = time.perf_counter()
    last_report = 0

    try:
        for it in range(tier.iters):
            games_so_far = it * tier.gpi
            is_bootstrap = tier.bootstrap > 0 and games_so_far < tier.bootstrap
            phase = "bootstrap" if is_bootstrap else "alphazero"

            # Self-play — parallel or sequential
            new_samples: list[AlphaZeroSample] = []

            if executor is not None:
                # --- parallel self-play ---
                tasks = [
                    (net, train_config, bootstrap_config,
                     tier.seed + it * tier.gpi + g,
                     it * tier.gpi + g,
                     is_bootstrap,
                     tier.pimc_samples,
                     tier.late_threshold)
                    for g in range(tier.gpi)
                ]
                for samples in executor.map(_hybrid_play_in_worker, tasks):
                    new_samples.extend(samples)
            else:
                # --- sequential self-play ---
                for g in range(tier.gpi):
                    game_seed = tier.seed + it * tier.gpi + g
                    samples = hybrid_play_one_game(
                        game, net,
                        train_config, bootstrap_config,
                        game_seed=game_seed,
                        ep_idx=it * tier.gpi + g,
                        is_bootstrap=is_bootstrap,
                        pimc_samples=tier.pimc_samples,
                        late_threshold=tier.late_threshold,
                    )
                    new_samples.extend(samples)

            # Replay buffer
            replay_buffer.extend(new_samples)
            if len(replay_buffer) > tier.buffer_cap:
                replay_buffer = replay_buffer[-tier.buffer_cap:]

            # Train
            vmse, pce = _train_on_buffer(
                net, replay_buffer, rng, tier.lr, tier.l2,
                tier.train_steps, tier.batch_size,
            )

            # Stats
            stats.iterations = it + 1
            stats.total_games += tier.gpi
            stats.total_samples += len(new_samples)
            stats.last_value_mse = vmse
            stats.last_policy_ce = pce
            stats.buffer_size = len(replay_buffer)
            stats.phase = phase

            if stats.iterations - last_report >= max(1, tier.iters // 20) or \
               stats.iterations == tier.iters:
                last_report = stats.iterations
                elapsed = time.perf_counter() - t0
                pct = stats.iterations / tier.iters * 100
                tag = "BOOT" if phase == "bootstrap" else "HYBR"
                print(
                    f"    [{tag}] iter {stats.iterations:>4d}/{tier.iters} ({pct:4.0f}%)  "
                    f"games={stats.total_games:>5d}  "
                    f"vmse={vmse:.4f}  pce={pce:.4f}  "
                    f"[{elapsed:.0f}s]",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    train_time = time.perf_counter() - t0

    # Save
    with open(model_dir / "net.pkl", "wb") as f:
        pickle.dump(net, f)

    spec = {
        "game": "snapszer",
        "kind": "alphazero",
        "method": "hybrid",
        "params": {
            "body_units": tier.body_units,
            "body_layers": tier.body_layers,
            "head_units": tier.head_units,
        },
    }
    info = {
        "method": "hybrid",
        "iterations": tier.iters,
        "games_per_iter": tier.gpi,
        "total_games": stats.total_games,
        "total_samples": stats.total_samples,
        "bootstrap_games": tier.bootstrap,
        "sims": tier.sims,
        "dets": tier.dets,
        "pimc_samples": tier.pimc_samples,
        "late_threshold": tier.late_threshold,
        "train_steps": tier.train_steps,
        "batch_size": tier.batch_size,
        "buffer_capacity": tier.buffer_cap,
        "lr": tier.lr,
        "l2": tier.l2,
        "seed": tier.seed,
        "vmse": round(stats.last_value_mse, 6),
        "pce": round(stats.last_policy_ce, 6),
        "train_time_s": round(train_time, 1),
    }
    (model_dir / "spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (model_dir / "train_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\n    Done in {train_time:.1f}s  →  {model_dir}")
    return net, stats, model_dir


# ---------------------------------------------------------------------------
#  Head-to-head evaluation
# ---------------------------------------------------------------------------


def play_match(
    game: SnapszerGame,
    agent_a, agent_b,
    deals: int = 200,
    seed: int = 0,
) -> tuple[int, int]:
    a_pts, b_pts = 0, 0
    for g in range(deals):
        a_idx = g % 2
        node = game.new_game(seed=seed + g, starting_leader=0)
        rng_a = random.Random(seed + g + 10000)
        rng_b = random.Random(seed + g + 20000)

        while not game.is_terminal(node):
            player = game.current_player(node)
            if player == a_idx:
                action = agent_a(node, player, rng_a)
            else:
                action = agent_b(node, player, rng_b)
            node = game.apply(node, action)

        winner, pts, _ = deal_awarded_game_points(node.gs)
        if winner == a_idx:
            a_pts += pts
        else:
            b_pts += pts
    return a_pts, b_pts


def load_az_agent(model_dir: Path, game: SnapszerGame, sims: int = 50, dets: int = 4):
    """Load a pure MCTS AlphaZero agent."""
    with open(model_dir / "net.pkl", "rb") as f:
        net = pickle.load(f)
    config = MCTSConfig(
        simulations=sims, determinizations=dets,
        use_value_head=True, use_policy_priors=True,
        dirichlet_alpha=0.0, visit_temp=0.1,
    )

    def agent(node, player, rng):
        actions = game.legal_actions(node)
        if len(actions) <= 1:
            return actions[0]
        return alpha_mcts_choose(node, game, net, player, config, rng)

    return agent, model_dir.name


def load_hybrid_agent(model_dir: Path, game: SnapszerGame, sims: int = 50, dets: int = 4):
    """Load a hybrid agent (MCTS + PIMC + Minimax)."""
    with open(model_dir / "net.pkl", "rb") as f:
        net = pickle.load(f)
    config = MCTSConfig(
        simulations=sims, determinizations=dets,
        use_value_head=True, use_policy_priors=True,
        dirichlet_alpha=0.0, visit_temp=0.1,
    )
    hybrid = HybridPlayer(net=net, mcts_config=config, game=game)

    def agent(node, player, rng):
        actions = game.legal_actions(node)
        if len(actions) <= 1:
            return actions[0]
        return hybrid.choose_action(node, player, rng)

    return agent, f"{model_dir.name} (hybrid)"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid AlphaZero Training + Eval")
    parser.add_argument("--tier", type=str, default="bishop",
                        choices=list(TIERS.keys()),
                        help="Training tier (default: bishop)")
    parser.add_argument("--eval-deals", type=int, default=DEFAULT_EVAL_DEALS,
                        help=f"Deals per evaluation matchup (default {DEFAULT_EVAL_DEALS})")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing hybrid models")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation (training only)")
    parser.add_argument("--sims", type=int, default=None,
                        help="Override MCTS simulations")
    parser.add_argument("--dets", type=int, default=None,
                        help="Override determinizations")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel self-play workers (default: tier default, usually 4)")
    args = parser.parse_args()

    game = SnapszerGame()
    tier = TIERS[args.tier]

    if args.sims:
        tier.sims = args.sims
    if args.dets:
        tier.dets = args.dets
    if args.workers is not None:
        tier.workers = args.workers

    # ── Training ──────────────────────────────────────────────────────
    model_dir = Path("models") / tier.name

    if not args.eval_only:
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║         HYBRID ALPHAZERO TRAINING                               ║")
        print("║         MCTS (opening) + PIMC/Minimax (endgame) self-play       ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print(f"  Tier: {tier.name} — {tier.description}")
        print(f"  Net: {tier.body_units}x{tier.body_layers}/h{tier.head_units}")
        print(f"  Budget: {tier.iters} iters × {tier.gpi} gpi = {tier.total_games:,} games")
        print(f"  MCTS: {tier.sims}s × {tier.dets}d | PIMC: {tier.pimc_samples} samples")
        print(f"  Bootstrap: first {tier.bootstrap} games use random rollouts")
        print(f"  Workers: {tier.workers} {'(parallel)' if tier.workers > 1 else '(sequential)'}")
        print()

        net, stats, model_dir = train_hybrid(tier, game)
        print(f"\n  Training complete: {stats.total_games} games, "
              f"{stats.total_samples} samples")
        print(f"  Final vmse={stats.last_value_mse:.4f}, pce={stats.last_policy_ce:.4f}")
    else:
        if not model_dir.exists() or not (model_dir / "net.pkl").exists():
            print(f"  No hybrid model found at {model_dir}. Train first.")
            return
        print(f"  Using existing model: {model_dir}")

    if args.no_eval:
        return

    # ── Evaluation: Hybrid vs Old Models ──────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         HEAD-TO-HEAD EVALUATION                                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    eval_sims = DEFAULT_EVAL_SIMS
    eval_dets = DEFAULT_EVAL_DETS
    deals = args.eval_deals

    # Build agent list
    agents: list[tuple] = []

    # Random baseline
    def rnd_agent(node, player, rng):
        actions = game.legal_actions(node)
        card_actions = [a for a in actions if a != "close_talon"]
        return rng.choice(card_actions) if card_actions else rng.choice(actions)
    agents.append((rnd_agent, "Random"))

    # Old models (Bishop, Rook, Captain)
    for old_name in ["T4-Bishop", "T5-Rook", "T6-Captain"]:
        old_dir = Path("models") / old_name
        if (old_dir / "net.pkl").exists():
            fn, label = load_az_agent(old_dir, game, eval_sims, eval_dets)
            agents.append((fn, label))

    # Hybrid model (current tier)
    if (model_dir / "net.pkl").exists():
        fn_h, label_h = load_hybrid_agent(model_dir, game, eval_sims, eval_dets)
        agents.append((fn_h, label_h))

    n = len(agents)
    print(f"\n  Round-robin: {n} agents, {deals} deals per matchup")
    print(f"  Eval MCTS: {eval_sims}s × {eval_dets}d\n")

    total_pts = [0] * n
    results: list[list[str]] = [["-"] * n for _ in range(n)]
    matchup_idx = 0
    total_matchups = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            matchup_idx += 1
            name_a = agents[i][1]
            name_b = agents[j][1]
            print(
                f"  [{matchup_idx}/{total_matchups}] "
                f"{name_a} vs {name_b}...",
                end="", flush=True,
            )
            t0 = time.perf_counter()
            pts_a, pts_b = play_match(
                game, agents[i][0], agents[j][0],
                deals=deals, seed=i * 1000 + j,
            )
            elapsed = time.perf_counter() - t0

            ppd_a = pts_a / deals
            ppd_b = pts_b / deals
            margin = ppd_a - ppd_b
            total_pts[i] += pts_a
            total_pts[j] += pts_b
            results[i][j] = f"{margin:+.2f}"
            results[j][i] = f"{-margin:+.2f}"

            winner = name_a if margin > 0 else name_b if margin < 0 else "DRAW"
            print(f"  {ppd_a:.2f} vs {ppd_b:.2f}  (Δ={margin:+.2f})  [{elapsed:.1f}s]")

    # ── Ranking ───────────────────────────────────────────────────────
    print()
    print("  ┌─ RANKING " + "─" * 55)
    print("  │")

    ranking = sorted(range(n), key=lambda k: total_pts[k], reverse=True)
    max_name = max(len(agents[k][1]) for k in ranking)
    total_deals = (n - 1) * deals

    print(f"  │  {'Rank':<5} {'Agent':<{max_name+2}} {'Total pts':>10}  {'pts/deal':>8}")
    print(f"  │  {'─'*5} {'─'*(max_name+2)} {'─'*10}  {'─'*8}")

    for rank, k in enumerate(ranking, 1):
        name = agents[k][1]
        pts = total_pts[k]
        ppd = pts / max(1, total_deals)
        medal = ">> " if rank == 1 else "   "
        print(f"  │  {medal}{rank:<3} {name:<{max_name+2}} {pts:>10}  {ppd:>8.3f}")

    print("  │")

    # Margin table
    print("  │  Margin table (row vs col, + = row stronger):")
    header = "  │  " + " " * (max_name + 2)
    for k in ranking:
        header += f" {agents[k][1][:10]:>10}"
    print(header)
    for k in ranking:
        row = f"  │  {agents[k][1]:<{max_name+2}}"
        for j in ranking:
            row += f" {results[k][j]:>10}"
        print(row)

    print("  │")
    print("  └" + "─" * 65)
    print()


if __name__ == "__main__":
    main()
