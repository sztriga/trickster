#!/usr/bin/env python3
"""Train Ulti Parti models at 3 strength tiers and evaluate head-to-head.

Thin configuration wrapper around the shared training engine
(``trickster.training.ulti_hybrid``).  Each tier defines a set of
hyperparameters; the engine handles the full training loop with
parallel workers, cosine LR, deal enrichment, etc.

Tiers:
  Scout  — 500 steps, 4k games  (baseline / fast iteration)
  Knight — 2000 steps, 16k games (medium)
  Bishop — 8000 steps, 64k games (strong, overnight run)

Usage:
    # Train all 3 tiers + eval (default)
    python scripts/train_ulti.py

    # Train only, skip eval
    python scripts/train_ulti.py --no-eval

    # Eval only (models must already exist)
    python scripts/train_ulti.py --eval-only

    # Single tier
    python scripts/train_ulti.py --tiers scout

    # Parallel self-play (4 workers)
    python scripts/train_ulti.py --workers 4

    # Override solver settings
    python scripts/train_ulti.py --endgame-tricks 7 --pimc-dets 30

    # Enable deal enrichment
    python scripts/train_ulti.py --enrichment

    # Continue from checkpoint
    python scripts/train_ulti.py --tiers knight --load models/ulti/U1-Scout/model.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.game import soloist_won_simple
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, UltiNetWrapper
from trickster.training.ulti_hybrid import (
    UltiTrainConfig,
    UltiTrainStats,
    train_ulti_hybrid,
)


# ---------------------------------------------------------------------------
#  Tier definitions (parameters only — no training logic)
# ---------------------------------------------------------------------------

@dataclass
class UltiTier:
    name: str
    steps: int
    games_per_step: int
    train_steps: int
    sol_sims: int
    def_sims: int
    endgame_tricks: int
    pimc_dets: int
    solver_temp: float
    body_units: int
    body_layers: int
    lr_start: float
    lr_end: float
    batch_size: int
    buffer_size: int
    description: str

    @property
    def total_games(self) -> int:
        return self.steps * self.games_per_step


TIERS: dict[str, UltiTier] = {
    "scout": UltiTier(
        name="U1-Scout",
        steps=500, games_per_step=8, train_steps=50,
        sol_sims=20, def_sims=8,
        endgame_tricks=6, pimc_dets=20, solver_temp=0.5,
        body_units=256, body_layers=4,
        lr_start=1e-3, lr_end=2e-4,
        batch_size=64, buffer_size=50_000,
        description="Scout (500 steps, 4k games) — baseline",
    ),
    "knight": UltiTier(
        name="U2-Knight",
        steps=2000, games_per_step=8, train_steps=80,
        sol_sims=30, def_sims=12,
        endgame_tricks=6, pimc_dets=20, solver_temp=0.5,
        body_units=256, body_layers=4,
        lr_start=1e-3, lr_end=1e-4,
        batch_size=64, buffer_size=50_000,
        description="Knight (2000 steps, 16k games) — medium",
    ),
    "bishop": UltiTier(
        name="U3-Bishop",
        steps=8000, games_per_step=8, train_steps=100,
        sol_sims=30, def_sims=12,
        endgame_tricks=6, pimc_dets=20, solver_temp=0.5,
        body_units=256, body_layers=4,
        lr_start=1e-3, lr_end=5e-5,
        batch_size=64, buffer_size=50_000,
        description="Bishop (8000 steps, 64k games) — strong",
    ),
}

TIER_ORDER = ["scout", "knight", "bishop"]


# ---------------------------------------------------------------------------
#  Tier -> UltiTrainConfig conversion
# ---------------------------------------------------------------------------


def _tier_to_config(
    tier: UltiTier,
    *,
    seed: int = 42,
    device: str = "cpu",
    num_workers: int = 1,
    enrichment: bool = False,
    endgame_tricks: int | None = None,
    pimc_dets: int | None = None,
    solver_temp: float | None = None,
) -> UltiTrainConfig:
    """Build a UltiTrainConfig from a tier definition + CLI overrides."""
    return UltiTrainConfig(
        steps=tier.steps,
        games_per_step=tier.games_per_step,
        train_steps=tier.train_steps,
        batch_size=tier.batch_size,
        buffer_size=tier.buffer_size,
        lr_start=tier.lr_start,
        lr_end=tier.lr_end,
        sol_sims=tier.sol_sims,
        sol_dets=1,
        def_sims=tier.def_sims,
        def_dets=1,
        endgame_tricks=endgame_tricks if endgame_tricks is not None else tier.endgame_tricks,
        pimc_dets=pimc_dets if pimc_dets is not None else tier.pimc_dets,
        solver_temp=solver_temp if solver_temp is not None else tier.solver_temp,
        body_units=tier.body_units,
        body_layers=tier.body_layers,
        num_workers=num_workers,
        enrichment=enrichment,
        seed=seed,
        device=device,
    )


# ---------------------------------------------------------------------------
#  Progress callback (pretty-printing)
# ---------------------------------------------------------------------------


def _make_progress_cb(
    tier: UltiTier,
) -> tuple[callable, dict]:
    """Create a progress callback and a mutable context dict."""
    report_interval = max(1, tier.steps // 20)
    ctx = {"report_interval": report_interval}

    def on_progress(stats: UltiTrainStats) -> None:
        step = stats.step
        ri = ctx["report_interval"]
        if step % ri != 0 and step != 1 and step != stats.total_steps:
            return

        pct = step / stats.total_steps * 100
        elapsed = stats.train_time_s

        # Trend arrows
        window = max(1, ri)
        h_v = stats.history_vloss
        h_p = stats.history_ploss
        recent_v = np.mean(h_v[-window:]) if h_v else 0
        prev_v = np.mean(h_v[-2*window:-window]) if len(h_v) > window else recent_v
        recent_p = np.mean(h_p[-window:]) if h_p else 0
        prev_p = np.mean(h_p[-2*window:-window]) if len(h_p) > window else recent_p
        recent_acc = np.mean(stats.history_pacc[-window:]) if stats.history_pacc else 0

        v_arrow = "↓" if recent_v < prev_v - 0.005 else ("↑" if recent_v > prev_v + 0.005 else "→")
        p_arrow = "↓" if recent_p < prev_p - 0.01 else ("↑" if recent_p > prev_p + 0.01 else "→")

        print(
            f"    step {step:>5d}/{stats.total_steps} ({pct:4.0f}%)  "
            f"games={stats.total_games:>5d}  "
            f"vloss={stats.vloss:.4f}{v_arrow} "
            f"(sol={stats.sol_vloss:.3f} def={stats.def_vloss:.3f})  "
            f"ploss={stats.ploss:.4f}{p_arrow}  "
            f"pacc={recent_acc:.0%}  "
            f"lr={stats.lr:.1e}  "
            f"[{elapsed:.0f}s]",
            flush=True,
        )

    return on_progress, ctx


# ---------------------------------------------------------------------------
#  Train one tier
# ---------------------------------------------------------------------------


def train_tier(
    tier: UltiTier,
    *,
    seed: int = 42,
    device: str = "cpu",
    num_workers: int = 1,
    enrichment: bool = False,
    load_path: str | None = None,
    endgame_tricks: int | None = None,
    pimc_dets: int | None = None,
    solver_temp: float | None = None,
) -> tuple[UltiNet, Path]:
    """Train one tier using the shared engine. Returns (net, model_dir)."""
    model_dir = Path("models") / "ulti" / tier.name
    model_dir.mkdir(parents=True, exist_ok=True)

    game = UltiGame()
    cfg = _tier_to_config(
        tier, seed=seed, device=device,
        num_workers=num_workers,
        enrichment=enrichment,
        endgame_tricks=endgame_tricks,
        pimc_dets=pimc_dets,
        solver_temp=solver_temp,
    )

    # Load checkpoint if provided
    initial_net = None
    if load_path is not None:
        cp = torch.load(load_path, weights_only=False, map_location=device)
        initial_net = UltiNet(
            input_dim=game.state_dim,
            body_units=cfg.body_units,
            body_layers=cfg.body_layers,
            action_dim=game.action_space_size,
        )
        initial_net.load_state_dict(cp["model_state_dict"])
        print(f"    Loaded checkpoint from {load_path}")

    # Print config
    param_count = sum(
        p.numel() for p in (initial_net or UltiNet(
            input_dim=game.state_dim,
            body_units=cfg.body_units,
            body_layers=cfg.body_layers,
            action_dim=game.action_space_size,
        )).parameters()
    )
    print(f"    Net: {cfg.body_units}x{cfg.body_layers} ({param_count:,} params)")
    print(f"    MCTS: sol={cfg.sol_sims}s def={cfg.def_sims}s | "
          f"Solver: {SOLVER_ENGINE} (endgame={cfg.endgame_tricks}t, "
          f"PIMC={cfg.pimc_dets}d)")
    print(f"    LR: {cfg.lr_start} → {cfg.lr_end} (cosine)  "
          f"batch={cfg.batch_size}  SGD/step={cfg.train_steps}  "
          f"buffer={cfg.buffer_size:,}")
    workers_str = f"{cfg.num_workers} (parallel)" if cfg.num_workers > 1 else "1 (sequential)"
    print(f"    Workers: {workers_str}")
    if cfg.enrichment:
        print(f"    Enrichment: ON (warmup={cfg.enrich_warmup}, "
              f"random_frac={cfg.enrich_random_frac:.0%})")
    print()

    # Train
    on_progress, _ctx = _make_progress_cb(tier)
    net, stats = train_ulti_hybrid(
        cfg,
        initial_net=initial_net,
        on_progress=on_progress,
    )

    # Summary
    ri = _ctx["report_interval"]
    final_v = np.mean(stats.history_vloss[-ri:]) if stats.history_vloss else 0
    final_p = np.mean(stats.history_ploss[-ri:]) if stats.history_ploss else 0
    final_acc = np.mean(stats.history_pacc[-ri:]) if stats.history_pacc else 0
    start_v = np.mean(stats.history_vloss[:ri]) if len(stats.history_vloss) > ri else final_v
    start_p = np.mean(stats.history_ploss[:ri]) if len(stats.history_ploss) > ri else final_p
    start_acc = np.mean(stats.history_pacc[:ri]) if len(stats.history_pacc) > ri else final_acc

    print()
    print(f"    ┌─ TRAINING SUMMARY ──────────────────────────────────────")
    print(f"    │  Games: {stats.total_games:,}  Samples: {stats.total_samples:,}  "
          f"SGD steps: {stats.total_sgd_steps:,}  Time: {stats.train_time_s:.0f}s")
    print(f"    │  Value head:   {start_v:.4f} → {final_v:.4f}  "
          f"({'improved' if final_v < start_v - 0.001 else 'plateau'})")
    print(f"    │  Policy head:  {start_p:.4f} → {final_p:.4f}  "
          f"({'improved' if final_p < start_p - 0.01 else 'plateau'})")
    print(f"    │  Policy acc:   {start_acc:.1%} → {final_acc:.1%}  "
          f"({'improved' if final_acc > start_acc + 0.01 else 'plateau'})")
    print(f"    └──────────────────────────────────────────────────────────")

    # Save
    save_path = model_dir / "model.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": cfg.body_units,
        "body_layers": cfg.body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "training_mode": cfg.training_mode,
        "method": "hybrid",
        "endgame_tricks": cfg.endgame_tricks,
        "pimc_dets": cfg.pimc_dets,
        "total_games": stats.total_games,
        "total_samples": stats.total_samples,
        "total_sgd_steps": stats.total_sgd_steps,
        "train_time_s": round(stats.train_time_s, 1),
        "final_vloss": round(final_v, 6),
        "final_ploss": round(final_p, 6),
        "final_pacc": round(final_acc, 4),
    }, save_path)

    info = {
        "tier": tier.name,
        "steps": tier.steps,
        "games_per_step": tier.games_per_step,
        "train_steps_per_iter": tier.train_steps,
        "total_games": stats.total_games,
        "total_samples": stats.total_samples,
        "total_sgd_steps": stats.total_sgd_steps,
        "sol_sims": tier.sol_sims,
        "def_sims": tier.def_sims,
        "endgame_tricks": cfg.endgame_tricks,
        "pimc_dets": cfg.pimc_dets,
        "body_units": tier.body_units,
        "body_layers": tier.body_layers,
        "lr_start": tier.lr_start,
        "lr_end": tier.lr_end,
        "num_workers": cfg.num_workers,
        "enrichment": cfg.enrichment,
        "solver": SOLVER_ENGINE,
        "train_time_s": round(stats.train_time_s, 1),
        "final_vloss": round(final_v, 6),
        "final_ploss": round(final_p, 6),
        "final_pacc": round(final_acc, 4),
    }
    (model_dir / "train_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8",
    )

    print(f"    Saved to {save_path}")
    return net, model_dir


# ---------------------------------------------------------------------------
#  Head-to-head evaluation
# ---------------------------------------------------------------------------


def load_model(model_dir: Path, device: str = "cpu") -> tuple[UltiNet, UltiNetWrapper]:
    """Load a saved UltiNet from a tier directory."""
    cp = torch.load(model_dir / "model.pt", weights_only=False, map_location=device)
    net = UltiNet(
        input_dim=cp.get("input_dim", 291),
        body_units=cp.get("body_units", 256),
        body_layers=cp.get("body_layers", 4),
        action_dim=cp.get("action_dim", 32),
    )
    net.load_state_dict(cp["model_state_dict"])
    wrapper = UltiNetWrapper(net, device=device)
    return net, wrapper


def play_eval_game(
    game: UltiGame,
    sol_wrapper: UltiNetWrapper | None,
    def_wrapper: UltiNetWrapper | None,
    mcts_cfg: MCTSConfig,
    seed: int,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
) -> tuple[bool, str | None]:
    """Play one eval game. Returns (soloist_won, trump_name)."""
    rng = random.Random(seed)
    state = game.new_game(
        seed=seed,
        training_mode="simple",
        starting_leader=seed % 3,
    )
    soloist_idx = state.gs.soloist
    trump_name = state.gs.trump.value if state.gs.trump else None

    sol_hybrid = None
    if sol_wrapper is not None:
        sol_hybrid = HybridPlayer(
            game, sol_wrapper,
            mcts_config=mcts_cfg,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,
        )

    def_hybrid = None
    if def_wrapper is not None:
        def_hybrid = HybridPlayer(
            game, def_wrapper,
            mcts_config=mcts_cfg,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,
        )

    rng_rand = random.Random(seed + 50000)

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        is_soloist = (player == soloist_idx)
        hybrid = sol_hybrid if is_soloist else def_hybrid

        if hybrid is not None:
            action = hybrid.choose_action(state, player, rng)
        else:
            action = rng_rand.choice(actions)

        state = game.apply(state, action)

    won = soloist_won_simple(state.gs)
    return won, trump_name


def play_match(
    game: UltiGame,
    wrapper_a: UltiNetWrapper | None,
    wrapper_b: UltiNetWrapper | None,
    deals: int = 200,
    seed: int = 0,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
) -> dict:
    """Play a match: A as soloist vs B as defenders, then swap."""
    mcts_cfg = MCTSConfig(
        simulations=20, determinizations=1,
        c_puct=1.5, dirichlet_alpha=0.0, dirichlet_weight=0.0,
        use_value_head=True, use_policy_priors=True, visit_temp=0.1,
    )

    a_sol_wins = 0
    a_sol_total = deals // 2
    for g in range(a_sol_total):
        won, _ = play_eval_game(
            game, wrapper_a, wrapper_b, mcts_cfg,
            seed=seed + g,
            endgame_tricks=endgame_tricks,
            pimc_dets=pimc_dets,
        )
        if won:
            a_sol_wins += 1

    b_sol_wins = 0
    b_sol_total = deals - a_sol_total
    for g in range(b_sol_total):
        won, _ = play_eval_game(
            game, wrapper_b, wrapper_a, mcts_cfg,
            seed=seed + a_sol_total + g,
            endgame_tricks=endgame_tricks,
            pimc_dets=pimc_dets,
        )
        if won:
            b_sol_wins += 1

    a_wins = a_sol_wins + (b_sol_total - b_sol_wins)
    b_wins = deals - a_wins

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "deals": deals,
        "a_wr": a_wins / deals,
        "a_sol_wr": a_sol_wins / max(1, a_sol_total),
        "b_sol_wr": b_sol_wins / max(1, b_sol_total),
    }


# ---------------------------------------------------------------------------
#  Round-robin evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    tier_names: list[str],
    eval_deals: int = 200,
    device: str = "cpu",
) -> None:
    """Round-robin evaluation of all trained tiers + random baseline."""
    game = UltiGame()
    agents: list[tuple[str, UltiNetWrapper | None]] = []
    agents.append(("Random", None))

    for tier_key in tier_names:
        tier = TIERS[tier_key]
        model_dir = Path("models") / "ulti" / tier.name
        if not (model_dir / "model.pt").exists():
            print(f"  Skipping {tier.name} — no model found at {model_dir}")
            continue
        _, wrapper = load_model(model_dir, device)

        info_path = model_dir / "train_info.json"
        label = tier.name
        if info_path.exists():
            info = json.loads(info_path.read_text())
            label = f"{tier.name} ({info.get('total_games', '?')}g)"
        agents.append((label, wrapper))

    n = len(agents)
    if n < 2:
        print("  Need at least 2 agents for evaluation.")
        return

    print(f"  Round-robin: {n} agents, {eval_deals} deals per matchup")
    print(f"  Solver: {SOLVER_ENGINE}")
    print()

    total_wins = [0] * n
    total_deals = [0] * n
    margin_table: list[list[str]] = [["-"] * n for _ in range(n)]
    matchup_idx = 0
    total_matchups = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            matchup_idx += 1
            name_a = agents[i][0]
            name_b = agents[j][0]
            print(
                f"  [{matchup_idx}/{total_matchups}] {name_a} vs {name_b}...",
                end="", flush=True,
            )
            t0 = time.perf_counter()
            result = play_match(
                game,
                agents[i][1], agents[j][1],
                deals=eval_deals,
                seed=i * 10000 + j * 100,
            )
            elapsed = time.perf_counter() - t0

            a_wr = result["a_wr"]
            b_wr = 1 - a_wr
            total_wins[i] += result["a_wins"]
            total_wins[j] += result["b_wins"]
            total_deals[i] += result["deals"]
            total_deals[j] += result["deals"]

            margin_table[i][j] = f"{a_wr:.0%}"
            margin_table[j][i] = f"{b_wr:.0%}"

            winner = name_a if a_wr > 0.5 else name_b if a_wr < 0.5 else "DRAW"
            print(
                f"  A={a_wr:.0%} "
                f"(sol: A={result['a_sol_wr']:.0%} B={result['b_sol_wr']:.0%})  "
                f"→ {winner}  [{elapsed:.1f}s]"
            )

    # Ranking
    print()
    print("  ┌─ RANKING " + "─" * 55)
    print("  │")

    ranking = sorted(
        range(n),
        key=lambda k: total_wins[k] / max(1, total_deals[k]),
        reverse=True,
    )
    max_name = max(len(agents[k][0]) for k in ranking)

    print(f"  │  {'Rank':<5} {'Agent':<{max_name+2}} {'Wins':>6} {'/ Deals':>8}  {'Win rate':>8}")
    print(f"  │  {'─'*5} {'─'*(max_name+2)} {'─'*6} {'─'*8}  {'─'*8}")

    for rank, k in enumerate(ranking, 1):
        name = agents[k][0]
        wins = total_wins[k]
        deals = total_deals[k]
        wr = wins / max(1, deals)
        medal = ">> " if rank == 1 else "   "
        print(f"  │  {medal}{rank:<3} {name:<{max_name+2}} {wins:>5} / {deals:<5}  {wr:>8.1%}")

    print("  │")

    # Margin table
    print("  │  Matchup table (cell = row's win rate vs column):")
    header = "  │  " + " " * (max_name + 2)
    for k in ranking:
        short = agents[k][0][:12]
        header += f" {short:>12}"
    print(header)
    for k in ranking:
        row = f"  │  {agents[k][0]:<{max_name+2}}"
        for j in ranking:
            row += f" {margin_table[k][j]:>12}"
        print(row)

    print("  │")
    print("  └" + "─" * 65)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Ulti Parti models at 3 tiers + head-to-head eval",
    )
    # Tier selection
    parser.add_argument("--tiers", type=str, nargs="+", default=TIER_ORDER,
                        choices=list(TIERS.keys()),
                        help="Which tiers to train (default: all 3)")

    # Overrides (applied to all tiers)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel self-play processes (default: 1)")
    parser.add_argument("--endgame-tricks", type=int, default=None,
                        help="Override endgame solver depth")
    parser.add_argument("--pimc-dets", type=int, default=None,
                        help="Override PIMC determinizations")
    parser.add_argument("--solver-temp", type=float, default=None,
                        help="Override solver temperature")
    parser.add_argument("--enrichment", action="store_true",
                        help="Enable deal enrichment via value head")

    # Checkpoint
    parser.add_argument("--load", type=str, default=None,
                        help="Load pre-trained model to continue training")

    # Evaluation
    parser.add_argument("--eval-deals", type=int, default=200,
                        help="Deals per evaluation matchup (default 200)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing models")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation (training only)")

    # General
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (default cpu)")
    args = parser.parse_args()

    tier_keys = args.tiers

    # ── Training ──────────────────────────────────────────────────────
    if not args.eval_only:
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║         ULTI PARTI — TIERED TRAINING                            ║")
        print("║         Hybrid self-play: MCTS + alpha-beta solver              ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        total_games = sum(TIERS[k].total_games for k in tier_keys)
        print(f"  Tiers: {', '.join(TIERS[k].name for k in tier_keys)}")
        print(f"  Total budget: {total_games:,} games")
        print(f"  Solver: {SOLVER_ENGINE}")
        workers_str = f"{args.workers} (parallel)" if args.workers > 1 else "1 (sequential)"
        print(f"  Workers: {workers_str}")
        if args.enrichment:
            print(f"  Deal enrichment: ON")
        print()

        for tier_key in tier_keys:
            tier = TIERS[tier_key]
            print(f"  ┌─ {tier.name}: {tier.description} " + "─" * 20)
            print(f"  │  Budget: {tier.steps} steps × {tier.games_per_step} gpi = "
                  f"{tier.total_games:,} games, {tier.train_steps} SGD/step")

            train_tier(
                tier,
                seed=args.seed,
                device=args.device,
                num_workers=args.workers,
                enrichment=args.enrichment,
                load_path=args.load,
                endgame_tricks=args.endgame_tricks,
                pimc_dets=args.pimc_dets,
                solver_temp=args.solver_temp,
            )

            print(f"  └─ {tier.name} complete")
            print()

            # Only apply --load to the first tier
            args.load = None

    # ── Evaluation ────────────────────────────────────────────────────
    if not args.no_eval:
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║         HEAD-TO-HEAD EVALUATION                                 ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

        run_evaluation(
            tier_keys,
            eval_deals=args.eval_deals,
            device=args.device,
        )


if __name__ == "__main__":
    main()
