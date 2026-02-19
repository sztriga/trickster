#!/usr/bin/env python3
"""Ulti training with bidding.

Trains all contract models together from scratch using value-head
bidding, kontra, and neural discard.

Multiple tiers can be trained sequentially in a single invocation.

Usage:
    python scripts/train_e2e.py knight_light                     # single tier
    python scripts/train_e2e.py knight_light knight_balanced      # batch
    python scripts/train_e2e.py bronze --steps 4000 -v            # override steps
    python scripts/train_e2e.py bronze --workers 4                # parallel
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from trickster.games.ulti.adapter import UltiGame
from trickster.hybrid import SOLVER_ENGINE
from trickster.train_utils import _GAME_PTS_MAX
from trickster.training.bidding_loop import (
    BiddingTrainConfig,
    DISPLAY_ORDER,
    train_with_bidding,
)
from trickster.model import _ort_available
from trickster.training.model_io import (
    DK_LABELS,
    auto_device,
    estimate_params,
)
from trickster.training.progress import (
    bidding_progress_bar,
    bidding_progress_verbose,
    fmt_pts,
    fmt_time,
)
from trickster.training.tiers import (
    CONTRACT_KEYS,
    CONTRACTS,
    TIERS,
)


def train_tier(tier_name: str, args) -> None:
    """Train a single tier from scratch."""
    tier = TIERS[tier_name]
    steps = args.steps or tier.steps
    resolved_device = auto_device(
        tier.body_units, tier.body_layers,
        force=args.device, gpu_tier=tier.gpu,
    )
    params = estimate_params(tier.body_units, tier.body_layers)

    # Inference/solver backend strings
    if resolved_device == "cpu":
        inf_str = "ONNX (CPU)" if _ort_available() else "PyTorch (CPU)"
    else:
        inf_str = f"PyTorch ({resolved_device.upper()})"
    solver_str = SOLVER_ENGINE

    # Warn if GPU tier falls back to CPU (no CUDA)
    if tier.gpu and resolved_device == "cpu":
        print("\n  WARNING: GPU tier selected but no CUDA device available — "
              "falling back to CPU.\n")

    t0 = time.perf_counter()

    # Banner
    print()
    lines = [
        "ULTI TRAINING",
        f"Tier: {tier_name} — {tier.description}",
        f"Budget: {steps:,} steps × {tier.games_per_step} games/step"
        f" = {steps * tier.games_per_step:,} deals",
    ]
    w = max(len(l) for l in lines) + 4
    print("╔" + "═" * w + "╗")
    for l in lines:
        print(f"║  {l:<{w-2}}║")
    print("╚" + "═" * w + "╝")
    print(f"  Net: {tier.body_units}×{tier.body_layers} (~{params:,} params)")
    print(f"  Inference: {inf_str}  |  Solver: {solver_str}")
    if tier.gpu and resolved_device != "cpu":
        print(f"  Self-play: {tier.games_per_step} concurrent games (GPU batching)")
    elif args.workers > 1:
        print(f"  Self-play: {args.workers} workers (process pool)")
    else:
        print(f"  Self-play: sequential")
    print()

    # ── Training ──
    print("  Starting from random weights")
    print()

    pool_sources = args.opponent_pool or []
    cfg = BiddingTrainConfig(
        steps=steps,
        games_per_step=tier.games_per_step,
        train_steps=tier.train_steps,
        sol_sims=tier.sol_sims,
        sol_dets=tier.sol_dets,
        def_sims=tier.def_sims,
        def_dets=tier.def_dets,
        leaf_batch_size=8,
        endgame_tricks=tier.endgame_tricks,
        pimc_dets=tier.pimc_dets,
        solver_temp=tier.solver_temp,
        body_units=tier.body_units,
        body_layers=tier.body_layers,
        lr_start=5e-4,
        lr_end=1e-4,
        buffer_size=tier.buffer_size,
        batch_size=tier.batch_size,
        max_discards=15,
        min_bid_pts=0.0,
        exploration_frac=0.2,
        contract_keys=CONTRACT_KEYS,
        num_workers=args.workers,
        seed=args.seed,
        device=resolved_device,
        opponent_pool=pool_sources,
        pool_frac=args.pool_frac,
    )

    e2e_reuse = (steps * cfg.train_steps * cfg.batch_size) / cfg.buffer_size
    print(f"  Budget: {steps} steps × {cfg.games_per_step} gpi "
          f"= {steps * cfg.games_per_step:,} deals")
    print(f"  SGD: {cfg.train_steps}/step  buffer={cfg.buffer_size:,}  "
          f"batch={cfg.batch_size}  reuse={e2e_reuse:.0f}x")
    print(f"  Contracts: {', '.join(DK_LABELS[dk] for dk in DISPLAY_ORDER)}")
    if pool_sources:
        print(f"  Opponent pool: {', '.join(pool_sources)} "
              f"({cfg.pool_frac:.0%} of games)")
    else:
        print("  Opponent pool: none (pure self-play)")
    print()

    progress_fn = (bidding_progress_verbose if args.verbose else bidding_progress_bar)(cfg)
    slots, final_stats = train_with_bidding(
        cfg,
        initial_nets={},
        on_progress=progress_fn,
    )

    # Save
    save_base = Path(f"models/ulti/{tier_name}/final")
    game = UltiGame()

    for key, slot in slots.items():
        out_dir = save_base / key
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": slot.net.state_dict(),
            "body_units": cfg.body_units,
            "body_layers": cfg.body_layers,
            "input_dim": game.state_dim,
            "action_dim": game.action_space_size,
            "training_mode": CONTRACTS[key].training_mode,
            "method": "bidding_hybrid",
            "total_games": slot.games,
            "total_samples": slot.samples,
            "total_sgd_steps": slot.sgd_steps,
        }, out_dir / "model.pt")

    # Final Summary
    elapsed = time.perf_counter() - t0

    total_games = sum(s.games for s in slots.values())
    total_deals = cfg.steps * cfg.games_per_step
    total_passes = total_deals - total_games

    print()
    w = 64
    print("  ┌─ TRAINING COMPLETE " + "─" * (w - 21))
    print("  │")
    print(f"  │  Tier:     {tier_name}")
    print(f"  │  Output:   {save_base}/")
    print(f"  │  Time:     {fmt_time(elapsed)} "
          f"({total_games:,} games, {total_passes:,} passes)")
    print("  │")

    print(f"  │  {'contract':<10} {'games':>5} {'avg_pts':>7} {'win%':>5}")
    print(f"  │  {'─'*10} {'─'*5} {'─'*7} {'─'*5}")
    for dk in DISPLAY_ORDER:
        g = final_stats.cumulative_games.get(dk, 0)
        if g == 0:
            continue
        pts = final_stats.cumulative_pts.get(dk, 0.0)
        wins = final_stats.cumulative_wins.get(dk, 0)
        label = DK_LABELS.get(dk, dk)
        print(f"  │  {label:<10} {g:>5} {fmt_pts(pts, g):>7} {wins/g*100:4.0f}%")

    print("  └" + "─" * w)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ulti training with bidding. "
                    "Trains from scratch — no base models needed.",
    )
    parser.add_argument(
        "tiers", nargs="+", choices=list(TIERS.keys()),
        metavar="TIER",
        help=f"Tier(s) to train: {', '.join(TIERS.keys())}",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--opponent-pool", nargs="*", default=None,
        help="Model sources for opponent pool (e.g. scout knight_light). "
             "Defenders in pool games use a randomly chosen pool model.",
    )
    parser.add_argument(
        "--pool-frac", type=float, default=0.5,
        help="Fraction of games played vs pool opponents (default 0.5)",
    )
    args = parser.parse_args()

    # Validate tier names
    valid = set(TIERS.keys())
    bad = [t for t in args.tiers if t not in valid]
    if bad:
        parser.error(f"unknown tier(s): {', '.join(bad)}. "
                     f"Choose from: {', '.join(valid)}")

    n = len(args.tiers)
    t_all = time.perf_counter()

    if n > 1:
        print()
        print(f"  Queued {n} tiers: {', '.join(args.tiers)}")

    for i, tier_name in enumerate(args.tiers, 1):
        if n > 1:
            print()
            print("=" * 64)
            print(f"  TIER {i}/{n}: {tier_name}")
            print("=" * 64)
        train_tier(tier_name, args)

    if n > 1:
        elapsed = time.perf_counter() - t_all
        print("=" * 64)
        print(f"  ALL {n} TIERS COMPLETE in {fmt_time(elapsed)}")
        print(f"  Trained: {', '.join(args.tiers)}")
        print("=" * 64)
        print()


if __name__ == "__main__":
    main()
