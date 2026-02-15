#!/usr/bin/env python3
"""End-to-end Ulti training: base models + bidding.

Phase 1 — Base training:
    Trains individual contract models (parti, ulti, betli, 40-100) at the
    chosen tier using hybrid self-play.

Phase 2 — Bidding training:
    Takes the Phase 1 models and trains them together with value-head
    bidding, kontra, and neural discard.  Produces the final e2e models.

Usage:
    python scripts/train_e2e.py knight                          # full pipeline
    python scripts/train_e2e.py bishop --workers 6              # heavier tier
    python scripts/train_e2e.py knight --skip-base              # Phase 2 only
    python scripts/train_e2e.py knight --bidding-steps 4000 -v  # verbose
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame
from trickster.hybrid import SOLVER_ENGINE
from trickster.train_utils import _GAME_PTS_MAX
from trickster.training.bidding_loop import (
    BiddingTrainConfig,
    DISPLAY_ORDER,
    train_with_bidding,
)
from trickster.training.contract_loop import train_one_tier
from trickster.training.model_io import (
    DK_LABELS,
    auto_device,
    estimate_params,
    load_net,
    resolve_paths,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Ulti training: base models → bidding",
    )
    parser.add_argument(
        "tier", choices=list(TIERS.keys()),
        help="Tier to train (defines net size, MCTS budget, etc.)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip Phase 1 (base models must already exist)")
    parser.add_argument("--base-from", type=str, default=None,
                        help="Load base models from a different tier (e.g. bronze)")
    parser.add_argument("--bidding-steps", type=int, default=None,
                        help="Override bidding training steps")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--opponent-pool", nargs="*", default=None,
        help="Model sources for opponent pool (e.g. scout knight). "
             "Defenders in pool games use a randomly chosen pool model.",
    )
    parser.add_argument(
        "--pool-frac", type=float, default=0.5,
        help="Fraction of games played vs pool opponents (default 0.5)",
    )
    args = parser.parse_args()

    tier = TIERS[args.tier]
    tier_name = args.tier
    bidding_steps = args.bidding_steps or tier.e2e_steps
    resolved_device = auto_device(tier.body_units, tier.body_layers, force=args.device)
    params = estimate_params(tier.body_units, tier.body_layers)

    t0 = time.perf_counter()

    # Banner
    print()
    lines = [
        "END-TO-END ULTI TRAINING",
        f"Tier: {tier_name} — {tier.description}",
        f"Phase 1: base training ({len(CONTRACT_KEYS)} contracts × {tier.total_games:,} games each)",
        f"Phase 2: bidding training ({bidding_steps:,} steps × {tier.e2e_gpi} games/step)",
    ]
    w = max(len(l) for l in lines) + 4
    print("╔" + "═" * w + "╗")
    for l in lines:
        print(f"║  {l:<{w-2}}║")
    print("╚" + "═" * w + "╝")
    print(f"  Net: {tier.body_units}×{tier.body_layers} (~{params:,} params)")
    print(f"  Device: {resolved_device}")
    print(f"  Workers: {args.workers}  Solver: {SOLVER_ENGINE}")
    print()

    # ── Phase 1: Base Training ──
    if not args.skip_base:
        print("━" * 60)
        print("  PHASE 1 — BASE CONTRACT TRAINING")
        print("━" * 60)
        print()

        for ckey in CONTRACT_KEYS:
            spec = CONTRACTS[ckey]
            name = tier.tier_name(spec.name_prefix)
            print(f"  ┌─ {name}: {spec.display_name} " + "─" * 30)
            print(f"  │  Budget: {tier.steps} steps × {tier.games_per_step} gpi = "
                  f"{tier.total_games:,} games")

            train_one_tier(
                tier, spec,
                seed=args.seed,
                device=args.device,
                num_workers=args.workers,
                enrichment=True,
                verbose=args.verbose,
            )

            print(f"  └─ {name} complete")
            print()

        elapsed = time.perf_counter() - t0
        print(f"  Phase 1 complete in {fmt_time(elapsed)}")
        print()
    else:
        base_tier = args.base_from or tier_name
        source = f"{base_tier}_base"
        paths = resolve_paths(source)
        missing = [k for k, p in paths.items() if not (p / "model.pt").exists()]
        if missing:
            print(f"  ERROR: --skip-base but missing base models ({source}): {', '.join(missing)}")
            sys.exit(1)
        if args.base_from:
            print(f"  Phase 1 skipped (borrowing base models from {source})")
        else:
            print(f"  Phase 1 skipped (using existing {source} models)")
        print()

    # ── Phase 2: Bidding Training ──
    print("━" * 60)
    print("  PHASE 2 — BIDDING TRAINING")
    print("━" * 60)
    print()

    t_phase2 = time.perf_counter()

    base_tier = args.base_from or tier_name
    source = f"{base_tier}_base"
    model_paths = resolve_paths(source)
    initial_nets = {}
    print(f"  Loading base models ({source}):")
    for key in CONTRACT_KEYS:
        path = model_paths[key]
        net = load_net(path, device="cpu")
        if net is not None:
            initial_nets[key] = net
            print(f"    {key:<8} ← {path}")
        else:
            print(f"    {key:<8} — not found, starting fresh")
    print()

    pool_sources = args.opponent_pool or []
    cfg = BiddingTrainConfig(
        steps=bidding_steps,
        games_per_step=tier.e2e_gpi,
        train_steps=tier.e2e_train_steps,
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
        buffer_size=tier.e2e_buffer_size,
        batch_size=tier.batch_size,
        max_discards=15,
        min_bid_pts=-2.0,
        exploration_frac=0.2,
        contract_keys=CONTRACT_KEYS,
        num_workers=args.workers,
        seed=args.seed,
        device=args.device,
        opponent_pool=pool_sources,
        pool_frac=args.pool_frac,
    )

    e2e_reuse = (bidding_steps * cfg.train_steps * cfg.batch_size) / cfg.buffer_size
    print(f"  Budget: {bidding_steps} steps × {cfg.games_per_step} gpi "
          f"= {bidding_steps * cfg.games_per_step:,} deals")
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
        initial_nets=initial_nets,
        on_progress=progress_fn,
    )

    # Save
    save_base = Path(f"models/e2e/{tier_name}")
    game = UltiGame()

    for key, slot in slots.items():
        cdef = CONTRACT_DEFS[key]
        out_dir = save_base / cdef.model_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": slot.net.state_dict(),
            "body_units": cfg.body_units,
            "body_layers": cfg.body_layers,
            "input_dim": game.state_dim,
            "action_dim": game.action_space_size,
            "training_mode": cdef.training_mode,
            "method": "bidding_hybrid",
            "total_games": slot.games,
            "total_samples": slot.samples,
            "total_sgd_steps": slot.sgd_steps,
        }, out_dir / "model.pt")

    # Final Summary
    elapsed_total = time.perf_counter() - t0
    elapsed_phase2 = time.perf_counter() - t_phase2

    total_games = sum(s.games for s in slots.values())
    total_deals = cfg.steps * cfg.games_per_step
    total_passes = total_deals - total_games

    print()
    w = 64
    print("  ┌─ E2E TRAINING COMPLETE " + "─" * (w - 25))
    print("  │")
    print(f"  │  Tier:     {tier_name}")
    print(f"  │  Output:   {save_base}/")
    print(f"  │  Phase 1:  {fmt_time(elapsed_total - elapsed_phase2) if not args.skip_base else 'skipped'}")
    print(f"  │  Phase 2:  {fmt_time(elapsed_phase2)} "
          f"({total_games:,} games, {total_passes:,} passes)")
    print(f"  │  Total:    {fmt_time(elapsed_total)}")
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


if __name__ == "__main__":
    main()
