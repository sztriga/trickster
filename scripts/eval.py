#!/usr/bin/env python3
"""CLI evaluation script — same interface as the GUI Evaluate tab.

Usage examples:

  # Evaluate a model vs random (auto-detects MCTS)
  python scripts/eval.py vs-random models/Direct --deals 2000 --workers 4

  # Head-to-head compare (auto-detects MCTS per side)
  python scripts/eval.py compare models/Direct models/MCTS --deals 2000 --workers 4

  # List available models
  python scripts/eval.py list
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.training.eval import (
    EvalStats,
    evaluate_head_to_head,
    evaluate_head_to_head_parallel,
    evaluate_policies,
    evaluate_policies_parallel,
    evaluate_policy_vs_random,
    evaluate_policy_vs_random_parallel,
    evaluate_policy_vs_random_mcts,
    evaluate_policy_vs_random_mcts_parallel,
)
from trickster.training.model_spec import list_model_dirs, model_label_from_dir, read_spec
from trickster.training.model_store import load_slot, slot_exists


def _detect_mcts(model_dir: Path):
    """Return an MCTSConfig if the model was MCTS-trained, else None."""
    try:
        if read_spec(model_dir / "spec.json").method == "mcts":
            from trickster.games.snapszer.mcts_agent import MCTSConfig
            return MCTSConfig(simulations=50, determinizations=4, c=1.4)
    except Exception:
        pass
    return None


def _print_stats(label_a: str, label_b: str, stats: EvalStats) -> None:
    print(f"  {label_a}: {stats.a_points} pts ({stats.a_ppd:.2f}/deal)")
    print(f"  {label_b}: {stats.b_points} pts ({stats.b_ppd:.2f}/deal)")
    diff = stats.a_ppd - stats.b_ppd
    print(f"  Difference: {diff:+.2f} pts/deal over {stats.deals} deals")


def cmd_list(args) -> int:
    dirs = list_model_dirs(root="models")
    if not dirs:
        print("No models found in models/")
        return 0
    for d in dirs:
        label = model_label_from_dir(d)
        has_latest = "ok" if (d / "latest.pkl").exists() else "no latest.pkl"
        print(f"  {d.name:30s}  {label}  [{has_latest}]")
    return 0


def cmd_vs_random(args) -> int:
    model_dir = Path(args.model)
    if not slot_exists("latest", models_dir=model_dir):
        print(f"Error: {model_dir}/latest.pkl not found")
        return 1

    mcts_cfg = _detect_mcts(model_dir)
    mode = "mcts" if mcts_cfg else "direct"
    print(f"Evaluating {model_dir.name} ({mode}) vs random — {args.deals} deals, {args.workers} workers")

    pol = load_slot("latest", models_dir=model_dir)
    t0 = time.perf_counter()

    if mcts_cfg:
        if args.workers <= 1:
            stats = evaluate_policy_vs_random_mcts(pol, games=args.deals, seed=args.seed, mcts_config=mcts_cfg)
        else:
            stats = evaluate_policy_vs_random_mcts_parallel(
                pol, games=args.deals, seed=args.seed, workers=args.workers, mcts_config=mcts_cfg,
            )
    else:
        if args.workers <= 1:
            stats = evaluate_policy_vs_random(pol, games=args.deals, seed=args.seed)
        else:
            stats = evaluate_policy_vs_random_parallel(pol, games=args.deals, seed=args.seed, workers=args.workers)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")
    _print_stats(model_dir.name, "random", stats)
    return 0


def cmd_compare(args) -> int:
    a_dir = Path(args.model_a)
    b_dir = Path(args.model_b)
    if not slot_exists("latest", models_dir=a_dir):
        print(f"Error: {a_dir}/latest.pkl not found")
        return 1
    if not slot_exists("latest", models_dir=b_dir):
        print(f"Error: {b_dir}/latest.pkl not found")
        return 1

    cfg_a = _detect_mcts(a_dir)
    cfg_b = _detect_mcts(b_dir)
    mode_a = "mcts" if cfg_a else "direct"
    mode_b = "mcts" if cfg_b else "direct"

    print(f"Comparing A={a_dir.name} ({mode_a}) vs B={b_dir.name} ({mode_b})")
    print(f"  {args.deals} deals, {args.workers} workers")

    pa = load_slot("latest", models_dir=a_dir)
    pb = load_slot("latest", models_dir=b_dir)
    t0 = time.perf_counter()

    if cfg_a is not None or cfg_b is not None:
        if args.workers <= 1:
            stats = evaluate_head_to_head(
                pa, pb, games=args.deals, seed=args.seed,
                mcts_config_a=cfg_a, mcts_config_b=cfg_b,
            )
        else:
            stats = evaluate_head_to_head_parallel(
                pa, pb, games=args.deals, seed=args.seed, workers=args.workers,
                mcts_config_a=cfg_a, mcts_config_b=cfg_b,
            )
    else:
        if args.workers <= 1:
            stats = evaluate_policies(pa, pb, games=args.deals, seed=args.seed)
        else:
            stats = evaluate_policies_parallel(pa, pb, games=args.deals, seed=args.seed, workers=args.workers)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")
    _print_stats(a_dir.name, b_dir.name, stats)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate Snapszer models")
    sub = p.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List available models")

    # vs-random
    vr = sub.add_parser("vs-random", help="Evaluate model vs random opponent")
    vr.add_argument("model", help="Path to model directory (e.g. models/Direct)")
    vr.add_argument("--deals", type=int, default=2000)
    vr.add_argument("--seed", type=int, default=0)
    vr.add_argument("--workers", type=int, default=4)

    # compare
    cmp = sub.add_parser("compare", help="Head-to-head compare two models")
    cmp.add_argument("model_a", help="Path to model A directory")
    cmp.add_argument("model_b", help="Path to model B directory")
    cmp.add_argument("--deals", type=int, default=2000)
    cmp.add_argument("--seed", type=int, default=0)
    cmp.add_argument("--workers", type=int, default=4)

    args = p.parse_args()
    if args.command == "list":
        return cmd_list(args)
    elif args.command == "vs-random":
        return cmd_vs_random(args)
    elif args.command == "compare":
        return cmd_compare(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
