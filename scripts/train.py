#!/usr/bin/env python3
"""CLI training script.

Usage examples:

  # Direct self-play MLP (h=64, relu), 20k episodes
  python scripts/train.py --kind mlp --method direct --episodes 20000 --hidden 64 --activation relu

  # Expert Iteration (MCTS expert generates training data)
  python scripts/train.py --kind mlp --method expert --hidden 128 --layers 2 \
      --iters 500 --games-per-iter 200 --train-steps 400 --sims 50 --dets 4 --lr 0.03

  # Custom output name
  python scripts/train.py --kind mlp --method expert --name Wolverine --hidden 128 --layers 2 \
      --iters 500 --games-per-iter 200 --train-steps 400 --sims 50 --dets 4 --lr 0.03
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.training.model_spec import ModelSpec, model_dir, write_spec
from trickster.training.model_store import save_latest_and_prev


def main() -> int:
    p = argparse.ArgumentParser(description="Train a Snapszer model")
    p.add_argument("--kind", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--method", choices=["direct", "expert"], default="direct")
    p.add_argument("--episodes", type=int, default=20000, help="Episodes (direct only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="Learning rate (default: 0.05 linear, 0.01 mlp)")
    p.add_argument("--l2", type=float, default=1e-6)
    # MLP params
    p.add_argument("--hidden", type=int, default=64, help="Hidden units per layer (mlp only)")
    p.add_argument("--layers", type=int, default=1, help="Number of hidden layers (mlp only)")
    p.add_argument("--activation", choices=["relu", "tanh"], default="relu", help="Activation (mlp only)")
    # Direct params
    p.add_argument("--eps-start", type=float, default=0.2, help="Epsilon start (direct only)")
    p.add_argument("--eps-end", type=float, default=0.02, help="Epsilon end (direct only)")
    # Expert iteration params
    p.add_argument("--sims", type=int, default=50, help="MCTS simulations per determinization (expert)")
    p.add_argument("--dets", type=int, default=4, help="MCTS determinizations (expert)")
    p.add_argument("--c", type=float, default=1.4, help="MCTS exploration constant (expert)")
    p.add_argument("--iters", type=int, default=100, help="Training iterations (expert)")
    p.add_argument("--games-per-iter", type=int, default=50, help="Self-play games per iteration (expert)")
    p.add_argument("--train-steps", type=int, default=200, help="SGD steps per iteration (expert)")
    p.add_argument("--buf-cap", type=int, default=50000, help="Replay buffer capacity (expert)")
    p.add_argument("--workers", type=int, default=1, help="Parallel self-play workers (expert)")
    # Output
    p.add_argument("--name", type=str, default=None, help="Custom model directory name (under models/)")
    args = p.parse_args()

    # Build spec
    if args.kind == "mlp":
        params = {"hidden_units": args.hidden, "hidden_layers": args.layers, "activation": args.activation}
    else:
        params = {}
    spec = ModelSpec(kind=args.kind, params=params, method=args.method)

    lr = args.lr if args.lr is not None else (0.01 if args.kind == "mlp" else 0.05)

    # Determine output directory
    if args.name:
        mdir = Path("models") / args.name
        mdir.mkdir(parents=True, exist_ok=True)
        spec_path = mdir / "spec.json"
        spec_path.write_text(
            json.dumps(spec.canonical(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        mdir = model_dir(spec, root="models")
        write_spec(spec, root="models")

    print(f"Model dir: {mdir}")
    print(f"Spec: kind={args.kind} method={args.method} lr={lr} l2={args.l2}")
    if args.kind == "mlp":
        print(f"  hidden={args.hidden} layers={args.layers} activation={args.activation}")

    t0 = time.perf_counter()

    if args.method == "expert":
        from trickster.mcts import MCTSConfig as ExpertMCTSConfig
        from trickster.training.expert_iteration import train_expert_iteration

        cfg = ExpertMCTSConfig(
            simulations=args.sims, determinizations=args.dets, c_puct=args.c,
            use_policy_priors=False,  # unbiased MCTS expert
            use_value_head=False,     # rollout-based evaluation
        )
        total_games = args.iters * args.games_per_iter
        print(f"  Expert Iteration: iters={args.iters} games/iter={args.games_per_iter}")
        print(f"  total_games={total_games} train_steps/iter={args.train_steps}")
        print(f"  sims={args.sims} dets={args.dets} c_puct={args.c}")

        def on_expert_progress(stats):
            now = time.perf_counter()
            elapsed = now - t0
            rate = stats.total_games / elapsed if elapsed > 0 else 0
            eta = (total_games - stats.total_games) / rate if rate > 0 else 0
            print(
                f"  iter {stats.iterations}/{args.iters}  "
                f"games={stats.total_games}  samples={stats.total_samples}  "
                f"lead_loss={stats.last_lead_loss:.4f}  follow_loss={stats.last_follow_loss:.4f}  "
                f"({rate:.1f} g/s, ETA {eta:.0f}s)"
            )

        policy, ex_stats = train_expert_iteration(
            spec=spec, iterations=args.iters, games_per_iter=args.games_per_iter,
            train_steps=args.train_steps, mcts_config=cfg,
            seed=args.seed, lr=lr, l2=args.l2,
            buffer_capacity=args.buf_cap,
            workers=args.workers,
            on_progress=on_expert_progress,
        )
        save_latest_and_prev(policy, models_dir=mdir)
        info = {
            "method": "expert", "seed": args.seed, "lr": lr, "l2": args.l2,
            "iterations": args.iters, "games_per_iter": args.games_per_iter,
            "total_games": total_games, "total_samples": ex_stats.total_samples,
        }
        (mdir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    else:
        # Direct self-play
        from trickster.training.self_play import train_self_play

        print(f"  episodes={args.episodes} eps={args.eps_start}->{args.eps_end}")

        last_print = [0]

        def on_progress(done, stats):
            now = time.perf_counter()
            if done < last_print[0] + max(1, args.episodes // 20) and done < args.episodes:
                return
            last_print[0] = done
            elapsed = now - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (args.episodes - done) / rate if rate > 0 else 0
            print(f"  {done}/{args.episodes}  ({rate:.0f} ep/s, ETA {eta:.0f}s)")

        policy, stats = train_self_play(
            spec=spec, episodes=args.episodes,
            seed=args.seed, lr=lr, l2=args.l2,
            epsilon_start=args.eps_start, epsilon_end=args.eps_end,
            on_progress=on_progress,
        )
        save_latest_and_prev(policy, models_dir=mdir)
        info = {"episodes": stats.episodes, "method": args.method, "seed": args.seed, "lr": lr, "l2": args.l2}
        (mdir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s. Saved to {mdir}/latest.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
