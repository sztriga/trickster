from __future__ import annotations

import argparse
from pathlib import Path

from trickster.training.model_store import load_slot, save_latest_and_prev, slot_exists
from trickster.training.self_play import load_policy, save_policy, train_self_play
from trickster.training.model_spec import ModelSpec, model_dir, write_spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Ulti Card — self-play trainer (headless)")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the selected model's rolling slot: models/<model_id>/latest.pkl",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Resume from an explicit model path (overrides --resume).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional extra output path (manual export). Rolling slots are always saved under models/<model_id>/{latest,prev}.pkl.",
    )
    args = parser.parse_args()

    spec = ModelSpec(
        kind=args.model,
        params={
            "hidden_units": int(args.hidden_units),
            "activation": str(args.activation),
        }
        if args.model == "mlp"
        else {},
    )
    mdir = model_dir(spec, root="models")
    write_spec(spec, root="models")

    initial_policy = None
    if args.resume_from:
        initial_policy = load_policy(args.resume_from)
        print(f"Resuming from {args.resume_from}")
    elif args.resume:
        if not slot_exists("latest", models_dir=mdir):
            raise SystemExit(f"No {mdir}/latest.pkl to resume from. Train once without --resume first.")
        initial_policy = load_slot("latest", models_dir=mdir)
        print(f"Resuming from {mdir}/latest.pkl")

    policy, stats = train_self_play(
        spec=spec,
        episodes=args.episodes,
        seed=args.seed,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        initial_policy=initial_policy,
    )

    slots = save_latest_and_prev(policy, models_dir=mdir)
    print(f"Saved rolling models: latest={slots.latest} prev={slots.prev}")

    if args.out:
        out = Path(args.out)
        if out != slots.latest:
            save_policy(policy, out)
            print(f"Also saved model to {out}")
    print(
        f"episodes={stats.episodes} winrate_p0={stats.winrate_p0:.3f} "
        f"last_score={stats.last_score0}-{stats.last_score1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

