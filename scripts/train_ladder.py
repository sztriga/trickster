#!/usr/bin/env python3
"""Train a ladder of progressively stronger models and evaluate them.

Usage:
    python scripts/train_ladder.py              # train all tiers
    python scripts/train_ladder.py --tiers 3    # train first 3 tiers only
    python scripts/train_ladder.py --from-tier 5        # train from tier 5 (Rook) upward
    python scripts/train_ladder.py --from-tier Bishop   # train from "Bishop" upward
    python scripts/train_ladder.py --only 5 --only 6    # train specific tiers only
    python scripts/train_ladder.py --range 5:6          # train tiers 5 through 6 (inclusive)
    python scripts/train_ladder.py --list       # show the ladder without training
    python scripts/train_ladder.py --eval-only  # skip training, just evaluate existing models

Each model is saved under models/<Name>/ with spec.json + train_info.json
so the GUI can see and use them immediately.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.games.snapszer.adapter import SnapszerGame
from trickster.games.snapszer.game import deal_awarded_game_points
from trickster.mcts import MCTSConfig, alpha_mcts_choose
from trickster.models.alpha_net import SharedAlphaNet, create_shared_alpha_net
from trickster.training.alpha_zero import train_alpha_zero


# ═══════════════════════════════════════════════════════════════════════
#  Tier definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AZTier:
    name: str
    tag: str
    body_units: int
    body_layers: int
    head_units: int
    iters: int
    gpi: int            # games per iteration
    sims: int
    dets: int
    bootstrap: int
    train_steps: int = 80
    batch_size: int = 32
    buffer_cap: int = 20000
    lr: float = 0.01
    l2: float = 1e-4
    workers: int = 4
    seed: int = 42
    description: str = ""

    @property
    def total_games(self) -> int:
        return self.iters * self.gpi


LADDER: list[AZTier] = [
    # ── Tier 1: AZ Pawn — tiny net, very light search ────────────────
    AZTier(
        name="T1-Pawn",
        tag="T1",
        body_units=32,
        body_layers=1,
        head_units=16,
        iters=100,
        gpi=20,
        sims=30,
        dets=3,
        bootstrap=1000,
        train_steps=50,
        buffer_cap=10000,
        description="Tiny AZ (32x1/16, 2k games, 30s×3d) — ~30s",
    ),
    # ── Tier 2: AZ Scout — small net, light search ───────────────────
    AZTier(
        name="T2-Scout",
        tag="T2",
        body_units=64,
        body_layers=2,
        head_units=32,
        iters=150,
        gpi=30,
        sims=40,
        dets=3,
        bootstrap=2000,
        train_steps=60,
        buffer_cap=15000,
        description="Small AZ (64x2/32, 4.5k games, 40s×3d) — ~1-2 min",
    ),
    # ── Tier 3: AZ Knight — medium net, moderate search ──────────────
    AZTier(
        name="T3-Knight",
        tag="T3",
        body_units=64,
        body_layers=2,
        head_units=32,
        iters=200,
        gpi=40,
        sims=50,
        dets=4,
        bootstrap=3000,
        train_steps=80,
        buffer_cap=20000,
        description="Medium AZ (64x2/32, 8k games, 50s×4d) — ~3-5 min",
    ),
    # ── Tier 4: AZ Bishop — full-size net, solid search ──────────────
    AZTier(
        name="T4-Bishop",
        tag="T4",
        body_units=128,
        body_layers=2,
        head_units=64,
        iters=300,
        gpi=50,
        sims=60,
        dets=5,
        bootstrap=6000,
        train_steps=100,
        batch_size=64,
        buffer_cap=30000,
        description="Full AZ (128x2/64, 15k games, 60s×5d) — ~10-15 min",
    ),
    # ── Tier 5: AZ Rook — full net, heavy search ─────────────────────
    AZTier(
        name="T5-Rook",
        tag="T5",
        body_units=128,
        body_layers=2,
        head_units=64,
        iters=400,
        gpi=50,
        sims=60,
        dets=5,
        bootstrap=8000,
        train_steps=100,
        batch_size=64,
        buffer_cap=40000,
        description="Full AZ (128x2/64, 20k games, 60s×5d) — ~15-20 min",
    ),
    # ── Tier 6: AZ Captain — deeper net, strong search ───────────────
    AZTier(
        name="T6-Captain",
        tag="T6",
        body_units=128,
        body_layers=3,
        head_units=64,
        iters=600,
        gpi=80,
        sims=80,
        dets=6,
        bootstrap=12000,
        train_steps=150,
        batch_size=64,
        buffer_cap=60000,
        lr=0.005,
        description="Deep AZ (128x3/64, 48k games, 80s×6d) — ~1-1.5 hr",
    ),
    # ── Tier 7: AZ General — wide+deep net, heavy search ─────────────
    AZTier(
        name="T7-General",
        tag="T7",
        body_units=256,
        body_layers=4,
        head_units=128,
        iters=1000,
        gpi=100,
        sims=150,
        dets=8,
        bootstrap=20000,
        train_steps=200,
        batch_size=128,
        buffer_cap=100000,
        lr=0.003,
        workers=8,
        description="Wide AZ (256x4/128, 100k games, 150s×8d) — ~5-8 hr (AWS)",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
#  Model registration (spec.json + train_info.json → visible in GUI)
# ═══════════════════════════════════════════════════════════════════════

def _register_az_model(
    tier: AZTier, model_dir: Path, stats, train_time: float,
) -> None:
    spec = {
        "game": "snapszer",
        "kind": "alphazero",
        "method": "alphazero",
        "params": {
            "body_units": tier.body_units,
            "body_layers": tier.body_layers,
            "head_units": tier.head_units,
        },
    }
    info = {
        "method": "alphazero",
        "iterations": tier.iters,
        "games_per_iter": tier.gpi,
        "total_games": stats.total_games,
        "total_samples": stats.total_samples,
        "bootstrap_games": tier.bootstrap,
        "sims": tier.sims,
        "dets": tier.dets,
        "train_steps": tier.train_steps,
        "batch_size": tier.batch_size,
        "buffer_capacity": tier.buffer_cap,
        "lr": tier.lr,
        "l2": tier.l2,
        "seed": tier.seed,
        "workers": tier.workers,
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


# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════

def train_az_tier(tier: AZTier, game: SnapszerGame) -> Path:
    model_dir = Path("models") / tier.name
    model_dir.mkdir(parents=True, exist_ok=True)

    mcts_config = MCTSConfig(
        simulations=tier.sims,
        determinizations=tier.dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        visit_temp=1.0,
    )

    last_report = [0]

    def on_progress(stats):
        if stats.iterations - last_report[0] >= max(1, tier.iters // 20) or \
           stats.iterations == tier.iters:
            last_report[0] = stats.iterations
            pct = stats.iterations / tier.iters * 100
            phase = "BOOT" if stats.phase == "bootstrap" else "AZ  "
            print(
                f"    [{phase}] iter {stats.iterations:>4d}/{tier.iters} ({pct:4.0f}%)  "
                f"games={stats.total_games:>5d}  "
                f"vmse={stats.last_value_mse:.4f}  "
                f"pce={stats.last_policy_ce:.4f}",
                flush=True,
            )

    t0 = time.perf_counter()
    net, stats = train_alpha_zero(
        game=game,
        iterations=tier.iters,
        games_per_iter=tier.gpi,
        train_steps=tier.train_steps,
        mcts_config=mcts_config,
        body_units=tier.body_units,
        body_layers=tier.body_layers,
        head_units=tier.head_units,
        lr=tier.lr,
        l2=tier.l2,
        batch_size=tier.batch_size,
        buffer_capacity=tier.buffer_cap,
        bootstrap_games=tier.bootstrap,
        seed=tier.seed,
        num_workers=tier.workers,
        on_progress=on_progress,
    )
    elapsed = time.perf_counter() - t0

    with open(model_dir / "net.pkl", "wb") as f:
        pickle.dump(net, f)
    _register_az_model(tier, model_dir, stats, elapsed)

    return model_dir


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation — round-robin with MCTS for AZ, learned policy for Direct
# ═══════════════════════════════════════════════════════════════════════

def _load_agent(model_dir: Path, game: SnapszerGame):
    """Return (agent_fn, label).

    agent_fn(node, player, rng) -> action
    """
    spec_path = model_dir / "spec.json"
    spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
    kind = spec_data.get("kind", "")

    net_path = model_dir / "net.pkl"
    with open(net_path, "rb") as f:
        net = pickle.load(f)
    eval_config = MCTSConfig(
        simulations=50,
        determinizations=4,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.0,
        visit_temp=0.1,
    )

    def az_agent(node, player, rng):
        actions = game.legal_actions(node)
        if len(actions) <= 1:
            return actions[0]
        return alpha_mcts_choose(node, game, net, player, eval_config, rng)

    return az_agent, model_dir.name


def _random_agent(node, game, rng):
    actions = game.legal_actions(node)
    card_actions = [a for a in actions if a != "close_talon"]
    return rng.choice(card_actions) if card_actions else rng.choice(actions)


def play_match(
    game: SnapszerGame,
    agent_a, agent_b,
    deals: int = 100,
    seed: int = 0,
) -> tuple[int, int]:
    """Play *deals* games. Returns (pts_a, pts_b)."""
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


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def show_ladder() -> None:
    print()
    print("  Strength Ladder")
    print("  " + "─" * 60)
    for i, tier in enumerate(LADDER):
        detail = (
            f"AZ {tier.body_units}x{tier.body_layers}/h{tier.head_units}, "
            f"{tier.total_games:,}g, {tier.sims}s×{tier.dets}d"
        )
        print(f"  {i}: {tier.name:<14s}  {detail}")
        print(f"     {tier.description}")
    print()


def _parse_tier_selector(sel: str, *, allow_end: bool) -> int:
    """Parse a tier selector into a ladder index.

    Accepts:
      - An integer index (0..len-1). If allow_end=True, also allows len(LADDER).
      - Exact tier name (e.g. "T4-Bishop")
      - Tier tag (e.g. "T4")
      - Tier short name (e.g. "Bishop")
    """
    s = sel.strip()
    if not s:
        raise argparse.ArgumentTypeError("Tier selector cannot be empty")

    if s.isdigit():
        idx = int(s)
        max_ok = len(LADDER) if allow_end else (len(LADDER) - 1)
        if 0 <= idx <= max_ok:
            return idx
        if allow_end:
            raise argparse.ArgumentTypeError(
                f"Tier index out of range: {idx} (expected 0..{len(LADDER)})",
            )
        raise argparse.ArgumentTypeError(
            f"Tier index out of range: {idx} (expected 0..{len(LADDER) - 1})",
        )

    needle = s.lower()
    for i, tier in enumerate(LADDER):
        name = tier.name.lower()
        tag = getattr(tier, "tag", "").lower()
        short = tier.name.split("-", 1)[-1].lower()
        if needle in {name, tag, short}:
            return i

    raise argparse.ArgumentTypeError(
        f"Unknown tier selector: {sel!r}. Use --list to see valid indices/names.",
    )


def _parse_tier_selector_one(sel: str) -> int:
    return _parse_tier_selector(sel, allow_end=False)


def _parse_tier_selector_end(sel: str) -> int:
    return _parse_tier_selector(sel, allow_end=True)


def _parse_range_spec(spec: str) -> tuple[int, int]:
    """Parse a human-friendly inclusive range `start:end`.

    Examples:
      - "5:6"       -> (5, 7)  # inclusive end, so end_exclusive=6+1
      - "5:"        -> (5, len(LADDER))
      - ":3"        -> (0, 4)
      - "Bishop:Rook" -> inclusive by name

    Returns (start_idx, end_exclusive_idx).
    """
    s = spec.strip()
    if ":" not in s:
        raise argparse.ArgumentTypeError("Range must contain ':' (example: 5:6)")

    left, right = s.split(":", 1)
    left = left.strip()
    right = right.strip()

    start = 0 if left == "" else _parse_tier_selector_one(left)

    if right == "":
        end_excl = len(LADDER)
        end_incl = len(LADDER) - 1
    else:
        r = right.lower()
        if r in {"end", "last"}:
            end_incl = len(LADDER) - 1
        else:
            end_incl = _parse_tier_selector_one(right)
        end_excl = end_incl + 1

    if not (0 <= start <= end_incl < len(LADDER)):
        raise argparse.ArgumentTypeError(
            f"Invalid range {spec!r}. Expected start<=end within 0..{len(LADDER) - 1}.",
        )
    return start, end_excl


def _model_is_loadable(model_dir: Path) -> bool:
    """Return True if the directory looks loadable by `_load_agent()`."""
    spec_path = model_dir / "spec.json"
    if not spec_path.exists():
        return False
    try:
        spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (model_dir / "net.pkl").exists()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ladder of models from weakest to strongest",
    )
    parser.add_argument(
        "--tiers", type=int, default=len(LADDER),
        help=f"Number of tiers to train (1-{len(LADDER)}, default: all)",
    )
    parser.add_argument(
        "--from-tier", type=_parse_tier_selector_one, default=None,
        help="Start from this tier (index/name/tag, e.g. 5, T5, Rook)",
    )
    parser.add_argument(
        "--to-tier", type=_parse_tier_selector_end, default=None,
        help="Stop before this tier (exclusive). Accepts index/name/tag; "
             f"use {len(LADDER)} to mean 'through the end'.",
    )
    parser.add_argument(
        "--range", type=_parse_range_spec, default=None,
        help="Inclusive tier range start:end (example: 5:6 includes both 5 and 6). "
             "Also supports names (e.g. Bishop:Rook), ':X', 'X:' and 'end'.",
    )
    parser.add_argument(
        "--only", type=_parse_tier_selector_one, action="append", default=None,
        help="Train/eval only these tiers (repeatable; index/name/tag)",
    )
    parser.add_argument(
        "--skip-trained", action="store_true",
        help="Skip training tiers that already exist under models/<Tier>/",
    )
    parser.add_argument(
        "--eval-deals", type=int, default=200,
        help="Deals per matchup in evaluation (default: 200)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Show the ladder and exit",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, evaluate existing models only",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation (training only)",
    )
    args = parser.parse_args()

    if args.list:
        show_ladder()
        return

    if args.eval_only and args.no_eval:
        parser.error("--eval-only and --no-eval are mutually exclusive")

    if args.range is not None and (args.only or args.from_tier is not None or args.to_tier is not None):
        parser.error("--range cannot be combined with --only/--from-tier/--to-tier")
    if args.only and (args.from_tier is not None or args.to_tier is not None):
        parser.error("--only cannot be combined with --from-tier/--to-tier")

    # Tier selection
    if args.only:
        idxs = sorted(set(args.only))
        tiers = [LADDER[i] for i in idxs]
        sel_desc = f"{len(tiers)} tier(s) via --only: {', '.join(map(str, idxs))}"
    elif args.range is not None:
        start, end_excl = args.range
        tiers = LADDER[start:end_excl]
        sel_desc = f"tiers [{start}..{end_excl - 1}] ({len(tiers)} tier(s)) via --range"
    elif args.from_tier is not None or args.to_tier is not None:
        start = args.from_tier if args.from_tier is not None else 0
        end = args.to_tier if args.to_tier is not None else len(LADDER)
        if not (0 <= start <= end <= len(LADDER)):
            parser.error(f"Invalid tier range: start={start}, end={end}")
        tiers = LADDER[start:end]
        sel_desc = f"tiers [{start}:{end}] ({len(tiers)} tier(s))"
    else:
        n_tiers = min(args.tiers, len(LADDER))
        tiers = LADDER[:n_tiers]
        sel_desc = f"first {n_tiers} tier(s) via --tiers"

    if not tiers:
        print("No tiers selected. Use --list to see available tiers.")
        return

    game = SnapszerGame()

    if not args.eval_only:
        # ── Training ─────────────────────────────────────────────────
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║              STRENGTH LADDER TRAINING                       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"  Selection: {sel_desc}")
        print()

        total_t0 = time.perf_counter()
        trained_dirs: list[Path] = []

        for i, tier in enumerate(tiers):
            print(f"┌─ Tier {i}: {tier.name} {'─' * (49 - len(tier.name))}")
            print(f"│  {tier.description}")

            print(f"│  Net {tier.body_units}x{tier.body_layers}/h{tier.head_units} | "
                  f"{tier.total_games:,} games ({tier.iters} iters × {tier.gpi} gpi) | "
                  f"{tier.sims}s×{tier.dets}d | "
                  f"bootstrap={tier.bootstrap:,} | workers={tier.workers}")
            print("│")

            model_dir = Path("models") / tier.name
            already_trained = _model_is_loadable(model_dir)
            if args.skip_trained and already_trained:
                trained_dirs.append(model_dir)
                print("│  ↷ Already trained, skipping (found loadable model on disk)")
                print(f"└{'─' * 62}")
                print()
                continue

            t0 = time.perf_counter()
            d = train_az_tier(tier, game)
            elapsed = time.perf_counter() - t0

            trained_dirs.append(d)

            mins = elapsed / 60
            if mins < 1:
                time_str = f"{elapsed:.1f}s"
            else:
                time_str = f"{mins:.1f} min"

            print(f"│")
            print(f"│  ✓ Done in {time_str}  →  {d}")
            print(f"└{'─' * 62}")
            print()

        total_elapsed = time.perf_counter() - total_t0
        total_min = total_elapsed / 60
        print(f"  All training complete: {total_min:.1f} min total")
        print()

    if args.no_eval:
        return

    # ── Evaluation ───────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              STRENGTH EVALUATION                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Collect existing model dirs for tiers we want to evaluate
    eval_dirs: list[Path] = []
    for tier in tiers:
        d = Path("models") / tier.name
        if d.exists() and _model_is_loadable(d):
            eval_dirs.append(d)
        else:
            print(f"  ⚠ {tier.name} not found, skipping")

    if len(eval_dirs) < 1:
        print("  No models to evaluate.")
        return

    # Load all agents
    agents: list[tuple[callable, str]] = []
    for d in eval_dirs:
        fn, label = _load_agent(d, game)
        if fn is not None:
            agents.append((fn, label))

    # Add random as floor
    def rnd_agent(node, player, rng):
        return _random_agent(node, game, rng)
    agents.insert(0, (rnd_agent, "Random"))

    n = len(agents)
    deals = args.eval_deals

    # Round-robin: each agent vs every other
    print(f"\n  Round-robin: {n} agents, {deals} deals per matchup")
    print(f"  (This may take a while for AZ agents using MCTS search)\n")

    # wins[i] = total game points scored by agent i
    total_pts = [0] * n
    results: list[list[str]] = [["-"] * n for _ in range(n)]  # margin table

    matchup_idx = 0
    total_matchups = n * (n - 1) // 2
    eval_t0 = time.perf_counter()

    for i in range(n):
        for j in range(i + 1, n):
            matchup_idx += 1
            name_a = agents[i][1]
            name_b = agents[j][1]
            print(
                f"  [{matchup_idx}/{total_matchups}] "
                f"{name_a} vs {name_b}  ({deals} deals)...",
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

    eval_elapsed = time.perf_counter() - eval_t0

    # ── Ranking ──────────────────────────────────────────────────────
    print()
    print("  ┌─ RESULTS " + "─" * 52)
    print("  │")

    # Sort by total points descending
    ranking = sorted(range(n), key=lambda k: total_pts[k], reverse=True)

    max_name = max(len(agents[k][1]) for k in ranking)

    print(f"  │  {'Rank':<5} {'Agent':<{max_name+2}} {'Total pts':>10}  {'pts/deal':>8}")
    print(f"  │  {'─'*5} {'─'*(max_name+2)} {'─'*10}  {'─'*8}")

    total_deals = (n - 1) * deals  # each agent plays this many deals total
    for rank, k in enumerate(ranking, 1):
        name = agents[k][1]
        pts = total_pts[k]
        ppd = pts / max(1, total_deals)
        medal = "👑" if rank == 1 else "  "
        print(f"  │  {medal}{rank:<3} {name:<{max_name+2}} {pts:>10}  {ppd:>8.3f}")

    print("  │")

    # Margin table
    print("  │  Margin table (row vs column, + means row is stronger):")
    print("  │")
    header = "  │  " + " " * (max_name + 2)
    for k in ranking:
        header += f" {agents[k][1][:8]:>8}"
    print(header)

    for k in ranking:
        row = f"  │  {agents[k][1]:<{max_name+2}}"
        for j in ranking:
            row += f" {results[k][j]:>8}"
        print(row)

    print("  │")
    print(f"  └─ Evaluation: {eval_elapsed:.0f}s ({eval_elapsed/60:.1f} min)")
    print()


if __name__ == "__main__":
    main()
