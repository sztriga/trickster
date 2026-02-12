#!/usr/bin/env python3
"""Evaluate Hybrid (MCTS+PIMC+Minimax) vs pure MCTS players.

Runs head-to-head matches and reports:
- Win Rate & Points-per-deal
- Move Latency (per-phase breakdown)
- Phase contribution analysis (which algorithm drove wins/losses)

Usage:
    python3 scripts/eval_hybrid.py                            # quick eval
    python3 scripts/eval_hybrid.py --deals 500 --sims 60      # thorough eval
    python3 scripts/eval_hybrid.py --model models/T5-Rook     # specific model
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.games.snapszer.adapter import SnapszerGame, SnapszerNode
from trickster.games.snapszer.constants import (
    DEFAULT_EVAL_DEALS,
    DEFAULT_EVAL_DETS,
    DEFAULT_EVAL_SIMS,
    DEFAULT_LATE_THRESHOLD,
    DEFAULT_MCTS_SIMS,
    DEFAULT_PIMC_SAMPLES,
)
from trickster.games.snapszer.game import deal_awarded_game_points
from trickster.games.snapszer.hybrid import DecisionStats, HybridPlayer
from trickster.games.snapszer.minimax import alphabeta, game_phase
from trickster.mcts import MCTSConfig, alpha_mcts_choose
from trickster.models.alpha_net import SharedAlphaNet


# ---------------------------------------------------------------------------
#  Agent helpers
# ---------------------------------------------------------------------------


def load_net(model_dir: Path) -> SharedAlphaNet:
    net_path = model_dir / "net.pkl"
    if not net_path.exists():
        raise FileNotFoundError(f"No net.pkl in {model_dir}")
    with open(net_path, "rb") as f:
        return pickle.load(f)


def make_hybrid_player(
    net: SharedAlphaNet,
    game: SnapszerGame,
    sims: int,
    dets: int,
    pimc_samples: int = DEFAULT_PIMC_SAMPLES,
    late_threshold: int = DEFAULT_LATE_THRESHOLD,
) -> HybridPlayer:
    config = MCTSConfig(
        simulations=sims,
        determinizations=dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.0,  # no noise at eval
        visit_temp=0.1,       # near-greedy
    )
    return HybridPlayer(
        net=net,
        mcts_config=config,
        game=game,
        pimc_samples=pimc_samples,
        late_threshold=late_threshold,
    )


def make_mcts_config(sims: int, dets: int) -> MCTSConfig:
    return MCTSConfig(
        simulations=sims,
        determinizations=dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.0,
        visit_temp=0.1,
    )


def random_agent_choose(node: SnapszerNode, game: SnapszerGame, rng: random.Random):
    actions = game.legal_actions(node)
    card_actions = [a for a in actions if a != "close_talon"]
    return rng.choice(card_actions) if card_actions else rng.choice(actions)


# ---------------------------------------------------------------------------
#  Match runner with phase tracking
# ---------------------------------------------------------------------------


def play_match(
    game: SnapszerGame,
    agent_a,   # callable(node, player, rng) -> action
    agent_b,
    deals: int = 200,
    seed: int = 0,
    label_a: str = "A",
    label_b: str = "B",
    verbose: bool = False,
) -> dict:
    """Play *deals* games. Returns detailed results dict."""
    a_pts, b_pts = 0, 0
    a_wins, b_wins = 0, 0
    a_schwarz, b_schwarz = 0, 0

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

        winner, pts, reason = deal_awarded_game_points(node.gs)
        if winner == a_idx:
            a_pts += pts
            a_wins += 1
            if reason == "schwarz":
                a_schwarz += 1
        else:
            b_pts += pts
            b_wins += 1
            if reason == "schwarz":
                b_schwarz += 1

        if verbose and (g + 1) % 50 == 0:
            print(f"    ... {g + 1}/{deals} deals done", flush=True)

    return {
        "deals": deals,
        "a_label": label_a,
        "b_label": label_b,
        "a_pts": a_pts,
        "b_pts": b_pts,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "a_schwarz": a_schwarz,
        "b_schwarz": b_schwarz,
        "a_ppd": a_pts / deals,
        "b_ppd": b_pts / deals,
        "margin": (a_pts - b_pts) / deals,
    }


def print_match_result(r: dict) -> None:
    a, b = r["a_label"], r["b_label"]
    print(f"    {a}: {r['a_ppd']:.3f} pts/deal  ({r['a_wins']} wins, "
          f"{r['a_schwarz']} schwarz)")
    print(f"    {b}: {r['b_ppd']:.3f} pts/deal  ({r['b_wins']} wins, "
          f"{r['b_schwarz']} schwarz)")
    margin = r["margin"]
    winner = a if margin > 0 else b if margin < 0 else "DRAW"
    print(f"    Margin: {margin:+.3f} pts/deal  →  {winner}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid vs MCTS evaluation")
    parser.add_argument("--model", type=str, default="models/T5-Rook",
                        help="Path to model dir with net.pkl")
    parser.add_argument("--deals", type=int, default=DEFAULT_EVAL_DEALS,
                        help=f"Deals per matchup (default {DEFAULT_EVAL_DEALS})")
    parser.add_argument("--sims", type=int, default=DEFAULT_EVAL_SIMS,
                        help=f"MCTS simulations per move (default {DEFAULT_EVAL_SIMS})")
    parser.add_argument("--dets", type=int, default=DEFAULT_EVAL_DETS,
                        help=f"Determinizations per move (default {DEFAULT_EVAL_DETS})")
    parser.add_argument("--pimc-samples", type=int, default=DEFAULT_PIMC_SAMPLES,
                        help=f"PIMC worlds for Phase 1 Late (default {DEFAULT_PIMC_SAMPLES})")
    parser.add_argument("--late-threshold", type=int, default=DEFAULT_LATE_THRESHOLD,
                        help=f"Talon cards for PIMC switch (default {DEFAULT_LATE_THRESHOLD})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-models", action="store_true",
                        help="Evaluate all models in models/ with net.pkl")
    args = parser.parse_args()

    game = SnapszerGame()

    # Discover models
    if args.all_models:
        model_dirs = sorted(
            [d for d in Path("models").iterdir()
             if d.is_dir() and (d / "net.pkl").exists()],
            key=lambda d: d.name,
        )
    else:
        model_dirs = [Path(args.model)]

    if not model_dirs:
        print("No models found.")
        return

    print("=" * 70)
    print("  HYBRID SEARCH EVALUATION")
    print("  MCTS PUCT + PIMC Minimax + Pure Minimax")
    print("=" * 70)
    print(f"  MCTS: {args.sims} sims × {args.dets} dets")
    print(f"  PIMC: {args.pimc_samples} samples, switch at talon ≤ {args.late_threshold}")
    print(f"  Deals per matchup: {args.deals}")
    print()

    for model_dir in model_dirs:
        if not (model_dir / "net.pkl").exists():
            print(f"  ⚠ {model_dir.name}: no net.pkl, skipping")
            continue

        print(f"┌─ Model: {model_dir.name} {'─' * (55 - len(model_dir.name))}")

        net = load_net(model_dir)
        mcts_config = make_mcts_config(args.sims, args.dets)
        hybrid = make_hybrid_player(
            net, game, args.sims, args.dets,
            pimc_samples=args.pimc_samples,
            late_threshold=args.late_threshold,
        )

        # ---- Matchup 1: Hybrid vs Random --------------------------------
        print("│")
        print(f"│  [1] Hybrid vs Random ({args.deals} deals)...")
        hybrid.reset_stats()
        t0 = time.perf_counter()

        def hybrid_agent(node, player, rng, _h=hybrid):
            return _h.choose_action(node, player, rng)

        def random_agent(node, player, rng):
            return random_agent_choose(node, game, rng)

        r1 = play_match(
            game, hybrid_agent, random_agent,
            deals=args.deals, seed=args.seed,
            label_a="Hybrid", label_b="Random",
        )
        t1 = time.perf_counter() - t0
        print_match_result(r1)
        print(f"│  Phase stats: {hybrid.stats.summary()}")
        print(f"│  Time: {t1:.1f}s ({t1/args.deals*1000:.0f}ms/deal)")

        # ---- Matchup 2: Pure MCTS vs Random ------------------------------
        print("│")
        print(f"│  [2] Pure MCTS vs Random ({args.deals} deals)...")
        t0 = time.perf_counter()

        def mcts_agent(node, player, rng, _net=net, _cfg=mcts_config):
            actions = game.legal_actions(node)
            if len(actions) <= 1:
                return actions[0]
            return alpha_mcts_choose(node, game, _net, player, _cfg, rng)

        r2 = play_match(
            game, mcts_agent, random_agent,
            deals=args.deals, seed=args.seed,
            label_a="Pure MCTS", label_b="Random",
        )
        t2 = time.perf_counter() - t0
        print_match_result(r2)
        print(f"│  Time: {t2:.1f}s ({t2/args.deals*1000:.0f}ms/deal)")

        # ---- Matchup 3: Hybrid vs Pure MCTS (same net) -------------------
        print("│")
        print(f"│  [3] Hybrid vs Pure MCTS ({args.deals} deals)...")
        hybrid.reset_stats()
        t0 = time.perf_counter()

        r3 = play_match(
            game, hybrid_agent, mcts_agent,
            deals=args.deals, seed=args.seed + 5000,
            label_a="Hybrid", label_b="Pure MCTS",
        )
        t3 = time.perf_counter() - t0
        print_match_result(r3)
        print(f"│  Phase stats: {hybrid.stats.summary()}")
        print(f"│  Time: {t3:.1f}s ({t3/args.deals*1000:.0f}ms/deal)")

        # ---- Matchup 4: Pure Minimax vs Random (endgame-only baseline) ---
        print("│")
        print(f"│  [4] Minimax-only vs Random ({args.deals} deals)...")
        t0 = time.perf_counter()
        minimax_stats = DecisionStats()

        def minimax_agent(node, player, rng):
            nonlocal minimax_stats
            phase = game_phase(node)
            if phase == "phase2":
                tt = time.perf_counter()
                _val, action = alphabeta(node, game, player)
                minimax_stats.minimax_decisions += 1
                minimax_stats.minimax_time += time.perf_counter() - tt
                if action is not None:
                    return action
            return random_agent_choose(node, game, rng)

        r4 = play_match(
            game, minimax_agent, random_agent,
            deals=args.deals, seed=args.seed,
            label_a="Minimax-only", label_b="Random",
        )
        t4 = time.perf_counter() - t0
        print_match_result(r4)
        print(f"│  Minimax decisions: {minimax_stats.minimax_decisions}, "
              f"time: {minimax_stats.minimax_time:.2f}s")
        print(f"│  Time: {t4:.1f}s ({t4/args.deals*1000:.0f}ms/deal)")

        # ---- Summary -----------------------------------------------------
        print("│")
        print("│  ┌─ SUMMARY " + "─" * 50)
        print(f"│  │  Hybrid vs Random:    {r1['margin']:+.3f} pts/deal")
        print(f"│  │  Pure MCTS vs Random: {r2['margin']:+.3f} pts/deal")
        print(f"│  │  Hybrid vs MCTS:      {r3['margin']:+.3f} pts/deal")
        print(f"│  │  Minimax-only vs Rnd: {r4['margin']:+.3f} pts/deal")

        hybrid_boost = r1["margin"] - r2["margin"]
        speed_ratio = t2 / max(t3, 0.01)
        print(f"│  │")
        print(f"│  │  Hybrid boost over MCTS: {hybrid_boost:+.3f} pts/deal")
        if t3 < t2:
            print(f"│  │  Speed: Hybrid is {speed_ratio:.1f}x FASTER (less compute in endgame)")
        else:
            print(f"│  │  Speed: Hybrid is {t3/t2:.1f}x slower (PIMC overhead)")
        print(f"│  └{'─' * 60}")
        print(f"└{'─' * 68}")
        print()


if __name__ == "__main__":
    main()
