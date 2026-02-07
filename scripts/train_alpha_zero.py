#!/usr/bin/env python3
"""Train a SharedAlphaNet via Hybrid Bootstrap AlphaZero and evaluate.

The bootstrap solves the cold-start problem:
  Phase 1 (bootstrap): MCTS uses random rollouts -> reliable search signal.
                        Both policy + value heads are trained on the results.
  Phase 2 (alphazero): MCTS switches to value-head evaluation + policy priors.
                        The network is now warm enough to guide itself.

All evaluations use MCTS + determinization (the "search-led" standard).

Usage:
    python scripts/train_alpha_zero.py [--bootstrap N]
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from trickster.games.snapszer.adapter import SnapszerGame, SnapszerNode
from trickster.mcts import MCTSConfig, alpha_mcts_choose
from trickster.models.alpha_net import SharedAlphaNet, create_shared_alpha_net
from trickster.training.alpha_zero import train_alpha_zero
from trickster.games.snapszer.game import (
    deal,
    deal_awarded_game_points,
    is_terminal,
    legal_actions,
    play_trick,
)


# ---------------------------------------------------------------------------
#  Evaluation helpers — ALL use MCTS for AlphaZero agent
# ---------------------------------------------------------------------------

def _alpha_mcts_agent_choose(
    game: SnapszerGame,
    node: SnapszerNode,
    net: SharedAlphaNet,
    player: int,
    config: MCTSConfig,
    rng: random.Random,
) -> object:
    """Pick the best action using MCTS search."""
    actions = game.legal_actions(node)
    if len(actions) <= 1:
        return actions[0]
    return alpha_mcts_choose(node, game, net, player, config, rng)


def eval_alpha_mcts_vs_random(
    game: SnapszerGame,
    net: SharedAlphaNet,
    config: MCTSConfig,
    *,
    deals: int = 200,
    seed: int = 0,
) -> tuple[int, int]:
    """AlphaZero (MCTS search) vs random.  Returns (alpha_pts, random_pts)."""
    a_pts, b_pts = 0, 0
    for g in range(deals):
        a_is_0 = g % 2 == 0
        a_idx = 0 if a_is_0 else 1
        base = seed + g // 2
        node = game.new_game(seed=base, starting_leader=0)
        rnd = random.Random(seed + g + 10000)
        rng = random.Random(seed + g + 15000)

        while not game.is_terminal(node):
            player = game.current_player(node)
            actions = game.legal_actions(node)
            if player == a_idx:
                action = _alpha_mcts_agent_choose(
                    game, node, net, player, config, rng,
                )
            else:
                card_actions = [a for a in actions if a != "close_talon"]
                action = rnd.choice(card_actions) if card_actions else rnd.choice(actions)
            node = game.apply(node, action)

        winner, pts, _ = deal_awarded_game_points(node.gs)
        if winner == a_idx:
            a_pts += pts
        else:
            b_pts += pts
    return a_pts, b_pts


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero Hybrid Bootstrap Training")
    parser.add_argument("--bootstrap", type=int, default=5000,
                        help="Games in bootstrap phase (rollout MCTS, default 5000)")
    parser.add_argument("--iters", type=int, default=200,
                        help="Total training iterations (default 200)")
    parser.add_argument("--games-per-iter", type=int, default=40,
                        help="Self-play games per iteration (default 40)")
    parser.add_argument("--sims", type=int, default=50,
                        help="MCTS simulations per move (default 50)")
    parser.add_argument("--dets", type=int, default=4,
                        help="Determinizations per move (default 4)")
    parser.add_argument("--eval-deals", type=int, default=200,
                        help="Deals per evaluation matchup (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default="models/AlphaZero/net.pkl",
                        help="Path to save the trained network")
    args = parser.parse_args()

    game = SnapszerGame()

    total_games = args.iters * args.games_per_iter
    bootstrap_iters = (args.bootstrap + args.games_per_iter - 1) // args.games_per_iter

    # ---- Training config ----
    train_config = MCTSConfig(
        simulations=args.sims,
        determinizations=args.dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        visit_temp=1.0,
    )

    print("=" * 64)
    print("  AlphaZero Hybrid Bootstrap Training")
    print("=" * 64)
    print(f"  MCTS: {args.sims} sims x {args.dets} dets")
    print(f"  Network: body=128x2, heads=64, ReLU")
    print(f"  Budget: {args.iters} iters x {args.games_per_iter} games = {total_games} games")
    print(f"  Bootstrap: first {args.bootstrap} games use random rollouts")
    print(f"    -> ~{bootstrap_iters} iters of rollout MCTS, then switch to value head")
    print()

    t0 = time.perf_counter()

    def on_progress(stats):
        elapsed = time.perf_counter() - t0
        phase_tag = "BOOT" if stats.phase == "bootstrap" else "AZ  "
        print(
            f"\r  [{phase_tag}] iter {stats.iterations:3d}/{args.iters}  "
            f"games={stats.total_games:5d}  "
            f"samples={stats.total_samples:6d}  "
            f"vmse={stats.last_value_mse:.4f}  "
            f"pce={stats.last_policy_ce:.4f}  "
            f"[{elapsed:.0f}s]",
            end="", flush=True,
        )

    net, stats = train_alpha_zero(
        game=game,
        iterations=args.iters,
        games_per_iter=args.games_per_iter,
        train_steps=100,
        mcts_config=train_config,
        body_units=128,
        body_layers=2,
        head_units=64,
        lr=0.01,
        l2=1e-4,
        batch_size=32,
        buffer_capacity=50_000,
        bootstrap_games=args.bootstrap,
        seed=args.seed,
        on_progress=on_progress,
    )
    train_time = time.perf_counter() - t0
    print()
    print(f"\n  Training done: {stats.total_games} games, "
          f"{stats.total_samples} samples, {train_time:.1f}s")
    print(f"  Final value MSE: {stats.last_value_mse:.4f}")
    print(f"  Final policy CE: {stats.last_policy_ce:.4f}")

    # ---- Save model ----
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(net, f)
    print(f"  Model saved to {save_path}")

    # ---- Evaluation (MCTS vs Random) ----
    eval_config = MCTSConfig(
        simulations=args.sims,
        determinizations=args.dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.0,  # no noise at eval time
        visit_temp=0.1,       # near-greedy at eval time
    )
    eval_deals = args.eval_deals

    print()
    print("=" * 64)
    print("  Evaluation (MCTS search)")
    print("=" * 64)

    print(f"\n  AlphaZero (MCTS) vs Random ({eval_deals} deals)...", end="", flush=True)
    t1 = time.perf_counter()
    az_pts, az_rnd_pts = eval_alpha_mcts_vs_random(
        game, net, eval_config, deals=eval_deals, seed=0,
    )
    az_time = time.perf_counter() - t1
    az_ppd = az_pts / eval_deals
    az_rnd_ppd = az_rnd_pts / eval_deals
    print(f" {az_time:.1f}s")
    print(f"    AlphaZero: {az_ppd:.2f} pts/deal | Random: {az_rnd_ppd:.2f} pts/deal "
          f"(margin: {az_ppd - az_rnd_ppd:+.2f})")

    # ---- Summary ----
    total_time = time.perf_counter() - t0

    print()
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  Training:  {stats.total_games} games ({train_time:.0f}s)")
    print(f"    Bootstrap phase: first {args.bootstrap} games (random rollouts)")
    print(f"    AlphaZero phase: remaining {max(0, stats.total_games - args.bootstrap)} games (value head)")
    print(f"  vs Random margin: {az_ppd - az_rnd_ppd:+.2f} pts/deal")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Eval config: {eval_config.simulations} sims, "
          f"{eval_config.determinizations} dets, "
          f"temp={eval_config.visit_temp}")


if __name__ == "__main__":
    main()
