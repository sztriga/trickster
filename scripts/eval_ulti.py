#!/usr/bin/env python3
"""Evaluate MCTS agent (soloist) vs. random defenders in Ulti.

Usage:
    python scripts/eval_ulti.py [--games N] [--sims S] [--dets D]

The script plays N games with an untrained MCTS soloist against two
random defenders, reporting the soloist win rate.  Increasing
``--sims`` and ``--dets`` should raise the win rate — confirming that
the AI pipeline (adapter + encoder + MCTS + determinization) is wired
correctly.

Example (quick smoke test):
    python scripts/eval_ulti.py --games 50 --sims 20 --dets 4

Example (strength comparison):
    python scripts/eval_ulti.py --games 200 --sims 50  --dets 6
    python scripts/eval_ulti.py --games 200 --sims 200 --dets 10
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.game import current_player, legal_actions
from trickster.mcts import MCTSConfig, alpha_mcts_choose
from trickster.models.alpha_net import create_shared_alpha_net


def play_one_game(
    game: UltiGame,
    net,
    config: MCTSConfig,
    seed: int,
    verbose: bool = False,
) -> bool:
    """Play one game: MCTS soloist vs. random defenders.

    Returns True if the soloist wins.
    """
    rng = random.Random(seed)
    state: UltiNode = game.new_game(seed=seed, starting_leader=seed % 3)
    soloist = state.gs.soloist
    contract = "Betli" if state.gs.betli else f"Trump={state.gs.trump}"

    if verbose:
        print(f"  Game {seed}: soloist=P{soloist}, {contract}")

    while not game.is_terminal(state):
        cp = game.current_player(state)

        if cp == soloist:
            # MCTS agent chooses
            action = alpha_mcts_choose(state, game, net, cp, config, rng)
        else:
            # Random defender
            actions = game.legal_actions(state)
            action = rng.choice(actions)

        state = game.apply(state, action)

    outcome = game.outcome(state, soloist)
    won = outcome > 0

    if verbose:
        pts = state.gs.scores[soloist]
        print(f"    → soloist {'WON' if won else 'LOST'} ({pts} pts)")

    return won


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MCTS soloist vs. random defenders in Ulti",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per determinization")
    parser.add_argument("--dets", type=int, default=6, help="Number of determinizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print per-game results")
    args = parser.parse_args()

    game = UltiGame()

    # Create an untrained network (random weights)
    net = create_shared_alpha_net(
        state_dim=game.state_dim,
        action_space_size=game.action_space_size,
        body_units=128,
        body_layers=2,
        head_units=64,
        seed=args.seed,
    )

    config = MCTSConfig(
        simulations=args.sims,
        determinizations=args.dets,
        c_puct=1.5,
        dirichlet_alpha=0.0,   # no noise for evaluation
        dirichlet_weight=0.0,
        use_value_head=False,  # untrained net → use rollouts
        use_policy_priors=False,
        visit_temp=0.0,        # greedy (pick most-visited)
    )

    print(f"Ulti MCTS Evaluation")
    print(f"  Games: {args.games}")
    print(f"  MCTS sims: {args.sims}, determinizations: {args.dets}")
    print(f"  Mode: random rollouts (untrained net)")
    print(f"  Seed: {args.seed}")
    print()

    wins = 0
    t0 = time.time()

    for i in range(args.games):
        game_seed = args.seed + i
        won = play_one_game(game, net, config, game_seed, verbose=args.verbose)
        if won:
            wins += 1

        # Progress
        done = i + 1
        if done % 10 == 0 or done == args.games:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            wr = wins / done * 100
            print(
                f"  [{done}/{args.games}] "
                f"wins={wins} ({wr:.1f}%) "
                f"elapsed={elapsed:.1f}s ({rate:.1f} games/s)"
            )

    elapsed = time.time() - t0
    wr = wins / args.games * 100
    print()
    print(f"Final: {wins}/{args.games} soloist wins ({wr:.1f}%)")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
