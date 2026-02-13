#!/usr/bin/env python3
"""Head-to-head evaluation: pit two Ulti models against each other.

Loads two checkpoint files and plays N games with every possible
role assignment (Model A as soloist, Model B as both defenders, and
vice versa).  Reports per-role, per-trump diagnostics.

Both players use the HybridPlayer (MCTS + solver) — same engine
used during training.

Usage:
    # Model vs random baseline
    python scripts/eval_head2head.py \
        --model-a models/ulti_simple.pt \
        --model-b random \
        --games 100

    # Two checkpoints against each other
    python scripts/eval_head2head.py \
        --model-a models/checkpoints/simple/step_00200.pt \
        --model-b models/ulti_simple.pt \
        --games 100

    # Quick test (fewer games, less MCTS)
    python scripts/eval_head2head.py \
        --model-a models/ulti_simple.pt \
        --model-b random \
        --games 20 --sims 8
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import Suit
from trickster.games.ulti.game import (
    soloist_points,
    defender_points,
    soloist_tricks,
    soloist_won_simple,
)
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, UltiNetWrapper


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------


def load_model(path: str) -> UltiNetWrapper | None:
    """Load a UltiNet checkpoint. Returns None for 'random'."""
    if path.lower() == "random":
        return None
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    net = UltiNet(
        input_dim=checkpoint.get("input_dim", 291),
        body_units=checkpoint.get("body_units", 256),
        body_layers=checkpoint.get("body_layers", 4),
        action_dim=checkpoint.get("action_dim", 32),
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    return UltiNetWrapper(net, device="cpu")


def model_label(path: str) -> str:
    if path.lower() == "random":
        return "Random"
    return Path(path).stem


# ---------------------------------------------------------------------------
#  Game record
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """Result of one evaluated game."""
    seed: int
    trump: str | None
    is_red: bool
    soloist_model: str   # "A" or "B"
    soloist_won: bool
    soloist_pts: int
    defender_pts: int
    soloist_trick_count: int
    game_point_reward: float  # normalised [-1,+1]


# ---------------------------------------------------------------------------
#  Play one game
# ---------------------------------------------------------------------------


def play_one_game(
    game: UltiGame,
    wrapper_soloist: UltiNetWrapper | None,
    wrapper_defender: UltiNetWrapper | None,
    mcts_config: MCTSConfig,
    seed: int,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
) -> tuple[UltiNode, int]:
    """Play one Parti game. Returns (terminal_state, soloist_idx)."""
    rng = random.Random(seed)
    dealer = seed % 3

    state = game.new_game(
        seed=seed,
        training_mode="simple",
        starting_leader=dealer,
    )
    soloist_idx = state.gs.soloist

    # Build players
    if wrapper_soloist is not None:
        sol_player = HybridPlayer(
            game, wrapper_soloist,
            mcts_config=mcts_config,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,  # near-greedy for eval
        )
    else:
        sol_player = None

    if wrapper_defender is not None:
        def_player = HybridPlayer(
            game, wrapper_defender,
            mcts_config=mcts_config,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,
        )
    else:
        def_player = None

    rng_rand = random.Random(seed + 50000)

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        is_soloist = (player == soloist_idx)
        hybrid = sol_player if is_soloist else def_player

        if hybrid is not None:
            action = hybrid.choose_action(state, player, rng)
        else:
            action = rng_rand.choice(actions)

        state = game.apply(state, action)

    return state, soloist_idx


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    model_a_path: str,
    model_b_path: str,
    num_games: int = 100,
    sims: int = 20,
    dets: int = 1,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
    seed: int = 42,
    show_highlights: int = 5,
) -> None:
    label_a = model_label(model_a_path)
    label_b = model_label(model_b_path)

    print("=" * 70)
    print("  Ulti Head-to-Head Evaluation (Parti)")
    print("=" * 70)
    print(f"  Model A: {label_a}  ({model_a_path})")
    print(f"  Model B: {label_b}  ({model_b_path})")
    print(f"  Games: {num_games} per role × 2 roles = {num_games * 2} total")
    print(f"  Engine: Hybrid ({SOLVER_ENGINE}, endgame={endgame_tricks}, "
          f"PIMC={pimc_dets})")
    print(f"  MCTS: {sims} sims × {dets} dets")
    print()

    wrapper_a = load_model(model_a_path)
    wrapper_b = load_model(model_b_path)
    game = UltiGame()

    config = MCTSConfig(
        simulations=sims,
        determinizations=dets,
        c_puct=1.5,
        dirichlet_alpha=0.0,
        dirichlet_weight=0.0,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.1,
    )

    records: list[GameRecord] = []
    t0 = time.perf_counter()
    game_count = 0
    total_total = num_games * 2

    for soloist_label, sol_wrap, def_wrap in [
        ("A", wrapper_a, wrapper_b),
        ("B", wrapper_b, wrapper_a),
    ]:
        for g in range(num_games):
            game_seed = seed + hash((soloist_label, g)) % (2**30)

            state, soloist_idx = play_one_game(
                game, sol_wrap, def_wrap, config, game_seed,
                endgame_tricks=endgame_tricks,
                pimc_dets=pimc_dets,
            )
            gs = state.gs
            won = soloist_won_simple(gs)
            trump_str = gs.trump.value if gs.trump else None
            is_red = gs.trump is not None and gs.trump == Suit.HEARTS

            rec = GameRecord(
                seed=game_seed,
                trump=trump_str,
                is_red=is_red,
                soloist_model=soloist_label,
                soloist_won=won,
                soloist_pts=soloist_points(gs),
                defender_pts=defender_points(gs),
                soloist_trick_count=soloist_tricks(gs),
                game_point_reward=game.outcome(state, soloist_idx),
            )
            records.append(rec)

            game_count += 1
            if game_count % 20 == 0:
                elapsed = time.perf_counter() - t0
                rate = game_count / elapsed if elapsed > 0 else 0
                print(
                    f"\r  [{game_count}/{total_total}] "
                    f"{elapsed:.0f}s ({rate:.1f} games/s)",
                    end="", flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"\r  Done: {game_count} games in {elapsed:.1f}s "
          f"({game_count / elapsed:.1f} games/s)           ")
    print()

    _print_results(records, label_a, label_b, show_highlights)


def _print_results(
    records: list[GameRecord],
    label_a: str,
    label_b: str,
    show_highlights: int,
) -> None:
    # ----- Overall head-to-head -----
    a_sol = [r for r in records if r.soloist_model == "A"]
    b_sol = [r for r in records if r.soloist_model == "B"]

    a_sol_wr = sum(r.soloist_won for r in a_sol) / max(1, len(a_sol))
    b_sol_wr = sum(r.soloist_won for r in b_sol) / max(1, len(b_sol))

    # A wins = A won as soloist + B lost as soloist (A defended successfully)
    a_wins = sum(r.soloist_won for r in a_sol) + sum(not r.soloist_won for r in b_sol)
    b_wins = sum(r.soloist_won for r in b_sol) + sum(not r.soloist_won for r in a_sol)
    total = len(records)

    print("=" * 70)
    print("  OVERALL RESULTS")
    print("=" * 70)
    print(f"  {label_a:>20s}  vs  {label_b}")
    print(f"  {'Wins':>20s}:  {a_wins} ({a_wins/total:.0%})  "
          f"vs  {b_wins} ({b_wins/total:.0%})")
    print()

    # ----- Per-role breakdown -----
    print("-" * 70)
    print("  PER-ROLE BREAKDOWN")
    print("-" * 70)
    print(f"  {'Model':<15s} {'Role':<12s} "
          f"{'W':>4s} {'L':>4s} {'Total':>5s} {'WR':>6s}  "
          f"{'Avg Sol Pts':>11s}  {'Avg Def Pts':>11s}")
    print(f"  {'-'*15:<15s} {'-'*12:<12s} "
          f"{'-'*4:>4s} {'-'*4:>4s} {'-'*5:>5s} {'-'*6:>6s}  "
          f"{'-'*11:>11s}  {'-'*11:>11s}")

    for model_label, model_key in [(label_a, "A"), (label_b, "B")]:
        # As soloist
        sol_games = [r for r in records if r.soloist_model == model_key]
        if sol_games:
            w = sum(r.soloist_won for r in sol_games)
            l = len(sol_games) - w
            wr = w / len(sol_games)
            avg_sp = sum(r.soloist_pts for r in sol_games) / len(sol_games)
            avg_dp = sum(r.defender_pts for r in sol_games) / len(sol_games)
            print(f"  {model_label:<15s} {'soloist':<12s} "
                  f"{w:>4d} {l:>4d} {len(sol_games):>5d} {wr:>5.0%}  "
                  f"{avg_sp:>11.1f}  {avg_dp:>11.1f}")

        # As defender (other model was soloist)
        other_key = "B" if model_key == "A" else "A"
        def_games = [r for r in records if r.soloist_model == other_key]
        if def_games:
            w = sum(not r.soloist_won for r in def_games)  # defender won
            l = len(def_games) - w
            wr = w / len(def_games)
            avg_sp = sum(r.soloist_pts for r in def_games) / len(def_games)
            avg_dp = sum(r.defender_pts for r in def_games) / len(def_games)
            print(f"  {model_label:<15s} {'defender':<12s} "
                  f"{w:>4d} {l:>4d} {len(def_games):>5d} {wr:>5.0%}  "
                  f"{avg_sp:>11.1f}  {avg_dp:>11.1f}")

    # ----- Trump suit breakdown -----
    print()
    print("-" * 70)
    print("  BY TRUMP SUIT")
    print("-" * 70)
    trumps = sorted(set(r.trump for r in records if r.trump))
    for trump in trumps:
        t_recs = [r for r in records if r.trump == trump]
        is_red = any(r.is_red for r in t_recs)
        tag = " (RED)" if is_red else ""
        a_t = [r for r in t_recs if r.soloist_model == "A"]
        b_t = [r for r in t_recs if r.soloist_model == "B"]
        a_wr = sum(r.soloist_won for r in a_t) / max(1, len(a_t))
        b_wr = sum(r.soloist_won for r in b_t) / max(1, len(b_t))
        print(f"  {trump:<10s}{tag:<7s}  "
              f"{label_a} sol: {sum(r.soloist_won for r in a_t)}/{len(a_t)} "
              f"({a_wr:.0%})  |  "
              f"{label_b} sol: {sum(r.soloist_won for r in b_t)}/{len(b_t)} "
              f"({b_wr:.0%})")

    # ----- Game-point analysis -----
    print()
    print("-" * 70)
    print("  GAME-POINT REWARD ANALYSIS (soloist perspective, normalised)")
    print("-" * 70)
    for model_label, model_key in [(label_a, "A"), (label_b, "B")]:
        sol_rewards = [r.game_point_reward for r in records
                       if r.soloist_model == model_key]
        if sol_rewards:
            print(f"  {model_label} as soloist:  "
                  f"avg={np.mean(sol_rewards):+.3f}  "
                  f"min={min(sol_rewards):+.3f}  "
                  f"max={max(sol_rewards):+.3f}")

    # ----- Points distribution -----
    print()
    print("-" * 70)
    print("  CARD-POINTS DISTRIBUTION (soloist pts when playing as soloist)")
    print("-" * 70)
    for model_label, model_key in [(label_a, "A"), (label_b, "B")]:
        sol_pts = [r.soloist_pts for r in records
                   if r.soloist_model == model_key]
        if sol_pts:
            print(f"  {model_label}:  "
                  f"avg={np.mean(sol_pts):.1f}  "
                  f"min={min(sol_pts)}  "
                  f"max={max(sol_pts)}  "
                  f"median={int(np.median(sol_pts))}  "
                  f"tricks={np.mean([r.soloist_trick_count for r in records if r.soloist_model == model_key]):.1f}/10")

    # ----- Highlight games -----
    if show_highlights > 0:
        print()
        print("-" * 70)
        print(f"  GAME HIGHLIGHTS (top {show_highlights})")
        print("-" * 70)

        # Biggest soloist wins
        won_games = sorted(
            [r for r in records if r.soloist_won],
            key=lambda r: r.soloist_pts,
            reverse=True,
        )[:show_highlights]

        if won_games:
            print("  Biggest soloist wins:")
            for r in won_games:
                red_tag = " RED" if r.is_red else ""
                print(f"    seed={r.seed:>10d}  {r.soloist_model}-soloist  "
                      f"trump={r.trump or '-':<8s}{red_tag:>4s}  "
                      f"pts={r.soloist_pts:>3d}-{r.defender_pts:<3d}  "
                      f"tricks={r.soloist_trick_count}/10  "
                      f"reward={r.game_point_reward:+.2f}")

        # Closest games
        close_games = sorted(
            records,
            key=lambda r: abs(r.soloist_pts - r.defender_pts),
        )[:show_highlights]

        if close_games:
            print("  Closest games:")
            for r in close_games:
                winner = f"{r.soloist_model}-sol {'WON' if r.soloist_won else 'LOST'}"
                print(f"    seed={r.seed:>10d}  {winner:<12s}  "
                      f"trump={r.trump or '-':<8s}  "
                      f"pts={r.soloist_pts:>3d}-{r.defender_pts:<3d}  "
                      f"margin={abs(r.soloist_pts - r.defender_pts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head Ulti evaluation (Parti, hybrid engine)",
    )
    parser.add_argument("--model-a", type=str, required=True,
                        help="Path to model A checkpoint (or 'random')")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Path to model B checkpoint (or 'random')")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per role (default 100, total = 2x)")
    parser.add_argument("--sims", type=int, default=20,
                        help="MCTS simulations per move (default 20)")
    parser.add_argument("--dets", type=int, default=1,
                        help="MCTS determinizations (default 1)")
    parser.add_argument("--endgame-tricks", type=int, default=6,
                        help="Solver endgame threshold (default 6)")
    parser.add_argument("--pimc-dets", type=int, default=20,
                        help="PIMC determinizations for solver (default 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--highlights", type=int, default=5,
                        help="Number of highlight games to show (default 5)")
    args = parser.parse_args()

    run_evaluation(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        num_games=args.games,
        sims=args.sims,
        dets=args.dets,
        endgame_tricks=args.endgame_tricks,
        pimc_dets=args.pimc_dets,
        seed=args.seed,
        show_highlights=args.highlights,
    )


if __name__ == "__main__":
    main()
