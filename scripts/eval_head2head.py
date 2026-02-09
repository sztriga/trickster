#!/usr/bin/env python3
"""Head-to-head evaluation: pit two Ulti models against each other.

Loads two checkpoint files and plays N games with every possible
role assignment (Model A as soloist, Model B as both defenders, and
vice versa).  Reports detailed per-contract, per-role diagnostics
plus sample game highlights.

Usage:
    # Compare a checkpoint against the final model
    python scripts/eval_head2head.py \\
        --model-a models/checkpoints/mixed/step_00050.pt \\
        --model-b models/ulti_mixed.pt \\
        --games 100

    # Compare checkpoint against random baseline
    python scripts/eval_head2head.py \\
        --model-a models/ulti_mixed.pt \\
        --model-b random \\
        --games 200

    # Compare two checkpoints from the same run
    python scripts/eval_head2head.py \\
        --model-a models/checkpoints/mixed/step_00050.pt \\
        --model-b models/checkpoints/mixed/step_00200.pt \\
        --games 100
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from trickster.games.ulti.adapter import UltiGame, UltiNode, _IDX_TO_CARD
from trickster.games.ulti.cards import ALL_SUITS, Suit, BETLI_STRENGTH
from trickster.games.ulti.game import (
    last_trick_ulti_check,
    soloist_lost_betli,
    soloist_points,
    defender_points,
    soloist_tricks,
    soloist_won_simple,
    soloist_won_durchmars,
)
from trickster.mcts import MCTSConfig, alpha_mcts_policy
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
        input_dim=checkpoint.get("input_dim", 259),
        body_units=checkpoint.get("body_units", 256),
        body_layers=checkpoint.get("body_layers", 4),
        action_dim=checkpoint.get("action_dim", 32),
    )
    model_dict = net.state_dict()
    pretrained = {
        k: v for k, v in checkpoint["model_state_dict"].items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained)
    net.load_state_dict(model_dict)
    return UltiNetWrapper(net, device="cpu")


def model_label(path: str) -> str:
    if path.lower() == "random":
        return "Random"
    p = Path(path)
    return p.stem


# ---------------------------------------------------------------------------
#  Greedy discard (same as train_baseline)
# ---------------------------------------------------------------------------


def _greedy_discard(hand, betli, trump):
    candidates = list(hand)
    if betli:
        candidates.sort(key=lambda c: -BETLI_STRENGTH[c.rank])
    else:
        def simple_key(c):
            is_trump = 1 if (trump is not None and c.suit == trump) else 0
            return (is_trump, c.rank.value)
        candidates.sort(key=simple_key)
    return candidates[:2]


# ---------------------------------------------------------------------------
#  Game record
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """Result of one evaluated game."""
    seed: int
    contract: str        # "simple" or "betli"
    trump: str | None    # suit name or None
    soloist_model: str   # "A" or "B"
    soloist_won: bool
    soloist_pts: int
    defender_pts: int
    soloist_tricks: int
    ulti_info: str       # "none", "soloist_won", "soloist_lost", "defender_won", "defender_lost"
    durchmars: bool


# ---------------------------------------------------------------------------
#  Play one game
# ---------------------------------------------------------------------------


def play_one_game(
    game: UltiGame,
    wrapper_soloist: UltiNetWrapper | None,
    wrapper_defender: UltiNetWrapper | None,
    config: MCTSConfig,
    seed: int,
    training_mode: str,
) -> tuple[UltiNode, int]:
    """Play one game. Returns (terminal_state, soloist_idx)."""
    rng = random.Random(seed)
    dealer = seed % 3

    state = game.new_game(
        seed=seed,
        training_mode=training_mode,
        starting_leader=dealer,
        _discard_fn=_greedy_discard,
    )
    soloist_idx = state.gs.soloist

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        is_soloist = (player == soloist_idx)
        wrapper = wrapper_soloist if is_soloist else wrapper_defender

        if wrapper is not None:
            pi, action = alpha_mcts_policy(
                state, game, wrapper, player, config, rng,
            )
        else:
            action = rng.choice(actions)

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
    dets: int = 2,
    seed: int = 42,
    modes: list[str] | None = None,
    show_highlights: int = 5,
) -> None:
    if modes is None:
        modes = ["simple", "betli"]

    label_a = model_label(model_a_path)
    label_b = model_label(model_b_path)

    print("=" * 70)
    print("  Ulti Head-to-Head Evaluation")
    print("=" * 70)
    print(f"  Model A: {label_a}  ({model_a_path})")
    print(f"  Model B: {label_b}  ({model_b_path})")
    print(f"  Games: {num_games} per matchup × {len(modes)} modes × 2 roles")
    print(f"         = {num_games * len(modes) * 2} total games")
    print(f"  MCTS: {sims} sims × {dets} dets")
    print(f"  Modes: {', '.join(modes)}")
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
    rng_mode = random.Random(seed)
    t0 = time.perf_counter()
    game_count = 0
    total_total = num_games * len(modes) * 2

    for training_mode in modes:
        for soloist_label, sol_wrap, def_wrap in [
            ("A", wrapper_a, wrapper_b),
            ("B", wrapper_b, wrapper_a),
        ]:
            for g in range(num_games):
                game_seed = seed + hash((training_mode, soloist_label, g)) % (2**30)

                state, soloist_idx = play_one_game(
                    game, sol_wrap, def_wrap, config, game_seed, training_mode,
                )
                gs = state.gs

                # Determine outcome
                if gs.betli:
                    won = not soloist_lost_betli(gs)
                    contract = "betli"
                else:
                    won = soloist_won_simple(gs)
                    contract = "simple"

                trump_str = gs.trump.value if gs.trump else None
                side, ulti_won = last_trick_ulti_check(gs)
                if side == "none":
                    ulti_info = "none"
                else:
                    ulti_info = f"{side}_{'won' if ulti_won else 'lost'}"

                rec = GameRecord(
                    seed=game_seed,
                    contract=contract,
                    trump=trump_str,
                    soloist_model=soloist_label,
                    soloist_won=won,
                    soloist_pts=soloist_points(gs),
                    defender_pts=defender_points(gs),
                    soloist_tricks=soloist_tricks(gs),
                    ulti_info=ulti_info,
                    durchmars=soloist_won_durchmars(gs),
                )
                records.append(rec)

                game_count += 1
                if game_count % 50 == 0:
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

    # ----- Aggregate results -----
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

    # A wins = A won as soloist + B lost as soloist (= A defended successfully)
    a_wins = sum(r.soloist_won for r in a_sol) + sum(not r.soloist_won for r in b_sol)
    b_wins = sum(r.soloist_won for r in b_sol) + sum(not r.soloist_won for r in a_sol)
    total = len(records)

    print("=" * 70)
    print("  OVERALL RESULTS")
    print("=" * 70)
    print(f"  {label_a:>20s}  vs  {label_b}")
    print(f"  {'Wins':>20s}:  {a_wins} ({a_wins/total:.0%})  vs  {b_wins} ({b_wins/total:.0%})")
    print()
    print(f"  As Soloist:")
    print(f"    {label_a:>18s}:  {sum(r.soloist_won for r in a_sol)}/{len(a_sol)} "
          f"({a_sol_wr:.0%})")
    print(f"    {label_b:>18s}:  {sum(r.soloist_won for r in b_sol)}/{len(b_sol)} "
          f"({b_sol_wr:.0%})")
    print()

    # ----- Per-contract breakdown -----
    contracts = sorted(set(r.contract for r in records))
    print("-" * 70)
    print("  PER-CONTRACT BREAKDOWN")
    print("-" * 70)
    print(f"  {'Contract':<12s} {'Model':<10s} {'Role':<10s} "
          f"{'W':>4s} {'L':>4s} {'Total':>5s} {'WR':>6s}  {'Avg Pts':>8s}")
    print(f"  {'-'*12:<12s} {'-'*10:<10s} {'-'*10:<10s} "
          f"{'-'*4:>4s} {'-'*4:>4s} {'-'*5:>5s} {'-'*6:>6s}  {'-'*8:>8s}")

    for ct in contracts:
        for model_label, model_key in [(label_a, "A"), (label_b, "B")]:
            # As soloist
            sol_games = [r for r in records if r.contract == ct and r.soloist_model == model_key]
            if sol_games:
                w = sum(r.soloist_won for r in sol_games)
                l = len(sol_games) - w
                wr = w / len(sol_games)
                avg_pts = sum(r.soloist_pts for r in sol_games) / len(sol_games)
                print(f"  {ct:<12s} {model_label:<10s} {'soloist':<10s} "
                      f"{w:>4d} {l:>4d} {len(sol_games):>5d} {wr:>5.0%}  {avg_pts:>8.1f}")

            # As defender (other model is soloist)
            other_key = "B" if model_key == "A" else "A"
            def_games = [r for r in records if r.contract == ct and r.soloist_model == other_key]
            if def_games:
                w = sum(not r.soloist_won for r in def_games)
                l = len(def_games) - w
                wr = w / len(def_games)
                avg_pts = sum(r.defender_pts for r in def_games) / len(def_games)
                print(f"  {'':<12s} {model_label:<10s} {'defender':<10s} "
                      f"{w:>4d} {l:>4d} {len(def_games):>5d} {wr:>5.0%}  {avg_pts:>8.1f}")

    # ----- Trump suit breakdown (simple only) -----
    simple_recs = [r for r in records if r.contract == "simple" and r.trump]
    if simple_recs:
        print()
        print("-" * 70)
        print("  SIMPLE BY TRUMP SUIT")
        print("-" * 70)
        trumps = sorted(set(r.trump for r in simple_recs if r.trump))
        for trump in trumps:
            t_recs = [r for r in simple_recs if r.trump == trump]
            a_t = [r for r in t_recs if r.soloist_model == "A"]
            b_t = [r for r in t_recs if r.soloist_model == "B"]
            a_wr = sum(r.soloist_won for r in a_t) / max(1, len(a_t))
            b_wr = sum(r.soloist_won for r in b_t) / max(1, len(b_t))
            print(f"  {trump:<10s}  "
                  f"{label_a} sol: {sum(r.soloist_won for r in a_t)}/{len(a_t)} ({a_wr:.0%})  |  "
                  f"{label_b} sol: {sum(r.soloist_won for r in b_t)}/{len(b_t)} ({b_wr:.0%})")

    # ----- Special events -----
    ulti_events = [r for r in records if r.ulti_info != "none"]
    durchmars = [r for r in records if r.durchmars]
    betli_games = [r for r in records if r.contract == "betli"]
    zero_trick = [r for r in records if r.contract == "simple" and r.soloist_tricks == 0]

    if ulti_events or durchmars or zero_trick:
        print()
        print("-" * 70)
        print("  NOTABLE EVENTS")
        print("-" * 70)
        if ulti_events:
            for kind in ["soloist_won", "soloist_lost", "defender_won", "defender_lost"]:
                n = sum(1 for r in ulti_events if r.ulti_info == kind)
                if n:
                    desc = kind.replace("_", " ")
                    print(f"  Ulti ({desc}): {n} times")
        if durchmars:
            print(f"  Durchmars (soloist took all 10): {len(durchmars)} times")
        if zero_trick:
            print(f"  Soloist 0 tricks (simple): {len(zero_trick)} times")

    # ----- Points distribution -----
    print()
    print("-" * 70)
    print("  POINTS DISTRIBUTION (soloist pts when playing as soloist)")
    print("-" * 70)
    for model_label, model_key in [(label_a, "A"), (label_b, "B")]:
        sol_pts = [r.soloist_pts for r in records if r.soloist_model == model_key
                   and r.contract == "simple"]
        if sol_pts:
            print(f"  {model_label} (simple):  "
                  f"avg={np.mean(sol_pts):.1f}  "
                  f"min={min(sol_pts)}  "
                  f"max={max(sol_pts)}  "
                  f"median={int(np.median(sol_pts))}")
        bet_pts = [r.soloist_tricks for r in records if r.soloist_model == model_key
                   and r.contract == "betli"]
        if bet_pts:
            print(f"  {model_label} (betli):   "
                  f"avg tricks={np.mean(bet_pts):.1f}  "
                  f"0-trick rate={sum(1 for t in bet_pts if t == 0)/len(bet_pts):.0%}")

    # ----- Highlight games -----
    if show_highlights > 0:
        print()
        print("-" * 70)
        print(f"  GAME HIGHLIGHTS (top {show_highlights})")
        print("-" * 70)

        # Biggest soloist wins
        won_games = sorted(
            [r for r in records if r.soloist_won and r.contract == "simple"],
            key=lambda r: r.soloist_pts,
            reverse=True,
        )[:show_highlights]

        if won_games:
            print("  Biggest soloist wins (simple):")
            for r in won_games:
                print(f"    seed={r.seed:>10d}  {r.soloist_model}-soloist  "
                      f"trump={r.trump or '-':<8s}  "
                      f"pts={r.soloist_pts:>3d}-{r.defender_pts:<3d}  "
                      f"tricks={r.soloist_tricks}/10"
                      f"{'  DURI!' if r.durchmars else ''}"
                      f"{'  ULTI: ' + r.ulti_info if r.ulti_info != 'none' else ''}")

        # Closest games
        close_games = sorted(
            [r for r in records if r.contract == "simple"],
            key=lambda r: abs(r.soloist_pts - r.defender_pts),
        )[:show_highlights]

        if close_games:
            print("  Closest games (simple):")
            for r in close_games:
                winner = f"{r.soloist_model}-sol {'WON' if r.soloist_won else 'LOST'}"
                print(f"    seed={r.seed:>10d}  {winner:<12s}  "
                      f"trump={r.trump or '-':<8s}  "
                      f"pts={r.soloist_pts:>3d}-{r.defender_pts:<3d}  "
                      f"margin={abs(r.soloist_pts - r.defender_pts)}")

        # Betli highlights
        betli_lost = [r for r in records if r.contract == "betli" and not r.soloist_won]
        if betli_lost:
            print(f"  Betli failures ({len(betli_lost)} total, showing up to {min(show_highlights, len(betli_lost))}):")
            for r in betli_lost[:show_highlights]:
                print(f"    seed={r.seed:>10d}  {r.soloist_model}-soloist  "
                      f"took {r.soloist_tricks} tricks (should be 0)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head Ulti model evaluation",
    )
    parser.add_argument("--model-a", type=str, required=True,
                        help="Path to model A checkpoint (or 'random')")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Path to model B checkpoint (or 'random')")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per matchup per mode (default 100)")
    parser.add_argument("--sims", type=int, default=20,
                        help="MCTS simulations per move (default 20)")
    parser.add_argument("--dets", type=int, default=2,
                        help="MCTS determinizations (default 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--modes", type=str, default="simple,betli",
                        help="Comma-separated modes: simple,betli (default both)")
    parser.add_argument("--highlights", type=int, default=5,
                        help="Number of highlight games to show (default 5)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]

    run_evaluation(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        num_games=args.games,
        sims=args.sims,
        dets=args.dets,
        seed=args.seed,
        modes=modes,
        show_highlights=args.highlights,
    )


if __name__ == "__main__":
    main()
