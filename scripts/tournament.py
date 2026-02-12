#!/usr/bin/env python3
"""Live tournament: watch two arenas play Snapszer in real time.

Arena 1: T5-Rook (MCTS)      vs  Pure PIMC-50
Arena 2: H2-Knight-v3 (hybrid) vs  Pure PIMC-50

Usage:
    python3 scripts/tournament.py                # default 40 min
    python3 scripts/tournament.py --minutes 10   # shorter run
    python3 scripts/tournament.py --pimc 30      # fewer PIMC samples

Press Ctrl+C to stop safely — results are printed before exit.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.games.snapszer.adapter import SnapszerGame, SnapszerNode
from trickster.games.snapszer.game import deal_awarded_game_points
from trickster.games.snapszer.constants import (
    DEFAULT_EVAL_DETS,
    DEFAULT_EVAL_SIMS,
    DEFAULT_LATE_THRESHOLD,
    DEFAULT_PIMC_SAMPLES,
)
from trickster.games.snapszer.minimax import alphabeta, pimc_minimax, is_phase2
from trickster.games.snapszer.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig, alpha_mcts_choose


# ---------------------------------------------------------------------------
#  Agents
# ---------------------------------------------------------------------------

def make_pimc_agent(n_samples: int = DEFAULT_PIMC_SAMPLES):
    """Pure PIMC engine — no neural network, just search."""
    game = SnapszerGame()

    def agent(node: SnapszerNode, player: int, rng: random.Random):
        actions = game.legal_actions(node)
        if len(actions) <= 1:
            return actions[0]
        # Phase 2: perfect info → exact minimax
        if is_phase2(node):
            _, best = alphabeta(node, game, player)
            return best if best is not None else rng.choice(actions)
        # Phase 1: PIMC
        best, _ = pimc_minimax(node, game, player, n_samples=n_samples, rng=rng)
        return best

    return agent, f"PIMC-{n_samples}"


def make_mcts_agent(model_dir: Path, sims: int = DEFAULT_EVAL_SIMS, dets: int = DEFAULT_EVAL_DETS):
    """Pure MCTS AlphaZero agent."""
    game = SnapszerGame()
    with open(model_dir / "net.pkl", "rb") as f:
        net = pickle.load(f)
    config = MCTSConfig(
        simulations=sims, determinizations=dets,
        use_value_head=True, use_policy_priors=True,
        dirichlet_alpha=0.0, visit_temp=0.1,
    )

    def agent(node, player, rng):
        actions = game.legal_actions(node)
        if len(actions) <= 1:
            return actions[0]
        return alpha_mcts_choose(node, game, net, player, config, rng)

    return agent, model_dir.name


def make_hybrid_agent(model_dir: Path, sims: int = DEFAULT_EVAL_SIMS,
                      dets: int = DEFAULT_EVAL_DETS,
                      pimc_samples: int = DEFAULT_PIMC_SAMPLES,
                      late_threshold: int = DEFAULT_LATE_THRESHOLD):
    """Hybrid agent (MCTS + PIMC + Minimax)."""
    game = SnapszerGame()
    with open(model_dir / "net.pkl", "rb") as f:
        net = pickle.load(f)
    config = MCTSConfig(
        simulations=sims, determinizations=dets,
        use_value_head=True, use_policy_priors=True,
        dirichlet_alpha=0.0, visit_temp=0.1,
    )
    hybrid = HybridPlayer(
        net=net, mcts_config=config, game=game,
        pimc_samples=pimc_samples, late_threshold=late_threshold,
    )

    def agent(node, player, rng):
        actions = game.legal_actions(node)
        if len(actions) <= 1:
            return actions[0]
        return hybrid.choose_action(node, player, rng)

    return agent, f"{model_dir.name} (hybrid)"


# ---------------------------------------------------------------------------
#  Arena state
# ---------------------------------------------------------------------------

@dataclass
class ArenaState:
    name: str
    label_a: str
    label_b: str
    games_played: int = 0
    game_in_progress: bool = False
    # Points
    pts_a: int = 0
    pts_b: int = 0
    # Wins (deal wins, not game points)
    wins_a: int = 0
    wins_b: int = 0
    # Last game result
    last_winner: str = ""
    last_pts: int = 0
    last_game_str: str = ""
    # Timing
    total_time: float = 0.0


# ---------------------------------------------------------------------------
#  Arena worker
# ---------------------------------------------------------------------------

def arena_worker(
    arena: ArenaState,
    agent_a, agent_b,
    stop_event: threading.Event,
    seed_offset: int = 0,
):
    """Play games until stop_event is set."""
    game = SnapszerGame()
    g = 0
    while not stop_event.is_set():
        seed = seed_offset + g
        a_idx = g % 2  # alternate sides
        rng_a = random.Random(seed + 10000)
        rng_b = random.Random(seed + 20000)

        arena.game_in_progress = True
        node = game.new_game(seed=seed, starting_leader=0)

        t0 = time.perf_counter()
        while not game.is_terminal(node):
            if stop_event.is_set():
                arena.game_in_progress = False
                return
            player = game.current_player(node)
            if player == a_idx:
                action = agent_a(node, player, rng_a)
            else:
                action = agent_b(node, player, rng_b)
            node = game.apply(node, action)

        elapsed = time.perf_counter() - t0

        # Score
        winner_idx, pts, _ = deal_awarded_game_points(node.gs)
        if winner_idx == a_idx:
            arena.pts_a += pts
            arena.wins_a += 1
            arena.last_winner = arena.label_a
        else:
            arena.pts_b += pts
            arena.wins_b += 1
            arena.last_winner = arena.label_b

        arena.last_pts = pts
        arena.games_played += 1
        arena.total_time += elapsed

        side_a = "P1" if a_idx == 0 else "P2"
        arena.last_game_str = (
            f"#{arena.games_played}: {arena.last_winner} won {pts}pt "
            f"({arena.label_a} as {side_a}) [{elapsed:.1f}s]"
        )
        arena.game_in_progress = False
        g += 1


# ---------------------------------------------------------------------------
#  Display
# ---------------------------------------------------------------------------

def render(arenas: list[ArenaState], elapsed: float, time_limit: float):
    """Render the live scoreboard."""
    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    limit_mins = int(time_limit) // 60
    pct = min(100, elapsed / time_limit * 100)

    lines = []
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════════╗")
    lines.append(f"║  SNAPSZER TOURNAMENT              {mins:02d}:{secs:02d} / {limit_mins}:00  ({pct:.0f}%)    ║")
    lines.append("╠══════════════════════════════════════════════════════════════════╣")

    for arena in arenas:
        n = arena.games_played
        ppd_a = arena.pts_a / max(1, n)
        ppd_b = arena.pts_b / max(1, n)
        avg_time = arena.total_time / max(1, n)

        status = "⏳ playing..." if arena.game_in_progress else "waiting"

        lines.append(f"║                                                                  ║")
        lines.append(f"║  {arena.name:<64s}║")
        lines.append(f"║  {'─' * 64}║")

        a_line = f"{arena.label_a:<22s}  {arena.wins_a:>3d}W {arena.wins_b:>3d}L  │  {ppd_a:.3f} pts/deal  │  {arena.pts_a:>4d} pts"
        b_line = f"{arena.label_b:<22s}  {arena.wins_b:>3d}W {arena.wins_a:>3d}L  │  {ppd_b:.3f} pts/deal  │  {arena.pts_b:>4d} pts"
        lines.append(f"║  {a_line:<64s}║")
        lines.append(f"║  {b_line:<64s}║")

        margin = ppd_a - ppd_b
        margin_str = f"Δ = {margin:+.3f}" if n > 0 else ""
        game_str = f"Game {n + 1} {status}  │  avg {avg_time:.1f}s/game  │  {margin_str}"
        lines.append(f"║  {game_str:<64s}║")

        if arena.last_game_str:
            lines.append(f"║  Last: {arena.last_game_str:<57s}║")
        else:
            lines.append(f"║  {'Waiting for first game...':<64s}║")

    lines.append(f"║                                                                  ║")
    lines.append("╠══════════════════════════════════════════════════════════════════╣")
    lines.append("║  Ctrl+C to stop safely — final summary will be printed          ║")
    lines.append("╚══════════════════════════════════════════════════════════════════╝")

    # Move to top and redraw
    sys.stdout.write("\033[H\033[J")
    sys.stdout.write("\n".join(lines))
    sys.stdout.write("\n")
    sys.stdout.flush()


def print_final(arenas: list[ArenaState], elapsed: float):
    """Print final summary (safe to call after display loop)."""
    mins = int(elapsed) // 60
    secs = int(elapsed) % 60

    print("\n")
    print("=" * 68)
    print(f"  FINAL RESULTS  ({mins:02d}:{secs:02d} elapsed)")
    print("=" * 68)

    for arena in arenas:
        n = arena.games_played
        if n == 0:
            print(f"\n  {arena.name}: no games completed")
            continue

        ppd_a = arena.pts_a / n
        ppd_b = arena.pts_b / n
        margin = ppd_a - ppd_b
        leader = arena.label_a if margin > 0 else arena.label_b if margin < 0 else "TIED"
        avg_time = arena.total_time / n

        print(f"\n  {arena.name}")
        print(f"  {'─' * 60}")
        print(f"  {arena.label_a:<22s}  {arena.wins_a:>3d}W  {ppd_a:.3f} pts/deal  {arena.pts_a:>4d} total pts")
        print(f"  {arena.label_b:<22s}  {arena.wins_b:>3d}W  {ppd_b:.3f} pts/deal  {arena.pts_b:>4d} total pts")
        print(f"  Games: {n}  |  Margin: {margin:+.3f}  |  Leader: {leader}")
        print(f"  Avg time/game: {avg_time:.1f}s  |  Total time: {arena.total_time:.0f}s")

    print()
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live Snapszer tournament")
    parser.add_argument("--minutes", type=int, default=40,
                        help="Time limit in minutes (default 40)")
    parser.add_argument("--pimc", type=int, default=DEFAULT_PIMC_SAMPLES,
                        help=f"PIMC samples for the pure engine (default {DEFAULT_PIMC_SAMPLES})")
    parser.add_argument("--sims", type=int, default=DEFAULT_EVAL_SIMS,
                        help=f"MCTS simulations (default {DEFAULT_EVAL_SIMS})")
    parser.add_argument("--dets", type=int, default=DEFAULT_EVAL_DETS,
                        help=f"MCTS determinizations (default {DEFAULT_EVAL_DETS})")
    args = parser.parse_args()

    time_limit = args.minutes * 60

    # Check models exist
    rook_dir = Path("models/T5-Rook")
    v3_dir = Path("models/H2-Knight-v3")
    for d in [rook_dir, v3_dir]:
        if not (d / "net.pkl").exists():
            print(f"  ERROR: model not found at {d}")
            return

    # Build agents
    rook_agent, rook_label = make_mcts_agent(rook_dir, args.sims, args.dets)
    v3_agent, v3_label = make_hybrid_agent(v3_dir, args.sims, args.dets)
    pimc_agent_1, pimc_label_1 = make_pimc_agent(args.pimc)
    pimc_agent_2, pimc_label_2 = make_pimc_agent(args.pimc)

    # Arenas
    arena1 = ArenaState(
        name=f"ARENA 1: {rook_label} vs {pimc_label_1}",
        label_a=rook_label, label_b=pimc_label_1,
    )
    arena2 = ArenaState(
        name=f"ARENA 2: {v3_label} vs {pimc_label_2}",
        label_a=v3_label, label_b=pimc_label_2,
    )

    stop = threading.Event()

    # Graceful shutdown
    def on_signal(signum, frame):
        stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Start workers
    t1 = threading.Thread(
        target=arena_worker,
        args=(arena1, rook_agent, pimc_agent_1, stop, 0),
        daemon=True,
    )
    t2 = threading.Thread(
        target=arena_worker,
        args=(arena2, v3_agent, pimc_agent_2, stop, 100000),
        daemon=True,
    )

    # Clear screen
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    t0 = time.perf_counter()
    t1.start()
    t2.start()

    # Display loop
    try:
        while not stop.is_set():
            elapsed = time.perf_counter() - t0
            if elapsed >= time_limit:
                stop.set()
                break
            render([arena1, arena2], elapsed, time_limit)
            # Sleep in small increments so Ctrl+C is responsive
            for _ in range(20):
                if stop.is_set():
                    break
                time.sleep(0.1)
    except KeyboardInterrupt:
        stop.set()

    # Wait for workers to finish current game
    t1.join(timeout=15)
    t2.join(timeout=15)

    elapsed = time.perf_counter() - t0

    # Clear live display and print final results
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    print_final([arena1, arena2], elapsed)


if __name__ == "__main__":
    main()
