"""Verify the Cython solver against the pure Python solver.

Runs both solvers on random positions and checks that they agree
on move values.  Also benchmarks the speedup.

Covers all pluggable contract evaluators:
  - Parti (default)
  - Betli (binary: win if 0 tricks, lose otherwise)
  - Durchmars (binary: win if all 10 tricks)
  - Parti + Ulti (card points + ulti bonus)

Usage:
    python scripts/test_cython_solver.py
"""

import random
import sys
import time

sys.path.insert(0, "src")

from trickster.games.ulti.adapter import UltiGame
from trickster.games.ulti.cards import TRICKS_PER_GAME
from trickster.games.ulti.game import (
    current_player,
    is_terminal,
    legal_actions,
    play_card,
)
from trickster.solver import solve_root as py_solve_root
from trickster.solver import solve_best as py_solve_best

try:
    from trickster._solver_core import (
        solve_root as cy_solve_root,
        solve_best as cy_solve_best,
        get_stats as cy_get_stats,
        CONTRACTS as cy_contracts,
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("WARNING: Cython solver not built. Run:")
    print("  pip install cython")
    print("  python setup_cython.py build_ext --inplace")
    print()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _play_to_trick(gs, rng, target_trick):
    """Play random moves until we reach the target trick number."""
    while gs.trick_no < target_trick and not is_terminal(gs):
        acts = legal_actions(gs)
        play_card(gs, rng.choice(acts))


# ---------------------------------------------------------------------------
#  Test 1: Correctness — Parti (Python vs Cython)
# ---------------------------------------------------------------------------

def test_correctness_parti(num_seeds=50, start_trick=5):
    """Compare Python and Cython solve_root on Parti positions."""
    print(f"=== Test 1: Parti correctness ({num_seeds} positions, "
          f"from trick {start_trick}) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    mismatches = 0

    for seed in range(1000, 1000 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="simple")
        gs = state.gs
        _play_to_trick(gs, rng, start_trick)

        if is_terminal(gs):
            continue

        py_vals = py_solve_root(gs, max_exact_tricks=10)
        cy_vals = cy_solve_root(gs, contract="parti")

        py_cards = set(py_vals.keys())
        cy_cards = set(cy_vals.keys())

        if py_cards != cy_cards:
            print(f"  MISMATCH seed={seed}: different move sets")
            mismatches += 1
            continue

        for card in py_cards:
            if abs(py_vals[card] - cy_vals[card]) > 0.01:
                print(f"  MISMATCH seed={seed}: {card.short()} "
                      f"py={py_vals[card]:.1f} cy={cy_vals[card]:.1f}")
                mismatches += 1
                break

    ok = mismatches == 0
    print(f"  Tested {num_seeds} seeds: {mismatches} mismatches")
    print(f"  PASS: {'YES' if ok else 'NO'}")
    return ok


# ---------------------------------------------------------------------------
#  Test 2: Betli — binary evaluation
# ---------------------------------------------------------------------------

def test_betli(num_seeds=50, start_trick=5):
    """Verify Betli evaluator: values are 10.0 (win) or 0.0 (lose)."""
    print(f"\n=== Test 2: Betli binary evaluation ({num_seeds} positions) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    bad = 0
    wins = 0
    tested = 0

    for seed in range(2000, 2000 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="betli")
        gs = state.gs
        _play_to_trick(gs, rng, start_trick)

        if is_terminal(gs):
            continue

        tested += 1
        vals = cy_solve_root(gs, contract="betli")

        for card, v in vals.items():
            if abs(v - 10.0) > 0.01 and abs(v - 0.0) > 0.01:
                print(f"  BAD VALUE seed={seed}: {card.short()} = {v:.2f} "
                      f"(expected 0.0 or 10.0)")
                bad += 1
                break

        # Count how many positions are Betli-winnable
        max_val = max(vals.values())
        if abs(max_val - 10.0) < 0.01:
            wins += 1

    ok = bad == 0
    print(f"  Tested {tested} seeds: {bad} bad values")
    print(f"  Betli-winnable: {wins}/{tested}")
    print(f"  PASS: {'YES' if ok else 'NO'}")
    return ok


# ---------------------------------------------------------------------------
#  Test 3: Betli win/loss agreement with Python solver
# ---------------------------------------------------------------------------

def test_betli_winloss(num_seeds=50, start_trick=5):
    """Check that Python and Cython agree on whether Betli is won or lost."""
    print(f"\n=== Test 3: Betli win/loss agreement ({num_seeds} positions) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    mismatches = 0
    tested = 0

    for seed in range(2100, 2100 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="betli")
        gs = state.gs
        _play_to_trick(gs, rng, start_trick)

        if is_terminal(gs):
            continue

        tested += 1
        py_vals = py_solve_root(gs, max_exact_tricks=10)
        cy_vals = cy_solve_root(gs, contract="betli")

        # Python: value == 10 means 0 tricks taken = Betli win
        # Cython: value == 10 means Betli win, 0 means loss
        player = current_player(gs)
        is_sol = (player == gs.soloist)

        if is_sol:
            py_best = max(py_vals.values())
            cy_best = max(cy_vals.values())
        else:
            py_best = min(py_vals.values())
            cy_best = min(cy_vals.values())

        py_win = abs(py_best - 10.0) < 0.01
        cy_win = abs(cy_best - 10.0) < 0.01

        if py_win != cy_win:
            print(f"  MISMATCH seed={seed}: py_win={py_win} cy_win={cy_win} "
                  f"(py_best={py_best:.1f} cy_best={cy_best:.1f})")
            mismatches += 1

    ok = mismatches == 0
    print(f"  Tested {tested} seeds: {mismatches} win/loss disagreements")
    print(f"  PASS: {'YES' if ok else 'NO'}")
    return ok


# ---------------------------------------------------------------------------
#  Test 4: Durchmars — binary evaluation
# ---------------------------------------------------------------------------

def test_durchmars(num_seeds=50, start_trick=7):
    """Verify Durchmars evaluator: values are 10.0 (win) or 0.0 (lose).

    Uses late endgame positions (trick 7+) to make durchmars more testable.
    """
    print(f"\n=== Test 4: Durchmars binary evaluation "
          f"({num_seeds} positions) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    bad = 0
    tested = 0
    dm_wins = 0

    for seed in range(3000, 3000 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="simple")
        gs = state.gs
        _play_to_trick(gs, rng, start_trick)

        if is_terminal(gs):
            continue

        tested += 1
        # Force durchmars evaluation on a normal game state
        vals = cy_solve_root(gs, contract="durchmars")

        for card, v in vals.items():
            if abs(v - 10.0) > 0.01 and abs(v - 0.0) > 0.01:
                print(f"  BAD VALUE seed={seed}: {card.short()} = {v:.2f}")
                bad += 1
                break

        if max(vals.values()) > 5:
            dm_wins += 1

    ok = bad == 0
    print(f"  Tested {tested} seeds: {bad} bad values")
    print(f"  Durchmars possible: {dm_wins}/{tested}")
    print(f"  PASS: {'YES' if ok else 'NO'}")
    return ok


# ---------------------------------------------------------------------------
#  Test 5: Parti + Ulti evaluation
# ---------------------------------------------------------------------------

def test_parti_ulti(num_seeds=50, start_trick=5):
    """Verify Parti+Ulti evaluator produces correct value range."""
    print(f"\n=== Test 5: Parti+Ulti evaluation ({num_seeds} positions) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    bad = 0
    tested = 0

    for seed in range(4000, 4000 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="ulti")
        gs = state.gs
        _play_to_trick(gs, rng, start_trick)

        if is_terminal(gs):
            continue

        tested += 1
        vals = cy_solve_root(gs, contract="parti_ulti")

        for card, v in vals.items():
            # Parti values: 0-90+10 bonus = 0-100
            # Ulti bonus: +100 / penalty: -100
            # Valid range: -100 to 200
            if v < -101 or v > 201:
                print(f"  BAD VALUE seed={seed}: {card.short()} = {v:.2f} "
                      f"(out of range)")
                bad += 1
                break

    ok = bad == 0
    print(f"  Tested {tested} seeds: {bad} out-of-range values")
    print(f"  PASS: {'YES' if ok else 'NO'}")
    return ok


# ---------------------------------------------------------------------------
#  Test 6: Contract switching — same state, different contracts
# ---------------------------------------------------------------------------

def test_contract_switching(num_seeds=20, start_trick=6):
    """Same position evaluated under different contracts should differ."""
    print(f"\n=== Test 6: Contract switching ({num_seeds} positions) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    differ_count = 0
    tested = 0

    for seed in range(5000, 5000 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="simple")
        gs = state.gs
        _play_to_trick(gs, rng, start_trick)

        if is_terminal(gs):
            continue

        tested += 1

        v_parti = cy_solve_root(gs, contract="parti")
        v_dm = cy_solve_root(gs, contract="durchmars")

        # Parti and durchmars should generally give different values
        # (parti returns card-point range, durchmars returns 0 or 10)
        some_card = next(iter(v_parti))
        if abs(v_parti[some_card] - v_dm[some_card]) > 0.01:
            differ_count += 1

    ok = differ_count > 0
    print(f"  Tested {tested} seeds: {differ_count} had different values")
    print(f"  PASS: {'YES' if ok else 'NO'} (expect at least some difference)")
    return ok


# ---------------------------------------------------------------------------
#  Test 7: solve_best consistency
# ---------------------------------------------------------------------------

def test_solve_best(num_seeds=30, start_trick=6):
    """Check that solve_best agrees with solve_root."""
    print(f"\n=== Test 7: solve_best consistency ({num_seeds} positions) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    ok_count = 0
    total = 0

    for seed in range(6000, 6000 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="simple")
        gs = state.gs
        _play_to_trick(gs, rng, start_trick)

        if is_terminal(gs):
            continue

        total += 1
        root_vals = cy_solve_root(gs)
        best_card, best_val = cy_solve_best(gs)

        player = current_player(gs)
        if player == gs.soloist:
            root_best = max(root_vals.values())
        else:
            root_best = min(root_vals.values())

        if abs(root_best - best_val) < 0.01:
            ok_count += 1
        else:
            print(f"  MISMATCH seed={seed}: root_best={root_best:.1f} "
                  f"solve_best={best_val:.1f}")

    ok = ok_count == total
    print(f"  Passed: {ok_count}/{total}")
    print(f"  PASS: {'YES' if ok else 'NO'}")
    return ok


# ---------------------------------------------------------------------------
#  Test 8: Benchmark — speedup over Python
# ---------------------------------------------------------------------------

def test_benchmark(num_seeds=100, start_trick=5):
    """Time both solvers on identical positions."""
    print(f"\n=== Test 8: Benchmark ({num_seeds} positions, "
          f"5-trick endgames) ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    game = UltiGame()
    py_times = []
    cy_times = []
    py_nodes_list = []
    cy_nodes_list = []

    for seed in range(7000, 7000 + num_seeds):
        rng = random.Random(seed)
        state = game.new_game(seed=seed, training_mode="simple")
        gs_orig = state.gs
        _play_to_trick(gs_orig, rng, start_trick)

        if is_terminal(gs_orig):
            continue

        # Python solver
        gs_py = gs_orig.clone()
        t0 = time.perf_counter()
        py_solve_root(gs_py, max_exact_tricks=10)
        py_times.append(time.perf_counter() - t0)
        from trickster.solver import get_solve_stats
        py_nodes_list.append(get_solve_stats().nodes_explored)

        # Cython solver
        gs_cy = gs_orig.clone()
        t0 = time.perf_counter()
        cy_solve_root(gs_cy)
        cy_times.append(time.perf_counter() - t0)
        stats = cy_get_stats()
        cy_nodes_list.append(stats["nodes_explored"])

    if not py_times:
        print("  No valid positions found")
        return True

    n = len(py_times)
    py_avg = sum(py_times) / n * 1000
    cy_avg = sum(cy_times) / n * 1000
    speedup = py_avg / max(cy_avg, 0.001)

    py_nodes_avg = sum(py_nodes_list) / n
    cy_nodes_avg = sum(cy_nodes_list) / n

    py_nps = py_nodes_avg / (py_avg / 1000)
    cy_nps = cy_nodes_avg / (cy_avg / 1000)

    print(f"  Positions: {n}")
    print(f"  Python:  avg {py_avg:.1f}ms  ({py_nps:,.0f} nodes/sec)")
    print(f"  Cython:  avg {cy_avg:.3f}ms  ({cy_nps:,.0f} nodes/sec)")
    print(f"  Speedup: {speedup:.1f}x")

    ok = speedup > 5.0
    print(f"  PASS: {'YES' if ok else 'NO'} (speedup > 5x)")
    return ok


# ---------------------------------------------------------------------------
#  Test 9: Hybrid player import check
# ---------------------------------------------------------------------------

def test_hybrid_import():
    """Verify the hybrid player can be imported and instantiated."""
    print("\n=== Test 9: Hybrid player import ===")

    try:
        from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
        print(f"  Solver engine: {SOLVER_ENGINE}")
        print(f"  PASS: YES")
        return True
    except Exception as e:
        print(f"  Import error: {e}")
        print(f"  PASS: NO")
        return False


# ---------------------------------------------------------------------------
#  Test 10: Available contracts list
# ---------------------------------------------------------------------------

def test_contracts_list():
    """Verify the CONTRACTS list is exposed."""
    print("\n=== Test 10: Contracts list ===")

    if not HAS_CYTHON:
        print("  SKIP (Cython solver not available)")
        return True

    expected = {"parti", "betli", "durchmars", "parti_ulti", "ulti"}
    available = set(cy_contracts)

    missing = expected - available
    if missing:
        print(f"  Missing contracts: {missing}")
        print(f"  PASS: NO")
        return False

    print(f"  Available: {sorted(available)}")
    print(f"  PASS: YES")
    return True


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Cython Solver Verification & Benchmark")
    print("  (with pluggable contract evaluators)")
    print("=" * 70)
    print()

    results = {}
    results["parti_correctness"] = test_correctness_parti()
    results["betli_binary"] = test_betli()
    results["betli_winloss"] = test_betli_winloss()
    results["durchmars_binary"] = test_durchmars()
    results["parti_ulti_range"] = test_parti_ulti()
    results["contract_switching"] = test_contract_switching()
    results["solve_best"] = test_solve_best()
    results["benchmark"] = test_benchmark()
    results["hybrid_import"] = test_hybrid_import()
    results["contracts_list"] = test_contracts_list()

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name:<25s} {'PASS' if ok else 'FAIL'}")

    total_pass = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {total_pass}/{total} tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
