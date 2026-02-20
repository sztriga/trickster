#!/usr/bin/env python3
"""Round-robin tournament between Ulti models.

Seats every combination of 3 players from the given model pool and plays
N deals per matchup.  Aggregates results into a strength ranking.

Key statistics reported:
  - Total game-points and average game-points per deal (with 95% CI)
  - Soloist performance: avg pts when soloist, bid rate
  - Head-to-head matrix
  - Final ranking by avg pts/deal

Models can include "random" for a baseline random player.
Each model can optionally specify a search speed (e.g. "knight:deep").

Usage:
    python scripts/tournament.py knight bronze random --games 500 --workers 6
    python scripts/tournament.py knight:fast bishop:deep random --games 1000
    python scripts/tournament.py scout knight bronze --games 300
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from trickster.bidding.auction_runner import run_auction, setup_bid_game
from trickster.bidding.constants import (
    KONTRA_THRESHOLD,
    MAX_DISCARDS,
    MIN_BID_PTS,
    PASS_PENALTY,
    REKONTRA_THRESHOLD,
)
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.game import deal
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
from trickster.mcts import MCTSConfig
from trickster.model import UltiNetWrapper
from trickster.train_utils import simple_outcome, _GAME_PTS_MAX
from trickster.training.bidding_loop import (
    DISPLAY_ORDER,
    _display_key,
    _kontrable_units,
)
from trickster.training.model_io import (
    DK_LABELS,
    load_wrappers,
    list_available_sources,
)


# ---------------------------------------------------------------------------
#  Search speed presets (same as eval_bidding.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchPreset:
    sims: int
    dets: int
    pimc_dets: int
    endgame_tricks: int

    def mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            simulations=self.sims,
            determinizations=self.dets,
            c_puct=1.5,
            dirichlet_alpha=0.0,
            dirichlet_weight=0.0,
            use_value_head=True,
            use_policy_priors=True,
            visit_temp=0.1,
            leaf_batch_size=8,
        )


SPEEDS: dict[str, SearchPreset] = {
    "fast":   SearchPreset(sims=40,  dets=2, pimc_dets=20, endgame_tricks=6),
    "normal": SearchPreset(sims=80,  dets=3, pimc_dets=25, endgame_tricks=6),
    "deep":   SearchPreset(sims=200, dets=6, pimc_dets=50, endgame_tricks=7),
}


def _parse_entrant(arg: str, default_speed: str) -> tuple[str, str]:
    """Parse 'model:speed' or 'model' into (source, speed_name)."""
    if ":" in arg:
        source, speed = arg.rsplit(":", 1)
        if speed not in SPEEDS:
            raise ValueError(f"Unknown speed '{speed}'. Choose from: {', '.join(SPEEDS)}")
        return source, speed
    return arg, default_speed




# ---------------------------------------------------------------------------
#  Deal result
# ---------------------------------------------------------------------------

@dataclass
class DealResult:
    matchup: tuple[str, str, str]
    seed: int
    dealer: int
    soloist: int
    soloist_model: str
    contract_dkey: str
    contract_mkey: str
    is_piros: bool
    model_pts: dict[str, float]
    kontrad: bool
    rekontrad: bool


# ---------------------------------------------------------------------------
#  Kontra (per-seat models)
# ---------------------------------------------------------------------------

def _decide_kontra(game, state, contract_key, seat_wrappers):
    gs = state.gs
    soloist = gs.soloist
    units = _kontrable_units(contract_key)
    if not units:
        return
    defenders = [i for i in range(3) if i != soloist]
    def_values = []
    for d in defenders:
        w = seat_wrappers[d].get(contract_key)
        if w is None:
            def_values.append(-1.0)
            continue
        feats = game.encode_state(state, d)
        def_values.append(w.predict_value(feats))
    if max(def_values) > KONTRA_THRESHOLD:
        for u in units:
            state.component_kontras[u] = 1
        sol_w = seat_wrappers[soloist].get(contract_key)
        if sol_w is not None:
            feats = game.encode_state(state, soloist)
            if sol_w.predict_value(feats) > REKONTRA_THRESHOLD:
                for u in units:
                    if state.component_kontras.get(u, 0) == 1:
                        state.component_kontras[u] = 2


def _play_one_deal(
    game: UltiGame,
    seat_wrappers: list[dict[str, UltiNetWrapper]],
    seat_presets: list[SearchPreset],
    seat_models: tuple[str, str, str],
    seed: int,
    deal_index: int,
    pass_penalty: float,
    min_bid_pts: float,
    max_discards: int,
) -> DealResult:
    rng = random.Random(seed)
    dealer = deal_index % 3

    gs, talon = deal(seed=seed, dealer=dealer)

    # --- Competitive auction (shared with training) ---
    auction_result = run_auction(
        gs, talon, dealer, seat_wrappers,
        min_bid_pts=min_bid_pts,
        max_discards=max_discards,
    )
    soloist = auction_result.soloist
    bid = auction_result.bid

    def _pts(raw: float) -> float:
        return raw * _GAME_PTS_MAX

    if bid is None:
        model_pts: dict[str, float] = defaultdict(float)
        sol_model = seat_models[soloist]
        model_pts[sol_model] += _pts((-pass_penalty * 2) / _GAME_PTS_MAX)
        for i in range(3):
            if i != soloist:
                model_pts[seat_models[i]] += _pts(pass_penalty / _GAME_PTS_MAX)
        return DealResult(
            matchup=seat_models, seed=seed, dealer=dealer, soloist=soloist,
            soloist_model=sol_model,
            contract_dkey="__pass__", contract_mkey="",
            is_piros=False, model_pts=dict(model_pts),
            kontrad=False, rekontrad=False,
        )

    dkey = _display_key(bid.contract_key, bid.is_piros)
    mkey = bid.contract_key

    state = setup_bid_game(
        game, gs, soloist, dealer, bid,
        initial_bidder=auction_result.initial_bidder,
    )

    players: list[HybridPlayer | None] = [None, None, None]
    for seat in range(3):
        w = seat_wrappers[seat].get(mkey)
        if w is not None:
            sp = seat_presets[seat]
            players[seat] = HybridPlayer(
                game, w,
                mcts_config=sp.mcts_config(),
                endgame_tricks=sp.endgame_tricks,
                pimc_determinizations=sp.pimc_dets,
                solver_temperature=0.1,
            )

    kontra_done = False
    rng_rand = random.Random(seed + 99999)

    while not game.is_terminal(state):
        if not kontra_done and state.gs.trick_no == 1:
            kontra_done = True
            _decide_kontra(game, state, mkey, seat_wrappers)

        player = game.current_player(state)
        actions = game.legal_actions(state)
        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        hybrid = players[player]
        action = hybrid.choose_action(state, player, rng) if hybrid else rng_rand.choice(actions)
        state = game.apply(state, action)

    model_pts = {}
    for seat in range(3):
        raw = simple_outcome(state, seat)
        m = seat_models[seat]
        model_pts[m] = model_pts.get(m, 0.0) + _pts(raw)

    kontras = state.component_kontras
    return DealResult(
        matchup=seat_models, seed=seed, dealer=dealer, soloist=soloist,
        soloist_model=seat_models[soloist],
        contract_dkey=dkey, contract_mkey=mkey,
        is_piros=bid.is_piros, model_pts=dict(model_pts),
        kontrad=any(v >= 1 for v in kontras.values()),
        rekontrad=any(v >= 2 for v in kontras.values()),
    )


# ---------------------------------------------------------------------------
#  Worker pool
# ---------------------------------------------------------------------------

_TW_GAME: UltiGame | None = None
_TW_ALL_WRAPPERS: dict[str, dict[str, UltiNetWrapper]] = {}
_TW_ALL_PRESETS: dict[str, SearchPreset] = {}


def _init_worker(
    model_sources: list[str],
    model_presets_raw: dict[str, tuple],
) -> None:
    global _TW_GAME, _TW_ALL_WRAPPERS, _TW_ALL_PRESETS
    _TW_GAME = UltiGame()
    _TW_ALL_WRAPPERS = {src: load_wrappers(src) for src in model_sources}
    _TW_ALL_PRESETS = {src: SearchPreset(*t) for src, t in model_presets_raw.items()}


def _worker_fn(args: tuple) -> DealResult:
    (seat_models, seed, deal_index, pass_penalty, min_bid_pts, max_discards) = args
    seat_wrappers = [_TW_ALL_WRAPPERS[m] for m in seat_models]
    seat_presets = [_TW_ALL_PRESETS[m] for m in seat_models]
    return _play_one_deal(
        _TW_GAME, seat_wrappers, seat_presets, seat_models,
        seed, deal_index, pass_penalty, min_bid_pts, max_discards,
    )


# ---------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------

def _ci95(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, stderr, half-width of 95% CI)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    hw = 1.96 * se
    return mean, se, hw


@dataclass
class ModelStats:
    """Accumulated stats for one model across all matchups."""
    name: str
    deal_pts: list[float] = field(default_factory=list)
    soloist_pts: list[float] = field(default_factory=list)
    soloist_deals: int = 0
    soloist_bids: int = 0
    total_deals: int = 0


def _collect_stats(
    results: list[DealResult],
    model_names: list[str],
) -> dict[str, ModelStats]:
    stats: dict[str, ModelStats] = {m: ModelStats(name=m) for m in model_names}

    for r in results:
        for m in model_names:
            if m not in r.matchup:
                continue
            pts = r.model_pts.get(m, 0.0)
            count_in_matchup = r.matchup.count(m)
            per_seat_pts = pts / count_in_matchup
            for _ in range(count_in_matchup):
                stats[m].deal_pts.append(per_seat_pts)
                stats[m].total_deals += 1

            if r.soloist_model == m:
                stats[m].soloist_pts.append(pts)
                stats[m].soloist_deals += 1
                if r.contract_dkey != "__pass__":
                    stats[m].soloist_bids += 1

    return stats


def _collect_h2h(
    results: list[DealResult],
    model_names: list[str],
) -> dict[tuple[str, str], list[float]]:
    """Head-to-head: h2h[(a, b)] = list of per-deal pts for a in deals where both played."""
    h2h: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        models_in = set(r.matchup)
        for a in model_names:
            if a not in models_in:
                continue
            for b in model_names:
                if b == a or b not in models_in:
                    continue
                pts_a = r.model_pts.get(a, 0.0) / r.matchup.count(a)
                h2h[(a, b)].append(pts_a)
    return dict(h2h)


# ---------------------------------------------------------------------------
#  Results printer
# ---------------------------------------------------------------------------

def _print_results(
    results: list[DealResult],
    model_names: list[str],
    matchups: list[tuple[str, str, str]],
    elapsed: float,
) -> None:
    N = len(results)
    stats = _collect_stats(results, model_names)
    h2h = _collect_h2h(results, model_names)

    col = max(16, *(len(m) + 2 for m in model_names))

    print()
    print("=" * 72)
    print("  TOURNAMENT RESULTS")
    print("=" * 72)

    # ── Ranking by avg pts/deal ───────────────────────────────────
    print()
    print("  RANKING (by avg game-points per deal)")
    print()
    print(f"  {'#':<3} {'Model':<{col}} {'Total pts':>10} {'Avg/deal':>16} "
          f"{'Deals':>6}")
    print(f"  {'─'*3} {'─'*col} {'─'*10} {'─'*16} {'─'*6}")

    ranked = sorted(model_names, key=lambda m: -sum(stats[m].deal_pts) / max(1, len(stats[m].deal_pts)))
    for rank, m in enumerate(ranked, 1):
        s = stats[m]
        total = sum(s.deal_pts)
        mean, se, hw = _ci95(s.deal_pts)
        ci_str = f"{mean:>+.3f} ± {hw:.3f}"
        print(f"  {rank:<3} {m:<{col}} {total:>+10.1f} {ci_str:>16} {s.total_deals:>6}")

    # ── Soloist performance ───────────────────────────────────────
    print()
    print("  SOLOIST PERFORMANCE (avg game-points when soloist)")
    print()
    print(f"  {'Model':<{col}} {'Avg sol pts':>18} {'Bid%':>6} {'Sol deals':>10}")
    print(f"  {'─'*col} {'─'*18} {'─'*6} {'─'*10}")

    for m in ranked:
        s = stats[m]
        if s.soloist_deals == 0:
            continue
        mean, se, hw = _ci95(s.soloist_pts)
        bid_pct = s.soloist_bids / s.soloist_deals * 100 if s.soloist_deals else 0
        ci_str = f"{mean:>+.3f} ± {hw:.3f}"
        print(f"  {m:<{col}} {ci_str:>18} {bid_pct:>5.0f}% {s.soloist_deals:>10}")

    # ── Head-to-head matrix ───────────────────────────────────────
    if len(model_names) > 1:
        print()
        print("  HEAD-TO-HEAD (avg pts/deal for row vs column)")
        print()
        hdr = f"  {'':>{col}}"
        for m in ranked:
            short = m[:col-1]
            hdr += f" {short:>{col}}"
        print(hdr)

        for a in ranked:
            row = f"  {a:>{col}}"
            for b in ranked:
                if a == b:
                    row += f" {'---':>{col}}"
                else:
                    vals = h2h.get((a, b), [])
                    if vals:
                        avg = sum(vals) / len(vals)
                        row += f" {avg:>+{col}.3f}"
                    else:
                        row += f" {'':>{col}}"
            print(row)

    # ── Per-contract breakdown ────────────────────────────────────
    played = [r for r in results if r.contract_dkey != "__pass__"]
    n_pass = N - len(played)

    print()
    print("  CONTRACT DISTRIBUTION")
    print()
    print(f"  {'Contract':<12} {'Deals':>6} {'%':>5}  {'Avg sol pts':>14} {'K%':>5}")
    print(f"  {'─'*12} {'─'*6} {'─'*5}  {'─'*14} {'─'*5}")

    if n_pass > 0:
        print(f"  {'Pass':<12} {n_pass:>6} {n_pass/N*100:>4.0f}%")

    for dk in DISPLAY_ORDER:
        dk_results = [r for r in results if r.contract_dkey == dk]
        n_dk = len(dk_results)
        if n_dk == 0:
            continue
        label = DK_LABELS.get(dk, dk)
        sol_pts_list = []
        for r in dk_results:
            raw_sol = r.model_pts.get(r.soloist_model, 0.0)
            sol_pts_list.append(raw_sol)
        avg_sol, _, hw_sol = _ci95(sol_pts_list)
        n_kontra = sum(1 for r in dk_results if r.kontrad)
        print(f"  {label:<12} {n_dk:>6} {n_dk/N*100:>4.0f}%  "
              f"{avg_sol:>+6.2f} ± {hw_sol:<5.2f} {n_kontra/n_dk*100:>4.0f}%")

    # ── Footer ────────────────────────────────────────────────────
    print()
    n_matchups = len(matchups)
    deals_per = N // max(1, n_matchups)
    print(f"  {N} deals across {n_matchups} matchup(s) "
          f"(~{deals_per}/matchup) in {elapsed:.0f}s "
          f"({N / elapsed:.1f} deals/s)")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin tournament between Ulti models. "
                    "Every 3-player combination is played.",
    )
    parser.add_argument(
        "models", nargs="+", metavar="MODEL",
        help="Model sources to enter (e.g. knight bronze random). "
             "Append :speed for per-model search speed (e.g. knight:deep).",
    )
    parser.add_argument(
        "--speed", default="fast", choices=list(SPEEDS.keys()),
        help="Default search speed for all models (default: fast)",
    )
    parser.add_argument("--games", type=int, default=500,
                        help="Deals per matchup (default: 500)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-bid-pts", type=float, default=MIN_BID_PTS)
    parser.add_argument("--pass-penalty", type=float, default=PASS_PENALTY)
    parser.add_argument("--max-discards", type=int, default=MAX_DISCARDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse entrants
    entrants: list[tuple[str, str]] = []
    for arg in args.models:
        source, speed = _parse_entrant(arg, args.speed)
        entrants.append((source, speed))

    model_names = [src for src, _ in entrants]
    model_speeds = {src: SPEEDS[spd] for src, spd in entrants}

    if len(model_names) < 2:
        parser.error("Need at least 2 models for a tournament.")

    # Generate matchups: all combinations of 3 (with repetition if < 3 models)
    unique = list(dict.fromkeys(model_names))
    if len(unique) >= 3:
        matchups = list(itertools.combinations(unique, 3))
    else:
        matchups = [tuple(sorted(combo)) for combo in itertools.combinations_with_replacement(unique, 3)]
        matchups = list(dict.fromkeys(matchups))
        matchups = [m for m in matchups if len(set(m)) >= min(2, len(unique))]

    if not matchups:
        matchups = [tuple(unique + [unique[0]] * (3 - len(unique)))]

    # Expand matchups with all seat permutations for fairness
    fair_matchups: list[tuple[str, str, str]] = []
    for combo in matchups:
        for perm in set(itertools.permutations(combo)):
            fair_matchups.append(perm)

    total_deals = len(fair_matchups) * args.games
    deals_per_perm = args.games

    # Banner
    print()
    w = 60
    print("╔" + "═" * w + "╗")
    print(f"║  {'ULTI TOURNAMENT — Round Robin':<{w-2}}║")
    print("╚" + "═" * w + "╝")
    print(f"  Models: {', '.join(unique)}")
    print(f"  Matchups: {len(matchups)} combinations × permutations "
          f"= {len(fair_matchups)} seat arrangements")
    print(f"  Deals: {deals_per_perm}/arrangement × {len(fair_matchups)} "
          f"= {total_deals:,} total")
    all_same = len(set(s for _, s in entrants)) == 1
    if all_same:
        sp = list(model_speeds.values())[0]
        print(f"  Speed: {entrants[0][1]}  "
              f"({sp.sims} sims, {sp.dets} dets, "
              f"{sp.pimc_dets} PIMC, endgame={sp.endgame_tricks}t)")
    else:
        for src, spd in entrants:
            sp = model_speeds[src]
            print(f"    {src}: {spd} "
                  f"({sp.sims} sims, {sp.dets} dets, {sp.pimc_dets} PIMC)")
    print(f"  Workers: {args.workers}  Solver: {SOLVER_ENGINE}")
    print()

    # Build work items
    deal_rng = random.Random(args.seed)
    work_args = []
    deal_idx = 0
    for perm in fair_matchups:
        for _ in range(deals_per_perm):
            seed = deal_rng.randint(0, 2**31)
            work_args.append((
                perm, seed, deal_idx,
                args.pass_penalty, args.min_bid_pts, args.max_discards,
            ))
            deal_idx += 1

    # Serialize presets for workers
    model_presets_raw = {
        src: (sp.sims, sp.dets, sp.pimc_dets, sp.endgame_tricks)
        for src, sp in model_speeds.items()
    }
    # "random" needs an entry too
    for m in unique:
        if m not in model_presets_raw:
            sp = SPEEDS[args.speed]
            model_presets_raw[m] = (sp.sims, sp.dets, sp.pimc_dets, sp.endgame_tricks)

    results: list[DealResult] = []
    t0 = time.perf_counter()

    if args.workers > 1:
        print(f"  Loading models in {args.workers} workers...", flush=True)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(unique, model_presets_raw),
        ) as pool:
            for i, result in enumerate(pool.map(_worker_fn, work_args, chunksize=4), 1):
                results.append(result)
                if i % 50 == 0 or i == total_deals:
                    elapsed = time.perf_counter() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"\r  [{i}/{total_deals}] {elapsed:.0f}s "
                          f"({rate:.1f} deals/s)", end="", flush=True)
        print()
    else:
        print("  Loading models...", flush=True)
        game = UltiGame()
        all_wrappers = {src: load_wrappers(src) for src in unique}
        for src in unique:
            if src != "random":
                n = len(all_wrappers[src])
                print(f"    {src}: {n} contract models loaded")
        print()

        for i, wa in enumerate(work_args, 1):
            seat_models = wa[0]
            seat_wrappers = [all_wrappers[m] for m in seat_models]
            seat_presets = [model_speeds[m] for m in seat_models]
            result = _play_one_deal(
                game, seat_wrappers, seat_presets, seat_models,
                wa[1], wa[2], args.pass_penalty, args.min_bid_pts, args.max_discards,
            )
            results.append(result)
            if i % 20 == 0 or i == total_deals:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"\r  [{i}/{total_deals}] {elapsed:.0f}s "
                      f"({rate:.1f} deals/s)", end="", flush=True)
        print()

    _print_results(results, unique, matchups, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
