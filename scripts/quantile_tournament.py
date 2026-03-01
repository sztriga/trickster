#!/usr/bin/env python3
"""Quantile tournament — same model, different pickup aggressiveness.

Each "player" is the same model but with a different pickup_quantile.
The quantile controls how optimistically the AI evaluates talon pickup:
  - Lower quantile (0.25) = cautious: needs a good median talon to pick up
  - Higher quantile (0.75) = aggressive: picks up on weaker hands
  - 1.0 = yolo: picks up if any sampled talon looks good

Plays all 3-seat permutations of selected presets and reports:
  - Overall ranking (avg pts/deal)
  - Contract distribution per preset
  - Head-to-head matrix

Usage:
    python scripts/quantile_tournament.py scout --games 200
    python scripts/quantile_tournament.py scout --games 500 --workers 4
    python scripts/quantile_tournament.py scout --presets cautious default aggressive
    python scripts/quantile_tournament.py scout --speed fast --games 300
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

from trickster.bidding.auction_runner import extract_player_bid_ranks, run_auction, setup_bid_game
from trickster.bidding.constants import (
    KONTRA_THRESHOLD,
    MIN_BID_PTS,
    PASS_PENALTY,
    PICKUP_QUANTILE_OVERRIDES,
    REKONTRA_THRESHOLD,
)
from trickster.games.ulti.adapter import UltiGame
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
)


# ---------------------------------------------------------------------------
#  Quantile presets
# ---------------------------------------------------------------------------

QUANTILE_PRESETS: dict[str, float] = {
    "cautious":   0.25,
    "default":    0.50,
    "aggressive": 0.75,
    "yolo":       1.00,
}

# ---------------------------------------------------------------------------
#  Search speed presets (same as tournament.py)
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


# ---------------------------------------------------------------------------
#  Deal result
# ---------------------------------------------------------------------------

@dataclass
class DealResult:
    seat_presets: tuple[str, str, str]   # preset name per seat
    seed: int
    dealer: int
    soloist: int
    soloist_preset: str
    contract_dkey: str
    contract_mkey: str
    is_piros: bool
    preset_pts: dict[str, float]         # preset → total pts in this deal
    kontrad: bool
    rekontrad: bool


# ---------------------------------------------------------------------------
#  Kontra logic (same as tournament.py)
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


# ---------------------------------------------------------------------------
#  Play one deal
# ---------------------------------------------------------------------------

def _play_one_deal(
    game: UltiGame,
    wrappers: dict[str, UltiNetWrapper],
    search_preset: SearchPreset,
    seat_preset_names: tuple[str, str, str],
    seat_quantiles: list[float],
    seed: int,
    deal_index: int,
    pass_penalty: float,
    min_bid_pts: float,
    quantile_overrides: dict[str, float] | None = None,
) -> DealResult:
    rng = random.Random(seed)
    dealer = deal_index % 3

    gs, talon = deal(seed=seed, dealer=dealer)

    # All seats use the same model — only pickup_quantile differs
    seat_wrappers = [wrappers, wrappers, wrappers]

    auction_result = run_auction(
        gs, talon, dealer, seat_wrappers,
        min_bid_pts=min_bid_pts,
        pickup_quantile=seat_quantiles,
        quantile_overrides=quantile_overrides,
    )
    soloist = auction_result.soloist
    bid = auction_result.bid

    def _pts(raw: float) -> float:
        return raw * _GAME_PTS_MAX

    if bid is None:
        preset_pts: dict[str, float] = defaultdict(float)
        sol_preset = seat_preset_names[soloist]
        preset_pts[sol_preset] += _pts((-pass_penalty * 2) / _GAME_PTS_MAX)
        for i in range(3):
            if i != soloist:
                preset_pts[seat_preset_names[i]] += _pts(pass_penalty / _GAME_PTS_MAX)
        return DealResult(
            seat_presets=seat_preset_names, seed=seed, dealer=dealer,
            soloist=soloist, soloist_preset=sol_preset,
            contract_dkey="__pass__", contract_mkey="",
            is_piros=False, preset_pts=dict(preset_pts),
            kontrad=False, rekontrad=False,
        )

    dkey = _display_key(bid.contract_key, bid.is_piros)
    mkey = bid.contract_key

    state = setup_bid_game(
        game, gs, soloist, dealer, bid,
        initial_bidder=auction_result.initial_bidder,
        player_bid_ranks=extract_player_bid_ranks(auction_result.auction),
    )

    # All seats use the same model & search config
    w = wrappers.get(mkey)
    players: list[HybridPlayer | None] = [None, None, None]
    if w is not None:
        for seat in range(3):
            players[seat] = HybridPlayer(
                game, w,
                mcts_config=search_preset.mcts_config(),
                endgame_tricks=search_preset.endgame_tricks,
                pimc_determinizations=search_preset.pimc_dets,
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

    preset_pts = defaultdict(float)
    for seat in range(3):
        raw = simple_outcome(state, seat)
        preset_pts[seat_preset_names[seat]] += _pts(raw)

    kontras = state.component_kontras
    return DealResult(
        seat_presets=seat_preset_names, seed=seed, dealer=dealer,
        soloist=soloist, soloist_preset=seat_preset_names[soloist],
        contract_dkey=dkey, contract_mkey=mkey,
        is_piros=bid.is_piros, preset_pts=dict(preset_pts),
        kontrad=any(v >= 1 for v in kontras.values()),
        rekontrad=any(v >= 2 for v in kontras.values()),
    )


# ---------------------------------------------------------------------------
#  Worker pool
# ---------------------------------------------------------------------------

_TW_GAME: UltiGame | None = None
_TW_WRAPPERS: dict[str, UltiNetWrapper] = {}
_TW_PRESET: SearchPreset | None = None
_TW_Q_OVERRIDES: dict[str, float] | None = None


def _init_worker(model_source: str, preset_raw: tuple, q_overrides: dict[str, float] | None) -> None:
    global _TW_GAME, _TW_WRAPPERS, _TW_PRESET, _TW_Q_OVERRIDES
    _TW_GAME = UltiGame()
    _TW_WRAPPERS = load_wrappers(model_source)
    _TW_PRESET = SearchPreset(*preset_raw)
    _TW_Q_OVERRIDES = q_overrides


def _worker_fn(args: tuple) -> DealResult:
    (seat_preset_names, seat_quantiles, seed, deal_index,
     pass_penalty, min_bid_pts) = args
    return _play_one_deal(
        _TW_GAME, _TW_WRAPPERS, _TW_PRESET,
        seat_preset_names, seat_quantiles,
        seed, deal_index, pass_penalty, min_bid_pts,
        quantile_overrides=_TW_Q_OVERRIDES,
    )


# ---------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------

def _ci95(values: list[float]) -> tuple[float, float, float]:
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
class PresetStats:
    name: str
    quantile: float
    deal_pts: list[float] = field(default_factory=list)
    soloist_pts: list[float] = field(default_factory=list)
    soloist_deals: int = 0
    soloist_bids: int = 0
    total_deals: int = 0
    # Per-contract soloist pts for this preset
    contract_soloist_pts: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    contract_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _collect_stats(
    results: list[DealResult],
    preset_names: list[str],
    quantiles: dict[str, float],
) -> dict[str, PresetStats]:
    stats = {p: PresetStats(name=p, quantile=quantiles[p]) for p in preset_names}

    for r in results:
        for p in preset_names:
            if p not in r.seat_presets:
                continue
            pts = r.preset_pts.get(p, 0.0)
            count_in = r.seat_presets.count(p)
            per_seat = pts / count_in
            for _ in range(count_in):
                stats[p].deal_pts.append(per_seat)
                stats[p].total_deals += 1

            if r.soloist_preset == p:
                stats[p].soloist_pts.append(pts)
                stats[p].soloist_deals += 1
                if r.contract_dkey != "__pass__":
                    stats[p].soloist_bids += 1
                    stats[p].contract_soloist_pts[r.contract_dkey].append(pts)
                    stats[p].contract_counts[r.contract_dkey] += 1

    return stats


def _collect_h2h(
    results: list[DealResult],
    preset_names: list[str],
) -> dict[tuple[str, str], list[float]]:
    h2h: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        presets_in = set(r.seat_presets)
        for a in preset_names:
            if a not in presets_in:
                continue
            for b in preset_names:
                if b == a or b not in presets_in:
                    continue
                pts_a = r.preset_pts.get(a, 0.0) / r.seat_presets.count(a)
                h2h[(a, b)].append(pts_a)
    return dict(h2h)


# ---------------------------------------------------------------------------
#  Print results
# ---------------------------------------------------------------------------

def _print_results(
    results: list[DealResult],
    preset_names: list[str],
    quantiles: dict[str, float],
    model_source: str,
    elapsed: float,
) -> None:
    N = len(results)
    stats = _collect_stats(results, preset_names, quantiles)
    h2h = _collect_h2h(results, preset_names)

    col = max(18, *(len(p) + 4 for p in preset_names))

    print()
    print("=" * 76)
    print(f"  QUANTILE TOURNAMENT — {model_source}")
    print("=" * 76)

    # ── Ranking ───────────────────────────────────────────────────
    print()
    print("  RANKING (by avg game-points per deal)")
    print()
    print(f"  {'#':<3} {'Preset':<{col}} {'Q':>4} {'Total pts':>10} "
          f"{'Avg/deal':>16} {'Deals':>6}")
    print(f"  {'─'*3} {'─'*col} {'─'*4} {'─'*10} {'─'*16} {'─'*6}")

    ranked = sorted(
        preset_names,
        key=lambda p: -sum(stats[p].deal_pts) / max(1, len(stats[p].deal_pts)),
    )
    for rank, p in enumerate(ranked, 1):
        s = stats[p]
        total = sum(s.deal_pts)
        mean, se, hw = _ci95(s.deal_pts)
        ci_str = f"{mean:>+.3f} ± {hw:.3f}"
        print(f"  {rank:<3} {p:<{col}} {s.quantile:>4.2f} "
              f"{total:>+10.1f} {ci_str:>16} {s.total_deals:>6}")

    # ── Soloist performance ───────────────────────────────────────
    print()
    print("  SOLOIST PERFORMANCE")
    print()
    print(f"  {'Preset':<{col}} {'Q':>4} {'Avg sol pts':>18} "
          f"{'Bid%':>6} {'Pass%':>6} {'Sol deals':>10}")
    print(f"  {'─'*col} {'─'*4} {'─'*18} {'─'*6} {'─'*6} {'─'*10}")

    for p in ranked:
        s = stats[p]
        if s.soloist_deals == 0:
            continue
        mean, _, hw = _ci95(s.soloist_pts)
        bid_pct = s.soloist_bids / s.soloist_deals * 100
        pass_pct = 100 - bid_pct
        ci_str = f"{mean:>+.3f} ± {hw:.3f}"
        print(f"  {p:<{col}} {s.quantile:>4.2f} {ci_str:>18} "
              f"{bid_pct:>5.0f}% {pass_pct:>5.0f}% {s.soloist_deals:>10}")

    # ── Head-to-head ──────────────────────────────────────────────
    if len(preset_names) > 1:
        print()
        print("  HEAD-TO-HEAD (avg pts/deal for row vs column)")
        print()
        hdr = f"  {'':>{col}}"
        for p in ranked:
            short = p[:col-1]
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

    # ── Per-contract breakdown per preset ─────────────────────────
    print()
    print("  CONTRACT DISTRIBUTION (when soloist)")
    print()

    # Global pass count
    n_pass = sum(1 for r in results if r.contract_dkey == "__pass__")
    n_played = N - n_pass
    print(f"  Overall: {N} deals, {n_pass} passes ({n_pass/N*100:.0f}%), "
          f"{n_played} played ({n_played/N*100:.0f}%)")
    print()

    for p in ranked:
        s = stats[p]
        sol_pass = s.soloist_deals - s.soloist_bids
        print(f"  {p} (q={s.quantile:.2f}):  "
              f"{s.soloist_bids} bids, {sol_pass} passes "
              f"({sol_pass/max(1,s.soloist_deals)*100:.0f}% pass)")

        if s.contract_counts:
            print(f"    {'Contract':<12} {'N':>5} {'Avg sol pts':>14}")
            print(f"    {'─'*12} {'─'*5} {'─'*14}")
            for dk in DISPLAY_ORDER:
                if dk not in s.contract_counts:
                    continue
                n_dk = s.contract_counts[dk]
                label = DK_LABELS.get(dk, dk)
                pts_list = s.contract_soloist_pts[dk]
                avg, _, hw = _ci95(pts_list)
                print(f"    {label:<12} {n_dk:>5} {avg:>+6.2f} ± {hw:<5.2f}")
        print()

    # ── Footer ────────────────────────────────────────────────────
    print(f"  {N} deals in {elapsed:.0f}s ({N/elapsed:.1f} deals/s)")
    print("=" * 76)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantile tournament — same model, different pickup aggressiveness.",
    )
    parser.add_argument(
        "model", metavar="MODEL",
        help="Model source (e.g. scout, knight)",
    )
    parser.add_argument(
        "--presets", nargs="+", default=list(QUANTILE_PRESETS.keys()),
        help=f"Quantile presets to compare (default: all). "
             f"Available: {', '.join(QUANTILE_PRESETS.keys())}",
    )
    parser.add_argument(
        "--speed", default="fast", choices=list(SPEEDS.keys()),
        help="Search speed (default: fast)",
    )
    parser.add_argument("--games", type=int, default=200,
                        help="Deals per seat arrangement (default: 200)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-bid-pts", type=float, default=MIN_BID_PTS)
    parser.add_argument("--pass-penalty", type=float, default=PASS_PENALTY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--betli-quantile", type=float, default=None,
        help=f"Override betli pickup quantile (default: {PICKUP_QUANTILE_OVERRIDES.get('betli', 'none')}). "
             f"Use 0 to disable the betli override.",
    )
    args = parser.parse_args()

    # Validate presets
    for p in args.presets:
        if p not in QUANTILE_PRESETS:
            parser.error(f"Unknown preset '{p}'. Choose from: {', '.join(QUANTILE_PRESETS)}")

    preset_names = args.presets
    quantiles = {p: QUANTILE_PRESETS[p] for p in preset_names}
    search = SPEEDS[args.speed]

    # Build per-contract quantile overrides
    q_overrides = dict(PICKUP_QUANTILE_OVERRIDES)
    if args.betli_quantile is not None:
        if args.betli_quantile == 0:
            q_overrides.pop("betli", None)
        else:
            q_overrides["betli"] = args.betli_quantile
    if not q_overrides:
        q_overrides = None

    # Generate all 3-seat permutations from the selected presets
    # (combinations_with_replacement for all combos, then all permutations)
    combos = list(itertools.combinations_with_replacement(preset_names, 3))
    fair_perms: list[tuple[str, str, str]] = []
    for combo in combos:
        for perm in set(itertools.permutations(combo)):
            fair_perms.append(perm)

    total_deals = len(fair_perms) * args.games

    # Banner
    print()
    w = 60
    print("╔" + "═" * w + "╗")
    print(f"║  {'QUANTILE TOURNAMENT':<{w-2}}║")
    print("╚" + "═" * w + "╝")
    print(f"  Model: {args.model}")
    print(f"  Presets: {', '.join(f'{p}(q={quantiles[p]:.2f})' for p in preset_names)}")
    if q_overrides:
        ovr_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(q_overrides.items()))
        print(f"  Quantile overrides: {ovr_str}")
    else:
        print(f"  Quantile overrides: none")
    print(f"  Arrangements: {len(fair_perms)} × {args.games} deals = {total_deals:,} total")
    print(f"  Speed: {args.speed}  "
          f"({search.sims} sims, {search.dets} dets, "
          f"{search.pimc_dets} PIMC, endgame={search.endgame_tricks}t)")
    print(f"  Workers: {args.workers}  Solver: {SOLVER_ENGINE}")
    print()

    # Build work items
    deal_rng = random.Random(args.seed)
    work_args = []
    deal_idx = 0
    for perm in fair_perms:
        seat_qs = [quantiles[p] for p in perm]
        for _ in range(args.games):
            seed = deal_rng.randint(0, 2**31)
            work_args.append((
                perm, seat_qs, seed, deal_idx,
                args.pass_penalty, args.min_bid_pts,
            ))
            deal_idx += 1

    results: list[DealResult] = []
    t0 = time.perf_counter()

    if args.workers > 1:
        print(f"  Loading model in {args.workers} workers...", flush=True)
        preset_raw = (search.sims, search.dets, search.pimc_dets, search.endgame_tricks)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.model, preset_raw, q_overrides),
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
        print("  Loading model...", flush=True)
        game = UltiGame()
        wrappers = load_wrappers(args.model)
        n = len(wrappers)
        print(f"    {args.model}: {n} contract models loaded")
        print()

        for i, wa in enumerate(work_args, 1):
            perm, seat_qs, seed, didx, pp, mbp = wa
            result = _play_one_deal(
                game, wrappers, search, perm, seat_qs,
                seed, didx, pp, mbp,
                quantile_overrides=q_overrides,
            )
            results.append(result)
            if i % 20 == 0 or i == total_deals:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"\r  [{i}/{total_deals}] {elapsed:.0f}s "
                      f"({rate:.1f} deals/s)", end="", flush=True)
        print()

    _print_results(results, preset_names, quantiles, args.model,
                   time.perf_counter() - t0)


if __name__ == "__main__":
    main()
