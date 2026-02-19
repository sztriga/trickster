#!/usr/bin/env python3
"""Evaluate models in full end-to-end Ulti with competitive bidding.

Seats 3 model sets at the table and plays N deals.  Each deal:
  1. Dealer rotates (0 → 1 → 2 → ...)
  2. Competitive auction: first bidder picks up the talon and bids;
     other players can pick up and overbid.  Each player uses their
     own NN value heads to evaluate contracts, filtering by legal bids.
  3. Auction winner plays as soloist with their chosen contract.
  4. Game is played with MCTS+solver; kontra/rekontra after trick 1.
  5. Game-point settlement computed (piros + kontra included).

Each seat can use a different model tier AND search speed.

Speed presets (estimated for 2000 deals @ 6 workers):
  fast   ~30s   — 40 sims, 2 dets, 20 PIMC
  normal ~3 min — 120 sims, 4 dets, 40 PIMC
  deep   ~10 min— 400 sims, 8 dets, 80 PIMC

Usage:
    python scripts/eval_bidding.py --seats knight_light knight_balanced scout --games 2000 --workers 6
    python scripts/eval_bidding.py --seats knight_light:deep knight_balanced:fast scout:normal
    python scripts/eval_bidding.py --seats knight_light knight_balanced scout --speed normal
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from trickster.bidding.auction_runner import run_auction, setup_bid_game
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
)

_BID_THRESHOLD = -2.0
_BID_MAX_DISCARDS = 15


# ---------------------------------------------------------------------------
#  Search speed presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchPreset:
    """MCTS + solver parameters for one seat."""
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


# Estimated wall-clock for 2000 deals @ 6 workers on Apple Silicon.
SPEEDS: dict[str, SearchPreset] = {
    "fast":   SearchPreset(sims=40,  dets=2, pimc_dets=20, endgame_tricks=6),  # ~1 min
    "normal": SearchPreset(sims=80,  dets=3, pimc_dets=25, endgame_tricks=6),  # ~5 min
    "deep":   SearchPreset(sims=200, dets=6, pimc_dets=50, endgame_tricks=7),  # ~15 min
}

SPEED_NAMES = list(SPEEDS.keys())


def _parse_seat(seat_arg: str, default_speed: str) -> tuple[str, str]:
    """Parse 'model:speed' or 'model' into (source, speed_name)."""
    if ":" in seat_arg:
        source, speed = seat_arg.rsplit(":", 1)
        if speed not in SPEEDS:
            raise ValueError(
                f"Unknown speed '{speed}'. Choose from: {', '.join(SPEEDS)}"
            )
        return source, speed
    return seat_arg, default_speed


# ---------------------------------------------------------------------------
#  Deal result
# ---------------------------------------------------------------------------

@dataclass
class DealResult:
    seed: int
    dealer: int
    soloist: int
    contract_dkey: str          # "p.parti", "ulti", ... or "__pass__"
    contract_mkey: str          # "parti", "ulti", ... or ""
    is_piros: bool
    seat_raw: tuple[float, float, float]  # simple_outcome per seat
    kontrad: bool
    rekontrad: bool


# ---------------------------------------------------------------------------
#  Kontra (per-seat models)
# ---------------------------------------------------------------------------


def _decide_kontra_eval(
    game: UltiGame,
    state: UltiNode,
    contract_key: str,
    seat_wrappers: list[dict[str, UltiNetWrapper]],
) -> None:
    """Kontra/rekontra with per-seat models."""
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

    kontrad = max(def_values) > 0.0
    if kontrad:
        for u in units:
            state.component_kontras[u] = 1

        sol_w = seat_wrappers[soloist].get(contract_key)
        if sol_w is not None:
            feats = game.encode_state(state, soloist)
            if sol_w.predict_value(feats) > 0.0:
                for u in units:
                    if state.component_kontras.get(u, 0) == 1:
                        state.component_kontras[u] = 2


# ---------------------------------------------------------------------------
#  Play one deal
# ---------------------------------------------------------------------------


def _play_one_deal(
    game: UltiGame,
    seat_wrappers: list[dict[str, UltiNetWrapper]],
    seat_presets: list[SearchPreset],
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
    result = run_auction(
        gs, talon, dealer, seat_wrappers,
        min_bid_pts=min_bid_pts,
        max_discards=max_discards,
    )
    soloist = result.soloist
    bid = result.bid

    if bid is None:
        # No real game (all Passz)
        seat_raw = [0.0, 0.0, 0.0]
        seat_raw[soloist] = (-pass_penalty * 2) / _GAME_PTS_MAX
        for i in range(3):
            if i != soloist:
                seat_raw[i] = pass_penalty / _GAME_PTS_MAX
        return DealResult(
            seed=seed, dealer=dealer, soloist=soloist,
            contract_dkey="__pass__", contract_mkey="",
            is_piros=False, seat_raw=tuple(seat_raw),
            kontrad=False, rekontrad=False,
        )

    dkey = _display_key(bid.contract_key, bid.is_piros)
    mkey = bid.contract_key

    state = setup_bid_game(
        game, gs, soloist, dealer, bid,
        initial_bidder=result.initial_bidder,
    )

    # Build per-seat players (each with their own search config)
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

    # Play
    kontra_done = False
    rng_rand = random.Random(seed + 99999)

    while not game.is_terminal(state):
        if not kontra_done and state.gs.trick_no == 1:
            kontra_done = True
            _decide_kontra_eval(game, state, mkey, seat_wrappers)

        player = game.current_player(state)
        actions = game.legal_actions(state)
        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        hybrid = players[player]
        action = hybrid.choose_action(state, player, rng) if hybrid else rng_rand.choice(actions)
        state = game.apply(state, action)

    # Settlement
    seat_raw = tuple(simple_outcome(state, seat) for seat in range(3))
    kontras = state.component_kontras
    return DealResult(
        seed=seed, dealer=dealer, soloist=soloist,
        contract_dkey=dkey, contract_mkey=mkey,
        is_piros=bid.is_piros, seat_raw=seat_raw,
        kontrad=any(v >= 1 for v in kontras.values()),
        rekontrad=any(v >= 2 for v in kontras.values()),
    )


# ---------------------------------------------------------------------------
#  Worker pool
# ---------------------------------------------------------------------------

_EW_GAME: UltiGame | None = None
_EW_SEATS: list[dict[str, UltiNetWrapper]] = []
_EW_PRESETS: list[SearchPreset] = []


def _init_worker(
    seat_sources: list[str],
    seat_presets_raw: list[tuple],
) -> None:
    global _EW_GAME, _EW_SEATS, _EW_PRESETS
    _EW_GAME = UltiGame()
    _EW_SEATS = [load_wrappers(src) for src in seat_sources]
    _EW_PRESETS = [SearchPreset(*t) for t in seat_presets_raw]


def _worker_fn(args: tuple) -> DealResult:
    (seed, deal_index, pass_penalty, min_bid_pts, max_discards) = args
    return _play_one_deal(
        _EW_GAME, _EW_SEATS, _EW_PRESETS, seed, deal_index,
        pass_penalty, min_bid_pts, max_discards,
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _pts(raw: float) -> float:
    return raw * _GAME_PTS_MAX


# ---------------------------------------------------------------------------
#  Results printer
# ---------------------------------------------------------------------------


def _print_results(
    results: list[DealResult],
    seat_labels: list[str],
    elapsed: float,
) -> None:
    import math

    N = len(results)
    seat_col = max(10, *(len(l) + 1 for l in seat_labels))
    w = 30 + seat_col * 3

    print()
    print("  ┌─ RESULTS " + "─" * (w - 11))
    print("  │")

    # ── Overall totals ──────────────────────────────────────────
    hdr = f"  │  {'':<12}"
    for sl in seat_labels:
        hdr += f" {sl:>{seat_col}}"
    print(hdr)
    sep = f"  │  {'─'*12}"
    for _ in seat_labels:
        sep += f" {'─'*seat_col}"
    print(sep)

    totals = [sum(_pts(r.seat_raw[s]) for r in results) for s in range(3)]
    avgs = [t / N for t in totals]
    row_tot = f"  │  {'Total pts':<12}"
    for s in range(3):
        row_tot += f" {totals[s]:>+{seat_col}.1f}"
    print(row_tot)
    row_avg = f"  │  {'Avg/deal':<12}"
    for s in range(3):
        row_avg += f" {avgs[s]:>+{seat_col}.3f}"
    print(row_avg)
    print("  │")

    # ── Soloist performance (primary metric) ───────────────────
    print(f"  │  SOLOIST STRENGTH (avg game-pts when soloist, including pass penalty)")
    print(f"  │  {'Seat':<12} {'Avg pts':>14} {'Bid%':>5} {'Deals':>5}")
    print(f"  │  {'─'*12} {'─'*14} {'─'*5} {'─'*5}")
    for seat in range(3):
        sol_all = [r for r in results if r.soloist == seat]
        sol_bid = [r for r in sol_all if r.contract_dkey != "__pass__"]
        n_sol = len(sol_all)
        if n_sol == 0:
            continue

        pts_list = [_pts(r.seat_raw[seat]) for r in sol_all]
        avg_pts = sum(pts_list) / n_sol
        std_pts = math.sqrt(sum((p - avg_pts) ** 2 for p in pts_list) / n_sol)
        se_pts = std_pts / math.sqrt(n_sol)
        bid_pct = len(sol_bid) / n_sol * 100

        print(f"  │  {seat_labels[seat]:<12} {avg_pts:>+6.2f} ± {se_pts:<5.2f} "
              f"{bid_pct:>4.0f}% {n_sol:>5}")

    # ── Per-contract breakdown ─────────────────────────────────
    print("  │")
    print(f"  │  PER-CONTRACT RESULTS (avg pts/deal per seat)")
    hdr = f"  │  {'Contract':<10} {'Deals':>5}"
    sep = f"  │  {'─'*10} {'─'*5}"
    for sl in seat_labels:
        hdr += f" {sl:>{seat_col}}"
        sep += f" {'─'*seat_col}"
    hdr += f"  {'K%':>4}"
    sep += f"  {'─'*4}"
    print(hdr)
    print(sep)

    n_pass = sum(1 for r in results if r.contract_dkey == "__pass__")
    if n_pass > 0:
        pass_results = [r for r in results if r.contract_dkey == "__pass__"]
        row = f"  │  {'Pass':<10} {n_pass:>5}"
        for seat in range(3):
            avg = sum(_pts(r.seat_raw[seat]) for r in pass_results) / n_pass
            row += f" {avg:>+{seat_col}.2f}"
        print(row)

    for dk in DISPLAY_ORDER:
        dk_results = [r for r in results if r.contract_dkey == dk]
        n_dk = len(dk_results)
        if n_dk == 0:
            continue
        label = DK_LABELS.get(dk, dk)
        row = f"  │  {label:<10} {n_dk:>5}"
        for seat in range(3):
            avg = sum(_pts(r.seat_raw[seat]) for r in dk_results) / n_dk
            row += f" {avg:>+{seat_col}.2f}"
        n_kontra = sum(1 for r in dk_results if r.kontrad)
        row += f"  {n_kontra/n_dk*100:>3.0f}%"
        print(row)

    # ── Summary line ───────────────────────────────────────────
    played = [r for r in results if r.contract_dkey != "__pass__"]
    if played:
        n_kontra = sum(1 for r in played if r.kontrad)
        n_rekontra = sum(1 for r in played if r.rekontrad)
        print("  │")
        print(f"  │  Kontra: {n_kontra}/{len(played)} ({n_kontra/len(played)*100:.0f}%)  "
              f"Rekontra: {n_rekontra}/{len(played)} ({n_rekontra/len(played)*100:.0f}%)")

    print(f"  │  {N} deals in {elapsed:.0f}s ({N/elapsed:.1f} deals/s)")
    print("  └" + "─" * w)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models in full end-to-end Ulti with bidding",
    )
    parser.add_argument(
        "--seats", nargs=3, required=True,
        metavar=("SEAT0", "SEAT1", "SEAT2"),
        help="Model source per seat, optionally with :speed suffix "
             "(e.g. knight_light:deep knight_balanced:fast scout)",
    )
    parser.add_argument(
        "--speed", default="fast", choices=SPEED_NAMES,
        help="Default search speed for all seats (default: fast)",
    )
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-bid-pts", type=float, default=0.0)
    parser.add_argument("--pass-penalty", type=float, default=2.0)
    parser.add_argument("--max-discards", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse seats + per-seat speeds
    seat_sources: list[str] = []
    seat_speed_names: list[str] = []
    seat_presets: list[SearchPreset] = []

    for seat_arg in args.seats:
        source, speed_name = _parse_seat(seat_arg, args.speed)
        seat_sources.append(source)
        seat_speed_names.append(speed_name)
        seat_presets.append(SPEEDS[speed_name])

    # Labels for display: "knight_light (fast)" or just "knight_light" if all same speed
    all_same_speed = len(set(seat_speed_names)) == 1
    if all_same_speed:
        seat_labels = seat_sources
    else:
        seat_labels = [f"{s} ({sp})" for s, sp in zip(seat_sources, seat_speed_names)]

    # Banner
    print()
    title = "BIDDING EVALUATION — 3-Seat Full Game"
    bw = len(title) + 4
    print("╔" + "═" * bw + "╗")
    print(f"║  {title:<{bw-2}}║")
    print("╚" + "═" * bw + "╝")

    for i in range(3):
        sp = seat_presets[i]
        speed_tag = f" [{seat_speed_names[i]}]" if not all_same_speed else ""
        print(f"  Seat {i}: {seat_sources[i]}{speed_tag}")

    if all_same_speed:
        sp = seat_presets[0]
        print(f"  Speed: {seat_speed_names[0]}  "
              f"({sp.sims} sims, {sp.dets} dets, "
              f"{sp.pimc_dets} PIMC, endgame={sp.endgame_tricks}t)")
    else:
        for i in range(3):
            sp = seat_presets[i]
            print(f"    Seat {i} search: {seat_speed_names[i]}  "
                  f"({sp.sims} sims, {sp.dets} dets, "
                  f"{sp.pimc_dets} PIMC, endgame={sp.endgame_tricks}t)")

    print(f"  Deals: {args.games}  Workers: {args.workers}  Solver: {SOLVER_ENGINE}")
    print(f"  Bid threshold: {args.min_bid_pts:+.1f}  "
          f"Pass penalty: {args.pass_penalty:.0f}/defender")
    print()

    # Validate all seat sources exist before running
    for src in set(seat_sources):
        if src == "random":
            continue
        w = load_wrappers(src)
        if not w:
            print(f"  ERROR: no models found for '{src}'. "
                  f"Check the name and that models/e2e/{src}/ exists.")
            sys.exit(1)

    # Serialize presets for workers (dataclass → tuple for pickling)
    seat_presets_raw = [
        (sp.sims, sp.dets, sp.pimc_dets, sp.endgame_tricks) for sp in seat_presets
    ]

    # Shuffle deal seeds so card distribution is independent of dealer rotation.
    # Without this, dealer=i%3 would correlate with card seed (seed+i),
    # causing persistent seat bias even in self-play.
    deal_rng = random.Random(args.seed)
    deal_seeds = [deal_rng.randint(0, 2**31) for _ in range(args.games)]

    work_args = [
        (deal_seeds[i], i, args.pass_penalty, args.min_bid_pts, args.max_discards)
        for i in range(args.games)
    ]

    results: list[DealResult] = []
    t0 = time.perf_counter()

    if args.workers > 1:
        print(f"  Loading models in {args.workers} workers...", flush=True)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(seat_sources, seat_presets_raw),
        ) as pool:
            for i, result in enumerate(pool.map(_worker_fn, work_args), 1):
                results.append(result)
                if i % 50 == 0 or i == args.games:
                    elapsed = time.perf_counter() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"\r  [{i}/{args.games}] {elapsed:.0f}s "
                          f"({rate:.1f} deals/s)", end="", flush=True)
        print()
    else:
        print("  Loading models...", flush=True)
        game = UltiGame()
        seats = [load_wrappers(src) for src in seat_sources]
        for src in set(seat_sources):
            if src != "random":
                n = len(load_wrappers(src))
                print(f"    {src}: {n} contract models loaded")
        print()

        for i, wa in enumerate(work_args, 1):
            result = _play_one_deal(
                game, seats, seat_presets, wa[0], wa[1],
                args.pass_penalty, args.min_bid_pts, args.max_discards,
            )
            results.append(result)
            if i % 20 == 0 or i == args.games:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"\r  [{i}/{args.games}] {elapsed:.0f}s "
                      f"({rate:.1f} deals/s)", end="", flush=True)
        print()

    _print_results(results, seat_labels, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
