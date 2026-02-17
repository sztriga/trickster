"""Hybrid MCTS + alpha-beta training engine for Ulti.

Shared training loop used by all Ulti training scripts.  Follows the
same pattern as ``trickster.training.alpha_zero`` for Snapszer: the
engine owns the full training loop (self-play → buffer → SGD) with
parallel workers, and callers pass configuration.

Features consolidated from train_baseline.py and train_parti.py:
  - Parallel self-play via ProcessPoolExecutor (``num_workers``)
  - Cosine LR annealing (``lr_start`` / ``lr_end``)
  - Decoupled SGD steps (fixed ``train_steps`` per iteration)
  - Deal enrichment via value head (``enrich_*`` params)
  - Per-role diagnostics (soloist vs defender value loss)
  - Policy accuracy tracking
  - Checkpoint load / resume (``initial_net``)
  - Progress callback (``on_progress``)

Usage (from a tier script)::

    from trickster.training.ulti_hybrid import train_ulti_hybrid, UltiTrainConfig

    cfg = UltiTrainConfig(steps=500, games_per_step=8, ...)
    net, stats = train_ulti_hybrid(cfg)
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
from trickster.mcts import MCTSConfig
from trickster.model import OnnxUltiWrapper, UltiNet, UltiNetWrapper, make_wrapper
from trickster.train_utils import ReplayBuffer, simple_outcome, solver_value_to_reward
from trickster.training.model_io import auto_device


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class UltiTrainConfig:
    """All hyperparameters for one training run.

    Tier scripts build this from their tier definitions and pass it
    to ``train_ulti_hybrid()``.
    """

    # -- Self-play budget --
    steps: int = 500
    games_per_step: int = 8

    # -- SGD --
    train_steps: int = 50          # SGD steps per iteration
    batch_size: int = 64
    buffer_size: int = 50_000
    lr_start: float = 1e-3
    lr_end: float = 2e-4           # cosine decay target (set = lr_start to disable)

    # -- MCTS (soloist / defender) --
    sol_sims: int = 20
    sol_dets: int = 1
    def_sims: int = 8
    def_dets: int = 1
    leaf_batch_size: int = 8       # batched NN eval in MCTS (1 = sequential)

    # -- Solver / hybrid --
    endgame_tricks: int = 6
    pimc_dets: int = 20
    solver_temp: float = 0.5

    # -- Network architecture --
    body_units: int = 256
    body_layers: int = 4

    # -- Parallelism --
    num_workers: int = 1
    concurrent_games: int = 1  # cross-game GPU batching (auto-set for GPU tiers)

    # -- Deal enrichment --
    enrichment: bool = False
    enrich_warmup: int = 20        # skip enrichment for first N steps
    enrich_random_frac: float = 0.3  # fraction of games always random

    # -- Value targets --
    solver_teacher: bool = False   # use solver value for MCTS positions

    # -- Kontra / Talon --
    kontra_enabled: bool = True    # apply kontra/rekontra after trick 1
    neural_discard: bool = True    # use value head for talon discard

    # -- General --
    seed: int = 42
    device: str = "cpu"
    training_mode: str = "simple"  # "simple" | "betli" | "ulti" | etc.

    @property
    def total_games(self) -> int:
        return self.steps * self.games_per_step


# ---------------------------------------------------------------------------
#  Training stats (returned to caller + passed to on_progress)
# ---------------------------------------------------------------------------


@dataclass
class UltiTrainStats:
    """Cumulative training diagnostics."""

    step: int = 0
    total_steps: int = 0
    total_games: int = 0
    total_samples: int = 0
    total_sgd_steps: int = 0

    # Current step losses
    vloss: float = 0.0
    ploss: float = 0.0
    pacc: float = 0.0
    sol_vloss: float = 0.0
    def_vloss: float = 0.0
    lr: float = 0.0

    # Rolling history (for trend detection)
    history_vloss: list = field(default_factory=list)
    history_ploss: list = field(default_factory=list)
    history_pacc: list = field(default_factory=list)

    # Self-play game-point tracking (current step)
    sp_sol_pts_sum: float = 0.0   # sum of soloist reward (normalised)
    sp_sol_total: int = 0

    train_time_s: float = 0.0


# ---------------------------------------------------------------------------
#  Multiprocessing helpers (module-level for pickling)
# ---------------------------------------------------------------------------

_WORKER_GAME: UltiGame | None = None
_WORKER_NET: UltiNet | None = None
_WORKER_WRAPPER: UltiNetWrapper | OnnxUltiWrapper | None = None


def _init_worker(net_kwargs: dict, device: str) -> None:
    """Called once per worker process to create game + network."""
    global _WORKER_GAME, _WORKER_NET, _WORKER_WRAPPER
    _WORKER_GAME = UltiGame()
    _WORKER_NET = UltiNet(**net_kwargs)
    _WORKER_WRAPPER = make_wrapper(_WORKER_NET, device=device)


def _play_game_in_worker(
    args: tuple,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Worker entry-point for parallel self-play."""
    (state_dict, sol_mcts_cfg, def_mcts_cfg,
     seed, endgame_tricks, pimc_dets, solver_temp,
     training_mode, enrich_thresh, solver_teacher,
     kontra_enabled, use_neural_discard) = args
    _WORKER_NET.load_state_dict(state_dict)
    if isinstance(_WORKER_WRAPPER, OnnxUltiWrapper):
        _WORKER_WRAPPER.sync_weights(_WORKER_NET)

    init_state = None
    if use_neural_discard:
        init_state = _new_game_with_neural_discard(
            _WORKER_GAME, _WORKER_WRAPPER, seed,
            training_mode=training_mode, dealer=seed % 3,
        )
    elif enrich_thresh > -999.0:
        init_state, _ = _value_enriched_new_game(
            _WORKER_GAME, _WORKER_WRAPPER, seed,
            min_value=enrich_thresh, training_mode=training_mode,
        )
    return _play_one_game(
        _WORKER_GAME, _WORKER_WRAPPER, sol_mcts_cfg, def_mcts_cfg, seed,
        endgame_tricks=endgame_tricks,
        pimc_dets=pimc_dets,
        solver_temp=solver_temp,
        training_mode=training_mode,
        initial_state=init_state,
        solver_teacher=solver_teacher,
        kontra_enabled=kontra_enabled,
    )


# ---------------------------------------------------------------------------
#  Deal enrichment via value head
# ---------------------------------------------------------------------------


def _value_enriched_new_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    seed: int,
    min_value: float,
    training_mode: str = "simple",
    max_attempts: int = 20,
) -> tuple[UltiNode, float]:
    """Deal games until the value head rates the soloist above *min_value*.

    Returns (state, soloist_value) — the best deal found, or the last
    attempt if none pass the threshold.
    """
    best_state = None
    best_val = -1.0
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt * 100_000
        state = game.new_game(
            seed=attempt_seed,
            training_mode=training_mode,
            starting_leader=seed % 3,
        )
        sol = state.gs.soloist
        feats = game.encode_state(state, sol)
        val = wrapper.predict_value(feats)
        if val > best_val:
            best_state, best_val = state, val
        if val >= min_value:
            return state, val
    return best_state, best_val


def _enrichment_threshold(step: int, total_steps: int) -> float:
    """Anneal the minimum soloist value from 0.0 -> -1.0 over training.

    Phase 1 (first 25%):   v >= 0.0   (only "even-ish or better" deals)
    Phase 2 (25%-60%):     v >= -0.25  (slightly losing included)
    Phase 3 (60%+):        disabled    (all deals accepted)

    Returns the threshold, or -999 to signal "accept everything".
    """
    frac = step / max(1, total_steps)
    if frac < 0.25:
        return 0.0
    if frac < 0.60:
        return -0.25
    return -999.0


# ---------------------------------------------------------------------------
#  Training mode → contract key (for kontra unit lookup)
# ---------------------------------------------------------------------------

_MODE_TO_CONTRACT_KEY: dict[str, str] = {
    "simple": "parti",
    "ulti":   "ulti",
    "40-100": "40-100",
    "betli":  "betli",
}


# ---------------------------------------------------------------------------
#  Kontra helper (same model for both sides in single-contract training)
# ---------------------------------------------------------------------------


def _decide_kontra_training(
    game: UltiGame,
    state: UltiNode,
    wrapper: UltiNetWrapper,
    training_mode: str,
) -> None:
    """Apply kontra/rekontra using the single shared value head.

    Same logic as ``bidding_loop._decide_kontra`` but adapted for
    single-contract self-play where all seats share one model.
    """
    from trickster.training.bidding_loop import _kontrable_units

    contract_key = _MODE_TO_CONTRACT_KEY.get(training_mode, "parti")
    units = _kontrable_units(contract_key)
    if not units:
        return

    gs = state.gs
    soloist = gs.soloist
    defenders = [i for i in range(3) if i != soloist]

    # Each defender evaluates with the same model
    def_values = []
    for d in defenders:
        feats = game.encode_state(state, d)
        v = wrapper.predict_value(feats)
        def_values.append(v)

    kontrad = max(def_values) > 0.0
    if kontrad:
        for u in units:
            state.component_kontras[u] = 1

    # Rekontra: soloist still expects to win
    if kontrad:
        feats = game.encode_state(state, soloist)
        sol_v = wrapper.predict_value(feats)
        if sol_v > 0.0:
            for u in units:
                if state.component_kontras.get(u, 0) == 1:
                    state.component_kontras[u] = 2


# ---------------------------------------------------------------------------
#  Neural talon discard for training
# ---------------------------------------------------------------------------


def _new_game_with_neural_discard(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    seed: int,
    training_mode: str,
    dealer: int,
) -> UltiNode:
    """Deal a new training game with value-head-driven talon discard.

    For adu contracts (parti, ulti, 40-100): picks up the talon,
    evaluates up to 20 discard pairs via the value head, and keeps
    the best.  For betli: same but with betli flag.

    Falls back to the standard ``game.new_game()`` for unknown modes.
    """
    from itertools import combinations

    from trickster.games.ulti.adapter import build_auction_constraints
    from trickster.games.ulti.cards import Card, Rank, Suit
    from trickster.games.ulti.game import (
        NUM_PLAYERS,
        deal,
        declare_all_marriages,
        discard_talon,
        next_player,
        pickup_talon,
        set_contract,
    )

    rng = random.Random(seed)
    gs, talon = deal(seed=seed, dealer=dealer)
    gs.training_mode = training_mode
    soloist = next_player(dealer)
    gs.soloist = soloist

    # Always pick up the talon so soloist sees 12 cards
    pickup_talon(gs, soloist, talon)

    if training_mode == "betli":
        trump = None
        betli = True
        set_contract(gs, soloist, trump=None, betli=True)
    else:
        betli = False
        suits_in_hand = list(set(c.suit for c in gs.hands[soloist]))
        trump = rng.choice(suits_in_hand)

        if training_mode == "40-100":
            # Ensure soloist holds trump K+Q
            trump_k = Card(trump, Rank.KING)
            trump_q = Card(trump, Rank.QUEEN)
            hand = gs.hands[soloist]
            for needed in (trump_k, trump_q):
                if needed not in hand:
                    # Find it in another hand or talon and swap
                    all_cards = []
                    for p in range(NUM_PLAYERS):
                        if p != soloist:
                            all_cards.extend((p, c) for c in gs.hands[p])
                    for owner, card in all_cards:
                        if card == needed:
                            swappable = [c for c in hand
                                         if c != trump_k and c != trump_q]
                            give = rng.choice(swappable)
                            hand.remove(give)
                            hand.append(needed)
                            gs.hands[owner].remove(needed)
                            gs.hands[owner].append(give)
                            break

        set_contract(gs, soloist, trump=trump)

    # ── Neural discard: evaluate up to 20 best discard pairs ──
    hand = gs.hands[soloist]
    all_pairs = list(combinations(range(len(hand)), 2))

    # Heuristic pruning for adu (keep all for betli — different strategy)
    max_pairs = 20
    if not betli and len(all_pairs) > max_pairs:
        def _discard_score(pair):
            score = 0.0
            for idx in pair:
                c = hand[idx]
                if c.suit == trump:
                    score += 100
                score -= c.points()
                if training_mode == "40-100":
                    if c == Card(trump, Rank.KING) or c == Card(trump, Rank.QUEEN):
                        score += 1000
                if training_mode == "ulti":
                    if trump and c == Card(trump, Rank.SEVEN):
                        score += 1000
            return score
        all_pairs.sort(key=_discard_score)
        all_pairs = all_pairs[:max_pairs]

    # Build contract components for encoding
    if betli:
        comps = frozenset({"betli"})
    else:
        comps = frozenset({"parti"})
        if training_mode == "ulti":
            comps = frozenset({"parti", "ulti"})
        elif training_mode == "40-100":
            comps = frozenset({"parti", "40", "100"})

    is_red = (trump is not None and trump == Suit.HEARTS)
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())

    # Batch-evaluate all discard candidates
    import copy
    import numpy as np

    feats_list: list[np.ndarray] = []
    valid_pairs: list[tuple[int, int]] = []

    for i, j in all_pairs:
        d0, d1 = hand[i], hand[j]

        # Simulate the discard
        gs_copy = copy.deepcopy(gs)
        discard_talon(gs_copy, [d0, d1])

        if training_mode == "ulti":
            gs_copy.has_ulti = True

        marriage_restrict = "40" if training_mode == "40-100" else None
        declare_all_marriages(gs_copy, soloist_marriage_restrict=marriage_restrict)

        constraints = build_auction_constraints(gs_copy, comps)

        node = UltiNode(
            gs=gs_copy,
            known_voids=empty_voids,
            bid_rank=1,
            is_red=is_red,
            contract_components=comps,
            dealer=dealer,
            must_have=constraints,
        )
        feats = game.encode_state(node, soloist)
        feats_list.append(feats)
        valid_pairs.append((i, j))

    if feats_list:
        batch = np.stack(feats_list)
        values = wrapper.batch_value(batch)
        best_idx = int(np.argmax(values))
        bi, bj = valid_pairs[best_idx]
        best_discards = [hand[bi], hand[bj]]
    else:
        best_discards = hand[:2]

    # Apply the chosen discard to the ORIGINAL gs
    discard_talon(gs, best_discards)

    if training_mode == "ulti":
        gs.has_ulti = True

    marriage_restrict = "40" if training_mode == "40-100" else None
    declare_all_marriages(gs, soloist_marriage_restrict=marriage_restrict)
    constraints = build_auction_constraints(gs, comps)

    return UltiNode(
        gs=gs,
        known_voids=empty_voids,
        bid_rank=1,
        is_red=is_red,
        contract_components=comps,
        dealer=dealer,
        must_have=constraints,
    )


# ---------------------------------------------------------------------------
#  Self-play: one full game -> training samples
# ---------------------------------------------------------------------------


def _play_one_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    sol_mcts_config: MCTSConfig,
    def_mcts_config: MCTSConfig,
    seed: int,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
    solver_temp: float = 1.0,
    training_mode: str = "simple",
    initial_state: UltiNode | None = None,
    solver_teacher: bool = False,
    kontra_enabled: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Play one hybrid self-play game.

    Uses HybridPlayer for all seats: neural MCTS for opening tricks,
    PIMC + exact alpha-beta for the endgame.  The same network plays
    both soloist and defenders (pure self-play).

    When ``solver_teacher`` is True, MCTS positions are labelled with
    the solver's position value at the handoff point (continuous,
    from the "all-seeing teacher").  When False (default), all
    positions use the binary game outcome from ``simple_outcome``.

    If *initial_state* is provided (e.g. from deal enrichment),
    it is used directly instead of dealing a new game.

    When ``kontra_enabled`` is True, defenders evaluate kontra after
    trick 1 using the value head (same ``v > 0`` rule as bidding
    training).
    """
    rng = random.Random(seed)

    if initial_state is not None:
        state = initial_state
    else:
        state = game.new_game(
            seed=seed,
            training_mode=training_mode,
            starting_leader=seed % 3,
        )
    soloist_idx = state.gs.soloist

    sol_hybrid = HybridPlayer(
        game, wrapper,
        mcts_config=sol_mcts_config,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )
    def_hybrid = HybridPlayer(
        game, wrapper,
        mcts_config=def_mcts_config,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )

    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]] = []
    handoff_solver_val: float | None = None
    kontra_done = False

    while not game.is_terminal(state):
        # ── Kontra decision after trick 1 ─────────────────────
        if kontra_enabled and not kontra_done and state.gs.trick_no == 1:
            kontra_done = True
            _decide_kontra_training(game, state, wrapper, training_mode)

        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        if player == soloist_idx:
            pi, action, sv = sol_hybrid.choose_action_with_policy(state, player, rng)
        else:
            pi, action, sv = def_hybrid.choose_action_with_policy(state, player, rng)

        used_solver = sv is not None
        if used_solver and handoff_solver_val is None:
            handoff_solver_val = sv

        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append((
            state_feats.copy(),
            mask.copy(),
            np.asarray(pi, dtype=np.float32).copy(),
            player,
            used_solver,
        ))

        state = game.apply(state, action)

    # ── Assign value targets ──────────────────────────────────────
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for state_feats, mask, pi, player, was_solver in trajectory:
        if solver_teacher and not was_solver and handoff_solver_val is not None:
            reward = solver_value_to_reward(
                handoff_solver_val, player, state.gs,
            )
        else:
            reward = simple_outcome(state, player)
        is_sol = (player == soloist_idx)
        samples.append((state_feats, mask, pi, reward, is_sol))

    return samples


# ---------------------------------------------------------------------------
#  Cosine LR schedule
# ---------------------------------------------------------------------------


def _cosine_lr(step: int, total_steps: int, lr_start: float, lr_end: float) -> float:
    if total_steps <= 1:
        return lr_start
    frac = step / total_steps
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * frac))


# ---------------------------------------------------------------------------
#  Main training entry point
# ---------------------------------------------------------------------------


def train_ulti_hybrid(
    cfg: UltiTrainConfig,
    *,
    initial_net: UltiNet | None = None,
    on_progress: Callable[[UltiTrainStats], None] | None = None,
) -> tuple[UltiNet, UltiTrainStats]:
    """Run hybrid self-play training for Ulti.

    Parameters
    ----------
    cfg : UltiTrainConfig
        All hyperparameters for the run.
    initial_net : UltiNet, optional
        Pass an existing net to resume training (checkpoint load).
        If None, a fresh network is created.
    on_progress : callable, optional
        Called after every step with an ``UltiTrainStats`` snapshot.

    Returns
    -------
    (UltiNet, UltiTrainStats)
    """
    # -- Resolve training device (auto-GPU for large nets) --
    device = auto_device(cfg.body_units, cfg.body_layers, force=cfg.device)
    use_gpu = device != "cpu"
    game = UltiGame()

    # -- Network --
    if initial_net is not None:
        net = initial_net
    else:
        net = UltiNet(
            input_dim=game.state_dim,
            body_units=cfg.body_units,
            body_layers=cfg.body_layers,
            action_dim=game.action_space_size,
        )
    net.to(device)
    # ONNX Runtime for self-play inference (CPU only, ~10x faster per call)
    wrapper = make_wrapper(net, device=device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg.lr_start, weight_decay=1e-4,
    )
    buffer = ReplayBuffer(capacity=cfg.buffer_size, seed=cfg.seed + 1)
    np_rng = np.random.default_rng(cfg.seed)

    # -- MCTS configs --
    sol_cfg = MCTSConfig(
        simulations=cfg.sol_sims,
        determinizations=cfg.sol_dets,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=1.0,
        leaf_batch_size=cfg.leaf_batch_size,
    )
    def_cfg = MCTSConfig(
        simulations=cfg.def_sims,
        determinizations=cfg.def_dets,
        c_puct=1.5,
        dirichlet_alpha=0.1,
        dirichlet_weight=0.15,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.5,
        leaf_batch_size=cfg.leaf_batch_size,
    )

    # -- Parallel self-play pool --
    executor = None
    batch_server = None
    net_kwargs = {
        "input_dim": game.state_dim,
        "body_units": cfg.body_units,
        "body_layers": cfg.body_layers,
        "action_dim": game.action_space_size,
    }
    if cfg.concurrent_games > 1:
        from trickster.batch_inference import BatchInferenceServer

        batch_server = BatchInferenceServer(wrapper, drain_ms=1.0)
        batch_server.start()
    elif cfg.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            initializer=_init_worker,
            initargs=(net_kwargs, "cpu"),
        )

    # -- Stats --
    stats = UltiTrainStats(total_steps=cfg.steps)
    t0 = time.perf_counter()

    _is_onnx = isinstance(wrapper, OnnxUltiWrapper)

    # Helpers for building self-play task tuples
    def _build_tasks(step: int):
        past_warmup = cfg.enrichment and step > cfg.enrich_warmup
        thresh = (
            _enrichment_threshold(step, cfg.steps)
            if past_warmup else -999.0
        )
        tasks = []
        for g in range(cfg.games_per_step):
            use_enrich = (
                past_warmup
                and thresh > -999.0
                and (g / cfg.games_per_step) >= cfg.enrich_random_frac
            )
            tasks.append((
                {k: v.cpu() for k, v in net.state_dict().items()},
                sol_cfg, def_cfg,
                cfg.seed + step * 1000 + g,
                cfg.endgame_tricks, cfg.pimc_dets, cfg.solver_temp,
                cfg.training_mode,
                thresh if use_enrich else -999.0,
                cfg.solver_teacher, cfg.kontra_enabled,
                cfg.neural_discard,
            ))
        return tasks, past_warmup, thresh

    def _collect_results(futures, buffer_, stats_):
        """Collect self-play results from completed futures."""
        sp_pts_sum_ = 0.0
        sp_total_ = 0
        n_samples = 0
        for f in futures:
            samples = f.result()
            sol_r = [r for _, _, _, r, is_sol in samples if is_sol]
            if sol_r:
                sp_total_ += 1
                sp_pts_sum_ += sol_r[0]
            for s, m, p, r, is_sol in samples:
                buffer_.push(s, m, p, r, is_soloist=is_sol)
            n_samples += len(samples)
            stats_.total_games += 1
        return sp_pts_sum_, sp_total_, n_samples

    # For double-buffering: futures from the previous iteration
    _pending_futures: list | None = None

    try:
        for step in range(1, cfg.steps + 1):
            # ---- 0. Cosine LR schedule ----
            lr = _cosine_lr(step, cfg.steps, cfg.lr_start, cfg.lr_end)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ---- 1. Self-play ----
            step_samples = 0
            sp_pts_sum = 0.0
            sp_total = 0

            past_warmup = cfg.enrichment and step > cfg.enrich_warmup
            thresh = (
                _enrichment_threshold(step, cfg.steps)
                if past_warmup else -999.0
            )

            if executor is not None:
                # --- Double-buffered parallel self-play ---
                #
                # On step 1, no previous futures exist — submit and
                # block (the buffer needs data before SGD can start).
                # On step 2+, the previous iteration already submitted
                # futures while SGD ran; collect those results now
                # (they're likely already done), then submit *next*
                # iteration's work before running SGD.

                if _pending_futures is not None:
                    sp_pts_sum, sp_total, step_samples = _collect_results(
                        _pending_futures, buffer, stats,
                    )
                    _pending_futures = None
                else:
                    # First iteration — submit and block
                    tasks, _, _ = _build_tasks(step)
                    futures = [executor.submit(_play_game_in_worker, t) for t in tasks]
                    sp_pts_sum, sp_total, step_samples = _collect_results(
                        futures, buffer, stats,
                    )
            elif batch_server is not None:
                # --- Threaded self-play with GPU batching ---
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _play_game_threaded(g_seed: int):
                    use_enrich = (
                        past_warmup
                        and thresh > -999.0
                        and (g_seed % cfg.games_per_step) / cfg.games_per_step
                        >= cfg.enrich_random_frac
                    )
                    init_state = None
                    if cfg.neural_discard:
                        init_state = _new_game_with_neural_discard(
                            UltiGame(), batch_server, g_seed,
                            training_mode=cfg.training_mode,
                            dealer=g_seed % 3,
                        )
                    elif use_enrich:
                        init_state, _ = _value_enriched_new_game(
                            UltiGame(), batch_server, g_seed,
                            min_value=thresh,
                            training_mode=cfg.training_mode,
                        )
                    return _play_one_game(
                        UltiGame(), batch_server, sol_cfg, def_cfg, g_seed,
                        endgame_tricks=cfg.endgame_tricks,
                        pimc_dets=cfg.pimc_dets,
                        solver_temp=cfg.solver_temp,
                        training_mode=cfg.training_mode,
                        initial_state=init_state,
                        solver_teacher=cfg.solver_teacher,
                        kontra_enabled=cfg.kontra_enabled,
                    )

                with ThreadPoolExecutor(max_workers=cfg.concurrent_games) as tpool:
                    game_seeds = [
                        cfg.seed + step * 1000 + g
                        for g in range(cfg.games_per_step)
                    ]
                    futs = [tpool.submit(_play_game_threaded, s) for s in game_seeds]
                    for f in as_completed(futs):
                        samples = f.result()
                        sol_r = [r for _, _, _, r, is_sol in samples if is_sol]
                        if sol_r:
                            sp_total += 1
                            sp_pts_sum += sol_r[0]
                        for s, m, p, r, is_sol in samples:
                            buffer.push(s, m, p, r, is_soloist=is_sol)
                        step_samples += len(samples)
                        stats.total_games += 1
            else:
                # --- Sequential self-play (no workers) ---
                for g in range(cfg.games_per_step):
                    game_seed = cfg.seed + step * 1000 + g

                    use_enrich = (
                        past_warmup
                        and thresh > -999.0
                        and (g / cfg.games_per_step) >= cfg.enrich_random_frac
                    )

                    if cfg.neural_discard:
                        init_state = _new_game_with_neural_discard(
                            game, wrapper, game_seed,
                            training_mode=cfg.training_mode,
                            dealer=game_seed % 3,
                        )
                    elif use_enrich:
                        init_state, _ = _value_enriched_new_game(
                            game, wrapper, game_seed,
                            min_value=thresh,
                            training_mode=cfg.training_mode,
                        )
                    else:
                        init_state = None

                    samples = _play_one_game(
                        game, wrapper, sol_cfg, def_cfg, game_seed,
                        endgame_tricks=cfg.endgame_tricks,
                        pimc_dets=cfg.pimc_dets,
                        solver_temp=cfg.solver_temp,
                        training_mode=cfg.training_mode,
                        initial_state=init_state,
                        solver_teacher=cfg.solver_teacher,
                        kontra_enabled=cfg.kontra_enabled,
                    )

                    sol_r = [r for _, _, _, r, is_sol in samples if is_sol]
                    if sol_r:
                        sp_total += 1
                        sp_pts_sum += sol_r[0]
                    for s, m, p, r, is_sol in samples:
                        buffer.push(s, m, p, r, is_soloist=is_sol)
                    step_samples += len(samples)
                    stats.total_games += 1

            stats.total_samples += step_samples

            # ---- 2. Train (decoupled SGD steps) ----
            avg_vloss = 0.0
            avg_ploss = 0.0
            avg_pacc = 0.0
            avg_sol_vloss = 0.0
            avg_def_vloss = 0.0

            # Submit next iteration's self-play *before* SGD so workers
            # run in parallel with training (double-buffering).
            if executor is not None and step < cfg.steps:
                next_tasks, _, _ = _build_tasks(step + 1)
                _pending_futures = [
                    executor.submit(_play_game_in_worker, t) for t in next_tasks
                ]

            if len(buffer) >= cfg.batch_size:
                net.train()
                n_train = cfg.train_steps
                total_vloss = 0.0
                total_ploss = 0.0
                total_pacc = 0.0
                total_sol_vloss = 0.0
                total_def_vloss = 0.0
                sol_count = 0
                def_count = 0

                for _ in range(n_train):
                    states, masks, policies, rewards, is_sol, _on_pol = buffer.sample(
                        cfg.batch_size, np_rng,
                    )

                    s_t = torch.from_numpy(states).float().to(device)
                    m_t = torch.from_numpy(masks).bool().to(device)
                    pi_t = torch.from_numpy(policies).float().to(device)
                    z_t = torch.from_numpy(rewards).float().to(device)
                    is_sol_t = torch.from_numpy(is_sol).bool().to(device)

                    log_probs, values = net.forward_dual(s_t, m_t, is_sol_t)

                    value_loss = F.huber_loss(values, z_t, delta=1.0)
                    policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
                    loss = value_loss + policy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

                    total_vloss += value_loss.item()
                    total_ploss += policy_loss.item()

                    with torch.no_grad():
                        pred_top1 = log_probs.argmax(dim=-1)
                        target_top1 = pi_t.argmax(dim=-1)
                        total_pacc += (pred_top1 == target_top1).float().mean().item()

                        sol_mask = is_sol_t
                        def_mask = ~is_sol_t
                        if sol_mask.any():
                            sv = F.mse_loss(values[sol_mask], z_t[sol_mask]).item()
                            total_sol_vloss += sv
                            sol_count += 1
                        if def_mask.any():
                            dv = F.mse_loss(values[def_mask], z_t[def_mask]).item()
                            total_def_vloss += dv
                            def_count += 1

                stats.total_sgd_steps += n_train
                avg_vloss = total_vloss / n_train
                avg_ploss = total_ploss / n_train
                avg_pacc = total_pacc / n_train
                avg_sol_vloss = total_sol_vloss / max(1, sol_count)
                avg_def_vloss = total_def_vloss / max(1, def_count)

            # Re-export ONNX sessions / sync batch server weights
            if _is_onnx:
                wrapper.sync_weights(net)
            elif batch_server is not None:
                batch_server.sync_weights(net)

            # ---- 3. Update stats ----
            stats.step = step
            stats.vloss = avg_vloss
            stats.ploss = avg_ploss
            stats.pacc = avg_pacc
            stats.sol_vloss = avg_sol_vloss
            stats.def_vloss = avg_def_vloss
            stats.lr = lr
            stats.sp_sol_pts_sum = sp_pts_sum
            stats.sp_sol_total = sp_total
            stats.train_time_s = time.perf_counter() - t0

            stats.history_vloss.append(avg_vloss)
            stats.history_ploss.append(avg_ploss)
            stats.history_pacc.append(avg_pacc)

            # ---- 4. Callback ----
            if on_progress is not None:
                on_progress(stats)

    finally:
        # Drain any pending futures before shutting down
        if _pending_futures is not None:
            for f in _pending_futures:
                f.cancel()
        if executor is not None:
            executor.shutdown(wait=False)
        if batch_server is not None:
            batch_server.stop()

    stats.train_time_s = time.perf_counter() - t0
    return net, stats
