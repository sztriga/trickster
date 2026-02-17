"""Multi-contract training loop with value-head-driven bidding.

Instead of training one contract with random deals, this loop:
  1. Deals a random 12-card hand
  2. Uses all models' value heads to pick the best contract+trump+discard
  3. Plays the chosen contract with that model
  4. Adds samples to that contract's replay buffer

Each contract model improves on hands that were *actually bid*, not
random garbage.  Better play → better value heads → smarter bidding →
more realistic training distribution → better play.

Usage:
    from trickster.training.bidding_loop import BiddingTrainConfig, train_with_bidding

    cfg = BiddingTrainConfig(steps=500, games_per_step=8, contracts={...})
    results = train_with_bidding(cfg)
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from trickster.bidding.evaluator import (
    ContractEval,
    evaluate_all_contracts,
    pick_best_bid,
)
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import Card
from trickster.games.ulti.game import (
    deal,
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
    Suit,
)
from trickster.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig
from trickster.model import OnnxUltiWrapper, UltiNet, UltiNetWrapper, make_wrapper
from trickster.train_utils import ReplayBuffer, simple_outcome, _GAME_PTS_MAX

from .model_io import auto_device, estimate_params
from .ulti_hybrid import _cosine_lr


# ---------------------------------------------------------------------------
#  Display key helpers
# ---------------------------------------------------------------------------

def _display_key(contract_key: str, is_piros: bool) -> str:
    """Create a display key that distinguishes red from non-red."""
    return f"p.{contract_key}" if is_piros else contract_key


def _model_key(display_key: str) -> str:
    """Strip the piros prefix to get the model/buffer key."""
    return display_key.removeprefix("p.")


# Ordered display keys (ascending bid rank).
# p.parti=2, 40-100=3, ulti=4, betli=5, p.40-100=8, p.ulti=10, p.betli=11
DISPLAY_ORDER: list[str] = [
    "p.parti", "40-100", "ulti", "betli",
    "p.40-100", "p.ulti", "p.betli",
]


# ---------------------------------------------------------------------------
#  Per-contract config
# ---------------------------------------------------------------------------


@dataclass
class ContractTrainSlot:
    """Per-contract training state.

    Holds the net, wrapper, replay buffer, and optimizer for one contract.
    """

    key: str                  # "parti", "ulti", "40-100", "betli"
    net: UltiNet
    wrapper: UltiNetWrapper
    optimizer: torch.optim.Adam
    buffer: ReplayBuffer

    # Cumulative stats (model-level: red+non-red combined)
    games: int = 0
    samples: int = 0
    sgd_steps: int = 0

    # Current step losses (updated each SGD round)
    vloss: float = 0.0
    ploss: float = 0.0

    # Loss history for trend detection
    vloss_history: list = field(default_factory=list)
    ploss_history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------


@dataclass
class BiddingTrainConfig:
    """Configuration for multi-contract training with bidding.

    All contracts share the same training schedule (steps, LR, etc.)
    but have separate models and replay buffers.
    """

    # -- Budget --
    steps: int = 500
    games_per_step: int = 8

    # -- SGD --
    train_steps: int = 50
    batch_size: int = 64
    buffer_size: int = 50_000
    lr_start: float = 1e-3
    lr_end: float = 2e-4

    # -- MCTS --
    sol_sims: int = 40
    sol_dets: int = 2
    def_sims: int = 20
    def_dets: int = 2
    leaf_batch_size: int = 8

    # -- Solver --
    endgame_tricks: int = 6
    pimc_dets: int = 20
    solver_temp: float = 0.5

    # -- Network --
    body_units: int = 256
    body_layers: int = 4

    # -- Bidding --
    max_discards: int = 15        # discard pairs per contract eval
    min_bid_pts: float = -2.0     # minimum expected pts/defender to bid (matches pass penalty)
    pass_penalty: float = 2.0     # pts per defender when everyone passes
    exploration_frac: float = 0.2  # fraction of games with random contract

    # -- Kontra --
    kontra_enabled: bool = True    # enable kontra/rekontra decisions after trick 1

    # -- Opponent pool --
    opponent_pool: list[str] = field(default_factory=list)  # e.g. ["scout", "knight"]
    pool_frac: float = 0.5         # fraction of games played vs pool opponents

    # -- Contracts to train (model keys) --
    # Parti is included because Piros Parti is the first playable game.
    # Plain (non-red) Parti cannot be played; the evaluator only
    # evaluates it with Hearts trump (piros_only flag in registry).
    contract_keys: list[str] = field(
        default_factory=lambda: ["parti", "ulti", "40-100", "betli"],
    )

    # -- Parallelism --
    num_workers: int = 1

    # -- General --
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
#  Stats
# ---------------------------------------------------------------------------


@dataclass
class BiddingTrainStats:
    """Per-step statistics for the bidding training loop."""

    step: int = 0
    total_steps: int = 0
    total_games: int = 0
    total_passes: int = 0         # deals where everyone passed
    train_time_s: float = 0.0
    lr: float = 0.0

    # Per-step
    step_passes: int = 0

    # Per-display-key step stats (e.g. "p.parti", "ulti", "p.ulti", …)
    step_games: dict[str, int] = field(default_factory=dict)
    step_pts: dict[str, float] = field(default_factory=dict)
    step_wins: dict[str, int] = field(default_factory=dict)

    # Per-model-key current losses (e.g. "parti", "ulti", …)
    model_vloss: dict[str, float] = field(default_factory=dict)
    model_ploss: dict[str, float] = field(default_factory=dict)

    # Cumulative per-display-key
    cumulative_games: dict[str, int] = field(default_factory=dict)
    cumulative_pts: dict[str, float] = field(default_factory=dict)
    cumulative_wins: dict[str, int] = field(default_factory=dict)

    # Cumulative per-model-key
    cumulative_samples: dict[str, int] = field(default_factory=dict)

    # Slots reference (for the callback to access histories)
    _slots: dict | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
#  Multiprocessing helpers (module-level for pickling)
# ---------------------------------------------------------------------------

_BW_GAME: UltiGame | None = None
_BW_NETS: dict[str, UltiNet] = {}
_BW_WRAPPERS: dict[str, UltiNetWrapper | OnnxUltiWrapper] = {}
_BW_POOL_WRAPPERS: list[dict[str, UltiNetWrapper | OnnxUltiWrapper]] = []


def _init_bidding_worker(
    net_kwargs: dict, contract_keys: list[str], device: str,
    pool_sources: list[str] | None = None,
) -> None:
    """Called once per worker process to create game + per-contract networks."""
    global _BW_GAME, _BW_NETS, _BW_WRAPPERS, _BW_POOL_WRAPPERS
    _BW_GAME = UltiGame()
    _BW_NETS = {}
    _BW_WRAPPERS = {}
    for key in contract_keys:
        net = UltiNet(**net_kwargs)
        _BW_NETS[key] = net
        _BW_WRAPPERS[key] = make_wrapper(net, device=device)

    _BW_POOL_WRAPPERS = []
    if pool_sources:
        from trickster.training.model_io import load_wrappers
        for source in pool_sources:
            pw = load_wrappers(source, device=device)
            if pw:
                _BW_POOL_WRAPPERS.append(pw)


def _play_bidding_game_in_worker(
    args: tuple,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]]:
    """Worker entry-point for parallel bidding self-play."""
    (all_state_dicts, sol_mcts_cfg, def_mcts_cfg, seed, cfg_dict, exploration) = args

    # Load latest weights into worker nets
    for key, sd in all_state_dicts.items():
        _BW_NETS[key].load_state_dict(sd)
        w = _BW_WRAPPERS.get(key)
        if isinstance(w, OnnxUltiWrapper):
            w.sync_weights(_BW_NETS[key])

    # Reconstruct a lightweight cfg from dict
    cfg = BiddingTrainConfig(**cfg_dict)

    # Pool opponent selection
    opp_w = None
    if _BW_POOL_WRAPPERS and random.Random(seed).random() < cfg.pool_frac:
        opp_w = random.Random(seed).choice(_BW_POOL_WRAPPERS)

    return _play_one_bidding_game(
        _BW_GAME, _BW_WRAPPERS, sol_mcts_cfg, def_mcts_cfg, seed, cfg,
        exploration=exploration,
        opp_wrappers=opp_w,
    )


# ---------------------------------------------------------------------------
#  Kontra helpers
# ---------------------------------------------------------------------------


def _kontrable_units(contract_key: str) -> list[str]:
    """Kontrable unit labels for a given contract model key."""
    _MAP: dict[str, list[str]] = {
        "parti": ["parti"],
        "ulti": ["parti", "ulti"],
        "40-100": ["40-100"],
        "betli": ["betli"],
    }
    return _MAP.get(contract_key, ["parti"])


def _decide_kontra(
    game: UltiGame,
    state: UltiNode,
    wrapper: UltiNetWrapper,
    contract_key: str,
) -> None:
    """Apply kontra/rekontra decisions after trick 1 using the value head.

    Modifies ``state.component_kontras`` in place.

    The decision is purely value-head-driven with **no threshold**:
      - Defender kontras when ``value > 0`` (expects to gain points).
      - Soloist rekontras when ``value > 0`` (still expects to win).

    This is the rational choice: doubling the stakes on a positive
    expected value always increases expected gain.  Early in training,
    noisy value heads produce ~50% kontra rates — natural exploration.
    As the value head improves, kontras converge to the correct
    frequency, creating a self-correcting equilibrium.

    For adu (trump) games: kontras are shared between defenders.
    If either defender expects to win, both kontra.
    """
    gs = state.gs
    soloist = gs.soloist
    units = _kontrable_units(contract_key)

    if not units:
        return

    # Encode state for each defender and evaluate.
    # predict_value reads is_soloist from the encoded features.
    defenders = [i for i in range(3) if i != soloist]
    def_values = []
    for d in defenders:
        feats = game.encode_state(state, d)
        v = wrapper.predict_value(feats)
        def_values.append(v)

    # Kontra when the most confident defender expects to gain.
    max_def_v = max(def_values)
    kontrad = max_def_v > 0.0

    if kontrad:
        for u in units:
            state.component_kontras[u] = 1

    # Rekontra: soloist still expects to win despite the kontra.
    if kontrad:
        feats = game.encode_state(state, soloist)
        sol_v = wrapper.predict_value(feats)
        if sol_v > 0.0:
            for u in units:
                if state.component_kontras.get(u, 0) == 1:
                    state.component_kontras[u] = 2


# ---------------------------------------------------------------------------
#  Setup a deal from a bid result
# ---------------------------------------------------------------------------


def _setup_bid_game(
    game: UltiGame,
    gs,
    soloist: int,
    dealer: int,
    bid: ContractEval,
) -> UltiNode:
    """Apply the bid result to a game state, returning a ready-to-play node.

    The gs must have the soloist's hand at 12 cards.
    """
    from trickster.bidding.evaluator import _make_eval_state

    cdef = CONTRACT_DEFS[bid.contract_key]
    return _make_eval_state(
        gs, soloist, bid.trump, bid.best_discard.discard,
        cdef, bid.is_piros, dealer,
    )


# ---------------------------------------------------------------------------
#  Play one game with bidding
# ---------------------------------------------------------------------------


def _play_one_bidding_game(
    game: UltiGame,
    wrappers: dict[str, UltiNetWrapper],
    sol_cfg: MCTSConfig,
    def_cfg: MCTSConfig,
    seed: int,
    cfg: BiddingTrainConfig,
    exploration: bool = False,
    opp_wrappers: dict[str, UltiNetWrapper] | None = None,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]]:
    """Play one game with value-head bidding.

    Parameters
    ----------
    opp_wrappers : optional
        When provided, defenders use these wrappers for play and kontra
        decisions instead of *wrappers*.  Only soloist samples are
        collected (defender policy is off-policy for the training model).

    Returns (display_key, samples) where display_key encodes both
    the contract and whether it's piros (e.g. "p.parti", "ulti").
    Returns ("__pass__", []) when everyone passes.
    """
    rng = random.Random(seed)
    dealer = seed % 3
    soloist = next_player(dealer)
    pool_game = opp_wrappers is not None

    # 1. Deal 12 cards
    gs, talon = deal(seed=seed, dealer=dealer)
    pickup_talon(gs, soloist, talon)

    # 2. Evaluate all contracts (always — exploration picks randomly)
    evals = evaluate_all_contracts(
        gs, soloist, dealer,
        wrappers=wrappers,
        max_discards=cfg.max_discards,
    )

    if not evals:
        return "__pass__", []

    # 3. Pick a bid
    if exploration:
        # Random exploration: pick any feasible contract
        bid = rng.choice(evals)
    else:
        bid = pick_best_bid(evals, min_stakes_pts=cfg.min_bid_pts)
        if bid is None:
            # Everyone passes — soloist pays penalty, no game played.
            return "__pass__", []

    dkey = _display_key(bid.contract_key, bid.is_piros)

    # 4. Setup game state from the bid
    state = _setup_bid_game(game, gs, soloist, dealer, bid)

    # 5. Get wrappers for play
    sol_wrapper = wrappers.get(bid.contract_key)
    if sol_wrapper is None:
        sol_wrapper = next(iter(wrappers.values()))

    if pool_game:
        def_wrapper = opp_wrappers.get(bid.contract_key)
        if def_wrapper is None:
            def_wrapper = sol_wrapper  # fallback to training model
    else:
        def_wrapper = sol_wrapper

    soloist_idx = state.gs.soloist

    # 6. Play the game
    sol_hybrid = HybridPlayer(
        game, sol_wrapper,
        mcts_config=sol_cfg,
        endgame_tricks=cfg.endgame_tricks,
        pimc_determinizations=cfg.pimc_dets,
        solver_temperature=cfg.solver_temp,
    )
    def_hybrid = HybridPlayer(
        game, def_wrapper,
        mcts_config=def_cfg,
        endgame_tricks=cfg.endgame_tricks,
        pimc_determinizations=cfg.pimc_dets,
        solver_temperature=cfg.solver_temp,
    )

    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]] = []
    kontra_done = False

    while not game.is_terminal(state):
        # ── Kontra decision after trick 1 ────────────────────────
        if (
            cfg.kontra_enabled
            and not kontra_done
            and state.gs.trick_no == 1
        ):
            kontra_done = True
            # Defenders decide kontra using their own model
            _decide_kontra(game, state, def_wrapper, bid.contract_key)

        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        if player == soloist_idx:
            pi, action, sv = sol_hybrid.choose_action_with_policy(
                state, player, rng,
            )
        else:
            pi, action, sv = def_hybrid.choose_action_with_policy(
                state, player, rng,
            )

        # Always collect trajectory.  In pool games, defender positions
        # are marked off-policy: their value target is still valid (the
        # game outcome is objective) but the policy target comes from a
        # different model.  The SGD loop will skip policy loss for these.
        is_on_policy = not pool_game or player == soloist_idx
        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append((
            state_feats.copy(),
            mask.copy(),
            np.asarray(pi, dtype=np.float32).copy(),
            player,
            player == soloist_idx,
            is_on_policy,
        ))

        state = game.apply(state, action)

    # 7. Label with outcome
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool]] = []
    for state_feats, mask, pi, player, is_sol, on_pol in trajectory:
        reward = simple_outcome(state, player)
        samples.append((state_feats, mask, pi, reward, is_sol, on_pol))

    return dkey, samples


# ---------------------------------------------------------------------------
#  Main training entry point
# ---------------------------------------------------------------------------


def train_with_bidding(
    cfg: BiddingTrainConfig,
    *,
    initial_nets: dict[str, UltiNet] | None = None,
    on_progress: Callable[[BiddingTrainStats], None] | None = None,
) -> tuple[dict[str, ContractTrainSlot], BiddingTrainStats]:
    """Run multi-contract training with value-head bidding.

    Parameters
    ----------
    cfg : BiddingTrainConfig
    initial_nets : contract_key → pre-trained UltiNet (optional)
    on_progress : called after each step

    Returns
    -------
    (slots, final_stats)
    """
    # -- Resolve training device (auto-GPU for large nets) --
    device = auto_device(cfg.body_units, cfg.body_layers, force=cfg.device)
    game = UltiGame()

    # -- Create per-contract slots (model-level) --
    slots: dict[str, ContractTrainSlot] = {}
    for key in cfg.contract_keys:
        if initial_nets and key in initial_nets:
            net = initial_nets[key]
        else:
            net = UltiNet(
                input_dim=game.state_dim,
                body_units=cfg.body_units,
                body_layers=cfg.body_layers,
                action_dim=game.action_space_size,
            )
        net.to(device)
        wrapper = make_wrapper(net, device=device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=cfg.lr_start, weight_decay=1e-4,
        )
        buf = ReplayBuffer(capacity=cfg.buffer_size, seed=cfg.seed + hash(key) % 10000)

        slots[key] = ContractTrainSlot(
            key=key, net=net, wrapper=wrapper,
            optimizer=optimizer, buffer=buf,
        )

    # Collect wrappers for bidding (main process, sequential fallback).
    # Sequential self-play uses CPU wrappers; we swap device around SGD.
    use_gpu = device != "cpu"
    wrappers = {key: slot.wrapper for key, slot in slots.items()}

    # -- Opponent pool (frozen, pre-trained models) --
    pool_wrappers_list: list[dict[str, UltiNetWrapper]] = []
    if cfg.opponent_pool:
        from trickster.training.model_io import load_wrappers
        for source in cfg.opponent_pool:
            pw = load_wrappers(source, device="cpu")
            if pw:
                pool_wrappers_list.append(pw)
                logging.info("Pool opponent loaded: %s (%d contracts)", source, len(pw))
            else:
                logging.warning("Pool opponent '%s' — no models found, skipping", source)
        if pool_wrappers_list:
            logging.info(
                "Opponent pool ready: %d sources, %.0f%% of games",
                len(pool_wrappers_list), cfg.pool_frac * 100,
            )

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

    np_rng = np.random.default_rng(cfg.seed)
    pool_rng = random.Random(cfg.seed + 777)   # for pool opponent selection
    t0 = time.perf_counter()

    stats = BiddingTrainStats(total_steps=cfg.steps)

    # Cumulative display-key tracking
    cum_dk_games: dict[str, int] = {dk: 0 for dk in DISPLAY_ORDER}
    cum_dk_pts: dict[str, float] = {dk: 0.0 for dk in DISPLAY_ORDER}
    cum_dk_wins: dict[str, int] = {dk: 0 for dk in DISPLAY_ORDER}

    # -- Parallel pool --
    executor = None
    net_kwargs = {
        "input_dim": game.state_dim,
        "body_units": cfg.body_units,
        "body_layers": cfg.body_layers,
        "action_dim": game.action_space_size,
    }
    # Serialisable subset of cfg for workers (dataclass fields only)
    cfg_dict = {
        "steps": cfg.steps,
        "games_per_step": cfg.games_per_step,
        "train_steps": cfg.train_steps,
        "batch_size": cfg.batch_size,
        "buffer_size": cfg.buffer_size,
        "lr_start": cfg.lr_start,
        "lr_end": cfg.lr_end,
        "sol_sims": cfg.sol_sims,
        "sol_dets": cfg.sol_dets,
        "def_sims": cfg.def_sims,
        "def_dets": cfg.def_dets,
        "leaf_batch_size": cfg.leaf_batch_size,
        "endgame_tricks": cfg.endgame_tricks,
        "pimc_dets": cfg.pimc_dets,
        "solver_temp": cfg.solver_temp,
        "body_units": cfg.body_units,
        "body_layers": cfg.body_layers,
        "max_discards": cfg.max_discards,
        "min_bid_pts": cfg.min_bid_pts,
        "pass_penalty": cfg.pass_penalty,
        "exploration_frac": cfg.exploration_frac,
        "kontra_enabled": cfg.kontra_enabled,
        "contract_keys": cfg.contract_keys,
        "num_workers": 1,
        "seed": cfg.seed,
        "device": "cpu",
        "pool_frac": cfg.pool_frac,
    }
    if cfg.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            initializer=_init_bidding_worker,
            initargs=(net_kwargs, cfg.contract_keys, "cpu",
                      cfg.opponent_pool if cfg.opponent_pool else None),
        )

    try:
        for step in range(1, cfg.steps + 1):
            # -- LR schedule --
            lr = _cosine_lr(step, cfg.steps, cfg.lr_start, cfg.lr_end)
            for slot in slots.values():
                for pg in slot.optimizer.param_groups:
                    pg["lr"] = lr

            # -- Self-play --
            step_dk_games: dict[str, int] = {dk: 0 for dk in DISPLAY_ORDER}
            step_dk_pts: dict[str, float] = {dk: 0.0 for dk in DISPLAY_ORDER}
            step_dk_wins: dict[str, int] = {dk: 0 for dk in DISPLAY_ORDER}
            step_passes = 0

            def _collect_result(
                dkey: str,
                samples: list,
            ) -> None:
                """Route one game's results to the correct slot."""
                nonlocal step_passes

                if dkey == "__pass__":
                    step_passes += 1
                    stats.total_passes += 1
                    return

                # Route samples to the model's buffer
                mkey = _model_key(dkey)
                slot = slots[mkey]
                for s, m, p, r, is_sol, on_pol in samples:
                    slot.buffer.push(s, m, p, r, is_soloist=is_sol, on_policy=on_pol)
                slot.samples += len(samples)
                slot.games += 1

                # Track by display key
                step_dk_games[dkey] = step_dk_games.get(dkey, 0) + 1
                cum_dk_games[dkey] = cum_dk_games.get(dkey, 0) + 1

                sol_r = [r for _, _, _, r, is_sol, _ in samples if is_sol]
                if sol_r:
                    step_dk_pts[dkey] = step_dk_pts.get(dkey, 0.0) + sol_r[0]
                    cum_dk_pts[dkey] = cum_dk_pts.get(dkey, 0.0) + sol_r[0]
                    sol_game_pts = sol_r[0] * _GAME_PTS_MAX / 2
                    if sol_game_pts > 0:
                        step_dk_wins[dkey] = step_dk_wins.get(dkey, 0) + 1
                        cum_dk_wins[dkey] = cum_dk_wins.get(dkey, 0) + 1

                stats.total_games += 1

            if executor is not None:
                # --- Parallel self-play ---
                all_state_dicts = {
                    key: {k: v.cpu() for k, v in slot.net.state_dict().items()}
                    for key, slot in slots.items()
                }
                tasks = []
                for g in range(cfg.games_per_step):
                    game_seed = cfg.seed + step * 1000 + g
                    exploration = (g / cfg.games_per_step) < cfg.exploration_frac
                    tasks.append((
                        all_state_dicts,
                        sol_cfg,
                        def_cfg,
                        game_seed,
                        cfg_dict,
                        exploration,
                    ))
                for dkey, samples in executor.map(
                    _play_bidding_game_in_worker, tasks,
                ):
                    _collect_result(dkey, samples)
            else:
                # --- Sequential self-play ---
                for g in range(cfg.games_per_step):
                    game_seed = cfg.seed + step * 1000 + g
                    exploration = (g / cfg.games_per_step) < cfg.exploration_frac

                    # Decide whether this game uses a pool opponent
                    opp_w = None
                    if pool_wrappers_list and pool_rng.random() < cfg.pool_frac:
                        opp_w = pool_rng.choice(pool_wrappers_list)

                    dkey, samples = _play_one_bidding_game(
                        game, wrappers, sol_cfg, def_cfg, game_seed, cfg,
                        exploration=exploration,
                        opp_wrappers=opp_w,
                    )
                    _collect_result(dkey, samples)

            # -- SGD for each contract that has enough data --
            step_model_vloss: dict[str, float] = {}
            step_model_ploss: dict[str, float] = {}

            for key, slot in slots.items():
                if len(slot.buffer) < min(cfg.batch_size, 16):
                    continue
                effective_batch = min(cfg.batch_size, len(slot.buffer))

                slot.net.train()
                n_train = cfg.train_steps
                acc_vloss = 0.0
                acc_ploss = 0.0

                for _ in range(n_train):
                    states, masks, policies, rewards, is_sol, on_pol = slot.buffer.sample(
                        effective_batch, np_rng,
                    )

                    s_t = torch.from_numpy(states).float().to(device)
                    m_t = torch.from_numpy(masks).bool().to(device)
                    pi_t = torch.from_numpy(policies).float().to(device)
                    z_t = torch.from_numpy(rewards).float().to(device)
                    is_sol_t = torch.from_numpy(is_sol).bool().to(device)
                    on_pol_t = torch.from_numpy(on_pol).float().to(device)

                    log_probs, values = slot.net.forward_dual(s_t, m_t, is_sol_t)

                    value_loss = F.huber_loss(values, z_t, delta=1.0)
                    # Off-policy samples (pool game defenders) contribute to
                    # value loss but not policy loss — the policy target came
                    # from a different model so it's not a valid gradient signal.
                    per_sample_ploss = -(pi_t * log_probs).sum(dim=-1)
                    policy_loss = (per_sample_ploss * on_pol_t).sum() / on_pol_t.sum().clamp(min=1)
                    loss = value_loss + policy_loss

                    slot.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(slot.net.parameters(), 5.0)
                    slot.optimizer.step()

                    acc_vloss += value_loss.item()
                    acc_ploss += policy_loss.item()

                slot.sgd_steps += n_train
                avg_v = acc_vloss / n_train
                avg_p = acc_ploss / n_train
                slot.vloss = avg_v
                slot.ploss = avg_p
                slot.vloss_history.append(avg_v)
                slot.ploss_history.append(avg_p)
                step_model_vloss[key] = avg_v
                step_model_ploss[key] = avg_p

                slot.net.eval()
                if isinstance(slot.wrapper, OnnxUltiWrapper):
                    slot.wrapper.sync_weights(slot.net)

            # -- Update stats --
            stats.step = step
            stats.lr = lr
            stats.train_time_s = time.perf_counter() - t0
            stats.step_passes = step_passes
            stats.step_games = dict(step_dk_games)
            stats.step_pts = dict(step_dk_pts)
            stats.step_wins = dict(step_dk_wins)
            stats.model_vloss = step_model_vloss
            stats.model_ploss = step_model_ploss
            stats.cumulative_games = dict(cum_dk_games)
            stats.cumulative_pts = dict(cum_dk_pts)
            stats.cumulative_wins = dict(cum_dk_wins)
            stats.cumulative_samples = {k: slots[k].samples for k in cfg.contract_keys}
            stats._slots = slots

            if on_progress:
                on_progress(stats)

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    return slots, stats
