"""Self-play engine for hybrid MCTS + alpha-beta training.

Contains the core game-playing logic used by ``train_ulti_hybrid``:
  - ``_play_one_game``: play a single self-play game and collect samples
  - ``_decide_kontra_training``: kontra/rekontra after trick 1
  - Worker helpers for ``ProcessPoolExecutor`` parallel self-play
"""
from __future__ import annotations

import random

import numpy as np

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig
from trickster.model import OnnxUltiWrapper, UltiNet, UltiNetWrapper, make_wrapper
from trickster.train_utils import simple_outcome, solver_value_to_reward


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

    def_values = []
    for d in defenders:
        feats = game.encode_state(state, d)
        v = wrapper.predict_value(feats)
        def_values.append(v)

    kontrad = max(def_values) > 0.0
    if kontrad:
        for u in units:
            state.component_kontras[u] = 1

    if kontrad:
        feats = game.encode_state(state, soloist)
        sol_v = wrapper.predict_value(feats)
        if sol_v > 0.0:
            for u in units:
                if state.component_kontras.get(u, 0) == 1:
                    state.component_kontras[u] = 2


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
    from trickster.training.deal_enrichment import (
        _new_game_with_neural_discard,
        _value_enriched_new_game,
    )

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
