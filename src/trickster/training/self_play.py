"""Self-play engine for hybrid MCTS + alpha-beta training.

Contains the core game-playing logic used by ``train_ulti_hybrid``:
  - ``_play_one_game``: play a single self-play game and collect samples
  - ``_new_game_with_neural_discard``: value-head-driven talon discard
  - ``_decide_kontra_training``: kontra/rekontra after trick 1
  - Worker helpers for ``ProcessPoolExecutor`` parallel self-play
"""
from __future__ import annotations

import copy
import random
from itertools import combinations

import numpy as np

from trickster.games.ulti.adapter import UltiGame, UltiNode, build_auction_constraints
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
from trickster.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig
from trickster.model import OnnxUltiWrapper, UltiNet, UltiNetWrapper, make_wrapper
from trickster.bidding.constants import KONTRA_THRESHOLD, REKONTRA_THRESHOLD
from trickster.train_utils import simple_outcome, solver_value_to_reward


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
    rng = random.Random(seed)
    gs, talon = deal(seed=seed, dealer=dealer)
    gs.training_mode = training_mode
    soloist = next_player(dealer)
    gs.soloist = soloist

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
            trump_k = Card(trump, Rank.KING)
            trump_q = Card(trump, Rank.QUEEN)
            hand = gs.hands[soloist]
            for needed in (trump_k, trump_q):
                if needed not in hand:
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

    feats_list: list[np.ndarray] = []
    valid_pairs: list[tuple[int, int]] = []

    for i, j in all_pairs:
        d0, d1 = hand[i], hand[j]

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

    kontrad = max(def_values) > KONTRA_THRESHOLD
    if kontrad:
        for u in units:
            state.component_kontras[u] = 1

    if kontrad:
        feats = game.encode_state(state, soloist)
        sol_v = wrapper.predict_value(feats)
        if sol_v > REKONTRA_THRESHOLD:
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
    (state_dict, sol_mcts_cfg, def_mcts_cfg,
     seed, endgame_tricks, pimc_dets, solver_temp,
     training_mode, _enrich_thresh, solver_teacher,
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
