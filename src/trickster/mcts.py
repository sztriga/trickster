"""AlphaZero-style MCTS — game-agnostic, uses PUCT + value evaluation.

Works with any game that implements
:class:`~trickster.games.interface.GameInterface`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from trickster.games.interface import GameInterface
from trickster.models.alpha_net import SharedAlphaNet as AlphaNet  # type alias for compat


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Tuning knobs for the AlphaZero MCTS."""

    simulations: int = 100  # iterations per determinization
    determinizations: int = 6  # sampled worlds (imperfect info)
    c_puct: float = 1.5  # exploration constant
    dirichlet_alpha: float = 0.3  # root noise (0 = off)
    dirichlet_weight: float = 0.25  # fraction of noise mixed in
    use_value_head: bool = False  # True = AlphaZero (value head eval)
                                  # False = AlphaGo  (random rollout eval)
    use_policy_priors: bool = True  # False = uniform priors (for training)
    visit_temp: float = 1.0  # temperature for visit distribution (< 1 = sharper)


# ---------------------------------------------------------------------------
#  Tree node
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Node:
    player: int
    action: Any = None
    parent: Optional["_Node"] = None
    children: list["_Node"] = field(default_factory=list)
    prior: float = 0.0  # P(a) from policy head
    visits: int = 0
    value_sum: float = 0.0  # cumulative value from *this player's* perspective


# ---------------------------------------------------------------------------
#  PUCT selection — fully inlined for speed
# ---------------------------------------------------------------------------

_sqrt = math.sqrt  # avoid module-level lookup in hot loop


def _select(node: _Node, c: float) -> _Node:
    """Walk down the tree picking the child with highest PUCT score.

    All PUCT math is inlined to avoid per-child function call overhead.
    sqrt(parent_visits) is cached per level.
    """
    children = node.children
    while children:
        sqrt_pv = _sqrt(node.visits)
        best = children[0]
        bv = best.visits
        best_score = (best.value_sum / bv if bv > 0 else 0.0) + c * best.prior * sqrt_pv / (1.0 + bv)
        for i in range(1, len(children)):
            ch = children[i]
            v = ch.visits
            score = (ch.value_sum / v if v > 0 else 0.0) + c * ch.prior * sqrt_pv / (1.0 + v)
            if score > best_score:
                best_score = score
                best = ch
        node = best
        children = node.children
    return node


# ---------------------------------------------------------------------------
#  Random rollout (used when use_value_head=False)
# ---------------------------------------------------------------------------


def _random_rollout(
    state: Any,
    game: GameInterface,
    perspective: int,
    rng: random.Random,
) -> float:
    """Play random moves to terminal, return outcome for *perspective*."""
    _is_terminal = game.is_terminal
    _legal = game.legal_actions
    _apply = game.apply
    _choice = rng.choice
    while not _is_terminal(state):
        actions = _legal(state)
        # Skip close_talon in rollouts (random agent wouldn't know when)
        card_actions = [a for a in actions if a != "close_talon"]
        state = _apply(state, _choice(card_actions) if card_actions else _choice(actions))
    return game.outcome(state, perspective)


# ---------------------------------------------------------------------------
#  Single-determinization MCTS (lazy expansion)
# ---------------------------------------------------------------------------


def _run_mcts(
    state: Any,
    game: GameInterface,
    net: AlphaNet,
    perspective: int,
    config: MCTSConfig,
    rollout_rng: random.Random | None = None,
) -> dict[Any, float]:
    """Run MCTS on one determinized state.  Returns action -> visit count.

    Uses lazy expansion: child states are only computed when a child is
    first selected as a leaf, not when its parent is expanded.  This
    avoids ~80% of game.apply calls in the tree.
    """
    actions = game.legal_actions(state)
    if len(actions) <= 1:
        return {a: 1.0 for a in actions}

    # Evaluate root: get policy priors
    player = game.current_player(state)
    K = len(actions)
    if config.use_policy_priors:
        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        full_priors = net.predict_policy(state_feats, mask)
        priors = np.array([full_priors[game.action_to_index(a)] for a in actions])
        p_sum = priors.sum()
        if p_sum > 0:
            priors /= p_sum
        else:
            priors = np.full(K, 1.0 / K)
    else:
        priors = np.full(K, 1.0 / K)

    # Add Dirichlet noise at root for exploration
    if config.dirichlet_alpha > 0:
        noise = np.random.dirichlet([config.dirichlet_alpha] * K)
        w = config.dirichlet_weight
        priors = (1.0 - w) * priors + w * noise

    root = _Node(player=player)
    root.visits = 1
    for i, a in enumerate(actions):
        child = _Node(player=player, action=a, parent=root, prior=float(priors[i]))
        root.children.append(child)

    # State cache: node id -> game state (lazy — computed on first visit)
    node_states: dict[int, Any] = {id(root): state}
    # Pre-seed root children states are NOT computed here (lazy)

    c_puct = config.c_puct
    use_policy = config.use_policy_priors
    use_value = config.use_value_head
    _game_apply = game.apply
    _game_is_terminal = game.is_terminal
    _game_legal = game.legal_actions
    _game_current = game.current_player
    _game_outcome = game.outcome

    for _ in range(config.simulations):
        # 1. Select
        leaf = _select(root, c_puct)

        # 2. Get or compute leaf state (lazy)
        leaf_id = id(leaf)
        leaf_state = node_states.get(leaf_id)
        if leaf_state is None:
            parent_state = node_states.get(id(leaf.parent))
            if parent_state is None:
                continue  # should not happen
            leaf_state = _game_apply(parent_state, leaf.action)
            node_states[leaf_id] = leaf_state

        # 3. Evaluate leaf
        if _game_is_terminal(leaf_state):
            value = _game_outcome(leaf_state, perspective)
        else:
            leaf_player = _game_current(leaf_state)
            leaf_actions = _game_legal(leaf_state)
            leaf_K = len(leaf_actions)

            # Encode state once (shared by both heads)
            leaf_value_from_both = None
            leaf_feats = None
            if use_policy or use_value:
                leaf_feats = game.encode_state(leaf_state, leaf_player)

            if use_policy:
                leaf_mask = game.legal_action_mask(leaf_state)
                if use_value:
                    # Combined forward — shared body computed once
                    leaf_value_from_both, full_p = net.predict_both(
                        leaf_feats, leaf_mask,
                    )
                else:
                    full_p = net.predict_policy(leaf_feats, leaf_mask)
                child_priors = np.array(
                    [full_p[game.action_to_index(a)] for a in leaf_actions],
                )
                cp_sum = child_priors.sum()
                if cp_sum > 0:
                    child_priors /= cp_sum
                else:
                    child_priors = np.full(leaf_K, 1.0 / leaf_K)
            else:
                child_priors = np.full(leaf_K, 1.0 / leaf_K)

            for i, a in enumerate(leaf_actions):
                child = _Node(
                    player=leaf_player, action=a, parent=leaf,
                    prior=float(child_priors[i]),
                )
                leaf.children.append(child)
                # Lazy: do NOT compute child states here

            # Evaluate the leaf position
            if leaf_value_from_both is not None:
                value = (leaf_value_from_both if leaf_player == perspective
                         else -leaf_value_from_both)
            elif use_value:
                leaf_value = net.predict_value(leaf_feats)
                value = leaf_value if leaf_player == perspective else -leaf_value
            else:
                value = _random_rollout(
                    leaf_state, game, perspective,
                    rollout_rng or random.Random(),
                )

        # 4. Backpropagate
        node: Optional[_Node] = leaf
        while node is not None:
            node.visits += 1
            if node.player == perspective:
                node.value_sum += value
            else:
                node.value_sum -= value
            node = node.parent

    return {ch.action: ch.visits for ch in root.children}


# ---------------------------------------------------------------------------
#  Public API — determinized AlphaZero MCTS
# ---------------------------------------------------------------------------


def alpha_mcts_choose(
    state: Any,
    game: GameInterface,
    net: AlphaNet,
    player: int,
    config: MCTSConfig,
    rng: random.Random,
) -> Any:
    """Choose an action using AlphaZero MCTS with determinization.

    Aggregates visit counts over ``config.determinizations`` sampled worlds
    and returns the most-visited action.
    """
    total: dict[Any, float] = {}
    for _ in range(config.determinizations):
        det = game.determinize(state, player, rng)
        rollout_rng = random.Random(rng.randrange(1 << 30))
        visits = _run_mcts(det, game, net, player, config, rollout_rng)
        for act, cnt in visits.items():
            total[act] = total.get(act, 0.0) + cnt

    if not total:
        actions = game.legal_actions(state)
        return rng.choice(actions)
    return max(total, key=total.__getitem__)


# ---------------------------------------------------------------------------
#  MCTS policy — returns visit distribution for AlphaZero training
# ---------------------------------------------------------------------------


def alpha_mcts_policy(
    state: Any,
    game: GameInterface,
    net: AlphaNet,
    player: int,
    config: MCTSConfig,
    rng: random.Random,
) -> tuple[np.ndarray, Any]:
    """MCTS search returning *(policy_target, action)* for training.

    ``policy_target`` is an ``(action_space_size,)`` distribution derived
    from aggregated visit counts (temperature-adjusted via
    ``config.visit_temp``).  ``action`` is sampled from that distribution.
    """
    total: dict[Any, float] = {}
    for _ in range(config.determinizations):
        det = game.determinize(state, player, rng)
        rollout_rng = random.Random(rng.randrange(1 << 30))
        visits = _run_mcts(det, game, net, player, config, rollout_rng)
        for act, cnt in visits.items():
            total[act] = total.get(act, 0.0) + cnt

    action_space = game.action_space_size

    if not total:
        actions = game.legal_actions(state)
        pi = np.zeros(action_space, dtype=np.float64)
        for a in actions:
            pi[game.action_to_index(a)] = 1.0 / len(actions)
        return pi, rng.choice(actions)

    # Build raw visit counts in action-space indices
    raw = np.zeros(action_space, dtype=np.float64)
    idx_to_action: dict[int, Any] = {}
    for act, cnt in total.items():
        idx = game.action_to_index(act)
        raw[idx] = cnt
        idx_to_action[idx] = act

    # Temperature-adjusted distribution  (π_a ∝ N(a)^{1/τ})
    nonzero = raw > 0
    pi = np.zeros(action_space, dtype=np.float64)
    temp = config.visit_temp
    if temp > 0 and nonzero.any():
        pi[nonzero] = raw[nonzero] ** (1.0 / temp)
    elif nonzero.any():
        # τ → 0: deterministic (pick most-visited)
        pi[np.argmax(raw)] = 1.0

    total_pi = pi.sum()
    if total_pi > 0:
        pi /= total_pi

    # Sample action from distribution
    legal_indices = [i for i in idx_to_action if pi[i] > 0]
    if legal_indices:
        weights = [pi[i] for i in legal_indices]
        chosen_idx = rng.choices(legal_indices, weights=weights, k=1)[0]
    else:
        chosen_idx = rng.choice(list(idx_to_action.keys()))

    return pi, idx_to_action[chosen_idx]
