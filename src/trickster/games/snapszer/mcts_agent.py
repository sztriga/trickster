"""
Monte Carlo Tree Search agent for Snapszer.

Uses Information Set MCTS (determinization) to handle imperfect information:
sample multiple possible game states consistent with what the player can
observe, run MCTS on each, and aggregate results by visit count.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

from trickster.games.snapszer.cards import Card, Color, make_deck
from trickster.games.snapszer.game import (
    GameState,
    can_close_talon,
    close_talon,
    deal_winner,
    is_terminal,
    legal_actions,
    play_trick,
)


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration for MCTS search."""

    simulations: int = 100  # MCTS iterations per determinization
    determinizations: int = 6  # number of sampled worlds
    c: float = 1.4  # UCB exploration constant


# An action is either a Card (play it) or the literal "close_talon".
Action = Union[Card, str]


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------


def _determinize(
    state: GameState,
    player: int,
    rng: random.Random,
    pending_lead: Optional[Card] = None,
) -> GameState:
    """
    Create a determinized copy of the game state.

    Unknown cards (opponent hand, face-down draw pile, hidden trump) are
    randomly redistributed while keeping all public information intact.
    If there is a pending_lead from the opponent, that card is pinned
    in the opponent's hand.
    """
    det = state.clone()
    opponent = 1 - player
    all_cards = set(make_deck())

    # Cards known to this player.
    known: set[Card] = set(det.hands[player])
    known.update(det.captured[0])
    known.update(det.captured[1])
    if det.trump_card is not None and det.trump_upcard_visible:
        known.add(det.trump_card)
    if pending_lead is not None and state.leader != player:
        known.add(pending_lead)

    unknown = list(all_cards - known)
    rng.shuffle(unknown)

    opp_size = len(det.hands[opponent])
    pile_size = len(det.draw_pile)
    idx = 0

    # If the opponent led a known card, pin it in their hand.
    if pending_lead is not None and state.leader == opponent:
        opp_hand = [pending_lead]
        opp_hand.extend(unknown[idx : idx + opp_size - 1])
        idx += opp_size - 1
    else:
        opp_hand = unknown[idx : idx + opp_size]
        idx += opp_size

    det.hands[opponent] = opp_hand
    det.draw_pile = deque(unknown[idx : idx + pile_size])
    idx += pile_size

    # Hidden trump card (talon closed but trump object still exists).
    if det.trump_card is not None and not det.trump_upcard_visible:
        if idx < len(unknown):
            det.trump_card = unknown[idx]

    return det


# ---------------------------------------------------------------------------
# Game simulation helpers
# ---------------------------------------------------------------------------


def _current_player(state: GameState, pending_lead: Optional[Card]) -> int:
    return state.leader if pending_lead is None else 1 - state.leader


def _get_actions(state: GameState, pending_lead: Optional[Card]) -> list[Action]:
    if pending_lead is not None:
        return legal_actions(state, 1 - state.leader, pending_lead)
    actions: list[Action] = list(legal_actions(state, state.leader, None))
    if can_close_talon(state, state.leader):
        actions.append("close_talon")
    return actions


def _apply(
    state: GameState, pending_lead: Optional[Card], action: Action
) -> tuple[GameState, Optional[Card]]:
    """Apply an action. Returns (new_state, new_pending_lead)."""
    if action == "close_talon":
        new = state.clone()
        close_talon(new, new.leader)
        return new, None
    card: Card = action  # type: ignore[assignment]
    if pending_lead is None:
        # Leader plays — state unchanged, card becomes pending.
        return state, card
    # Responder plays — resolve trick.
    new = state.clone()
    new, _ = play_trick(new, pending_lead, card)
    return new, None


# ---------------------------------------------------------------------------
# MCTS tree
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    player: int  # who acts at this node
    action: Optional[Action] = None  # action taken to reach this node
    parent: Optional[_Node] = None
    children: list[_Node] = field(default_factory=list)
    untried: list[Action] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # cumulative value from root-perspective


def _ucb1(child: _Node, parent_visits: int, c: float, invert: bool) -> float:
    if child.visits == 0:
        return float("inf")
    q = child.value / child.visits
    if invert:
        q = 1.0 - q
    return q + c * math.sqrt(math.log(parent_visits) / child.visits)


def _select(root: _Node, c: float, perspective: int) -> _Node:
    node = root
    while node.children and not node.untried:
        opp = node.player != perspective
        node = max(node.children, key=lambda ch, _o=opp: _ucb1(ch, node.visits, c, _o))
    return node


def _rollout(
    state: GameState, pending_lead: Optional[Card], rng: random.Random
) -> Optional[int]:
    """Random playout to terminal. Returns winner (0, 1, or None)."""
    st = state.clone()
    pl = pending_lead
    while not is_terminal(st):
        if pl is not None:
            resp = 1 - st.leader
            rl = legal_actions(st, resp, pl)
            st, _ = play_trick(st, pl, rng.choice(rl))
            pl = None
        else:
            ll = legal_actions(st, st.leader, None)
            pl = rng.choice(ll)
    return deal_winner(st)


def _model_leaf_value(
    state: GameState,
    pending_lead: Optional[Card],
    perspective: int,
    lead_model: object,
    follow_model: object,
) -> float:
    """Estimate state value using trained action-value models.

    For each legal action, ask the model for P(win|state, action) and take
    the maximum as the position value from the acting player's perspective.

    All legal actions are batched into a single ``predict_proba_batch`` call
    so the forward pass runs as one matrix multiply instead of N separate
    ones.
    """
    from trickster.games.snapszer.features import (
        follow_action_features,
        lead_action_features,
        lead_close_talon_features,
    )

    if is_terminal(state):
        w = deal_winner(state)
        if w is None:
            return 0.5
        return 1.0 if w == perspective else 0.0

    player = _current_player(state, pending_lead)
    actions = _get_actions(state, pending_lead)
    if not actions:
        return 0.5

    trump_up = state.trump_card if state.trump_upcard_visible else None

    # Pre-compute shared context once (avoid repeated attribute lookups)
    hand = state.hands[player]
    draw_sz = len(state.draw_pile)
    cap_self = state.captured[player]
    cap_opp = state.captured[1 - player]
    trump_col = state.trump_color

    # Separate actions by model (lead vs follow) and build feature lists
    lead_feats: list[dict[str, float]] = []
    follow_feats: list[dict[str, float]] = []

    for action in actions:
        if action == "close_talon":
            lead_feats.append(lead_close_talon_features(
                hand, draw_pile_size=draw_sz, captured_self=cap_self,
                captured_opp=cap_opp, trump_color=trump_col, trump_upcard=trump_up,
            ))
        elif pending_lead is None:
            lead_feats.append(lead_action_features(
                hand, action, draw_pile_size=draw_sz, captured_self=cap_self,
                captured_opp=cap_opp, trump_color=trump_col, trump_upcard=trump_up,
                exchanged_trump=False,
            ))
        else:
            follow_feats.append(follow_action_features(
                hand, pending_lead, action, draw_pile_size=draw_sz,
                captured_self=cap_self, captured_opp=cap_opp,
                trump_color=trump_col, trump_upcard=trump_up,
            ))

    # Batched forward pass — one matrix multiply per model
    best_p = 0.0
    if lead_feats:
        probs = lead_model.predict_proba_batch(lead_feats)
        p = float(probs.max())
        if p > best_p:
            best_p = p
    if follow_feats:
        probs = follow_model.predict_proba_batch(follow_feats)
        p = float(probs.max())
        if p > best_p:
            best_p = p

    # best_p is P(win) from the acting player's view; flip if needed.
    if player == perspective:
        return best_p
    return 1.0 - best_p


def _backprop(node: Optional[_Node], val: float) -> None:
    while node is not None:
        node.visits += 1
        node.value += val
        node = node.parent


# ---------------------------------------------------------------------------
# Single-determinization MCTS
# ---------------------------------------------------------------------------


def _mcts(
    state: GameState,
    pending_lead: Optional[Card],
    perspective: int,
    config: MCTSConfig,
    rng: random.Random,
    lead_model: object = None,
    follow_model: object = None,
) -> dict[Action, int]:
    """Run MCTS on one determinized state. Returns action -> visit count.

    If *lead_model* and *follow_model* are provided, leaf nodes are evaluated
    using the models (fast).  Otherwise a random rollout is used.
    """
    actions = _get_actions(state, pending_lead)
    if len(actions) <= 1:
        return {a: 1 for a in actions}

    use_model = lead_model is not None and follow_model is not None

    cur = _current_player(state, pending_lead)
    root = _Node(player=cur, untried=list(actions))
    root.visits = 1

    # Map node id -> (state, pending_lead) for simulation.
    node_states: dict[int, tuple[GameState, Optional[Card]]] = {
        id(root): (state, pending_lead)
    }

    for _ in range(config.simulations):
        # Selection
        node = _select(root, config.c, perspective)
        pair = node_states.get(id(node))
        if pair is None:
            continue
        n_st, n_pl = pair

        # Expansion
        if node.untried and not is_terminal(n_st):
            act = node.untried.pop()
            new_st, new_pl = _apply(n_st, n_pl, act)
            term = is_terminal(new_st)
            child = _Node(
                player=_current_player(new_st, new_pl) if not term else node.player,
                action=act,
                parent=node,
                untried=list(_get_actions(new_st, new_pl)) if not term else [],
            )
            node.children.append(child)
            node_states[id(child)] = (new_st, new_pl)
            node = child
            n_st, n_pl = new_st, new_pl

        # Simulation / leaf evaluation
        if is_terminal(n_st):
            winner = deal_winner(n_st)
            val = 0.5 if winner is None else (1.0 if winner == perspective else 0.0)
        elif use_model:
            val = _model_leaf_value(n_st, n_pl, perspective, lead_model, follow_model)
        else:
            winner = _rollout(n_st, n_pl, rng)
            val = 0.5 if winner is None else (1.0 if winner == perspective else 0.0)

        # Backpropagation
        _backprop(node, val)

    return {ch.action: ch.visits for ch in root.children if ch.action is not None}


# ---------------------------------------------------------------------------
# Public API — determinized MCTS
# ---------------------------------------------------------------------------


def mcts_choose(
    state: GameState,
    pending_lead: Optional[Card],
    player: int,
    config: MCTSConfig,
    rng: random.Random,
    *,
    lead_model: object = None,
    follow_model: object = None,
) -> Action:
    """
    Choose an action using MCTS with determinization.

    Runs ``config.determinizations`` independent MCTS searches on randomly
    sampled game states, aggregates visit counts, and returns the most-visited
    action.

    When *lead_model* / *follow_model* are provided the search uses them for
    leaf evaluation (fast, informed).  Otherwise it falls back to random
    rollouts.
    """
    total: dict[Action, int] = {}
    for _ in range(config.determinizations):
        det = _determinize(state, player, rng, pending_lead)
        visits = _mcts(
            det, pending_lead, player, config, rng,
            lead_model=lead_model, follow_model=follow_model,
        )
        for act, cnt in visits.items():
            total[act] = total.get(act, 0) + cnt

    if not total:
        actions = _get_actions(state, pending_lead)
        return rng.choice(actions) if actions else list(legal_actions(state, player, None))[0]

    return max(total, key=lambda a: total[a])
