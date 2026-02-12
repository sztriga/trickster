"""Alpha-Beta Minimax solver for Schnapsen / Snapszer endgame.

Provides two solvers:

* ``alphabeta``  — exact minimax with alpha-beta pruning for
  perfect-information states (Phase 2: talon empty or closed).

* ``pimc_minimax`` — Perfect Information Monte Carlo: samples
  N possible worlds from the current imperfect-information state
  and runs alpha-beta on each to produce a robust move
  recommendation (Phase 1 Late).

The objective is **game-point maximisation** (1, 2 or 3 points),
not just win/loss.  Values are normalised to [-1, +1] via the
existing ``SnapszerGame.outcome()`` (which divides game points
by 3).

If the Cython-accelerated ``_fast_minimax`` module is available
(~130x faster), it is used transparently.  Build it with::

    python src/trickster/games/snapszer/_build_fast_minimax.py
"""

from __future__ import annotations

import random
from typing import Any

from trickster.games.snapszer.adapter import Action, SnapszerGame, SnapszerNode
from trickster.games.snapszer.constants import DEFAULT_LATE_THRESHOLD, DEFAULT_PIMC_SAMPLES

# ---------------------------------------------------------------------------
#  Try to import Cython-accelerated implementations
# ---------------------------------------------------------------------------
try:
    from trickster.games.snapszer._fast_minimax import (
        c_alphabeta as _c_alphabeta,
        c_pimc_minimax as _c_pimc_minimax,
    )
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


# ---------------------------------------------------------------------------
#  Alpha-Beta Minimax
# ---------------------------------------------------------------------------


def alphabeta(
    node: SnapszerNode,
    game: SnapszerGame,
    maximizing_player: int,
    alpha: float = -2.0,
    beta: float = 2.0,
    _depth: int = 0,
    _max_depth: int = 30,
) -> tuple[float, Action | None]:
    """Exact minimax with alpha-beta pruning.

    Parameters
    ----------
    node : SnapszerNode
        Current game state (perfect information assumed).
    game : SnapszerGame
        Game interface for legal_actions / apply / is_terminal / outcome.
    maximizing_player : int
        The player whose game-point outcome we maximise.
    alpha, beta : float
        Pruning bounds (callers should leave at default ±2).

    Returns
    -------
    (value, best_action)
        ``value`` is in [-1, +1] (normalised game points for
        *maximizing_player*).  ``best_action`` is ``None`` at
        terminal states.
    """
    # Dispatch to Cython if available (root call only)
    if _HAS_CYTHON and _depth == 0:
        return _c_alphabeta(node, game, maximizing_player)
    if game.is_terminal(node):
        return game.outcome(node, maximizing_player), None

    if _depth >= _max_depth:
        # Safety fallback — should never happen in normal play.
        return 0.0, None

    actions = game.legal_actions(node)
    if not actions:
        return 0.0, None

    # Forced move — skip directly (no branching overhead).
    if len(actions) == 1:
        child = game.apply(node, actions[0])
        val, _ = alphabeta(
            child, game, maximizing_player, alpha, beta,
            _depth + 1, _max_depth,
        )
        return val, actions[0]

    player = game.current_player(node)
    is_max = player == maximizing_player
    best_action = actions[0]

    if is_max:
        value = -2.0
        for action in actions:
            child = game.apply(node, action)
            child_val, _ = alphabeta(
                child, game, maximizing_player, alpha, beta,
                _depth + 1, _max_depth,
            )
            if child_val > value:
                value = child_val
                best_action = action
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break  # β cutoff
        return value, best_action
    else:
        value = 2.0
        for action in actions:
            child = game.apply(node, action)
            child_val, _ = alphabeta(
                child, game, maximizing_player, alpha, beta,
                _depth + 1, _max_depth,
            )
            if child_val < value:
                value = child_val
                best_action = action
            if value < beta:
                beta = value
            if alpha >= beta:
                break  # α cutoff
        return value, best_action


# ---------------------------------------------------------------------------
#  PIMC Minimax (Perfect Information Monte Carlo)
# ---------------------------------------------------------------------------


def pimc_minimax(
    node: SnapszerNode,
    game: SnapszerGame,
    player: int,
    n_samples: int = DEFAULT_PIMC_SAMPLES,
    rng: random.Random | None = None,
) -> tuple[Action, float]:
    """Perfect Information Monte Carlo with Minimax.

    For each of *n_samples* sampled worlds (via ``game.determinize``),
    runs a single root-level alpha-beta search which finds the best
    action for that world.  Aggregates results by voting: the action
    that is optimal in the most worlds wins (ties broken by cumulative
    minimax value).

    This is ~|actions|× faster than the per-action approach because
    alpha-beta prunes across siblings at the root level.

    Parameters
    ----------
    node : SnapszerNode
        Current (imperfect-information) game state.
    game : SnapszerGame
        Game interface.
    player : int
        The player choosing an action (we maximise their outcome).
    n_samples : int
        Number of PIMC worlds to sample (default from constants).
    rng : random.Random, optional
        Random source for determinisation.

    Returns
    -------
    (best_action, avg_value)
        ``best_action`` is the recommended move.
        ``avg_value`` is its average minimax value across samples.
    """
    # Dispatch to Cython if available
    if _HAS_CYTHON:
        return _c_pimc_minimax(node, game, player, n_samples, rng)

    if rng is None:
        rng = random.Random()

    actions = game.legal_actions(node)
    if len(actions) <= 1:
        return actions[0] if actions else None, 0.0

    # Vote counting: how many worlds chose each action as optimal,
    # plus cumulative value for tie-breaking.
    action_wins: dict[Any, int] = {a: 0 for a in actions}
    action_value: dict[Any, float] = {a: 0.0 for a in actions}

    for _ in range(n_samples):
        det = game.determinize(node, player, rng)
        val, best = alphabeta(det, game, player)
        if best is not None and best in action_wins:
            action_wins[best] += 1
            action_value[best] += val
        else:
            # Determinization changed legal actions (shouldn't happen
            # for the current player, but be defensive).
            det_val, det_best = alphabeta(det, game, player)
            if det_best is not None:
                # Map to closest known action
                for a in actions:
                    child = game.apply(det, a)
                    cv, _ = alphabeta(child, game, player)
                    action_value[a] += cv
                    action_wins[a] += 1
                break

    # Pick action with most wins (ties broken by total value).
    best = max(actions, key=lambda a: (action_wins[a], action_value[a]))
    n_best = action_wins[best] or 1
    avg_val = action_value[best] / n_best
    return best, avg_val


# ---------------------------------------------------------------------------
#  Phase detection helpers
# ---------------------------------------------------------------------------


def is_phase2(node: SnapszerNode) -> bool:
    """True if the game is in Phase 2 (talon closed or exhausted).

    In Phase 2 all remaining cards are deducible from public
    information, so pure minimax is applicable.
    """
    gs = node.gs
    return gs.talon_closed or (
        len(gs.draw_pile) == 0 and gs.trump_card is None
    )


def talon_cards_left(node: SnapszerNode) -> int:
    """Number of cards remaining in the talon (including trump upcard)."""
    gs = node.gs
    return len(gs.draw_pile) + (1 if gs.trump_card is not None else 0)


def game_phase(node: SnapszerNode, late_threshold: int = DEFAULT_LATE_THRESHOLD) -> str:
    """Classify the current game phase.

    Returns
    -------
    ``"phase2"`` — talon closed / exhausted (perfect info).
    ``"phase1_late"`` — talon open but ≤ *late_threshold* cards left.
    ``"phase1_early"`` — everything else.
    """
    if is_phase2(node):
        return "phase2"
    if talon_cards_left(node) <= late_threshold:
        return "phase1_late"
    return "phase1_early"
