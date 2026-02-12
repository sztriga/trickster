"""Hybrid MCTS + alpha-beta player for Ulti.

Uses neural network-guided MCTS for early/mid-game decisions,
switching to PIMC (Perfect Information Monte Carlo) + exact
alpha-beta solving when the endgame is small enough.

The switch point is controlled by ``endgame_tricks``.  With the
Cython solver (~50x faster than pure Python), 6-8 exact tricks are
feasible per determinization, meaning the NN only needs to be good
for the first few tricks.

Usage (play):
    player = HybridPlayer(game, net_wrapper, endgame_tricks=6)
    action = player.choose_action(state, player_idx, rng)

Usage (training — returns policy target):
    pi, action = player.choose_action_with_policy(state, player_idx, rng)
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import Card, TRICKS_PER_GAME
from trickster.mcts import MCTSConfig, alpha_mcts_choose, alpha_mcts_policy

# ---------------------------------------------------------------------------
#  Solver backend: prefer Cython, fall back to pure Python
# ---------------------------------------------------------------------------

try:
    from trickster._solver_core import solve_root as _solve_root
    SOLVER_ENGINE = "cython"
except ImportError:
    from trickster.solver import solve_root as _solve_root
    SOLVER_ENGINE = "python"


def _detect_contract(state: UltiNode) -> str:
    """Infer the contract type from UltiNode metadata."""
    gs = state.gs
    if gs.betli:
        return "betli"
    comps = state.contract_components or frozenset()
    if "durchmars" in comps:
        return "durchmars"
    if "ulti" in comps:
        return "parti_ulti"
    return "parti"


# ---------------------------------------------------------------------------
#  Hybrid player
# ---------------------------------------------------------------------------


class HybridPlayer:
    """Neural MCTS for early game, exact alpha-beta for endgame.

    The handover point (``endgame_tricks``) determines how many tricks
    remain when the player switches from MCTS to exact solving.  Fewer
    remaining tricks → faster solves; more → stronger play overall.

    Parameters
    ----------
    game : UltiGame
        Game interface (for legal actions, determinize, encode, etc.).
    net : AlphaNet-compatible wrapper
        Neural network with ``predict_policy`` / ``predict_value``.
    mcts_config : MCTSConfig, optional
        MCTS hyperparameters.  Defaults to ``MCTSConfig()``.
    endgame_tricks : int
        Switch to exact solving when this many tricks (or fewer) remain.
        Default 6 (practical limit for Cython solver with PIMC).
    pimc_determinizations : int
        Number of determinizations for PIMC in the endgame solver.
    solver_temperature : float
        Temperature for converting solver values to a policy distribution.
        Lower = more peaked on the best move.  Default 1.0.
    """

    def __init__(
        self,
        game: UltiGame,
        net: Any,
        mcts_config: MCTSConfig | None = None,
        endgame_tricks: int = 6,
        pimc_determinizations: int = 20,
        solver_temperature: float = 1.0,
    ) -> None:
        self.game = game
        self.net = net
        self.mcts_config = mcts_config or MCTSConfig()
        self.endgame_tricks = endgame_tricks
        self.pimc_dets = pimc_determinizations
        self.solver_temp = solver_temperature

    @property
    def uses_fast_solver(self) -> bool:
        """True if the Cython solver is loaded."""
        return SOLVER_ENGINE == "cython"

    def _in_endgame(self, state: UltiNode) -> bool:
        """Should we use the exact solver for this position?"""
        remaining = TRICKS_PER_GAME - state.gs.trick_no
        return remaining <= self.endgame_tricks

    # ------------------------------------------------------------------
    #  Public API: play
    # ------------------------------------------------------------------

    def choose_action(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> Card:
        """Pick the best action for play / evaluation.

        Returns a single Card (the chosen action).
        """
        legal = self.game.legal_actions(state)
        if len(legal) <= 1:
            return legal[0]

        if self._in_endgame(state):
            return self._solve_action(state, player, rng)
        return alpha_mcts_choose(
            state, self.game, self.net, player, self.mcts_config, rng,
        )

    # ------------------------------------------------------------------
    #  Public API: training (returns policy + action)
    # ------------------------------------------------------------------

    def choose_action_with_policy(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, Card]:
        """Pick action and return a policy target for training.

        Returns ``(pi, action)`` where ``pi`` is an ``(action_space,)``
        probability distribution and ``action`` is sampled from it.
        """
        legal = self.game.legal_actions(state)
        action_space = self.game.action_space_size

        if len(legal) <= 1:
            pi = np.zeros(action_space, dtype=np.float64)
            if legal:
                pi[self.game.action_to_index(legal[0])] = 1.0
            return pi, legal[0] if legal else None

        if self._in_endgame(state):
            return self._solve_policy(state, player, rng)
        return alpha_mcts_policy(
            state, self.game, self.net, player, self.mcts_config, rng,
        )

    # ------------------------------------------------------------------
    #  PIMC + exact solver internals
    # ------------------------------------------------------------------

    def _pimc_values(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> dict[Card, float]:
        """Average exact-solve values over PIMC determinizations."""
        legal = self.game.legal_actions(state)
        totals: dict[Card, float] = {c: 0.0 for c in legal}
        counts: dict[Card, int] = {c: 0 for c in legal}

        contract = _detect_contract(state)

        for _ in range(self.pimc_dets):
            det = self.game.determinize(state, player, rng)
            vals = _solve_root(det.gs, contract=contract)
            for card in legal:
                if card in vals:
                    totals[card] += vals[card]
                    counts[card] += 1

        return {c: totals[c] / max(1, counts[c]) for c in legal}

    def _solve_action(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> Card:
        """Pick best card using PIMC + exact solver."""
        avg = self._pimc_values(state, player, rng)
        if player == state.gs.soloist:
            return max(avg, key=avg.__getitem__)
        return min(avg, key=avg.__getitem__)

    def _solve_policy(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, Card]:
        """Solver-derived policy + sampled action for training."""
        legal = self.game.legal_actions(state)
        action_space = self.game.action_space_size
        avg = self._pimc_values(state, player, rng)

        is_max = (player == state.gs.soloist)
        pi = self._values_to_policy(avg, action_space, is_max)

        # Sample action from the policy
        indices = [self.game.action_to_index(c) for c in legal]
        weights = [max(pi[i], 1e-12) for i in indices]
        chosen_idx = rng.choices(indices, weights=weights, k=1)[0]

        # Find corresponding Card
        for c in legal:
            if self.game.action_to_index(c) == chosen_idx:
                return pi, c

        return pi, legal[0]  # fallback

    def _values_to_policy(
        self,
        values: dict[Card, float],
        action_space: int,
        is_maximising: bool,
    ) -> np.ndarray:
        """Convert solver values → softmax policy distribution.

        For the defender (minimising), values are negated so that
        lower values map to higher probabilities.
        """
        pi = np.zeros(action_space, dtype=np.float64)
        if not values:
            return pi

        T = max(self.solver_temp, 1e-8)

        # Align so higher = better for both sides
        adjusted = {
            c: (v if is_maximising else -v) for c, v in values.items()
        }

        max_val = max(adjusted.values())
        exp_vals: dict[Card, float] = {}
        total = 0.0
        for card, val in adjusted.items():
            ev = math.exp((val - max_val) / T)
            exp_vals[card] = ev
            total += ev

        if total > 0:
            for card, ev in exp_vals.items():
                idx = self.game.action_to_index(card)
                pi[idx] = ev / total

        return pi
