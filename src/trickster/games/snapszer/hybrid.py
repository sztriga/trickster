"""HybridPlayer — phase-dependent search for Schnapsen / Snapszer.

Combines three algorithms based on the game phase:

* **Phase 1 Early** (high uncertainty) -> MCTS guided by the NN.
* **Phase 1 Late**  (moderate uncertainty) -> PIMC Minimax.
* **Phase 2**       (perfect information) -> Pure Alpha-Beta Minimax.

Phase boundaries are controlled by ``constants.DEFAULT_LATE_THRESHOLD``.
The ``HybridPlayer`` is usable in both evaluation and self-play training.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from trickster.games.snapszer.adapter import Action, SnapszerGame, SnapszerNode
from trickster.games.snapszer.constants import DEFAULT_LATE_THRESHOLD, DEFAULT_PIMC_SAMPLES
from trickster.games.snapszer.minimax import alphabeta, game_phase, pimc_minimax
from trickster.mcts import MCTSConfig, alpha_mcts_choose, alpha_mcts_policy
from trickster.models.alpha_net import SharedAlphaNet


# ---------------------------------------------------------------------------
#  Decision statistics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DecisionStats:
    """Tracks which algorithm contributed to each decision."""
    mcts_decisions: int = 0
    pimc_decisions: int = 0
    minimax_decisions: int = 0
    mcts_time: float = 0.0
    pimc_time: float = 0.0
    minimax_time: float = 0.0

    @property
    def total_decisions(self) -> int:
        return self.mcts_decisions + self.pimc_decisions + self.minimax_decisions

    @property
    def total_time(self) -> float:
        return self.mcts_time + self.pimc_time + self.minimax_time

    def summary(self) -> str:
        n = self.total_decisions or 1
        parts = [
            f"MCTS: {self.mcts_decisions} ({self.mcts_decisions/n:.0%}, "
            f"{self.mcts_time:.2f}s)",
            f"PIMC: {self.pimc_decisions} ({self.pimc_decisions/n:.0%}, "
            f"{self.pimc_time:.2f}s)",
            f"Minimax: {self.minimax_decisions} ({self.minimax_decisions/n:.0%}, "
            f"{self.minimax_time:.2f}s)",
        ]
        return " | ".join(parts)


# ---------------------------------------------------------------------------
#  HybridPlayer
# ---------------------------------------------------------------------------


class HybridPlayer:
    """Phase-dependent search: MCTS -> PIMC -> Minimax.

    Parameters
    ----------
    net : SharedAlphaNet
        Neural network for MCTS (policy + value heads).
    mcts_config : MCTSConfig
        MCTS settings (used only in Phase 1 Early).
    game : SnapszerGame
        Game interface.
    pimc_samples : int
        PIMC worlds per move (default from ``constants``).
    late_threshold : int
        Talon cards at which we switch to PIMC (default from ``constants``).
    """

    def __init__(
        self,
        net: SharedAlphaNet,
        mcts_config: MCTSConfig,
        game: SnapszerGame,
        pimc_samples: int = DEFAULT_PIMC_SAMPLES,
        late_threshold: int = DEFAULT_LATE_THRESHOLD,
    ) -> None:
        self.net = net
        self.mcts_config = mcts_config
        self.game = game
        self.pimc_samples = pimc_samples
        self.late_threshold = late_threshold
        self.stats = DecisionStats()

    def reset_stats(self) -> None:
        self.stats = DecisionStats()

    # -- public interface --------------------------------------------------

    def choose_action(
        self,
        node: SnapszerNode,
        player: int,
        rng: random.Random,
    ) -> Action:
        """Pick the best action using the phase-appropriate algorithm."""
        actions = self.game.legal_actions(node)
        if len(actions) <= 1:
            return actions[0]

        phase = game_phase(node, self.late_threshold)

        if phase == "phase1_early":
            return self._mcts_choose(node, player, rng)
        elif phase == "phase1_late":
            return self._pimc_choose(node, player, rng)
        else:
            return self._minimax_choose(node, player)

    def choose_action_for_training(
        self,
        node: SnapszerNode,
        player: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, Action]:
        """Pick action and return a policy distribution for training.

        For MCTS: returns the visit distribution.
        For PIMC/Minimax: returns a one-hot vector on the best action.
        """
        actions = self.game.legal_actions(node)
        if len(actions) <= 1:
            pi = np.zeros(self.game.action_space_size, dtype=np.float64)
            idx = self.game.action_to_index(actions[0])
            pi[idx] = 1.0
            return pi, actions[0]

        phase = game_phase(node, self.late_threshold)

        if phase == "phase1_early":
            return self._mcts_policy(node, player, rng)

        # PIMC or Minimax — both produce one-hot targets
        if phase == "phase1_late":
            action = self._pimc_choose(node, player, rng)
        else:
            action = self._minimax_choose(node, player)
        pi = np.zeros(self.game.action_space_size, dtype=np.float64)
        pi[self.game.action_to_index(action)] = 1.0
        return pi, action

    # -- internal methods --------------------------------------------------

    def _mcts_choose(
        self, node: SnapszerNode, player: int, rng: random.Random,
    ) -> Action:
        t0 = time.perf_counter()
        action = alpha_mcts_choose(
            node, self.game, self.net, player, self.mcts_config, rng,
        )
        self.stats.mcts_decisions += 1
        self.stats.mcts_time += time.perf_counter() - t0
        return action

    def _mcts_policy(
        self, node: SnapszerNode, player: int, rng: random.Random,
    ) -> tuple[np.ndarray, Action]:
        t0 = time.perf_counter()
        pi, action = alpha_mcts_policy(
            node, self.game, self.net, player, self.mcts_config, rng,
        )
        self.stats.mcts_decisions += 1
        self.stats.mcts_time += time.perf_counter() - t0
        return pi, action

    def _pimc_choose(
        self, node: SnapszerNode, player: int, rng: random.Random,
    ) -> Action:
        t0 = time.perf_counter()
        action, _val = pimc_minimax(
            node, self.game, player,
            n_samples=self.pimc_samples, rng=rng,
        )
        self.stats.pimc_decisions += 1
        self.stats.pimc_time += time.perf_counter() - t0
        return action

    def _minimax_choose(
        self, node: SnapszerNode, player: int,
    ) -> Action:
        t0 = time.perf_counter()
        _val, action = alphabeta(node, self.game, player)
        self.stats.minimax_decisions += 1
        self.stats.minimax_time += time.perf_counter() - t0
        if action is None:
            return self.game.legal_actions(node)[0]
        return action
