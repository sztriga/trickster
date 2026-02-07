"""Generic game interface for AlphaZero-style training.

Any game (Snapszer, Ulti, …) implements this protocol so that MCTS,
training, and evaluation code can be fully game-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import random


# Generic type aliases — concrete games define their own State / Action types.
State = Any
Action = Any


@runtime_checkable
class GameInterface(Protocol):
    """Protocol that every game must implement."""

    # ------------------------------------------------------------------
    #  Game rules
    # ------------------------------------------------------------------

    @property
    def num_players(self) -> int:
        """Number of players (2 for Snapszer, 3 for Ulti, …)."""
        ...

    def current_player(self, state: State) -> int:
        """Index of the player who acts next."""
        ...

    def legal_actions(self, state: State) -> list[Action]:
        """Legal actions for the current player."""
        ...

    def apply(self, state: State, action: Action) -> State:
        """Apply *action* and return a **new** state (no mutation)."""
        ...

    def is_terminal(self, state: State) -> bool:
        ...

    def outcome(self, state: State, player: int) -> float:
        """Normalised outcome in ``[-1, +1]`` for *player*.

        +1 = best possible result, −1 = worst, 0 = draw / neutral.
        Called only on terminal states.
        """
        ...

    # ------------------------------------------------------------------
    #  Imperfect information
    # ------------------------------------------------------------------

    def determinize(self, state: State, player: int, rng: random.Random) -> State:
        """Sample a concrete state consistent with *player*'s observations.

        For perfect-information games, just return a clone of *state*.
        """
        ...

    # ------------------------------------------------------------------
    #  Neural-network encoding
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Length of the state feature vector."""
        ...

    @property
    def action_dims(self) -> dict[str, int]:
        """Feature dimensions per decision type.

        E.g. ``{"lead": 138, "follow": 145}`` for Snapszer.
        """
        ...

    def decision_type(self, state: State) -> str:
        """Which policy head should act in this state.

        E.g. ``"lead"`` or ``"follow"`` for Snapszer.
        """
        ...

    def encode_state(self, state: State, player: int) -> np.ndarray:
        """Encode observable state into a 1-D float array for the value head."""
        ...

    def encode_actions(
        self,
        state: State,
        player: int,
        actions: Sequence[Action],
    ) -> np.ndarray:
        """Encode *(state, action)* pairs into an ``(N, action_dim)`` array.

        One row per legal action — used by the policy head.
        """
        ...

    @property
    def action_space_size(self) -> int:
        """Total number of distinct actions across all decision types."""
        ...

    def action_to_index(self, action: Action) -> int:
        """Map an action to its fixed index in [0, action_space_size)."""
        ...

    def legal_action_mask(self, state: State) -> np.ndarray:
        """Boolean mask of shape (action_space_size,) — True = legal."""
        ...

    # ------------------------------------------------------------------
    #  New game
    # ------------------------------------------------------------------

    def new_game(self, seed: int, **kwargs: Any) -> State:
        """Deal / set up a new game."""
        ...
