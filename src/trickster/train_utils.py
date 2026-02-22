"""Training utilities: ReplayBuffer and reward re-exports.

Ulti-specific reward functions (``simple_outcome``,
``solver_value_to_reward``) now live in
``trickster.games.ulti.rewards`` and are re-exported here for
backward compatibility.
"""
from __future__ import annotations

import numpy as np

# Re-export reward functions (canonical home: games.ulti.rewards)
from trickster.games.ulti.rewards import (  # noqa: F401
    _GAME_PTS_MAX,
    simple_outcome,
    solver_value_to_reward,
)


# ---------------------------------------------------------------------------
#  Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity replay buffer with reservoir sampling.

    Uses **reservoir sampling** (Algorithm R, Vitter 1985) instead of
    FIFO eviction.  Every sample ever pushed has an equal probability
    of being retained, preserving diversity across training history.
    This is what the NFSP paper recommends for the average strategy
    memory (Lanctot et al. 2017).
    """

    def __init__(
        self,
        capacity: int = 50_000,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity

        self._size = 0
        self._total_seen = 0
        self._allocated = False
        self._states: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self._policies: np.ndarray | None = None
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._is_soloist = np.zeros(capacity, dtype=bool)
        self._on_policy = np.ones(capacity, dtype=bool)  # True = on-policy (default)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        policy: np.ndarray,
        reward: float,
        is_soloist: bool = False,
        on_policy: bool = True,
    ) -> None:
        if not self._allocated:
            self._states = np.zeros(
                (self.capacity,) + state.shape, dtype=state.dtype,
            )
            self._masks = np.zeros(
                (self.capacity,) + mask.shape, dtype=mask.dtype,
            )
            self._policies = np.zeros(
                (self.capacity,) + policy.shape, dtype=policy.dtype,
            )
            self._allocated = True

        self._total_seen += 1

        if self._size < self.capacity:
            # Buffer not full — add directly
            pos = self._size
            self._size += 1
        else:
            # Reservoir sampling: replace random slot with prob capacity/total_seen
            j = int(self._rng.integers(0, self._total_seen))
            if j < self.capacity:
                pos = j
            else:
                return  # discard this sample

        self._states[pos] = state
        self._masks[pos] = mask
        self._policies[pos] = policy
        self._rewards[pos] = reward
        self._is_soloist[pos] = is_soloist
        self._on_policy[pos] = on_policy

    def stats(self) -> dict[str, float]:
        """Return buffer composition diagnostics."""
        n = self._size
        if n == 0:
            return {}
        sol = self._is_soloist[:n]
        rew = self._rewards[:n]
        n_sol = int(sol.sum())
        n_def = n - n_sol
        sol_win = int((sol & (rew > 0)).sum())
        sol_lose = n_sol - sol_win
        def_win = int((~sol & (rew > 0)).sum())
        def_lose = n_def - def_win
        return {
            "size": n,
            "total_seen": self._total_seen,
            "soloist_frac": n_sol / n,
            "sol_win": sol_win,
            "sol_lose": sol_lose,
            "def_win": def_win,
            "def_lose": def_lose,
            "sol_win_rate": sol_win / max(1, n_sol),
        }

    def sample(
        self, batch_size: int, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform random mini-batch sampling.

        Returns
        -------
        states : (B, state_dim)
        masks : (B, action_dim)
        policies : (B, action_dim)
        rewards : (B,)
        is_soloist : (B,) bool
        on_policy : (B,) bool
        """
        n = self._size
        B = min(batch_size, n)
        indices = rng.choice(n, size=B, replace=False)

        return (
            self._states[indices].copy(),
            self._masks[indices].copy(),
            self._policies[indices].copy(),
            self._rewards[indices].copy(),
            self._is_soloist[indices].copy(),
            self._on_policy[indices].copy(),
        )

