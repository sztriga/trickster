"""Asymmetric reward engine for Ulti AlphaZero training.

Translates the Hungarian Ulti scoring system into normalised
rewards suitable for neural network training (values in [-1, +1]).

Key properties:
  - **Asymmetry**: Soloist pays double on loss (standard Ulti rule).
  - **Coalition**: Both defenders receive the same reward.
  - **Normalization**: Rewards are squashed to [-1, +1] so the value
    head output range matches.

Contract-value scaling:
  Higher-value contracts (Ulti, 100, Durchmars) produce proportionally
  larger raw rewards before normalisation.  This teaches the network
  that losing an Ulti hurts more than losing a Simple.
"""

from __future__ import annotations

import numpy as np

from trickster.games.ulti.adapter import UltiNode

# ---------------------------------------------------------------------------
#  Contract value table (for reward scaling)
# ---------------------------------------------------------------------------

# Maps contract component sets to (win_value, loss_value) per defender.
# These are the "base" points before kontra/red multipliers.
_CONTRACT_VALUES: dict[frozenset[str], tuple[float, float]] = {
    frozenset({"parti"}):                     (1.0,  2.0),
    frozenset({"parti", "ulti"}):             (5.0, 10.0),  # 4+1
    frozenset({"betli"}):                     (5.0, 10.0),
    frozenset({"parti", "durchmars"}):        (7.0, 14.0),  # 6+1
    frozenset({"parti", "40", "100"}):        (5.0, 10.0),  # 4+1
    frozenset({"parti", "20", "100"}):        (9.0, 18.0),  # 8+1
    frozenset({"parti", "ulti", "durchmars"}): (11.0, 22.0), # 6+4+1
}

# Fallback for unknown/compound contracts
_DEFAULT_WIN = 1.0
_DEFAULT_LOSS = 2.0

# Maximum possible raw reward (for normalisation)
_MAX_RAW = 22.0


def _contract_scale(components: frozenset[str] | None) -> tuple[float, float]:
    """Look up (win_value, loss_value) for a contract."""
    if components is None:
        return _DEFAULT_WIN, _DEFAULT_LOSS
    entry = _CONTRACT_VALUES.get(components)
    if entry is not None:
        return entry
    # Fallback: sum up known component values
    win = 0.0
    loss = 0.0
    if "parti" in components:
        win += 1.0; loss += 2.0
    if "ulti" in components:
        win += 4.0; loss += 8.0
    if "betli" in components:
        win += 5.0; loss += 10.0
    if "durchmars" in components:
        win += 6.0; loss += 12.0
    if "100" in components and "40" in components:
        win += 4.0; loss += 8.0
    elif "100" in components and "20" in components:
        win += 8.0; loss += 16.0
    return max(win, _DEFAULT_WIN), max(loss, _DEFAULT_LOSS)


# ---------------------------------------------------------------------------
#  Reward calculation
# ---------------------------------------------------------------------------


def calculate_reward(
    state: UltiNode,
    soloist_won: bool,
) -> tuple[float, float, float]:
    """Compute normalised rewards for all 3 players.

    Parameters
    ----------
    state : UltiNode
        Terminal game state.
    soloist_won : bool
        Whether the soloist won the contract.

    Returns
    -------
    (r0, r1, r2) : tuple of float
        Reward for each player, normalised to [-1, +1].

    Reward model:
      Soloist wins  → soloist gets +win_val, each defender gets -win_val/2
      Soloist loses → soloist gets -loss_val, each defender gets +loss_val/2

    Normalised by dividing by ``_MAX_RAW`` to keep everything in [-1, 1].
    """
    gs = state.gs
    comps = state.contract_components
    win_val, loss_val = _contract_scale(comps)

    rewards = [0.0, 0.0, 0.0]
    soloist = gs.soloist

    if soloist_won:
        rewards[soloist] = win_val
        for i in range(3):
            if i != soloist:
                rewards[i] = -win_val / 2.0
    else:
        rewards[soloist] = -loss_val
        for i in range(3):
            if i != soloist:
                rewards[i] = loss_val / 2.0

    # Normalise to [-1, +1]
    rewards = [r / _MAX_RAW for r in rewards]
    # Clamp (shouldn't be needed, but safety)
    rewards = [max(-1.0, min(1.0, r)) for r in rewards]

    return (rewards[0], rewards[1], rewards[2])


def outcome_for_player(
    state: UltiNode,
    player: int,
    soloist_won: bool,
) -> float:
    """Normalised reward for a single player.  Used to fill ``z`` in training samples."""
    r = calculate_reward(state, soloist_won)
    return r[player]


# ---------------------------------------------------------------------------
#  Simple outcome (used when we don't need contract scaling)
# ---------------------------------------------------------------------------


def simple_outcome(state: UltiNode, player: int) -> float:
    """Basic +1 / -1 outcome for curriculum training.

    Uses the UltiGame.outcome() logic directly:
      +1 if player's side won, -1 otherwise.
    """
    from trickster.games.ulti.game import soloist_lost_betli, soloist_won_simple

    gs = state.gs
    if gs.betli:
        soloist_wins = not soloist_lost_betli(gs)
    else:
        soloist_wins = soloist_won_simple(gs)

    if player == gs.soloist:
        return 1.0 if soloist_wins else -1.0
    return -1.0 if soloist_wins else 1.0


# ---------------------------------------------------------------------------
#  Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer with priority-weighted sampling.

    Each sample is a tuple:
      (state_feats, mask, mcts_policy, reward, is_soloist)
    stored as numpy arrays for efficient batch sampling.

    Soloist experiences are sampled ``soloist_weight`` times more
    frequently than defender experiences.  This corrects the
    "defender bias" where ~67% of games are easy defender wins.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        soloist_weight: float = 3.0,
    ) -> None:
        self.capacity = capacity
        self.soloist_weight = soloist_weight
        self.states: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.is_soloist: list[bool] = []

    def __len__(self) -> int:
        return len(self.states)

    def push(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        policy: np.ndarray,
        reward: float,
        is_soloist: bool = False,
    ) -> None:
        self.states.append(state)
        self.masks.append(mask)
        self.policies.append(policy)
        self.rewards.append(reward)
        self.is_soloist.append(is_soloist)

        # FIFO eviction
        if len(self.states) > self.capacity:
            self.states = self.states[-self.capacity:]
            self.masks = self.masks[-self.capacity:]
            self.policies = self.policies[-self.capacity:]
            self.rewards = self.rewards[-self.capacity:]
            self.is_soloist = self.is_soloist[-self.capacity:]

    def sample(
        self, batch_size: int, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Priority-weighted mini-batch sampling.

        Soloist samples receive ``soloist_weight`` times higher
        probability than defender samples.  Weights are normalised
        to form a proper distribution.

        Returns
        -------
        states : (B, state_dim)
        masks : (B, action_dim)
        policies : (B, action_dim)
        rewards : (B,)
        """
        n = len(self.states)
        B = min(batch_size, n)

        # Build sampling weights: soloist_weight for soloists, 1.0 for defenders
        weights = np.array(
            [self.soloist_weight if s else 1.0 for s in self.is_soloist],
            dtype=np.float64,
        )
        weights /= weights.sum()

        indices = rng.choice(n, size=B, replace=False, p=weights)
        return (
            np.stack([self.states[i] for i in indices]),
            np.stack([self.masks[i] for i in indices]),
            np.stack([self.policies[i] for i in indices]),
            np.array([self.rewards[i] for i in indices], dtype=np.float64),
        )
