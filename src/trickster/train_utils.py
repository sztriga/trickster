"""Asymmetric reward engine for Ulti AlphaZero training.

Translates the Hungarian Ulti scoring system into normalised
rewards suitable for neural network training (values in [-1, +1]).

Key properties:
  - **Per-component evaluation**: Compound contracts (e.g. Ulti + Parti)
    are scored independently — the soloist can win Parti but lose Ulti.
  - **Asymmetry**: Soloist pays double on loss per component.
  - **Coalition**: Both defenders receive the same reward.
  - **Normalization**: Rewards are squashed to [-1, +1].

Component point values (win / loss):
  parti:      1 /  2
  ulti:       4 /  8
  betli:      5 / 10
  durchmars:  6 / 12
  40-100:     4 /  8
  20-100:     8 / 16
"""

from __future__ import annotations

import numpy as np

from trickster.games.ulti.adapter import UltiNode

# ---------------------------------------------------------------------------
#  Per-component value table
# ---------------------------------------------------------------------------

# Each component: (win_value, loss_value)
# These are additive — a "Parti + Ulti" contract evaluates both.
_COMPONENT_VALUES: dict[str, tuple[float, float]] = {
    "parti":      (1.0,  2.0),
    "ulti":       (4.0,  8.0),
    "betli":      (5.0, 10.0),
    "durchmars":  (6.0, 12.0),
    "40-100":     (4.0,  8.0),   # 40 + 100 combined
    "20-100":     (8.0, 16.0),   # 20 + 100 combined
}

# Maximum possible raw reward (for normalisation)
_MAX_RAW = 22.0


def _check_component_won(
    component: str,
    state: UltiNode,
) -> bool:
    """Did the soloist win a specific component?

    Each component has its own win condition evaluated from
    the terminal game state.
    """
    from trickster.games.ulti.game import (
        last_trick_ulti_check,
        soloist_lost_betli,
        soloist_won_durchmars,
        soloist_won_simple,
    )

    gs = state.gs
    if component == "parti":
        return soloist_won_simple(gs)
    if component == "ulti":
        side, won = last_trick_ulti_check(gs)
        return side == "soloist" and won
    if component == "betli":
        return not soloist_lost_betli(gs)
    if component == "durchmars":
        return soloist_won_durchmars(gs)
    if component in ("40-100", "20-100"):
        # 100 = soloist collected ≥100 card points
        # The 40/20 marriage part is auto-satisfied if they declared it
        return soloist_won_simple(gs)  # points > defender = auto 100+
    return False


# ---------------------------------------------------------------------------
#  Reward calculation
# ---------------------------------------------------------------------------


def calculate_reward(
    state: UltiNode,
    soloist_won: bool,
) -> tuple[float, float, float]:
    """Compute normalised per-component rewards for all 3 players.

    For compound contracts (e.g. ``{"parti", "ulti"}``), each component
    is evaluated independently:
      - Parti won + Ulti won  → full win (+5 for soloist)
      - Parti won + Ulti lost → partial (+1 parti, -8 ulti → net -7)
      - Parti lost + Ulti won → partial (-2 parti, +4 ulti → net +2)
      - Both lost             → full loss (-10 for soloist)

    The ``soloist_won`` parameter is used as a simple fallback when
    there's only one component or for non-compound contracts.

    Returns
    -------
    (r0, r1, r2) : tuple of float in [-1, +1]
    """
    gs = state.gs
    comps = state.contract_components
    soloist = gs.soloist
    raw_reward = 0.0

    if comps is None or len(comps) == 0:
        # Unknown contract — simple win/loss
        raw_reward = 1.0 if soloist_won else -2.0
    elif len(comps) == 1:
        # Single component — use simple evaluation
        comp = next(iter(comps))
        # Map compound keys like "40"+"100" to lookup key
        if comp == "betli":
            w, l = _COMPONENT_VALUES["betli"]
        elif comp == "parti":
            w, l = _COMPONENT_VALUES["parti"]
        elif comp == "durchmars":
            w, l = _COMPONENT_VALUES["durchmars"]
        else:
            w, l = (1.0, 2.0)
        raw_reward = w if soloist_won else -l
    else:
        # Multi-component: evaluate each independently
        # Build lookup keys from component set
        eval_components: list[str] = []
        if "parti" in comps:
            eval_components.append("parti")
        if "ulti" in comps:
            eval_components.append("ulti")
        if "durchmars" in comps:
            eval_components.append("durchmars")
        if "betli" in comps:
            eval_components.append("betli")
        if "100" in comps and "40" in comps:
            eval_components.append("40-100")
        elif "100" in comps and "20" in comps:
            eval_components.append("20-100")

        if not eval_components:
            raw_reward = 1.0 if soloist_won else -2.0
        else:
            for comp in eval_components:
                w, l = _COMPONENT_VALUES.get(comp, (1.0, 2.0))
                comp_won = _check_component_won(comp, state)
                raw_reward += w if comp_won else -l

    # Distribute to players
    rewards = [0.0, 0.0, 0.0]
    rewards[soloist] = raw_reward
    for i in range(3):
        if i != soloist:
            rewards[i] = -raw_reward / 2.0

    # Normalise to [-1, +1]
    rewards = [r / _MAX_RAW for r in rewards]
    rewards = [max(-1.0, min(1.0, r)) for r in rewards]
    return (rewards[0], rewards[1], rewards[2])


def outcome_for_player(
    state: UltiNode,
    player: int,
    soloist_won: bool,
) -> float:
    """Normalised reward for a single player."""
    r = calculate_reward(state, soloist_won)
    return r[player]


# ---------------------------------------------------------------------------
#  Simple outcome (used when we don't need contract scaling)
# ---------------------------------------------------------------------------


def simple_outcome(state: UltiNode, player: int) -> float:
    """Basic +1 / -1 outcome for curriculum training."""
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
