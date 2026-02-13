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
#  Game-point outcome (matches UltiGame.outcome)
# ---------------------------------------------------------------------------

# Max game-point outcome (piros Parti soloist win: 2 × 2 defenders)
_GAME_PTS_MAX = 4


def simple_outcome(state: UltiNode, player: int) -> float:
    """Game-point reward normalised to [-1, +1].

    Parti: soloist collects/pays 1 point per defender (2 total).
    Piros (Hearts trump): all stakes doubled (4 total).
    Normalised by _GAME_PTS_MAX so piros soloist maps to +/-1.0.
    """
    from trickster.games.ulti.cards import Suit
    from trickster.games.ulti.game import soloist_lost_betli, soloist_won_simple

    gs = state.gs
    if gs.betli:
        soloist_wins = not soloist_lost_betli(gs)
    else:
        soloist_wins = soloist_won_simple(gs)

    is_red = gs.trump is not None and gs.trump == Suit.HEARTS
    stake = 2 if is_red else 1  # per-defender stake

    if player == gs.soloist:
        raw = (stake * 2) if soloist_wins else -(stake * 2)
    else:
        raw = -stake if soloist_wins else stake

    return raw / _GAME_PTS_MAX


# ---------------------------------------------------------------------------
#  Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity replay buffer with reservoir sampling and outcome-balanced batching.

    Uses **reservoir sampling** (Algorithm R, Vitter 1985) instead of
    FIFO eviction.  Every sample ever pushed has an equal probability
    of being retained, preserving diversity across training history.
    This is what the NFSP paper recommends for the average strategy
    memory (Lanctot et al. 2017).

    Sampling strategy (prevents self-play collapse):
      1. Soloist experiences are sampled ``soloist_weight`` times more
         frequently than defender experiences.
      2. Each mini-batch guarantees at least ``min_positive_frac`` of
         its samples come from *winning* soloist outcomes (reward > 0).
    """

    def __init__(
        self,
        capacity: int = 50_000,
        soloist_weight: float = 3.0,
        min_positive_frac: float = 0.15,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.soloist_weight = soloist_weight
        self.min_positive_frac = min_positive_frac

        self._size = 0
        self._total_seen = 0
        self._allocated = False
        self._states: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self._policies: np.ndarray | None = None
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._is_soloist = np.zeros(capacity, dtype=bool)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Outcome-balanced mini-batch sampling.

        Guarantees a minimum fraction of soloist-win samples in each
        batch to prevent self-play collapse.  Falls back to pure
        weighted sampling when there aren't enough winning samples.

        Returns
        -------
        states : (B, state_dim)
        masks : (B, action_dim)
        policies : (B, action_dim)
        rewards : (B,)
        is_soloist : (B,) bool
        """
        n = self._size
        B = min(batch_size, n)

        sol = self._is_soloist[:n]
        rew = self._rewards[:n]

        # Identify soloist-win indices (the scarce, valuable samples)
        sol_win_mask = sol & (rew > 0)
        sol_win_idx = np.where(sol_win_mask)[0]
        other_idx = np.where(~sol_win_mask)[0]

        # Guarantee a minimum number of soloist-win samples per batch
        min_sol_win = max(1, int(B * self.min_positive_frac))

        if len(sol_win_idx) >= min_sol_win and len(other_idx) >= (B - min_sol_win):
            # Stratified draw: reserve min_sol_win slots for soloist wins
            sw_chosen = rng.choice(
                sol_win_idx, size=min_sol_win, replace=False,
            )

            # Fill remaining slots from other samples with role weighting
            remaining = B - min_sol_win
            other_weights = np.where(
                self._is_soloist[other_idx], self.soloist_weight, 1.0,
            )
            other_weights /= other_weights.sum()
            ot_chosen = rng.choice(
                other_idx, size=remaining, replace=False, p=other_weights,
            )

            indices = np.concatenate([sw_chosen, ot_chosen])
        else:
            # Not enough soloist-win samples — fall back to weighted sampling
            weights = np.where(sol, self.soloist_weight, 1.0)
            weights /= weights.sum()
            indices = rng.choice(n, size=B, replace=False, p=weights)

        return (
            self._states[indices].copy(),
            self._masks[indices].copy(),
            self._policies[indices].copy(),
            self._rewards[indices].copy(),
            self._is_soloist[indices].copy(),
        )


# ---------------------------------------------------------------------------
#  Checkpoint pool with PFSP opponent selection
# ---------------------------------------------------------------------------


class CheckpointPool:
    """Pool of frozen model checkpoints for league-style training.

    Stores snapshots of model weights taken at intervals.  During
    self-play the soloist uses the latest network while defenders
    are sampled from this pool.

    Implements **Prioritized Fictitious Self-Play** (PFSP):
    opponents are selected with probability weighted toward a ~50%
    win rate against the current agent — neither too easy nor too
    hard.  Based on Vinyals et al., "Grandmaster level in StarCraft
    II using multi-agent reinforcement learning" (Nature, 2019).
    """

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._pool: list[dict] = []

    def __len__(self) -> int:
        return len(self._pool)

    def add(self, step: int, state_dict: dict) -> None:
        """Snapshot current weights into the pool."""
        entry = {
            "step": step,
            "state_dict": {k: v.cpu().clone() for k, v in state_dict.items()},
            "games": 0,
            "wins": 0,  # current agent's soloist wins against this opponent
        }
        self._pool.append(entry)

        if len(self._pool) > self.max_size:
            # Evict the checkpoint with WR furthest from 50%
            worst_idx = 0
            worst_dist = -1.0
            for i, cp in enumerate(self._pool):
                if cp["games"] < 5:
                    continue
                wr = cp["wins"] / cp["games"]
                dist = abs(wr - 0.5)
                if dist > worst_dist:
                    worst_dist = dist
                    worst_idx = i
            self._pool.pop(worst_idx)

    def select(self, rng) -> dict | None:
        """Select an opponent via PFSP weighting.

        New/untested checkpoints get uniform weight (exploration).
        Tested checkpoints are weighted by a bell curve peaking at
        50% win rate.
        """
        if not self._pool:
            return None

        weights = []
        for cp in self._pool:
            if cp["games"] < 5:
                w = 1.0  # explore new checkpoints
            else:
                wr = cp["wins"] / cp["games"]
                # Bell curve: 1.0 at WR=50%, ~0 at WR=0% or 100%
                w = max(0.01, 1.0 - 4.0 * (wr - 0.5) ** 2)
            weights.append(w)

        return rng.choices(self._pool, weights=weights, k=1)[0]

    def update(self, entry: dict, current_won: bool) -> None:
        """Record a game result for PFSP weight updates."""
        entry["games"] += 1
        if current_won:
            entry["wins"] += 1

    def summary(self) -> str:
        """One-line summary for logging."""
        if not self._pool:
            return "empty"
        parts = []
        for cp in self._pool:
            wr = cp["wins"] / max(1, cp["games"])
            parts.append(f"s{cp['step']}:{wr:.0%}({cp['games']}g)")
        return " ".join(parts)
