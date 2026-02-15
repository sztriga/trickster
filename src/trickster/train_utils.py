"""Reward engine and training utilities for Ulti.

Core components:
  - ``simple_outcome``: per-component game-point scoring with kontra/piros
  - ``solver_value_to_reward``: continuous reward from endgame solver
  - ``ReplayBuffer``: reservoir-sampled replay buffer with outcome balancing
"""

from __future__ import annotations

import numpy as np

from trickster.games.ulti.adapter import UltiNode


# ---------------------------------------------------------------------------
#  Game-point outcome
# ---------------------------------------------------------------------------

# Normalisation constant for game-point rewards.
#
# Chosen so that common outcomes land in a useful range for the MSE
# value loss.  The piros multiplier (×2) is included in the reward,
# so the value head learns expected STAKES directly:
#
#   Non-piros Parti win:   +0.20     Piros Parti win:   +0.40
#   Non-piros Ulti win:    +1.00     Piros Ulti win:    +2.00
#   Betli win:             +1.00     Rebetli win:       +2.00
#
# Extreme combinations (piros + kontra + silents) can reach ±5 or
# beyond.  The value head is unbounded (no tanh) so it can represent
# the full range.  Gradient clipping (clip_grad_norm) in the training
# loop provides stability for these rare high-stakes outliers.
_GAME_PTS_MAX = 10


def simple_outcome(state: UltiNode, player: int) -> float:
    """Game-point reward normalised by _GAME_PTS_MAX.

    Scoring is driven by ``state.contract_components``.  Announced
    components use their full value; non-announced components can
    still trigger as *silent* bonuses at half value.

    Kontra multipliers are read from ``state.component_kontras``
    and applied to announced components.  Bukott (fallen) ulti uses
    the special formula: ``win_value × (2^k + 1)`` instead of the
    standard ``loss_value × 2^k``.  Silent bonuses are unaffected
    by kontra.

    The **piros multiplier** (×2 for red games) is applied to ALL
    components so the value head directly learns the expected stakes.
    This eliminates the need for external piros multiplication in the
    bidding evaluator and allows the AI to learn the asymmetric risk
    of piros games (e.g. a marginal piros hand that gets kontrad).

    Announced component values (per defender, win / loss):
      Parti:    +1 / −1          (only when no 100 is announced)
      40-100:   +2 / −4          (replaces Parti)
      20-100:   +4 / −8          (replaces Parti)
      Ulti:     +4 / −8          (bukott: 4 × (2^k + 1))
      Betli:    +5 / −5
      Durchmars: +3 / −6         (for future use)

    Silent values are half the announced equivalents:
      Silent 40-100:  ±2   (replaces Parti)
      Silent 20-100:  ±4   (replaces Parti)
      Silent Ulti:    ±2   / Fallen Ulti: ±4
      Silent Durchmars: ±3

    A component already announced is never re-scored as silent.
    """
    from trickster.games.ulti.game import (
        defender_has_20,
        defender_has_40,
        defender_points,
        defender_won_durchmars,
        last_trick_ulti_check,
        soloist_has_20,
        soloist_has_40,
        soloist_lost_betli,
        soloist_points,
        soloist_won_durchmars,
        soloist_won_simple,
    )

    gs = state.gs
    comps = state.contract_components or frozenset()
    kontras = state.component_kontras  # dict[str, int] or empty dict
    piros_mult = 2.0 if state.is_red else 1.0

    # ── Betli: standalone, no silent bonuses ───────────────────────
    if gs.betli:
        soloist_wins = not soloist_lost_betli(gs)
        k = kontras.get("betli", 0)
        mult = 2 ** k
        raw_per_def = (5.0 if soloist_wins else -5.0) * mult * piros_mult
        if player == gs.soloist:
            return (raw_per_def * 2) / _GAME_PTS_MAX
        else:
            return -raw_per_def / _GAME_PTS_MAX

    # Per-defender game points from soloist's perspective
    sol_pts = 0.0
    announced_100 = "100" in comps

    # Kontra helpers — map the kontrable unit to its kontra level.
    # 40-100 and 20-100 are separate kontrable units; parti is its own.
    def _kontra_mult(unit: str) -> int:
        return 2 ** kontras.get(unit, 0)

    # ── Base game: announced 100 OR Parti (+ silent 100) ──────────
    if announced_100:
        # Announced 40-100 or 20-100 — replaces Parti.
        sol_won_100 = soloist_points(gs) >= 100
        if "20" in comps:
            m = _kontra_mult("20-100")
            sol_pts += (4.0 if sol_won_100 else -8.0) * m
        else:  # 40-100
            m = _kontra_mult("40-100")
            sol_pts += (2.0 if sol_won_100 else -4.0) * m
    else:
        # Parti is the base.  Silent 100 can replace it.
        parti_won = soloist_won_simple(gs)
        m_parti = _kontra_mult("parti")

        sol_100 = soloist_points(gs) >= 100
        sol_base = 0.0
        if sol_100 and soloist_has_20(gs):
            sol_base = 4.0       # soloist silent 20-100
        elif sol_100 and soloist_has_40(gs):
            sol_base = 2.0       # soloist silent 40-100

        def_100 = defender_points(gs) >= 100
        def_base = 0.0
        if def_100 and defender_has_20(gs):
            def_base = 4.0       # defender silent 20-100
        elif def_100 and defender_has_40(gs):
            def_base = 2.0       # defender silent 40-100

        if sol_base > 0 or def_base > 0:
            # Silent 100 replaces parti — inherits parti kontra
            sol_pts += (sol_base - def_base) * m_parti
        else:
            sol_pts += (1.0 if parti_won else -1.0) * m_parti

    # ── Ulti component ─────────────────────────────────────────────
    announced_ulti = "ulti" in comps
    side, ulti_won = last_trick_ulti_check(gs)

    if announced_ulti:
        k = kontras.get("ulti", 0)
        mult = 2 ** k
        if side == "soloist" and ulti_won:
            sol_pts += 4.0 * mult
        else:
            # Bukott ulti: loss = win_value × (2^k + 1)
            # No kontra: 4×2=8, Kontra: 4×3=12, Rekontra: 4×5=20
            sol_pts -= 4.0 * (mult + 1)
    else:
        # Silent: only scores when trump 7 was played on last trick.
        # Silent bonuses are NOT affected by kontra.
        if side == "soloist":
            sol_pts += 2.0 if ulti_won else -4.0
        elif side == "defender":
            sol_pts += -2.0 if ulti_won else 4.0

    # ── Durchmars component ────────────────────────────────────────
    # (Currently always silent — announced durchmars is for future use)
    if soloist_won_durchmars(gs):
        sol_pts += 3.0
    if defender_won_durchmars(gs):
        sol_pts -= 3.0

    # ── Apply piros multiplier ─────────────────────────────────────
    sol_pts *= piros_mult

    # ── Distribute to player ──────────────────────────────────────
    if player == gs.soloist:
        raw = sol_pts * 2    # soloist pays/receives from 2 defenders
    else:
        raw = -sol_pts       # each defender is the mirror

    return raw / _GAME_PTS_MAX


def solver_value_to_reward(
    solver_val: float,
    player: int,
    gs,
) -> float:
    """Convert a PIMC solver position value to a player reward.

    The solver returns values from the **soloist's perspective**:
      - Parti:  soloist's card points (0–90, continuous)
      - Betli:  0.0 (lost) or 10.0 (won), continuous after PIMC avg
      - Durchmars: 0.0 (lost) or 10.0 (won), continuous after PIMC avg

    This function normalises the solver value to the same scale
    as ``simple_outcome`` (using _GAME_PTS_MAX), providing a
    continuous reward signal.  The "all-seeing teacher" approach:
    the solver has perfect information and labels positions accurately,
    while the network learns to predict the expected value given only
    partial observations.

    Parameters
    ----------
    solver_val : float
        Position value from the soloist's perspective, as returned by
        ``HybridPlayer.choose_action_with_policy``.
    player : int
        The player index whose reward we want.
    gs : GameState
        The game state (for contract type, trump, soloist index).
    """
    # ── Normalise to soloist advantage ∈ [-1, +1] ──────────────────
    if gs.betli:
        # Betli: 10.0 = perfect (0 tricks), 0.0 = lost (took tricks)
        # Midpoint = 5.0 → break-even probability
        advantage = (solver_val - 5.0) / 5.0
    else:
        # Parti: 0–90 card points, 45 = break-even
        advantage = (solver_val - 45.0) / 45.0

    advantage = max(-1.0, min(1.0, advantage))  # clamp

    # ── Scale to match simple_outcome range ──────────────────────
    # Note: solver_value_to_reward is used by single-contract training
    # scripts which don't use piros.  Piros scaling is handled by
    # simple_outcome in the bidding training loop.
    if player == gs.soloist:
        return advantage * 2 / _GAME_PTS_MAX   # ×2 for 2 defenders
    else:
        return -advantage / _GAME_PTS_MAX


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
        on_policy : (B,) bool
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
            self._on_policy[indices].copy(),
        )

