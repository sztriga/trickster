"""Ulti-specific reward and scoring functions.

Core components:
  - ``simple_outcome``: per-component game-point scoring with kontra/piros
  - ``solver_value_to_reward``: continuous reward from endgame solver

These functions convert Ulti game results into scalar training targets
for the value head.  They are Ulti-specific (card points, betli, kontra)
and live under ``games.ulti`` rather than the generic ``train_utils``.
"""
from __future__ import annotations

from trickster.games.ulti.adapter import UltiNode

# Normalisation constant for game-point rewards.
_GAME_PTS_MAX = 10


def simple_outcome(state: UltiNode, player: int) -> float:
    """Game-point reward normalised by _GAME_PTS_MAX.

    Scoring is driven by ``state.contract_components``.  Announced
    components use their full value; non-announced components can
    still trigger as *silent* bonuses at half value.

    Kontra multipliers are read from ``state.component_kontras``
    and applied to announced components.  Bukott (fallen) ulti uses
    the special formula: ``win_value * (2^k + 1)`` instead of the
    standard ``loss_value * 2^k``.  Silent bonuses are unaffected
    by kontra.

    The **piros multiplier** (x2 for red games) is applied to ALL
    components so the value head directly learns the expected stakes.
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
    kontras = state.component_kontras
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

    sol_pts = 0.0
    announced_100 = "100" in comps

    def _kontra_mult(unit: str) -> int:
        return 2 ** kontras.get(unit, 0)

    # ── Base game: announced 100 OR Parti (+ silent 100) ──────────
    if announced_100:
        sol_won_100 = soloist_points(gs) >= 100
        if "20" in comps:
            m = _kontra_mult("20-100")
            sol_pts += (4.0 if sol_won_100 else -8.0) * m
        else:
            m = _kontra_mult("40-100")
            sol_pts += (2.0 if sol_won_100 else -4.0) * m
    else:
        parti_won = soloist_won_simple(gs)
        m_parti = _kontra_mult("parti")

        sol_100 = soloist_points(gs) >= 100
        sol_base = 0.0
        if sol_100 and soloist_has_20(gs):
            sol_base = 4.0
        elif sol_100 and soloist_has_40(gs):
            sol_base = 2.0

        def_100 = defender_points(gs) >= 100
        def_base = 0.0
        if def_100 and defender_has_20(gs):
            def_base = 4.0
        elif def_100 and defender_has_40(gs):
            def_base = 2.0

        if sol_base > 0 or def_base > 0:
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
            sol_pts -= 4.0 * (mult + 1)
    else:
        if side == "soloist":
            sol_pts += 2.0 if ulti_won else -4.0
        elif side == "defender":
            sol_pts += -2.0 if ulti_won else 4.0

    # ── Durchmars component ────────────────────────────────────────
    if soloist_won_durchmars(gs):
        sol_pts += 3.0
    if defender_won_durchmars(gs):
        sol_pts -= 3.0

    # ── Apply piros multiplier ─────────────────────────────────────
    sol_pts *= piros_mult

    # ── Distribute to player ──────────────────────────────────────
    if player == gs.soloist:
        raw = sol_pts * 2
    else:
        raw = -sol_pts

    return raw / _GAME_PTS_MAX


def solver_value_to_reward(
    solver_val: float,
    player: int,
    gs,
) -> float:
    """Convert a PIMC solver position value to a player reward.

    The solver returns values from the **soloist's perspective**:
      - Parti:  soloist's card points (0-90, continuous)
      - Betli:  0.0 (lost) or 10.0 (won), continuous after PIMC avg

    This function normalises the solver value to the same scale
    as ``simple_outcome`` (using _GAME_PTS_MAX).
    """
    if gs.betli:
        advantage = (solver_val - 5.0) / 5.0
    else:
        advantage = (solver_val - 45.0) / 45.0

    advantage = max(-1.0, min(1.0, advantage))

    if player == gs.soloist:
        return advantage * 2 / _GAME_PTS_MAX
    else:
        return -advantage / _GAME_PTS_MAX
