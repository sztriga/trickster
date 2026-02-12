"""Alpha-beta minimax solver for Ulti (pure-Python fallback).

Provides exact-play analysis for perfect-information deals, and a
PIMC (Perfect Information Monte Carlo) wrapper that samples
determinizations and solves each one exactly.

This module is the pure-Python fallback for the Cython solver in
``_solver_core.pyx``.  The Cython version is 50-100x faster and
supports pluggable contract evaluators (parti, betli, durchmars,
parti_ulti).  When the Cython extension is compiled, ``hybrid.py``
loads it automatically; otherwise it falls back to this module.

The solver treats the game as 2-player: the soloist MAXIMISES and
both defenders MINIMISE.  Alpha-beta pruning with context-sensitive
move ordering keeps the search tractable.

Optimisations:
  - **Context-sensitive move ordering**: soloist (MAX) tries trumps
    / Aces first; defenders (MIN) try ducking / cheap cards first.
  - **Score-bounds (futility) pruning**: prune when the soloist's
    max/min possible score can't affect the alpha-beta window.

Components:
  solve_root(gs)      - exact value for every legal move
  solve_best(gs)      - best move with full root pruning
  get_solve_stats()   - diagnostics from the last solve call
  SolverPIMC          - determinize + solve (used by the web API)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import (
    BETLI_STRENGTH,
    Card,
    LAST_TRICK_BONUS,
    NUM_PLAYERS,
    Rank,
    Suit,
    TRICKS_PER_GAME,
)
from trickster.games.ulti.game import (
    GameState,
    current_player,
    legal_actions,
)

_INF = float("inf")

# Points per card (indexed by Rank.value for speed)
_CARD_PTS: tuple[int, ...] = tuple(
    10 if r in (Rank.ACE, Rank.TEN) else 0 for r in Rank
)


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SolveStats:
    """Per-solve diagnostics."""

    nodes_explored: int = 0
    cutoffs: int = 0
    max_depth_reached: int = 0
    solve_time_ms: float = 0.0

    @property
    def pruning_ratio(self) -> float:
        """Ratio of cutoffs to nodes explored (higher = better pruning)."""
        if self.nodes_explored == 0:
            return 0.0
        return self.cutoffs / self.nodes_explored


# Module-level counters (fast — avoids object passing in hot recursion)
_nodes: int = 0
_cutoffs: int = 0
_last_stats: SolveStats = SolveStats()


def get_solve_stats() -> SolveStats:
    """Diagnostics from the most recent solve call."""
    return _last_stats


# ---------------------------------------------------------------------------
#  Inline trick resolution (avoids TrickResult allocation)
# ---------------------------------------------------------------------------


def _trick_winner(
    tc: list[tuple[int, Card]], trump: Optional[Suit], betli: bool,
) -> int:
    """Determine trick winner without allocating a TrickResult."""
    led_suit = tc[0][1].suit
    best_p = tc[0][0]
    best_c = tc[0][1]
    best_trump = trump is not None and best_c.suit == trump

    for p, c in tc[1:]:
        c_trump = trump is not None and c.suit == trump
        if c_trump and not best_trump:
            best_p, best_c, best_trump = p, c, True
        elif c_trump and best_trump:
            s = BETLI_STRENGTH[c.rank] if betli else c.rank.value
            bs = BETLI_STRENGTH[best_c.rank] if betli else best_c.rank.value
            if s > bs:
                best_p, best_c = p, c
        elif not c_trump and not best_trump:
            if c.suit == led_suit:
                if best_c.suit == led_suit:
                    s = BETLI_STRENGTH[c.rank] if betli else c.rank.value
                    bs = BETLI_STRENGTH[best_c.rank] if betli else best_c.rank.value
                    if s > bs:
                        best_p, best_c = p, c
                else:
                    best_p, best_c = p, c
    return best_p


# ---------------------------------------------------------------------------
#  Reversible apply / undo
# ---------------------------------------------------------------------------

# Undo token layout (tuple for speed):
#   Non-completing: (player, card, False)
#   Completing:     (player, card, True, tc0, tc1,
#                    prev_leader, prev_trick_no,
#                    prev_s0, prev_s1, prev_s2, winner)


def _apply(gs: GameState, card: Card) -> tuple:
    """Play *card* into the current trick.  Returns an undo token."""
    player = current_player(gs)
    gs.hands[player].remove(card)
    gs.trick_cards.append((player, card))

    if len(gs.trick_cards) < NUM_PLAYERS:
        return (player, card, False)

    # --- Trick complete ---
    tc0, tc1 = gs.trick_cards[0], gs.trick_cards[1]
    prev_leader = gs.leader
    prev_tno = gs.trick_no
    s0, s1, s2 = gs.scores[0], gs.scores[1], gs.scores[2]

    winner = _trick_winner(gs.trick_cards, gs.trump, gs.betli)

    pts = 0
    for _, c in gs.trick_cards:
        pts += c.points()
    gs.scores[winner] += pts

    for _, c in gs.trick_cards:
        gs.captured[winner].append(c)

    gs.trick_no += 1
    if gs.trick_no == TRICKS_PER_GAME:
        gs.scores[winner] += LAST_TRICK_BONUS

    gs.leader = winner
    gs.trick_cards = []

    return (player, card, True, tc0, tc1,
            prev_leader, prev_tno, s0, s1, s2, winner)


def _undo(gs: GameState, tok: tuple) -> None:
    """Reverse the effect of _apply."""
    player = tok[0]
    card = tok[1]
    completed = tok[2]

    if completed:
        tc0, tc1 = tok[3], tok[4]
        gs.trick_cards = [tc0, tc1]
        gs.leader = tok[5]
        gs.trick_no = tok[6]
        gs.scores[0] = tok[7]
        gs.scores[1] = tok[8]
        gs.scores[2] = tok[9]
        winner = tok[10]
        del gs.captured[winner][-NUM_PLAYERS:]
    else:
        gs.trick_cards.pop()

    gs.hands[player].append(card)


# ---------------------------------------------------------------------------
#  Terminal evaluation
# ---------------------------------------------------------------------------


def _terminal_value(gs: GameState) -> float:
    """Score at a terminal node, from the soloist's perspective.

    Parti:  soloist's card points (0-90).  Higher = better for soloist.
    Betli:  (10 - soloist_tricks).  10 = perfect Betli, 0 = took all.
    """
    if gs.betli:
        sol_tricks = len(gs.captured[gs.soloist]) // NUM_PLAYERS
        return float(TRICKS_PER_GAME - sol_tricks)
    return float(gs.scores[gs.soloist])


# ---------------------------------------------------------------------------
#  Move ordering (critical for alpha-beta efficiency)
# ---------------------------------------------------------------------------


def _ordered_moves(gs: GameState, maximising: bool) -> list[Card]:
    """Legal moves sorted for best-first search.

    Context-sensitive ordering maximises alpha-beta cutoffs:

    Maximiser (soloist): trumps first, then high-point cards (Aces,
    Tens), then high strength.  Aggressive play raises alpha fast.

    Minimiser (defenders): non-trumps first, low-point cards, low
    strength.  Ducking/cheap moves first produces moderate values
    that set initial bounds, allowing subsequent strong moves to
    generate tight cutoffs via progressive bound narrowing.
    """
    moves = legal_actions(gs)
    if len(moves) <= 1:
        return moves

    trump = gs.trump
    betli = gs.betli

    def _key(c: Card) -> tuple[int, int, int]:
        is_trump = 1 if (trump is not None and c.suit == trump) else 0
        strength = BETLI_STRENGTH[c.rank] if betli else c.rank.value
        points = c.points()
        return (is_trump, points, strength)

    moves.sort(key=_key, reverse=maximising)
    return moves


# ---------------------------------------------------------------------------
#  Fast greedy playout (for depth-limited evaluation)
# ---------------------------------------------------------------------------


def _greedy_playout(gs: GameState) -> float:
    """Evaluate by greedy playout: each player plays best legal card.

    Uses _apply/_undo so the GameState is restored afterwards.
    Only used when max_exact_tricks < TRICKS_PER_GAME (depth-limited mode).
    """
    undos: list[tuple] = []
    while gs.trick_no < TRICKS_PER_GAME:
        player = current_player(gs)
        moves = _ordered_moves(gs, player == gs.soloist)
        tok = _apply(gs, moves[0])
        undos.append(tok)

    value = _terminal_value(gs)

    for tok in reversed(undos):
        _undo(gs, tok)

    return value


# ---------------------------------------------------------------------------
#  Alpha-beta minimax
# ---------------------------------------------------------------------------


def _remaining_card_points(gs: GameState) -> int:
    """Points still in play (hands + current trick + possible last trick bonus).

    Fast computation used for futility pruning.
    """
    pts = 0
    for h in gs.hands:
        for c in h:
            pts += _CARD_PTS[c.rank.value]
    for _, c in gs.trick_cards:
        pts += _CARD_PTS[c.rank.value]
    # Last trick bonus goes to whoever wins the final trick
    if gs.trick_no < TRICKS_PER_GAME:
        pts += LAST_TRICK_BONUS
    return pts


def _alphabeta(
    gs: GameState, alpha: float, beta: float,
    exact_from_trick: int,
) -> float:
    """Recursive alpha-beta from soloist's perspective.

    Soloist maximises, both defenders minimise.

    When ``gs.trick_no < exact_from_trick``, the position is
    evaluated with a fast greedy playout instead of exact search.
    Set ``exact_from_trick=0`` for full-game exact solving.

    """
    global _nodes, _cutoffs
    _nodes += 1

    if gs.trick_no >= TRICKS_PER_GAME:
        return _terminal_value(gs)

    # Depth limit: use greedy playout for positions in the approximate zone.
    if exact_from_trick > 0:
        tc_len = len(gs.trick_cards)
        if gs.trick_no < exact_from_trick:
            if tc_len == 0 or gs.trick_no + 1 < exact_from_trick:
                return _greedy_playout(gs)

    # --- Score-bounds (futility) pruning ---
    if not gs.betli:
        sol_score = gs.scores[gs.soloist]
        remaining = _remaining_card_points(gs)
        max_possible = float(sol_score + remaining)
        min_possible = float(sol_score)

        # If the soloist can't possibly reach alpha even with everything:
        if max_possible <= alpha:
            return max_possible
        # If the soloist is already above beta even with nothing more:
        if min_possible >= beta:
            return min_possible

    player = current_player(gs)
    maximising = (player == gs.soloist)
    moves = _ordered_moves(gs, maximising)

    if not moves:
        return _terminal_value(gs)

    if maximising:
        # MAX node
        value = -_INF
        for card in moves:
            tok = _apply(gs, card)
            v = _alphabeta(gs, alpha, beta, exact_from_trick)
            _undo(gs, tok)
            if v > value:
                value = v
            if value > alpha:
                alpha = value
            if alpha >= beta:
                _cutoffs += 1
                break
        return value
    else:
        # MIN node (either defender)
        value = _INF
        for card in moves:
            tok = _apply(gs, card)
            v = _alphabeta(gs, alpha, beta, exact_from_trick)
            _undo(gs, tok)
            if v < value:
                value = v
            if value < beta:
                beta = value
            if alpha >= beta:
                _cutoffs += 1
                break
        return value


# ---------------------------------------------------------------------------
#  Public: solve from a position
# ---------------------------------------------------------------------------


def solve_root(
    gs: GameState,
    max_exact_tricks: int = TRICKS_PER_GAME,
    contract: str | None = None,
) -> dict[Card, float]:
    """Compute value for every legal move.

    Each child subtree gets an independent alpha-beta search with
    fresh bounds.  This guarantees exact values for all moves —
    required for PIMC score aggregation.

    Returns {card: value} where value is from soloist's perspective.

    Parameters
    ----------
    max_exact_tricks : number of tricks to solve exactly from the end.
        Default is TRICKS_PER_GAME (10) for full-game exact solving.
    contract : str or None
        Accepted for API compatibility with the Cython solver.
        The Python fallback always uses the built-in evaluation.
    """
    global _nodes, _cutoffs, _last_stats
    _nodes = 0
    _cutoffs = 0

    exact_from_trick = max(0, TRICKS_PER_GAME - max_exact_tricks)

    t0 = time.perf_counter()

    player = current_player(gs)
    maximising = (player == gs.soloist)
    moves = _ordered_moves(gs, maximising)
    values: dict[Card, float] = {}

    for card in moves:
        tok = _apply(gs, card)
        v = _alphabeta(gs, -_INF, _INF, exact_from_trick)
        _undo(gs, tok)
        values[card] = v

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Compute max depth from starting position
    remaining_plays = sum(len(h) for h in gs.hands) + len(gs.trick_cards)
    _last_stats = SolveStats(
        nodes_explored=_nodes,
        cutoffs=_cutoffs,
        max_depth_reached=remaining_plays,
        solve_time_ms=elapsed_ms,
    )

    return values


def solve_best(
    gs: GameState,
    max_exact_tricks: int = TRICKS_PER_GAME,
    contract: str | None = None,
) -> tuple[Card | None, float]:
    """Find the best move and its value for the current player.

    Uses alpha-beta pruning at the root for maximum efficiency.
    Returns (best_card, best_value_from_soloist_perspective).
    """
    global _nodes, _cutoffs, _last_stats
    _nodes = 0
    _cutoffs = 0

    exact_from_trick = max(0, TRICKS_PER_GAME - max_exact_tricks)

    t0 = time.perf_counter()

    player = current_player(gs)
    maximising = (player == gs.soloist)
    moves = _ordered_moves(gs, maximising)

    if not moves:
        return None, 0.0
    if len(moves) == 1:
        return moves[0], 0.0

    best_card = moves[0]

    if maximising:
        best_val = -_INF
        for card in moves:
            tok = _apply(gs, card)
            v = _alphabeta(gs, best_val, _INF, exact_from_trick)
            _undo(gs, tok)
            if v > best_val:
                best_val = v
                best_card = card
        result = (best_card, best_val)
    else:
        best_val = _INF
        for card in moves:
            tok = _apply(gs, card)
            v = _alphabeta(gs, -_INF, best_val, exact_from_trick)
            _undo(gs, tok)
            if v < best_val:
                best_val = v
                best_card = card
        result = (best_card, best_val)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    remaining_plays = sum(len(h) for h in gs.hands) + len(gs.trick_cards)
    _last_stats = SolveStats(
        nodes_explored=_nodes,
        cutoffs=_cutoffs,
        max_depth_reached=remaining_plays,
        solve_time_ms=elapsed_ms,
    )

    return result


def get_node_count() -> int:
    """Node count from the last solve call (backward compat)."""
    return _last_stats.nodes_explored


# ---------------------------------------------------------------------------
#  PIMC: determinize + solve for imperfect information
# ---------------------------------------------------------------------------


# Practical depth limit for PIMC (profiled: 6 exact tricks ≈ 217ms avg)
PIMC_DEFAULT_EXACT_TRICKS: int = 6


class SolverPIMC:
    """Perfect Information Monte Carlo with exact solving.

    For each move decision:
    1. Sample N determinizations (plausible opponent hands)
    2. Solve each one with alpha-beta (exact for last N tricks)
    3. Average the scores per legal move
    4. Pick the move with the best average score

    Default uses 6 exact tricks (profiled practical limit in pure
    Python).  Full 10-trick exact solving from trick 0 is infeasible
    in Python (~100s/solve) but works fine for endgame positions
    (≤6 tricks remaining).
    """

    def __init__(
        self,
        num_determinizations: int = 20,
        max_exact_tricks: int = PIMC_DEFAULT_EXACT_TRICKS,
        game: UltiGame | None = None,
    ) -> None:
        self.num_dets = num_determinizations
        self.max_exact_tricks = max_exact_tricks
        self._game = game or UltiGame()

    def choose_action(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> Card:
        """Pick the best card using PIMC + exact solving."""
        legal = self._game.legal_actions(state)
        if len(legal) <= 1:
            return legal[0]

        best, _scores = self.choose_action_with_scores(state, player, rng)
        return best

    def choose_action_with_scores(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> tuple[Card, dict[Card, float]]:
        """Like choose_action but also returns average scores per move."""
        legal = self._game.legal_actions(state)
        if len(legal) <= 1:
            return legal[0], {legal[0]: 0.0}

        totals: dict[Card, float] = {c: 0.0 for c in legal}
        counts: dict[Card, int] = {c: 0 for c in legal}

        for _ in range(self.num_dets):
            det = self._game.determinize(state, player, rng)
            card_values = solve_root(det.gs, self.max_exact_tricks)
            for card in legal:
                if card in card_values:
                    totals[card] += card_values[card]
                    counts[card] += 1

        # Average scores
        avg: dict[Card, float] = {
            c: totals[c] / max(1, counts[c]) for c in legal
        }

        # Soloist maximises, defenders minimise
        if player == state.gs.soloist:
            best = max(avg, key=avg.__getitem__)
        else:
            best = min(avg, key=avg.__getitem__)

        return best, avg


