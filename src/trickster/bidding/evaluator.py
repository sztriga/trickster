"""Hand evaluation using the soloist value head.

Given a 12-card hand (after talon pickup), evaluates all feasible
(contract, trump, discard) combinations and returns the expected
stakes for each.

The soloist value head (value_fc_sol) is trained on all game
positions including trick-0 pre-game states, predicting the
expected game-point payoff from the soloist's perspective.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from trickster.games.ulti.adapter import UltiGame, UltiNode, build_auction_constraints
from trickster.games.ulti.cards import Card, Rank, Suit
from trickster.games.ulti.game import (
    GameState,
    NUM_PLAYERS,
    declare_all_marriages,
    discard_talon,
    set_contract,
)
from trickster.model import UltiNetWrapper
from trickster.train_utils import _GAME_PTS_MAX

from .registry import CONTRACT_DEFS, ContractDef


# ---------------------------------------------------------------------------
#  Result types
# ---------------------------------------------------------------------------


@dataclass
class DiscardChoice:
    """Result of evaluating a single (contract, trump, discard) combo."""

    discard: tuple[Card, Card]
    value: float          # raw value head output (normalised)
    game_pts: float       # un-normalised per-defender game points


@dataclass
class ContractEval:
    """Result of evaluating a contract + best trump/discard for it."""

    contract_key: str
    trump: Suit | None    # None for betli
    is_piros: bool
    best_discard: DiscardChoice
    game_pts: float       # mean NN prediction across discards (for bid decision)
    stakes_pts: float     # same as game_pts (for bid threshold comparison)


# ---------------------------------------------------------------------------
#  Setup helpers
# ---------------------------------------------------------------------------

_GAME = UltiGame()


def _make_eval_state(
    gs: GameState,
    soloist: int,
    trump: Suit | None,
    discards: tuple[Card, Card],
    contract_def: ContractDef,
    is_piros: bool,
    dealer: int,
    initial_bidder: int = -1,
) -> UltiNode:
    """Build a UltiNode ready for value-head evaluation.

    Creates a deep copy of *gs*, applies discard + contract + marriages,
    and wraps in a UltiNode with the right metadata.
    """
    gs2 = copy.deepcopy(gs)
    gs2.soloist = soloist

    # Discard
    discard_talon(gs2, list(discards))

    # Contract
    betli = contract_def.is_betli
    set_contract(gs2, soloist, trump=trump, betli=betli)

    # Flags
    if contract_def.key == "ulti":
        gs2.has_ulti = True

    gs2.training_mode = contract_def.training_mode

    # Marriages (with restriction for 100-games)
    restrict = None
    if contract_def.key == "40-100":
        restrict = "40"
    elif contract_def.key == "20-100":
        restrict = "20"
    declare_all_marriages(gs2, soloist_marriage_restrict=restrict)

    # Contract components
    if betli:
        comps = frozenset({"betli"})
    else:
        comps: set[str] = {"parti"}
        if "ulti" in contract_def.key or contract_def.key == "ulti":
            comps.add("ulti")
        if "40" in contract_def.key:
            comps.update({"40", "100"})
        if "20" in contract_def.key:
            comps.update({"20", "100"})

    comps_frozen = frozenset(comps)

    constraints = build_auction_constraints(gs2, comps_frozen)
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())

    return UltiNode(
        gs=gs2,
        known_voids=empty_voids,
        bid_rank=1,
        is_red=is_piros,
        contract_components=comps_frozen,
        dealer=dealer,
        initial_bidder=initial_bidder,
        must_have=constraints,
    )


def _hand_has_kq(hand: Sequence[Card], suit: Suit) -> bool:
    """Check if hand contains both King and Queen of the given suit."""
    has_k = any(c.suit == suit and c.rank == Rank.KING for c in hand)
    has_q = any(c.suit == suit and c.rank == Rank.QUEEN for c in hand)
    return has_k and has_q


def _hand_has_trump7(hand: Sequence[Card], trump: Suit) -> bool:
    """Check if hand contains the trump 7 (needed for ulti)."""
    return Card(trump, Rank.SEVEN) in hand


# ---------------------------------------------------------------------------
#  Core evaluator
# ---------------------------------------------------------------------------


def evaluate_contract(
    gs: GameState,
    soloist: int,
    dealer: int,
    contract_def: ContractDef,
    trump: Suit | None,
    is_piros: bool,
    wrapper: UltiNetWrapper,
    max_discards: int = 20,
) -> ContractEval | None:
    """Evaluate a single contract + trump for the soloist's 12-card hand.

    Tries up to *max_discards* discard pairs (heuristically pruned),
    returns the best one.  Returns None if the contract is infeasible.
    """
    hand = gs.hands[soloist]
    assert len(hand) == 12, f"Expected 12-card hand, got {len(hand)}"

    # ── Feasibility checks ─────────────────────────────────────────
    if contract_def.is_betli:
        trump = None
    else:
        if trump is None:
            return None
        if contract_def.key == "40-100" and not _hand_has_kq(hand, trump):
            return None
        if contract_def.key == "ulti" and not _hand_has_trump7(hand, trump):
            return None

    # ── Generate candidate discards ────────────────────────────────
    all_discards = list(combinations(hand, 2))

    if len(all_discards) > max_discards and not contract_def.is_betli:
        # Heuristic pruning: prefer discarding low-value non-trump cards.
        # Also rewards discards that create voids (enabling future trumping).
        # Score each pair: lower is better to discard.
        hand_nontump_suits = set(c.suit for c in hand if c.suit != trump)

        def _discard_score(pair: tuple[Card, Card]) -> float:
            score = 0.0
            for c in pair:
                # Penalise discarding trump cards
                if c.suit == trump:
                    score += 100
                # Penalise discarding high-value cards (they score for defenders)
                score -= c.points()
                # Penalise discarding cards needed for contract
                if contract_def.key == "40-100":
                    if c == Card(trump, Rank.KING) or c == Card(trump, Rank.QUEEN):
                        score += 1000  # never discard these
                if contract_def.key == "ulti":
                    if c == Card(trump, Rank.SEVEN):
                        score += 1000
            # Reward void creation: discarding both cards of a side suit
            # lets us trump that suit later — a major strategic advantage.
            remaining_suits = set(
                c.suit for c in hand
                if c not in pair and c.suit != trump
            )
            new_voids = hand_nontump_suits - remaining_suits
            score -= len(new_voids) * 20
            return score

        all_discards.sort(key=_discard_score)
        all_discards = all_discards[:max_discards]

    # ── Batch evaluate all discards ────────────────────────────────
    feats_batch = []
    valid_discards: list[tuple[Card, Card]] = []

    for pair in all_discards:
        try:
            node = _make_eval_state(
                gs, soloist, trump, pair, contract_def, is_piros, dealer,
            )
        except (ValueError, AssertionError):
            continue
        feats = _GAME.encode_state(node, soloist)
        feats_batch.append(feats)
        valid_discards.append(pair)

    if not feats_batch:
        return None

    # Batch soloist-value-head inference
    states_np = np.stack(feats_batch)
    values = wrapper.batch_value(states_np)  # (N,) normalised values

    # Best discard (max) — used for actual card selection
    best_idx = int(np.argmax(values))
    best_val = float(values[best_idx])
    best_pts = best_val * _GAME_PTS_MAX / 2

    # Mean value — used for bid/no-bid decision.
    # Taking the max over N noisy predictions inflates the estimate;
    # the mean is unbiased and reflects the true hand strength.
    mean_val = float(np.mean(values))
    mean_pts = mean_val * _GAME_PTS_MAX / 2

    return ContractEval(
        contract_key=contract_def.key,
        trump=trump,
        is_piros=is_piros,
        best_discard=DiscardChoice(
            discard=valid_discards[best_idx],
            value=best_val,
            game_pts=best_pts,
        ),
        game_pts=mean_pts,
        stakes_pts=mean_pts,
    )


def evaluate_all_contracts(
    gs: GameState,
    soloist: int,
    dealer: int,
    wrappers: dict[str, UltiNetWrapper],
    max_discards: int = 20,
) -> list[ContractEval]:
    """Evaluate all feasible contracts for the soloist's 12-card hand.

    Parameters
    ----------
    gs : GameState with 12-card soloist hand (after talon pickup)
    soloist : player index
    dealer : dealer index
    wrappers : contract_key → UltiNetWrapper (one per trained contract)
    max_discards : max discard pairs to evaluate per (contract, trump)

    Returns
    -------
    List of ContractEval, sorted by stakes_pts descending (best first).
    """
    hand = gs.hands[soloist]
    suits_in_hand = list(set(c.suit for c in hand))
    results: list[ContractEval] = []

    for contract_key, wrapper in wrappers.items():
        cdef = CONTRACT_DEFS[contract_key]

        if cdef.is_betli:
            # Betli: no trump. Evaluate both normal and rebetli (piros).
            # The value head predicts different stakes for each because
            # is_red is encoded in the state features and piros is in
            # the training rewards.
            for is_piros in (False, True):
                ev = evaluate_contract(
                    gs, soloist, dealer, cdef,
                    trump=None, is_piros=is_piros,
                    wrapper=wrapper, max_discards=max_discards,
                )
                if ev is not None:
                    results.append(ev)
        elif cdef.piros_only:
            # Piros-only contract (e.g. Parti): only evaluate Hearts trump.
            # Plain (non-red) version cannot be played.
            ev = evaluate_contract(
                gs, soloist, dealer, cdef,
                trump=Suit.HEARTS, is_piros=True,
                wrapper=wrapper, max_discards=max_discards,
            )
            if ev is not None:
                results.append(ev)
        else:
            # Colored contracts: evaluate each possible trump suit.
            # Hearts = piros (2x stakes), others = normal.
            for suit in suits_in_hand:
                is_piros = (suit == Suit.HEARTS)

                ev = evaluate_contract(
                    gs, soloist, dealer, cdef,
                    trump=suit, is_piros=is_piros,
                    wrapper=wrapper, max_discards=max_discards,
                )
                if ev is not None:
                    results.append(ev)

    # Sort by stakes_pts (best first)
    results.sort(key=lambda e: e.stakes_pts, reverse=True)
    return results


def pick_best_bid(
    evals: list[ContractEval],
    min_stakes_pts: float = 0.0,
) -> ContractEval | None:
    """Pick the contract with the highest expected payoff.

    Compares by ``stakes_pts``, which is the expected per-defender
    payoff as predicted directly by the value head (including piros
    and kontra dynamics).  A confident rebetli beats a marginal parti.

    Parameters
    ----------
    evals : output of evaluate_all_contracts (sorted by stakes_pts desc)
    min_stakes_pts : minimum expected payoff to place a bid (vs pass)

    Returns
    -------
    Best ContractEval, or None if nothing beats the threshold (→ pass).
    """
    for ev in evals:
        if ev.stakes_pts >= min_stakes_pts:
            return ev
    return None
