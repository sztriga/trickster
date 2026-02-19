"""Shared auction logic for training and evaluation.

Both ``bidding_loop.py`` (training) and ``eval_bidding.py`` (evaluation)
import from this module so that the bidding/game-setup code is identical.

Key design decision — **blind pickup**: in real Ulti the talon is
face-down.  A player must commit to picking up *before* seeing the
cards.  The pickup decision is driven by **constrained Monte Carlo**:
sample K possible talons from the unknown card pool, evaluate each
12-card hand with the value head, and compare E[pickup payoff] vs
E[defender payoff].
"""
from __future__ import annotations

import copy
import random as _random
from dataclasses import dataclass
from typing import Sequence

from trickster.bidding.evaluator import (
    ContractEval,
    DiscardChoice,
    _make_eval_state,
    evaluate_all_contracts,
)
from trickster.bidding.registry import (
    BID_TO_CONTRACT,
    CONTRACT_DEFS,
    MAX_SUPPORTED_RANK,
    SUPPORTED_BID_RANKS,
)
from trickster.games.ulti.adapter import UltiGame, UltiNode, build_auction_constraints
from trickster.games.ulti.auction import (
    AuctionState,
    BID_BY_RANK,
    BID_PASSZ,
    can_pickup,
    create_auction,
    legal_bids,
    submit_bid,
    submit_pass,
    submit_pickup,
)
from trickster.games.ulti.cards import Card, Rank, Suit, make_deck
from trickster.games.ulti.game import GameState, next_player
from trickster.model import UltiNetWrapper
from trickster.train_utils import _GAME_PTS_MAX

_ALL_CARDS: frozenset[Card] = frozenset(make_deck())


# ---------------------------------------------------------------------------
#  Reverse mapping: (contract_key, is_piros) → bid rank
# ---------------------------------------------------------------------------

CONTRACT_TO_BID_RANK: dict[tuple[str, bool], int] = {
    v: k for k, v in BID_TO_CONTRACT.items()
}


# ---------------------------------------------------------------------------
#  Result type
# ---------------------------------------------------------------------------


@dataclass
class AuctionResult:
    """Outcome of :func:`run_auction`."""

    soloist: int
    bid: ContractEval | None   # None ⇒ all-pass (Passz penalty)
    auction: AuctionState
    initial_bidder: int = -1   # first player to pick up the talon


# ---------------------------------------------------------------------------
#  MC-based pickup decision (replaces heuristic)
# ---------------------------------------------------------------------------

# Default number of talon samples for pickup evaluation.
_PICKUP_MC_SAMPLES = 8
# Max discards per (contract, trump) during pickup MC evaluation.
# Kept low for speed since we're evaluating many talons.
_PICKUP_MAX_DISCARDS = 8


def estimate_pickup_value(
    gs: GameState,
    player: int,
    dealer: int,
    wrappers: dict[str, UltiNetWrapper],
    auction: AuctionState,
    num_samples: int = _PICKUP_MC_SAMPLES,
    max_discards: int = _PICKUP_MAX_DISCARDS,
    rng: _random.Random | None = None,
) -> bool:
    """Decide whether to pick up the talon using Monte Carlo evaluation.

    Samples *num_samples* possible 2-card talons from the unknown card
    pool (constrained by auction info).  For each sample, evaluates
    only contracts that map to **legal overbids** — there's no point
    considering a contract the player can't actually bid.

    **E[pickup]** = average best legal-overbid value across MC samples.
    **Threshold** = 0 for non-Passz (only pick up if positive expected
    payoff as soloist), pass penalty for Passz (defenders are already
    getting paid if nobody bids).

    The value head drives the entire decision — no heuristics.
    """
    if rng is None:
        rng = _random.Random()

    hand = gs.hands[player]
    current_rank = auction.current_bid.rank if auction.current_bid else 0

    # Threshold: what do I need to beat by picking up?
    # Against Passz: defenders get +2 pts each, so need E[pickup] > that.
    # Against real bids: only pick up if I expect positive soloist payoff.
    if current_rank <= BID_PASSZ.rank:
        threshold = 2.0 * 2 / _GAME_PTS_MAX  # normalised pass penalty
    else:
        threshold = 0.0

    # Build the unknown card pool, constrained by auction info.
    known = set(hand)
    if auction.current_bid is not None:
        bid_info = BID_TO_CONTRACT.get(auction.current_bid.rank)
        if bid_info is not None:
            contract_key, is_piros = bid_info
            cdef = CONTRACT_DEFS.get(contract_key)
            if cdef is not None and not cdef.is_betli:
                _comps: set[str] = {"parti"}
                if "ulti" in contract_key:
                    _comps.add("ulti")
                if "40" in contract_key:
                    _comps.update({"40", "100"})
                constraints = build_auction_constraints(gs, frozenset(_comps))
                for cards in constraints.values():
                    known.update(cards)

    unknown = [c for c in _ALL_CARDS if c not in known]
    if len(unknown) < 2:
        return False

    # Pre-compute which bid ranks are legal overbids.
    legal_ranks = {b.rank for b in legal_bids(auction)}

    pickup_values: list[float] = []
    original_hand = gs.hands[player]
    for _ in range(num_samples):
        rng.shuffle(unknown)
        sampled_talon = unknown[:2]

        # Temporarily give the player a 12-card hand.
        # evaluate_all_contracts → _make_eval_state deepcopies gs
        # internally, so the original state is never mutated.
        gs.hands[player] = list(hand) + list(sampled_talon)

        evals = evaluate_all_contracts(
            gs, player, dealer,
            wrappers=wrappers,
            max_discards=max_discards,
        )

        # Filter to only contracts that map to legal overbids.
        best_legal_pts = None
        for ev in evals:
            bid_rank = CONTRACT_TO_BID_RANK.get(
                (ev.contract_key, ev.is_piros),
            )
            if bid_rank is not None and bid_rank in legal_ranks:
                best_legal_pts = ev.game_pts * 2 / _GAME_PTS_MAX
                break  # evals sorted desc — first legal match is best

        if best_legal_pts is not None:
            pickup_values.append(best_legal_pts)
        # If no legal overbid found for this talon, skip it —
        # don't penalise, just fewer samples.

    # Restore original hand.
    gs.hands[player] = original_hand

    if not pickup_values:
        return False

    e_pickup = sum(pickup_values) / len(pickup_values)
    return e_pickup > threshold


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _eval_to_auction_bid(
    ev: ContractEval,
    auction: AuctionState,
) -> tuple | None:
    """Convert a *ContractEval* to ``(Bid, discards)`` if legal."""
    bid_rank = CONTRACT_TO_BID_RANK.get((ev.contract_key, ev.is_piros))
    if bid_rank is None or bid_rank not in SUPPORTED_BID_RANKS:
        return None
    bid_obj = BID_BY_RANK.get(bid_rank)
    if bid_obj is None:
        return None
    if bid_obj not in legal_bids(auction):
        return None
    return bid_obj, list(ev.best_discard.discard)


def best_legal_bid(
    gs: GameState,
    player: int,
    dealer: int,
    wrappers: dict[str, UltiNetWrapper],
    auction: AuctionState,
    min_bid_pts: float,
    max_discards: int,
) -> tuple[ContractEval, object, list[Card], list[ContractEval]] | None:
    """Evaluate contracts and return the best *legal* auction bid.

    Returns ``(ContractEval, Bid, discards, all_evals)`` or ``None``
    if nothing exceeds the profitability threshold.  The full eval
    list is returned so callers can use the NN-chosen discard even
    for Passz / forced bids.
    """
    evals = evaluate_all_contracts(
        gs, player, dealer,
        wrappers=wrappers,
        max_discards=max_discards,
    )
    if not evals:
        return None
    for ev in evals:
        if ev.stakes_pts < min_bid_pts:
            break  # sorted desc — rest are worse
        r = _eval_to_auction_bid(ev, auction)
        if r is not None:
            bid_obj, discards = r
            return ev, bid_obj, discards, evals
    return None


def _nn_discard(evals: list[ContractEval]) -> list[Card]:
    """Pick the best NN-evaluated discard from a list of contract evals.

    Uses the top evaluation's discard — the NN's best judgement of
    which 2 cards to remove, even if no contract is profitable.
    """
    return list(evals[0].best_discard.discard)


def _infer_trump(hand: Sequence[Card], is_piros: bool) -> Suit | None:
    """Pick a trump suit for a forced (fallback) bid."""
    if is_piros:
        return Suit.HEARTS
    suit_counts: dict[Suit, int] = {}
    for c in hand:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    best = max(
        (s for s in suit_counts if s != Suit.HEARTS),
        key=lambda s: suit_counts[s],
        default=None,
    )
    return best or Suit.HEARTS


def _fallback_discards(hand: list[Card]) -> list[Card]:
    """Pick 2 weakest cards to discard (fallback when NN eval failed)."""
    return sorted(hand, key=lambda c: (c.points(), c.strength()))[:2]


def _forced_bid_eval(
    hand: list[Card],
    bid_obj,
) -> ContractEval | None:
    """Build a minimal ContractEval for a forced (unprofitable) bid."""
    contract_info = BID_TO_CONTRACT.get(bid_obj.rank)
    if contract_info is None:
        return None
    contract_key, is_piros = contract_info
    trump = _infer_trump(hand, is_piros)
    discards = _fallback_discards(hand)
    return ContractEval(
        contract_key=contract_key,
        trump=trump,
        is_piros=is_piros,
        best_discard=DiscardChoice(
            discard=tuple(discards), value=0.0, game_pts=0.0,
        ),
        game_pts=0.0,
        stakes_pts=0.0,
    )


# ---------------------------------------------------------------------------
#  Core auction runner
# ---------------------------------------------------------------------------


def run_auction(
    gs: GameState,
    talon: list[Card],
    dealer: int,
    seat_wrappers: list[dict[str, UltiNetWrapper]],
    *,
    min_bid_pts: float = 0.5,
    max_discards: int = 15,
) -> AuctionResult:
    """Run a competitive 3-player auction.

    Mutates ``gs.hands`` to reflect talon pickups and discards.
    On return the soloist has 10 cards (final discard already applied).

    Parameters
    ----------
    gs : GameState
        Must have 10-card hands for all players.
    talon : list[Card]
        The 2 initial talon cards.
    dealer : int
        Dealer seat index.
    seat_wrappers : list of 3 dicts
        ``seat_wrappers[seat]`` maps contract_key → UltiNetWrapper.
        For self-play pass the same dict three times.
    min_bid_pts : float
        Minimum expected per-defender stakes to place a bid.
    max_discards : int
        Max discard pairs evaluated per (contract, trump) combination.
    """
    first_bidder = next_player(dealer)

    # First bidder picks up the talon → 12 cards.
    gs.hands[first_bidder].extend(talon)
    a = create_auction(first_bidder, talon)

    winning_eval: ContractEval | None = None

    while not a.done:
        player = a.turn

        # Auto-pass when current bid is at or above the highest we support.
        if (
            a.current_bid is not None
            and a.current_bid.rank >= MAX_SUPPORTED_RANK
            and not a.awaiting_bid
        ):
            submit_pass(a, player)
            continue

        if a.awaiting_bid:
            # ── Player has 12 cards → evaluate contracts and bid ──
            wrappers = seat_wrappers[player]
            result = best_legal_bid(
                gs, player, dealer, wrappers, a,
                min_bid_pts, max_discards,
            ) if wrappers else None

            if result is not None:
                ev, bid_obj, discards, _all_evals = result
                for c in discards:
                    gs.hands[player].remove(c)
                submit_bid(a, player, bid_obj, discards)
                winning_eval = ev

            elif a.current_bid is None:
                # First bidder must bid at least Passz.
                # Use NN-evaluated discard: run evaluate_all_contracts
                # to find the best discard even though no bid is
                # profitable.  The NN decides which cards are least
                # harmful to put in the talon (avoid enabling opponent
                # ulti, keep defender-unfriendly cards, etc.).
                nn_evals = evaluate_all_contracts(
                    gs, player, dealer,
                    wrappers=wrappers if wrappers else {},
                    max_discards=max_discards,
                ) if wrappers else []

                if nn_evals:
                    discards = _nn_discard(nn_evals)
                else:
                    discards = _fallback_discards(gs.hands[player])
                for c in discards:
                    gs.hands[player].remove(c)
                submit_bid(a, player, BID_PASSZ, discards)
                winning_eval = None

            else:
                # Picked up but can't overbid profitably — forced bid.
                # Use NN discards from evaluate_all_contracts.
                nn_evals = evaluate_all_contracts(
                    gs, player, dealer,
                    wrappers=wrappers if wrappers else {},
                    max_discards=max_discards,
                ) if wrappers else []

                legal = legal_bids(a)
                supported = [
                    b for b in legal if b.rank in SUPPORTED_BID_RANKS
                ]

                if supported and nn_evals:
                    # Use the NN's best evaluation to pick both the
                    # forced contract and discards.
                    bid_obj = supported[0]
                    # Find an eval that matches this forced bid, or use
                    # the best eval's discard.
                    contract_info = BID_TO_CONTRACT.get(bid_obj.rank)
                    matched_eval = None
                    if contract_info:
                        ck, ip = contract_info
                        for ev in nn_evals:
                            if ev.contract_key == ck and ev.is_piros == ip:
                                matched_eval = ev
                                break
                    if matched_eval is None:
                        matched_eval = nn_evals[0]

                    discards = list(matched_eval.best_discard.discard)
                    for c in discards:
                        gs.hands[player].remove(c)
                    submit_bid(a, player, bid_obj, discards)
                    winning_eval = matched_eval
                    continue

                elif supported:
                    # No NN evals — true fallback.
                    bid_obj = supported[0]
                    forced = _forced_bid_eval(gs.hands[player], bid_obj)
                    if forced is not None:
                        discards = list(forced.best_discard.discard)
                        for c in discards:
                            gs.hands[player].remove(c)
                        submit_bid(a, player, bid_obj, discards)
                        winning_eval = forced
                        continue

                # No supported legal bid available — last resort.
                if legal:
                    if nn_evals:
                        discards = _nn_discard(nn_evals)
                    else:
                        discards = _fallback_discards(gs.hands[player])
                    for c in discards:
                        gs.hands[player].remove(c)
                    submit_bid(a, player, legal[0], discards)
                    winning_eval = None
                else:
                    # Should never happen; defensive guard.
                    submit_pass(a, player)

        else:
            # ── Player has 10 cards → MC pickup decision ──
            should_pickup = False
            if can_pickup(a) and seat_wrappers[player]:
                should_pickup = estimate_pickup_value(
                    gs, player, dealer,
                    wrappers=seat_wrappers[player],
                    auction=a,
                    rng=_random.Random(hash((player, len(a.history)))),
                )

            if should_pickup:
                gs.hands[player].extend(a.talon)
                submit_pickup(a, player)
            else:
                submit_pass(a, player)

    soloist = a.winner

    # Passz → no real game; first bidder pays the pass penalty.
    if a.current_bid is not None and a.current_bid.rank <= BID_PASSZ.rank:
        return AuctionResult(
            soloist=soloist, bid=None, auction=a,
            initial_bidder=first_bidder,
        )

    return AuctionResult(
        soloist=soloist, bid=winning_eval, auction=a,
        initial_bidder=first_bidder,
    )


# ---------------------------------------------------------------------------
#  Game setup from auction result
# ---------------------------------------------------------------------------


def setup_bid_game(
    game: UltiGame,
    gs: GameState,
    soloist: int,
    dealer: int,
    bid: ContractEval,
    initial_bidder: int = -1,
) -> UltiNode:
    """Build a ready-to-play :class:`UltiNode` from an auction result.

    Handles both 10-card (post-auction) and 12-card (pre-discard)
    soloist hands.  The original *gs* is **not** mutated — a deep
    copy is made internally by :func:`_make_eval_state`.
    """
    cdef = CONTRACT_DEFS[bid.contract_key]
    discard_cards = bid.best_discard.discard
    need_restore = False

    # _make_eval_state expects a 12-card soloist hand.
    if len(gs.hands[soloist]) == 10:
        gs.hands[soloist].extend(list(discard_cards))
        need_restore = True

    try:
        return _make_eval_state(
            gs, soloist, bid.trump, discard_cards,
            cdef, bid.is_piros, dealer,
            initial_bidder=initial_bidder,
        )
    finally:
        if need_restore:
            for c in discard_cards:
                gs.hands[soloist].remove(c)
