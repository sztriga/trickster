"""Shared auction logic for training and evaluation.

Both ``bidding_loop.py`` (training) and ``eval_bidding.py`` (evaluation)
import from this module so that the bidding/game-setup code is identical.

Key design decision — **blind pickup**: in real Ulti the talon is
face-down.  A player must commit to picking up *before* seeing the
cards.  The pickup decision uses per-contract value heads: encode
the 10-card hand with each contract's info → batch_value → pick up
if the best prediction exceeds the threshold.
"""
from __future__ import annotations

import copy as _copy
import random as _random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

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
from trickster.games.ulti.cards import Card, Suit
from trickster.games.ulti.game import (
    GameState,
    NUM_PLAYERS,
    declare_all_marriages,
    next_player,
    set_contract,
)
from trickster.model import UltiNetWrapper
from trickster.train_utils import _GAME_PTS_MAX


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
#  Per-contract pickup evaluation (~4-8 forward passes)
# ---------------------------------------------------------------------------


_PICKUP_GAME = UltiGame()


@dataclass
class PickupEval:
    """Result of 10-card pickup evaluation."""
    value: float          # best normalised value across eligible contracts
    bid_rank: int         # bid rank of the best contract
    contract_key: str     # contract key of the best contract
    is_piros: bool        # whether the best contract is piros
    trump: Suit | None    # trump suit of the best contract


def _evaluate_pickup(
    gs: GameState,
    player: int,
    dealer: int,
    seat_wrappers: dict[str, UltiNetWrapper],
    bid_rank: int = 0,
) -> PickupEval | None:
    """Evaluate a 10-card hand for pickup using per-contract value heads.

    Builds a proper game state for each eligible contract (with trump,
    marriages, constraints — same as the 12-card evaluator, just without
    discards).  Only considers contracts that can legally overbid the
    current bid.  Returns the best evaluation, or None if no contract
    is eligible.
    """
    hand = gs.hands[player]

    # Count suits for trump selection
    suit_counts: dict[Suit, int] = {}
    for c in hand:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    sorted_suits = sorted(suit_counts, key=lambda s: suit_counts[s], reverse=True)

    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())
    best: PickupEval | None = None

    for contract_key, wrapper in seat_wrappers.items():
        cdef = CONTRACT_DEFS[contract_key]

        if cdef.is_betli:
            variants = [(None, False), (None, True)]  # betli, P.betli
        elif cdef.piros_only:
            variants = [(Suit.HEARTS, True)]
        else:
            variants = []
            for suit in sorted_suits[:2]:
                variants.append((suit, suit == Suit.HEARTS))
            if Suit.HEARTS not in sorted_suits[:2] and Suit.HEARTS in suit_counts:
                variants.append((Suit.HEARTS, True))

        feats_list: list[np.ndarray] = []
        variant_info: list[tuple[int, str, bool, Suit | None]] = []
        for trump, is_piros in variants:
            overbid_rank = CONTRACT_TO_BID_RANK.get((contract_key, is_piros))
            if overbid_rank is None or overbid_rank <= bid_rank:
                continue

            gs2 = _copy.deepcopy(gs)
            gs2.soloist = player
            set_contract(gs2, player, trump=trump, betli=cdef.is_betli)

            if cdef.key == "ulti":
                gs2.has_ulti = True
            gs2.training_mode = cdef.training_mode

            restrict = None
            if cdef.key == "40-100":
                restrict = "40"
            declare_all_marriages(gs2, soloist_marriage_restrict=restrict)

            if cdef.is_betli:
                comps_frozen = frozenset({"betli"})
            else:
                comps: set[str] = {"parti"}
                if cdef.key == "ulti":
                    comps.add("ulti")
                if "40" in cdef.key:
                    comps.update({"40", "100"})
                comps_frozen = frozenset(comps)

            constraints = build_auction_constraints(gs2, comps_frozen)

            node = UltiNode(
                gs=gs2,
                known_voids=empty_voids,
                bid_rank=overbid_rank,
                is_red=is_piros,
                contract_components=comps_frozen,
                dealer=dealer,
                must_have=constraints,
            )
            feats = _PICKUP_GAME.encode_state(node, player)
            feats_list.append(feats)
            variant_info.append((overbid_rank, contract_key, is_piros, trump))

        if feats_list:
            states = np.stack(feats_list)
            vals = wrapper.batch_value(states)
            idx = int(np.argmax(vals))
            val = float(vals[idx])
            if best is None or val > best.value:
                rank, ck, ip, tr = variant_info[idx]
                best = PickupEval(
                    value=val, bid_rank=rank,
                    contract_key=ck, is_piros=ip, trump=tr,
                )

    return best


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
        if ev.game_pts < min_bid_pts:
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
    min_bid_pts: float = 0.0,
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
    # Track which contract the 10-card eval found positive per player,
    # so the stuck fallback can bid that same contract.
    _pickup_evals: dict[int, int] = {}  # player → bid_rank

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
                # Picked up but can't overbid profitably — put the
                # talon back and bid the contract the 10-card eval
                # found positive.  Discarding the talon restores
                # the original hand that was already approved.
                pickup_bid_rank = _pickup_evals.get(player)
                bid_obj = BID_BY_RANK.get(pickup_bid_rank) if pickup_bid_rank else None

                if bid_obj is None or bid_obj not in legal_bids(a):
                    # Fallback: lowest legal overbid
                    legal = legal_bids(a)
                    supported = [
                        b for b in legal if b.rank in SUPPORTED_BID_RANKS
                    ]
                    bid_obj = supported[0] if supported else (legal[0] if legal else None)

                if bid_obj is None:
                    submit_pass(a, player)
                    continue

                discards = list(a.talon)
                for c in discards:
                    gs.hands[player].remove(c)
                forced = _forced_bid_eval(gs.hands[player], bid_obj)
                submit_bid(a, player, bid_obj, discards)
                winning_eval = forced

        else:
            # ── Player has 10 cards → per-contract value-head pickup ──
            pickup_eval: PickupEval | None = None
            threshold = min_bid_pts * 2 / _GAME_PTS_MAX
            if can_pickup(a) and seat_wrappers[player]:
                current_rank = a.current_bid.rank if a.current_bid else 0
                pickup_eval = _evaluate_pickup(
                    gs, player, dealer, seat_wrappers[player],
                    bid_rank=current_rank,
                )

            if pickup_eval is not None and pickup_eval.value > threshold:
                _pickup_evals[player] = pickup_eval.bid_rank
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
