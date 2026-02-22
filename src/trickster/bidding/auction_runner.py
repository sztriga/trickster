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
from dataclasses import dataclass

import numpy as np

from trickster.bidding.evaluator import (
    ContractEval,
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


def evaluate_pickup(
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


def nn_discard(evals: list[ContractEval]) -> list[Card]:
    """Pick the best NN-evaluated discard from a list of contract evals.

    Uses the top evaluation's discard — the NN's best judgement of
    which 2 cards to remove, even if no contract is profitable.
    """
    return list(evals[0].best_discard.discard)


def fallback_discards(hand: list[Card]) -> list[Card]:
    """Pick 2 weakest cards to discard (fallback when NN eval failed)."""
    return sorted(hand, key=lambda c: (c.points(), c.strength()))[:2]


# ---------------------------------------------------------------------------
#  High-level per-turn decision functions
#
#  Both ``run_auction`` (training/eval) and the live API call these
#  so the decision logic is identical everywhere.
# ---------------------------------------------------------------------------


def decide_pickup(
    gs: GameState,
    player: int,
    dealer: int,
    wrappers: dict[str, UltiNetWrapper],
    auction: AuctionState,
    min_bid_pts: float = 0.0,
) -> PickupEval | None:
    """Decide whether to pick up the talon (10-card hand).

    Returns a :class:`PickupEval` (with the intended bid rank and trump)
    if the player should pick up, or ``None`` to pass.
    """
    if not can_pickup(auction) or not wrappers:
        return None
    current_rank = auction.current_bid.rank if auction.current_bid else 0
    threshold = min_bid_pts * 2 / _GAME_PTS_MAX
    result = evaluate_pickup(gs, player, dealer, wrappers, bid_rank=current_rank)
    if result is not None and result.value > threshold:
        return result
    return None


def decide_bid(
    gs: GameState,
    player: int,
    dealer: int,
    wrappers: dict[str, UltiNetWrapper],
    auction: AuctionState,
    min_bid_pts: float,
) -> tuple[object, list[Card], ContractEval | None]:
    """Decide what to bid with a 12-card hand.

    Returns ``(Bid, discards, ContractEval | None)``.
    *ContractEval* is ``None`` only for Passz (no real game).
    """
    evals = evaluate_all_contracts(
        gs, player, dealer,
        wrappers=wrappers,
    ) if wrappers else []

    # 1. First bidder (no current bid) — profitable bid or Passz.
    if auction.current_bid is None:
        for ev in evals:
            if ev.game_pts < min_bid_pts:
                break  # sorted desc — rest are worse
            r = _eval_to_auction_bid(ev, auction)
            if r is not None:
                bid_obj, discards = r
                return bid_obj, discards, ev
        discards = nn_discard(evals) if evals else fallback_discards(gs.hands[player])
        return BID_PASSZ, discards, None

    # 2. Picked up (must overbid) — best legal overbid from full NN eval.
    for ev in evals:
        r = _eval_to_auction_bid(ev, auction)
        if r is not None:
            bid_obj, discards = r
            return bid_obj, discards, ev
    raise AssertionError("No legal overbid found after pickup")


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
            bid_obj, discards, winning_eval = decide_bid(
                gs, player, dealer, seat_wrappers[player], a,
                min_bid_pts=min_bid_pts,
            )
            for c in discards:
                gs.hands[player].remove(c)
            submit_bid(a, player, bid_obj, discards)

        else:
            pe = decide_pickup(
                gs, player, dealer, seat_wrappers[player], a,
                min_bid_pts=min_bid_pts,
            )
            if pe is not None:
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
