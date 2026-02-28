"""Shared auction logic for training and evaluation.

Both ``bidding_loop.py`` (training) and ``eval_bidding.py`` (evaluation)
import from this module so that the bidding/game-setup code is identical.

Key design decision — **blind pickup**: in real Ulti the talon is
face-down.  A player must commit to picking up *before* seeing the
cards.  The pickup decision uses per-contract value heads: encode
the 10-card hand with each contract's info → batch_bid_value → pick up
if the best prediction exceeds the threshold.
"""
from __future__ import annotations

import copy as _copy
import math
import random
from dataclasses import dataclass

import numpy as np

from trickster.bidding.constants import _display_key
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
    bid_train_data: list[tuple[str, np.ndarray, float]] | None = None  # [(contract_key, feats, game_pts)]


# ---------------------------------------------------------------------------
#  Per-contract pickup evaluation (~4-8 forward passes)
# ---------------------------------------------------------------------------


_PICKUP_GAME = UltiGame()


def encode_bid_features(
    gs: GameState,
    hand: list[Card],
    contract_key: str,
    trump: Suit | None,
    is_piros: bool,
    dealer: int,
    player: int,
    bid_rank: int = 0,
) -> np.ndarray:
    """Encode a 10-card hand for bid value head evaluation.

    Builds a game state with the contract's settings and encodes it.
    The bid value head is trained on 10-card pre-pickup states.
    """
    cdef = CONTRACT_DEFS[contract_key]
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())

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
        bid_rank=bid_rank,
        is_red=is_piros,
        contract_components=comps_frozen,
        dealer=dealer,
        must_have=constraints,
    )
    feats = _PICKUP_GAME.encode_state(node, player)
    return feats


@dataclass
class PickupEval:
    """Result of 10-card pickup evaluation."""
    value: float          # best normalised soloist value across eligible contracts
    bid_rank: int         # bid rank of the best contract
    contract_key: str     # contract key of the best contract
    is_piros: bool        # whether the best contract is piros
    trump: Suit | None    # trump suit of the best contract
    def_value: float = 0.0  # conservative defender value (min over trump suits)


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
            vals = wrapper.batch_bid_value(states)
            idx = int(np.argmax(vals))
            val = float(vals[idx])
            if best is None or val > best.value:
                rank, ck, ip, tr = variant_info[idx]
                best = PickupEval(
                    value=val, bid_rank=rank,
                    contract_key=ck, is_piros=ip, trump=tr,
                )

    if best is None:
        return None

    # ── Defender evaluation ───────────────────────────────────────
    # Encode the player as a defender of the current contract (from
    # bid_rank) and evaluate with the defender value head. Try each
    # trump suit and take the minimum (conservative: assume the
    # soloist picked the strongest trump for themselves).
    current_contract = BID_TO_CONTRACT.get(bid_rank)
    if current_contract is not None:
        cur_ck, cur_piros = current_contract
        cur_cdef = CONTRACT_DEFS[cur_ck]
        cur_wrapper = seat_wrappers.get(cur_ck)
        if cur_wrapper is not None:
            def_feats_list: list[np.ndarray] = []

            if cur_cdef.is_betli:
                # Betli: no trump, single eval
                trump_variants = [None]
            else:
                # Try each possible trump suit
                trump_variants = list(Suit)

            for def_trump in trump_variants:
                gs_d = _copy.deepcopy(gs)
                # Encode player as defender (someone else is soloist)
                dummy_soloist = (player + 1) % 3
                gs_d.soloist = dummy_soloist
                set_contract(gs_d, dummy_soloist, trump=def_trump, betli=cur_cdef.is_betli)
                gs_d.training_mode = cur_cdef.training_mode

                if cur_cdef.is_betli:
                    def_comps = frozenset({"betli"})
                else:
                    def_comps_set: set[str] = {"parti"}
                    if cur_ck == "ulti":
                        def_comps_set.add("ulti")
                    if "40" in cur_ck:
                        def_comps_set.update({"40", "100"})
                    def_comps = frozenset(def_comps_set)

                declare_all_marriages(gs_d)
                constraints = build_auction_constraints(gs_d, def_comps)

                node_d = UltiNode(
                    gs=gs_d,
                    known_voids=empty_voids,
                    bid_rank=bid_rank,
                    is_red=cur_piros,
                    contract_components=def_comps,
                    dealer=dealer,
                    must_have=constraints,
                )
                feats_d = _PICKUP_GAME.encode_state(node_d, player)
                def_feats_list.append(feats_d)

            if def_feats_list:
                def_states = np.stack(def_feats_list)
                def_vals = cur_wrapper.batch_value_defender(def_states)
                best.def_value = float(np.min(def_vals))

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


def extract_player_bid_ranks(auction: AuctionState) -> tuple[int, int, int]:
    """Extract per-player highest bid rank from auction history.

    Returns a 3-tuple indexed by absolute player index.
    0 means the player never placed a bid (passed on pickup or Passz).
    """
    ranks = [0, 0, 0]
    for player, action, bid in auction.history:
        if action == "bid" and bid is not None:
            # Passz (rank 1) is treated as 0 — it signals weakness, not strength
            if bid.rank > 1:
                ranks[player] = max(ranks[player], bid.rank)
    return (ranks[0], ranks[1], ranks[2])


# ---------------------------------------------------------------------------
#  UCB+softmax contract sampling (for exploratory bidding)
# ---------------------------------------------------------------------------


def _ucb_sample(
    evals: list[ContractEval],
    bid_temp: float,
    c_explore: float,
    dk_game_counts: dict[str, int] | None,
    rng: random.Random,
) -> ContractEval:
    """Sample a contract from *evals* using UCB scoring + softmax.

    Same logic previously inlined in ``_play_one_bidding_game``.
    """
    pts = np.array([ev.game_pts / (_GAME_PTS_MAX / 2) for ev in evals])

    if dk_game_counts is not None and c_explore > 0:
        total_games = sum(dk_game_counts.values()) + 1
        ln_total = math.log(total_games)
        bonus = np.array([
            c_explore * math.sqrt(ln_total / (dk_game_counts.get(
                _display_key(ev.contract_key, ev.is_piros), 0) + 1))
            for ev in evals
        ])
        scores = pts + bonus
    else:
        scores = pts

    logits = scores / max(bid_temp, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = rng.choices(range(len(evals)), weights=probs.tolist(), k=1)[0]
    return evals[idx]


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
    *,
    pickup_explore: float = 0.0,
    rng: random.Random | None = None,
) -> PickupEval | None:
    """Decide whether to pick up the talon (10-card hand).

    Returns a :class:`PickupEval` (with the intended bid rank and trump)
    if the player should pick up, or ``None`` to pass.

    When *pickup_explore* > 0, with that probability the player picks up
    even when ``sol_value <= def_value`` (epsilon-greedy exploration).
    """
    if not can_pickup(auction) or not wrappers:
        return None
    current_rank = auction.current_bid.rank if auction.current_bid else 0
    threshold = min_bid_pts * 2 / _GAME_PTS_MAX
    result = evaluate_pickup(gs, player, dealer, wrappers, bid_rank=current_rank)
    if result is None or result.value <= threshold:
        return None
    if result.value > result.def_value:
        return result
    # Epsilon-greedy: pick up despite defender advantage
    if pickup_explore > 0 and rng is not None and rng.random() < pickup_explore:
        return result
    return None


def decide_bid(
    gs: GameState,
    player: int,
    dealer: int,
    wrappers: dict[str, UltiNetWrapper],
    auction: AuctionState,
    min_bid_pts: float,
    *,
    bid_temp: float = 0.0,
    c_explore: float = 0.0,
    dk_game_counts: dict[str, int] | None = None,
    rng: random.Random | None = None,
) -> tuple[object, list[Card], ContractEval | None, list[ContractEval]]:
    """Decide what to bid with a 12-card hand.

    Returns ``(Bid, discards, ContractEval | None, all_evals)``.
    *ContractEval* is ``None`` only for Passz (no real game).
    *all_evals* contains all evaluated contracts (empty for Passz
    without wrappers).

    When *bid_temp* > 0, UCB+softmax sampling is used instead of
    greedy selection — both for first bids and overbids.
    """
    evals = evaluate_all_contracts(
        gs, player, dealer,
        wrappers=wrappers,
    ) if wrappers else []

    # Exploratory: UCB+softmax sample from all legal contracts.
    if bid_temp > 0 and rng is not None and evals:
        legal_evals = [ev for ev in evals if _eval_to_auction_bid(ev, auction) is not None]
        if legal_evals:
            ev = _ucb_sample(legal_evals, bid_temp, c_explore, dk_game_counts, rng)
            r = _eval_to_auction_bid(ev, auction)
            if r is not None:
                bid_obj, discards = r
                return bid_obj, discards, ev, evals

    # Greedy fallback.
    if auction.current_bid is None:
        # First bidder — profitable bid or Passz.
        for ev in evals:
            if ev.game_pts < min_bid_pts:
                break  # sorted desc — rest are worse
            r = _eval_to_auction_bid(ev, auction)
            if r is not None:
                bid_obj, discards = r
                return bid_obj, discards, ev, evals
        discards = nn_discard(evals) if evals else fallback_discards(gs.hands[player])
        return BID_PASSZ, discards, None, evals

    # Picked up (must overbid) — best legal overbid.
    for ev in evals:
        r = _eval_to_auction_bid(ev, auction)
        if r is not None:
            bid_obj, discards = r
            return bid_obj, discards, ev, evals
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
    bid_temp: float = 0.0,
    c_explore: float = 0.0,
    dk_game_counts: dict[str, int] | None = None,
    pickup_explore: float = 0.0,
    rng: random.Random | None = None,
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
    bid_temp : float
        Softmax temperature for first-bidder contract selection.
        0 = greedy (eval default), >0 = exploratory (training).
    c_explore : float
        UCB exploration constant.  Only used when *bid_temp* > 0.
    dk_game_counts : dict
        Per-display-key cumulative game counts for UCB bonus.
    pickup_explore : float
        Epsilon-greedy pickup exploration probability.
    rng : random.Random
        RNG for stochastic decisions.  Required when any exploration
        parameter is non-zero.
    """
    first_bidder = next_player(dealer)

    # First bidder picks up the talon → 12 cards.
    gs.hands[first_bidder].extend(talon)
    a = create_auction(first_bidder, talon)

    winning_eval: ContractEval | None = None
    all_evals: list[ContractEval] = []
    # Track 10-card hand of pickup player (not first bidder) for bid training
    pickup_player: int = -1
    pickup_hand_10: list[Card] | None = None

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
            bid_obj, discards, winning_eval, all_evals = decide_bid(
                gs, player, dealer, seat_wrappers[player], a,
                min_bid_pts=min_bid_pts,
                bid_temp=bid_temp,
                c_explore=c_explore,
                dk_game_counts=dk_game_counts,
                rng=rng,
            )
            for c in discards:
                gs.hands[player].remove(c)
            submit_bid(a, player, bid_obj, discards)

        elif player == a.holder:
            # Holder's turn came back — nobody challenged.  Stand.
            submit_pass(a, player)

        else:
            pe = decide_pickup(
                gs, player, dealer, seat_wrappers[player], a,
                min_bid_pts=min_bid_pts,
                pickup_explore=pickup_explore,
                rng=rng,
            )
            if pe is not None:
                # Save 10-card hand before extending with talon
                pickup_player = player
                pickup_hand_10 = list(gs.hands[player])
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

    # ── Build bid training data from pickup player's 10-card hand ─────
    bid_train_data: list[tuple[str, np.ndarray, float]] | None = None
    if pickup_hand_10 is not None and all_evals:
        bid_train_data = []
        # Build a temporary GameState with the 10-card hand for encoding
        gs_10 = _copy.deepcopy(gs)
        gs_10.hands[pickup_player] = list(pickup_hand_10)
        for ev in all_evals:
            feats = encode_bid_features(
                gs_10, pickup_hand_10,
                contract_key=ev.contract_key,
                trump=ev.trump,
                is_piros=ev.is_piros,
                dealer=dealer,
                player=pickup_player,
            )
            bid_train_data.append((ev.contract_key, feats, ev.game_pts))

    return AuctionResult(
        soloist=soloist, bid=winning_eval, auction=a,
        initial_bidder=first_bidder,
        bid_train_data=bid_train_data,
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
    player_bid_ranks: tuple[int, int, int] = (0, 0, 0),
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
            player_bid_ranks=player_bid_ranks,
        )
    finally:
        if need_restore:
            for c in discard_cards:
                gs.hands[soloist].remove(c)
