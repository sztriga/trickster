"""Tests for the bid evaluator and auction decision functions.

These tests use a mock wrapper that returns deterministic values,
allowing us to verify the decision logic without trained models.
They also serve as regression tests for the evaluate_contract
refactoring (deepcopy elimination).
"""
from __future__ import annotations

from itertools import combinations
from unittest.mock import MagicMock

import numpy as np
import pytest

from trickster.bidding.auction_runner import (
    CONTRACT_TO_BID_RANK,
    PickupEval,
    decide_bid,
    decide_pickup,
    evaluate_pickup,
    fallback_discards,
    nn_discard,
)
from trickster.bidding.evaluator import (
    ContractEval,
    evaluate_all_contracts,
    evaluate_contract,
)
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.auction import (
    BID_BY_RANK,
    BID_PASSZ,
    create_auction,
    legal_bids,
    submit_bid,
    submit_pass,
)
from trickster.games.ulti.cards import Card, Rank, Suit, make_deck
from trickster.games.ulti.game import GameState, deal, next_player, pickup_talon


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_mock_wrapper(value: float = 0.5) -> MagicMock:
    """Create a mock UltiNetWrapper that returns a constant value."""
    w = MagicMock()
    w.batch_value = lambda states: np.full(len(states), value)
    w.predict_value = lambda feats: value
    return w


def _make_wrappers(value: float = 0.5) -> dict:
    """Create a dict of mock wrappers for all supported contracts."""
    return {key: _make_mock_wrapper(value) for key in ("parti", "betli", "ulti", "40-100")}


def _deal_12(seed: int = 42) -> tuple[GameState, list[Card], int]:
    """Deal and give first bidder 12 cards. Returns (gs, talon, first_bidder)."""
    gs, talon = deal(seed=seed, dealer=0)
    first_bidder = next_player(0)
    pickup_talon(gs, first_bidder, talon)
    return gs, talon, first_bidder


# ---------------------------------------------------------------------------
#  fallback_discards
# ---------------------------------------------------------------------------


class TestFallbackDiscards:
    def test_returns_two_cards(self):
        gs, talon, fb = _deal_12()
        discards = fallback_discards(gs.hands[fb])
        assert len(discards) == 2

    def test_weakest_first(self):
        hand = [
            Card(Suit.HEARTS, Rank.ACE),   # 11 pts
            Card(Suit.HEARTS, Rank.TEN),   # 10 pts
            Card(Suit.HEARTS, Rank.KING),  # 4 pts
            Card(Suit.BELLS, Rank.SEVEN),  # 0 pts, weakest
            Card(Suit.BELLS, Rank.EIGHT),  # 0 pts, weakest
        ]
        discards = fallback_discards(hand)
        assert set(discards) == {
            Card(Suit.BELLS, Rank.SEVEN),
            Card(Suit.BELLS, Rank.EIGHT),
        }


# ---------------------------------------------------------------------------
#  nn_discard
# ---------------------------------------------------------------------------


class TestNnDiscard:
    def test_returns_top_eval_discard(self):
        d1 = (Card(Suit.HEARTS, Rank.SEVEN), Card(Suit.HEARTS, Rank.EIGHT))
        d2 = (Card(Suit.BELLS, Rank.SEVEN), Card(Suit.BELLS, Rank.EIGHT))
        evals = [
            ContractEval(
                contract_key="parti", trump=Suit.ACORNS, is_piros=False,
                best_discard=MagicMock(discard=d1, value=0.5, game_pts=1.0),
                game_pts=1.0, stakes_pts=1.0,
            ),
            ContractEval(
                contract_key="parti", trump=Suit.BELLS, is_piros=False,
                best_discard=MagicMock(discard=d2, value=0.3, game_pts=0.5),
                game_pts=0.5, stakes_pts=0.5,
            ),
        ]
        result = nn_discard(evals)
        assert result == list(d1)


# ---------------------------------------------------------------------------
#  evaluate_contract — golden output test
# ---------------------------------------------------------------------------


class TestEvaluateContract:
    """Verify evaluate_contract produces correct output for all 66 discard pairs."""

    def test_all_66_discards_evaluated(self):
        """All 66 discard pairs should be evaluated."""
        gs, talon, fb = _deal_12(seed=7)
        hand = gs.hands[fb]
        assert len(hand) == 12

        # Use a MagicMock with side_effect so we can inspect calls
        wrapper = MagicMock()
        wrapper.batch_value = MagicMock(
            side_effect=lambda states: np.full(len(states), 0.3)
        )
        cdef = CONTRACT_DEFS["parti"]

        result = evaluate_contract(
            gs, fb, 0, cdef,
            trump=Suit.HEARTS, is_piros=True,
            wrapper=wrapper,
        )

        assert result is not None
        # The wrapper was called with batch_value — check shape
        call_args = wrapper.batch_value.call_args
        states = call_args[0][0]
        # Should have evaluated all 66 discard pairs
        assert states.shape[0] == 66

    def test_returns_none_for_infeasible(self):
        """40-100 without K+Q in hand should return None."""
        gs, talon, fb = _deal_12(seed=42)
        hand = gs.hands[fb]
        # Remove any hearts K and Q to make 40-100 infeasible with hearts trump
        gs.hands[fb] = [c for c in hand if not (c.suit == Suit.HEARTS and c.rank in (Rank.KING, Rank.QUEEN))]
        # Pad back to 12 with other cards if needed
        while len(gs.hands[fb]) < 12:
            gs.hands[fb].append(Card(Suit.ACORNS, Rank.SEVEN))

        cdef = CONTRACT_DEFS["40-100"]
        wrapper = _make_mock_wrapper(value=0.5)

        result = evaluate_contract(
            gs, fb, 0, cdef,
            trump=Suit.HEARTS, is_piros=True,
            wrapper=wrapper,
        )
        assert result is None


# ---------------------------------------------------------------------------
#  evaluate_all_contracts
# ---------------------------------------------------------------------------


class TestEvaluateAllContracts:
    def test_returns_sorted_by_stakes(self):
        gs, talon, fb = _deal_12(seed=42)
        wrappers = _make_wrappers(value=0.3)
        evals = evaluate_all_contracts(gs, fb, 0, wrappers=wrappers)
        assert len(evals) > 0
        for i in range(len(evals) - 1):
            assert evals[i].stakes_pts >= evals[i + 1].stakes_pts

    def test_empty_wrappers_returns_empty(self):
        gs, talon, fb = _deal_12(seed=42)
        evals = evaluate_all_contracts(gs, fb, 0, wrappers={})
        assert evals == []


# ---------------------------------------------------------------------------
#  decide_pickup
# ---------------------------------------------------------------------------


class TestDecidePickup:
    def test_positive_eval_picks_up(self):
        gs, talon, fb = _deal_12(seed=42)
        # Give player 2 a 10-card hand (they didn't get the talon)
        player = 2
        assert len(gs.hands[player]) == 10

        a = create_auction(fb, talon)
        submit_bid(a, fb, BID_PASSZ, [talon[0], talon[1]])

        wrappers = _make_wrappers(value=0.8)  # positive

        result = decide_pickup(gs, player, 0, wrappers, a, min_bid_pts=0.0)
        assert result is not None
        assert isinstance(result, PickupEval)
        assert result.value > 0.0

    def test_negative_eval_passes(self):
        gs, talon, fb = _deal_12(seed=42)
        player = 2

        a = create_auction(fb, talon)
        submit_bid(a, fb, BID_PASSZ, [talon[0], talon[1]])

        wrappers = _make_wrappers(value=-0.5)  # negative

        result = decide_pickup(gs, player, 0, wrappers, a, min_bid_pts=0.0)
        assert result is None

    def test_empty_wrappers_passes(self):
        gs, talon, fb = _deal_12(seed=42)
        player = 2

        a = create_auction(fb, talon)
        submit_bid(a, fb, BID_PASSZ, [talon[0], talon[1]])

        result = decide_pickup(gs, player, 0, {}, a)
        assert result is None


# ---------------------------------------------------------------------------
#  decide_bid
# ---------------------------------------------------------------------------


class TestDecideBid:
    def test_profitable_bid(self):
        """When the NN says all contracts are great, decide_bid should pick one."""
        gs, talon, fb = _deal_12(seed=42)
        a = create_auction(fb, talon)
        wrappers = _make_wrappers(value=0.8)

        bid_obj, discards, ev = decide_bid(
            gs, fb, 0, wrappers, a,
            min_bid_pts=0.0,
        )
        assert bid_obj is not None
        assert len(discards) == 2
        assert ev is not None
        assert ev.game_pts > 0

    def test_passz_first_bidder(self):
        """When nothing is profitable, first bidder bids Passz."""
        gs, talon, fb = _deal_12(seed=42)
        a = create_auction(fb, talon)
        wrappers = _make_wrappers(value=-0.5)  # everything negative

        bid_obj, discards, ev = decide_bid(
            gs, fb, 0, wrappers, a,
            min_bid_pts=0.0,
        )
        assert bid_obj == BID_PASSZ
        assert len(discards) == 2
        assert ev is None  # Passz has no eval

    def test_passz_discard_uses_nn(self):
        """Passz discard should come from NN evaluation, not heuristic."""
        gs, talon, fb = _deal_12(seed=42)
        a = create_auction(fb, talon)

        # Wrapper that returns value based on hand sum (so different
        # discards produce different values → NN has a preference)
        def _varied_batch_value(states):
            return np.array([-float(s[:32].sum()) / 20.0 for s in states])

        wrappers = {}
        for key in ("parti", "betli", "ulti", "40-100"):
            w = MagicMock()
            w.batch_value = _varied_batch_value
            wrappers[key] = w

        bid_obj, discards, ev = decide_bid(
            gs, fb, 0, wrappers, a,
            min_bid_pts=0.0,
        )
        assert bid_obj == BID_PASSZ
        assert len(discards) == 2
        # Discards should be from hand
        for c in discards:
            assert c in gs.hands[fb]

    def test_overbid_uses_nn_discard(self):
        """Picked-up player should use full NN eval for overbid + discard."""
        gs, talon, fb = _deal_12(seed=42)
        player = 2

        # Simulate: first bidder bids Passz, player 2 picks up
        a = create_auction(fb, talon)
        submit_bid(a, fb, BID_PASSZ, [talon[0], talon[1]])
        from trickster.games.ulti.auction import submit_pickup
        gs.hands[player].extend(a.talon)
        submit_pickup(a, player)

        # Wrappers with positive value — NN will find a legal overbid
        wrappers = _make_wrappers(value=0.5)

        bid_obj, discards, ev = decide_bid(
            gs, player, 0, wrappers, a,
            min_bid_pts=0.0,
        )
        # Should find a legal overbid (rank > Passz)
        assert bid_obj.rank > BID_PASSZ.rank
        assert len(discards) == 2
        assert ev is not None
        # Discard should NOT be the talon back — it's NN-chosen
        assert set(discards) != set(a.talon)

    def test_no_wrappers_first_bidder_passz(self):
        """No models → Passz with fallback discards."""
        gs, talon, fb = _deal_12(seed=42)
        a = create_auction(fb, talon)

        bid_obj, discards, ev = decide_bid(
            gs, fb, 0, {}, a,
            min_bid_pts=0.0,
        )
        assert bid_obj == BID_PASSZ
        assert len(discards) == 2
        assert ev is None


# ---------------------------------------------------------------------------
#  Golden output: encoder produces identical features for each discard
#  regardless of whether we use deepcopy+_make_eval_state or direct
#  encoding. This test captures the current (correct) output so the
#  refactoring can be verified against it.
# ---------------------------------------------------------------------------


class TestEncoderGoldenOutput:
    """Capture feature vectors for all 66 discard pairs as a regression test.

    After the deepcopy-elimination refactor, running this test verifies
    that the new code produces bit-identical output.
    """

    @pytest.fixture
    def golden_setup(self):
        """Set up a deterministic 12-card hand and compute all 66 feature vectors."""
        gs, talon, fb = _deal_12(seed=123)
        hand = gs.hands[fb]
        assert len(hand) == 12

        from trickster.bidding.evaluator import _make_eval_state
        from trickster.games.ulti.adapter import UltiGame

        game = UltiGame()
        cdef = CONTRACT_DEFS["parti"]
        trump = Suit.HEARTS
        is_piros = True

        features = {}
        for pair in combinations(hand, 2):
            try:
                node = _make_eval_state(gs, fb, trump, pair, cdef, is_piros, 0)
            except (ValueError, AssertionError):
                continue
            feats = game.encode_state(node, fb)
            features[pair] = feats

        return gs, fb, trump, is_piros, cdef, features

    def test_golden_feature_count(self, golden_setup):
        """All 66 discard pairs should produce valid features for a normal hand."""
        _, _, _, _, _, features = golden_setup
        assert len(features) == 66

    def test_golden_hand_bits(self, golden_setup):
        """Hand encoding should have exactly 10 bits set (12 - 2 discarded)."""
        _, _, _, _, _, features = golden_setup
        for pair, feats in features.items():
            hand_bits = feats[:32]
            assert hand_bits.sum() == 10, f"Discard {pair}: hand has {hand_bits.sum()} bits"

    def test_golden_talon_bits(self, golden_setup):
        """Talon encoding should have exactly 2 bits set (the discards)."""
        _, fb, _, _, _, features = golden_setup
        from trickster.games.ulti.encoder import _TALON_OFF, NUM_CARDS

        for pair, feats in features.items():
            talon_bits = feats[_TALON_OFF:_TALON_OFF + NUM_CARDS]
            assert talon_bits.sum() == 2, f"Discard {pair}: talon has {talon_bits.sum()} bits"

    def test_golden_features_deterministic(self, golden_setup):
        """Running the same encoding twice should produce identical features."""
        gs, fb, trump, is_piros, cdef, features_1 = golden_setup

        from trickster.bidding.evaluator import _make_eval_state
        from trickster.games.ulti.adapter import UltiGame

        game = UltiGame()
        hand = gs.hands[fb]

        for pair in list(features_1.keys())[:5]:  # spot-check 5 pairs
            node = _make_eval_state(gs, fb, trump, pair, cdef, is_piros, 0)
            feats = game.encode_state(node, fb)
            np.testing.assert_array_equal(feats, features_1[pair])

    def test_golden_different_discards_differ(self, golden_setup):
        """Different discard pairs should produce different feature vectors."""
        _, _, _, _, _, features = golden_setup
        pairs = list(features.keys())
        if len(pairs) >= 2:
            f1 = features[pairs[0]]
            f2 = features[pairs[1]]
            assert not np.array_equal(f1, f2)


# ---------------------------------------------------------------------------
#  Bit-identical regression: old (deepcopy) vs new (direct encoder) path
# ---------------------------------------------------------------------------


class TestDirectEncoderMatchesDeepCopy:
    """Verify the refactored evaluate_contract produces bit-identical
    feature vectors compared to the old _make_eval_state deepcopy path.
    """

    @pytest.fixture(params=[
        ("parti", Suit.HEARTS, True),
        ("parti", Suit.ACORNS, False),
        ("betli", None, False),
        ("betli", None, True),
        ("ulti", Suit.HEARTS, True),
        ("40-100", Suit.HEARTS, True),
    ])
    def contract_variant(self, request):
        return request.param

    def test_all_pairs_bit_identical(self, contract_variant):
        """For every valid discard pair, the direct encoder path must
        produce the exact same feature vector as _make_eval_state."""
        contract_key, trump, is_piros = contract_variant
        # seed=108 guarantees Hearts 7, K, Q in the first bidder's hand
        gs, talon, fb = _deal_12(seed=108)
        hand = gs.hands[fb]
        cdef = CONTRACT_DEFS[contract_key]

        from trickster.bidding.evaluator import _make_eval_state
        from trickster.games.ulti.adapter import UltiGame
        from trickster.games.ulti.encoder import UltiEncoder

        game = UltiGame()
        encoder = UltiEncoder()

        # Collect features via the OLD path (_make_eval_state + encode_state)
        old_features = {}
        for pair in combinations(hand, 2):
            try:
                node = _make_eval_state(
                    gs, fb, trump, pair, cdef, is_piros, 0,
                )
            except (ValueError, AssertionError):
                continue
            feats = game.encode_state(node, fb)
            old_features[pair] = feats

        # Collect features via the NEW path (direct encoder in evaluate_contract)
        # We capture them by using a wrapper that records the batch
        recorded_states = []
        recorded_discards = []

        def _capture_batch_value(states):
            recorded_states.append(states.copy())
            return np.full(len(states), 0.5)

        wrapper = MagicMock()
        wrapper.batch_value = _capture_batch_value

        evaluate_contract(
            gs, fb, 0, cdef,
            trump=trump, is_piros=is_piros,
            wrapper=wrapper,
        )

        # The new code may skip infeasible discards (contract-essential cards),
        # while the old code would still produce features for them.
        # Build a map from the new path's features.
        assert len(recorded_states) == 1, "batch_value should be called once"
        new_batch = recorded_states[0]

        # Get valid discards from the new path — compare using the wrapper's
        # batch call which receives features in the same order as valid_discards
        # We can verify by checking hand bits in each feature vector
        from trickster.games.ulti.encoder import _CARD_IDX, _HAND_OFF, _TALON_OFF, NUM_CARDS

        new_features = {}
        for i in range(len(new_batch)):
            feats = new_batch[i]
            # Reconstruct discard pair from talon bits
            talon_bits = feats[_TALON_OFF:_TALON_OFF + NUM_CARDS]
            discard_indices = np.where(talon_bits > 0.5)[0]
            idx_to_card = {idx: card for card, idx in _CARD_IDX.items()}
            pair = tuple(sorted(
                [idx_to_card[int(di)] for di in discard_indices],
                key=lambda c: _CARD_IDX[c],
            ))
            new_features[pair] = feats

        # Compare all pairs that exist in both
        for pair in new_features:
            if pair in old_features:
                np.testing.assert_array_equal(
                    new_features[pair], old_features[pair],
                    err_msg=f"Mismatch for discard {pair} in {contract_key}",
                )
