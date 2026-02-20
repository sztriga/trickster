"""Tests for the Ulti GameInterface adapter, encoder, and determinization."""

from __future__ import annotations

import random

import numpy as np
import pytest

from trickster.games.ulti.adapter import (
    UltiGame,
    UltiNode,
    _CARD_IDX,
    _IDX_TO_CARD,
    build_auction_constraints,
)
from trickster.games.ulti.cards import ALL_SUITS, Card, NUM_PLAYERS, Rank, Suit, make_deck
from trickster.games.ulti.encoder import (
    STATE_DIM,
    _HAND_OFF,
    _CAP0_OFF,
    _TRUMP_OFF,
    _SCALAR_OFF,
    _CONTRACT_OFF,
    _LEAD_HIST_OFF,
    _WIN_HIST_OFF,
    _MAR_MEM_OFF,
)
from trickster.games.ulti.game import deal, discard_talon, pickup_talon, set_contract


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_play_state(seed: int = 0) -> UltiNode:
    """Create a fully set-up play-phase UltiNode."""
    game = UltiGame()
    return game.new_game(seed=seed)


# ---------------------------------------------------------------------------
#  Card index mapping
# ---------------------------------------------------------------------------


class TestCardIndex:
    def test_all_cards_have_index(self):
        deck = make_deck()
        for c in deck:
            assert c in _CARD_IDX

    def test_indices_are_0_to_31(self):
        assert set(_CARD_IDX.values()) == set(range(32))

    def test_round_trip(self):
        for card, idx in _CARD_IDX.items():
            assert _IDX_TO_CARD[idx] == card


# ---------------------------------------------------------------------------
#  Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_state_dim(self):
        assert STATE_DIM == 293

    def test_encode_state_shape(self):
        state = _make_play_state(seed=1)
        game = UltiGame()
        player = game.current_player(state)
        feats = game.encode_state(state, player)
        assert feats.shape == (STATE_DIM,)
        assert feats.dtype == np.float64

    def test_hand_encoded_correctly(self):
        state = _make_play_state(seed=2)
        game = UltiGame()
        player = game.current_player(state)
        feats = game.encode_state(state, player)

        hand = state.gs.hands[player]
        for c in hand:
            idx = _CARD_IDX[c]
            assert feats[_HAND_OFF + idx] == 1.0, f"Card {c} should be in hand section"

        # Non-hand cards should be 0 in hand section
        hand_set = set(hand)
        for c in make_deck():
            if c not in hand_set:
                idx = _CARD_IDX[c]
                assert feats[_HAND_OFF + idx] == 0.0, f"Card {c} should NOT be in hand"

    def test_trump_encoded(self):
        state = _make_play_state(seed=3)
        game = UltiGame()
        player = game.current_player(state)
        feats = game.encode_state(state, player)

        trump = state.gs.trump
        if trump is not None:
            suit_idx = list(ALL_SUITS).index(trump)
            assert feats[_TRUMP_OFF + suit_idx] == 1.0
        else:
            # Betli: all zero
            assert feats[_TRUMP_OFF:_TRUMP_OFF + 4].sum() == 0.0

    def test_soloist_flag(self):
        state = _make_play_state(seed=4)
        game = UltiGame()
        soloist = state.gs.soloist
        feats_sol = game.encode_state(state, soloist)
        assert feats_sol[_SCALAR_OFF + 1] == 1.0  # am_soloist

        defender = (soloist + 1) % 3
        feats_def = game.encode_state(state, defender)
        assert feats_def[_SCALAR_OFF + 1] == 0.0

    def test_per_player_captured_cards(self):
        """After playing a trick, captured cards appear in the correct bitmap."""
        game = UltiGame()
        rng = random.Random(77)
        state = game.new_game(seed=77)

        # Play enough moves to complete at least one trick
        for _ in range(30):
            if game.is_terminal(state):
                break
            actions = game.legal_actions(state)
            state = game.apply(state, rng.choice(actions))
            if state.gs.trick_no > 0:
                break

        if state.gs.trick_no > 0:
            feats = game.encode_state(state, 0)
            # Verify captured cards are in the correct player bitmap
            cap_offsets = [_CAP0_OFF, _CAP0_OFF + 32, _CAP0_OFF + 64]
            for p in range(3):
                for c in state.gs.captured[p]:
                    idx = _CARD_IDX[c]
                    assert feats[cap_offsets[p] + idx] == 1.0

    def test_contract_dna(self):
        """Contract DNA flags should reflect the contract components."""
        game = UltiGame()
        state = game.new_game(seed=55)
        feats = game.encode_state(state, 0)

        # The training setup creates "parti" by default
        if state.contract_components and "parti" in state.contract_components:
            assert feats[_CONTRACT_OFF + 0] == 1.0  # Is_Simple
        if state.contract_components and "betli" in state.contract_components:
            assert feats[_CONTRACT_OFF + 2] == 1.0  # Is_Betli

    def test_trick_history_after_play(self):
        """Trick momentum should record leader and winner after tricks."""
        game = UltiGame()
        rng = random.Random(88)
        state = game.new_game(seed=88)

        # Play a full game
        while not game.is_terminal(state):
            actions = game.legal_actions(state)
            state = game.apply(state, rng.choice(actions))

        feats = game.encode_state(state, 0)
        # All 10 tricks should have non-zero leader/winner entries
        for i in range(10):
            assert feats[_LEAD_HIST_OFF + i] > 0.0, f"Trick {i} leader not set"
            assert feats[_WIN_HIST_OFF + i] > 0.0, f"Trick {i} winner not set"

    def test_marriage_memory(self):
        """Marriage memory should encode declared marriage values."""
        game = UltiGame()
        state = game.new_game(seed=33)
        feats = game.encode_state(state, 0)

        # Check that marriage memory section exists and is finite
        for i in range(6):
            assert np.isfinite(feats[_MAR_MEM_OFF + i])


# ---------------------------------------------------------------------------
#  UltiGame — GameInterface basics
# ---------------------------------------------------------------------------


class TestUltiGame:
    def test_num_players(self):
        assert UltiGame().num_players == 3

    def test_action_space(self):
        assert UltiGame().action_space_size == 32

    def test_new_game_hands(self):
        state = _make_play_state(seed=5)
        for h in state.gs.hands:
            assert len(h) == 10

    def test_legal_actions_subset_of_hand(self):
        state = _make_play_state(seed=6)
        game = UltiGame()
        player = game.current_player(state)
        actions = game.legal_actions(state)
        hand = state.gs.hands[player]
        for a in actions:
            assert a in hand

    def test_legal_action_mask(self):
        state = _make_play_state(seed=7)
        game = UltiGame()
        mask = game.legal_action_mask(state)
        assert mask.shape == (32,)
        assert mask.dtype == bool
        assert mask.sum() == len(game.legal_actions(state))

    def test_apply_creates_new_state(self):
        state = _make_play_state(seed=8)
        game = UltiGame()
        actions = game.legal_actions(state)
        new_state = game.apply(state, actions[0])
        assert new_state is not state
        assert new_state.gs is not state.gs

    def test_full_game_terminates(self):
        """Play a full game with random moves and verify termination."""
        game = UltiGame()
        rng = random.Random(99)

        for seed in range(10):
            state = game.new_game(seed=seed)
            moves = 0
            while not game.is_terminal(state):
                actions = game.legal_actions(state)
                state = game.apply(state, rng.choice(actions))
                moves += 1
                assert moves <= 100, "Game didn't terminate"

            # All tricks played
            assert state.gs.trick_no == 10

    def test_outcome_soloist_vs_defenders(self):
        """Outcome should be opposite for soloist and defenders."""
        game = UltiGame()
        rng = random.Random(42)

        state = game.new_game(seed=42)
        while not game.is_terminal(state):
            state = game.apply(state, rng.choice(game.legal_actions(state)))

        sol = state.gs.soloist
        defs = [i for i in range(3) if i != sol]

        o_sol = game.outcome(state, sol)
        o_d0 = game.outcome(state, defs[0])
        o_d1 = game.outcome(state, defs[1])

        assert o_d0 == o_d1, "Both defenders should have equal outcome"
        # Soloist receives 2x stake, each defender receives 1x stake
        assert o_sol == -2 * o_d0, "Soloist outcome should be -2x each defender's"
        assert o_sol != 0.0, "Game should have a winner"


# ---------------------------------------------------------------------------
#  same_team
# ---------------------------------------------------------------------------


class TestSameTeam:
    def test_same_player(self):
        state = _make_play_state(seed=10)
        game = UltiGame()
        for p in range(3):
            assert game.same_team(state, p, p)

    def test_defenders_are_allies(self):
        state = _make_play_state(seed=11)
        game = UltiGame()
        sol = state.gs.soloist
        defs = [i for i in range(3) if i != sol]
        assert game.same_team(state, defs[0], defs[1])
        assert game.same_team(state, defs[1], defs[0])

    def test_soloist_vs_defender(self):
        state = _make_play_state(seed=12)
        game = UltiGame()
        sol = state.gs.soloist
        defs = [i for i in range(3) if i != sol]
        for d in defs:
            assert not game.same_team(state, sol, d)
            assert not game.same_team(state, d, sol)


# ---------------------------------------------------------------------------
#  Determinization
# ---------------------------------------------------------------------------


class TestDeterminize:
    def test_preserves_own_hand(self):
        state = _make_play_state(seed=20)
        game = UltiGame()
        rng = random.Random(123)

        for player in range(3):
            det = game.determinize(state, player, rng)
            assert set(det.gs.hands[player]) == set(state.gs.hands[player])

    def test_preserves_hand_sizes(self):
        state = _make_play_state(seed=21)
        game = UltiGame()
        rng = random.Random(456)

        for player in range(3):
            det = game.determinize(state, player, rng)
            for i in range(3):
                assert len(det.gs.hands[i]) == len(state.gs.hands[i])

    def test_no_duplicate_cards(self):
        state = _make_play_state(seed=22)
        game = UltiGame()
        rng = random.Random(789)

        # Determinize as soloist — talon discards are known and excluded
        sol = state.gs.soloist
        det = game.determinize(state, sol, rng)
        all_cards: list[Card] = []
        for h in det.gs.hands:
            all_cards.extend(h)
        for pile in det.gs.captured:
            all_cards.extend(pile)
        for _, c in det.gs.trick_cards:
            all_cards.append(c)
        all_cards.extend(det.gs.talon_discards)

        assert len(all_cards) == len(set(all_cards)), "Duplicate card detected"

        # Determinize as defender — in-play cards (hands + captured + trick)
        # should have no duplicates. Talon discards may overlap with hands
        # since defenders don't know the discards.
        defs = [i for i in range(3) if i != sol]
        det_d = game.determinize(state, defs[0], rng)
        in_play: list[Card] = []
        for h in det_d.gs.hands:
            in_play.extend(h)
        for pile in det_d.gs.captured:
            in_play.extend(pile)
        for _, c in det_d.gs.trick_cards:
            in_play.append(c)
        assert len(in_play) == len(set(in_play)), "Duplicate in-play card"

    def test_talon_hidden_from_defenders(self):
        """Defenders should not know the talon discards; soloist should."""
        game = UltiGame()
        rng = random.Random(555)

        for seed in range(5):
            state = game.new_game(seed=seed)
            gs = state.gs
            sol = gs.soloist
            talon = gs.talon_discards
            if not talon:
                continue

            defs = [i for i in range(3) if i != sol]

            # For soloist: talon cards should NOT appear in any opponent's hand
            # because they're known to be discarded.
            det_sol = game.determinize(state, sol, rng)
            for d in defs:
                for c in talon:
                    assert c not in det_sol.gs.hands[d], (
                        f"Talon card {c} appeared in defender {d}'s "
                        f"hand during soloist determinization"
                    )

            # For defenders: talon cards COULD appear in opponent hands
            # (they don't know the discards), so we just verify hand sizes
            # are correct and no duplicates.
            for d in defs:
                det_def = game.determinize(state, d, rng)
                for i in range(3):
                    assert len(det_def.gs.hands[i]) == len(gs.hands[i])

    def test_respects_void_constraints(self):
        """If a player is void in Hearts, determinize should not give them Hearts."""
        state = _make_play_state(seed=30)
        game = UltiGame()

        # Artificially mark player 1 as void in Hearts
        voids = list(state.known_voids)
        voids[1] = frozenset({Suit.HEARTS})
        state = UltiNode(
            gs=state.gs,
            known_voids=tuple(voids),
            bid_rank=state.bid_rank,
            contract_components=state.contract_components,
            must_have=state.must_have,
        )

        rng = random.Random(111)
        for _ in range(20):  # repeat to test randomness
            det = game.determinize(state, 0, rng)
            for c in det.gs.hands[1]:
                assert c.suit != Suit.HEARTS, (
                    f"Player 1 got {c} but is void in Hearts"
                )

    def test_void_tracking_on_apply(self):
        """When a player fails to follow suit, void is recorded."""
        game = UltiGame()

        for seed in range(50):
            state = game.new_game(seed=seed)
            rng = random.Random(seed)

            # Play until someone fails to follow
            moves = 0
            found_void = False
            while not game.is_terminal(state) and moves < 30:
                actions = game.legal_actions(state)
                action = rng.choice(actions)
                cp = game.current_player(state)

                # Check if this is a response that doesn't follow suit
                if state.gs.trick_cards:
                    led_suit = state.gs.trick_cards[0][1].suit
                    if action.suit != led_suit:
                        new_state = game.apply(state, action)
                        assert led_suit in new_state.known_voids[cp], (
                            f"Player {cp} didn't follow {led_suit} "
                            f"but void not recorded"
                        )
                        found_void = True
                        break

                state = game.apply(state, action)
                moves += 1

            if found_void:
                break

        assert found_void, "No void inference found in 50 games — unlikely"


# ---------------------------------------------------------------------------
#  Auction constraints
# ---------------------------------------------------------------------------


class TestAuctionConstraints:
    def test_40_100_assigns_trump_kq(self):
        """With a 40-100 bid, the soloist must hold trump K+Q."""
        game = UltiGame()
        state = game.new_game(seed=42)
        gs = state.gs

        # Simulate a 40-100 contract with known trump
        comps = frozenset({"parti", "40", "100"})
        constraints = build_auction_constraints(gs, comps)

        if gs.trump is not None:
            k = Card(gs.trump, Rank.KING)
            q = Card(gs.trump, Rank.QUEEN)
            # If K/Q haven't been captured/played, they should be in constraints
            played = set()
            for pile in gs.captured:
                played.update(pile)
            if k not in played:
                assert k in constraints[gs.soloist]
            if q not in played:
                assert q in constraints[gs.soloist]

    def test_marriage_constraints(self):
        """Declared marriages should be reflected in constraints."""
        game = UltiGame()
        state = game.new_game(seed=55)
        gs = state.gs

        comps = frozenset({"parti"})
        constraints = build_auction_constraints(gs, comps)

        for player, suit, _pts in gs.marriages:
            k = Card(suit, Rank.KING)
            q = Card(suit, Rank.QUEEN)
            # If not captured, should be in player's constraints
            played = set()
            for pile in gs.captured:
                played.update(pile)
            if k not in played:
                assert k in constraints[player]
            if q not in played:
                assert q in constraints[player]

    def test_determinize_respects_must_have(self):
        """Determinize should assign must-have cards to the correct player."""
        game = UltiGame()
        rng = random.Random(999)

        # Create a state with must-have constraints
        state = game.new_game(seed=50)
        gs = state.gs
        soloist = gs.soloist

        if gs.trump is not None:
            k = Card(gs.trump, Rank.KING)
            q = Card(gs.trump, Rank.QUEEN)
            # Add must-have constraint for soloist
            must_have = {soloist: frozenset({k, q})}
            state = UltiNode(
                gs=gs,
                known_voids=state.known_voids,
                bid_rank=state.bid_rank,
                contract_components=state.contract_components,
                must_have=must_have,
            )

            observer = (soloist + 1) % 3
            for _ in range(20):
                det = game.determinize(state, observer, rng)
                hand = set(det.gs.hands[soloist])
                played = set()
                for pile in det.gs.captured:
                    played.update(pile)
                if k not in played:
                    assert k in hand, f"Soloist should hold {k}"
                if q not in played:
                    assert q in hand, f"Soloist should hold {q}"


class TestCurriculumMode:
    def test_betli_mode(self):
        """training_mode='betli' should force betli games."""
        game = UltiGame()
        for seed in range(10):
            state = game.new_game(seed=seed, training_mode="betli")
            assert state.gs.betli is True
            assert state.gs.trump is None

    def test_simple_mode(self):
        """training_mode='simple' should force non-betli games."""
        game = UltiGame()
        for seed in range(10):
            state = game.new_game(seed=seed, training_mode="simple")
            assert state.gs.betli is False

    def test_ulti_mode(self):
        """training_mode='ulti' should set has_ulti."""
        game = UltiGame()
        for seed in range(10):
            state = game.new_game(seed=seed, training_mode="ulti")
            assert state.gs.has_ulti is True
            assert "ulti" in state.contract_components

    def test_training_mode_stored(self):
        game = UltiGame()
        state = game.new_game(seed=1, training_mode="betli")
        assert state.gs.training_mode == "betli"


class TestTrickHistory:
    def test_history_recorded(self):
        """Trick history should record (leader, winner) for each completed trick."""
        game = UltiGame()
        # Use seed 1 which produces a parti (plays all 10 tricks)
        rng = random.Random(1)
        state = game.new_game(seed=1)

        while not game.is_terminal(state):
            actions = game.legal_actions(state)
            state = game.apply(state, rng.choice(actions))

        assert len(state.gs.trick_history) == state.gs.trick_no
        assert len(state.gs.trick_history) >= 1
        for leader, winner in state.gs.trick_history:
            assert 0 <= leader < 3
            assert 0 <= winner < 3


# ---------------------------------------------------------------------------
#  Auction encoder
# ---------------------------------------------------------------------------


class TestAuctionEncoder:
    def test_shape(self):
        from trickster.games.ulti.auction_encoder import AuctionEncoder, AUCTION_STATE_DIM

        enc = AuctionEncoder()
        assert enc.state_dim == AUCTION_STATE_DIM

        hand = [Card(Suit.HEARTS, r) for r in list(Rank)[:10]]
        feats = enc.encode_state(
            hand=hand,
            player=0,
            dealer=2,
            current_bid_rank=None,
            holder=None,
            consecutive_passes=0,
            history=[],
        )
        assert feats.shape == (AUCTION_STATE_DIM,)
        assert feats.dtype == np.float64

    def test_hand_encoded(self):
        from trickster.games.ulti.auction_encoder import AuctionEncoder, _HAND_OFF

        enc = AuctionEncoder()
        hand = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.BELLS, Rank.SEVEN)]
        feats = enc.encode_state(
            hand=hand, player=0, dealer=0,
            current_bid_rank=None, holder=None,
            consecutive_passes=0, history=[],
        )
        for c in hand:
            idx = _CARD_IDX[c]
            assert feats[_HAND_OFF + idx] == 1.0

    def test_talon_encoded(self):
        from trickster.games.ulti.auction_encoder import AuctionEncoder, _TALON_OFF, _TALON_FLAG_OFF

        enc = AuctionEncoder()
        talon = [Card(Suit.LEAVES, Rank.TEN)]
        feats = enc.encode_state(
            hand=[], player=0, dealer=0,
            current_bid_rank=None, holder=None,
            consecutive_passes=0, history=[],
            talon_cards=talon,
        )
        idx = _CARD_IDX[talon[0]]
        assert feats[_TALON_OFF + idx] == 1.0
        assert feats[_TALON_FLAG_OFF] == 1.0

    def test_suit_strength(self):
        from trickster.games.ulti.auction_encoder import AuctionEncoder, _SUIT_STR_OFF

        enc = AuctionEncoder()
        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TEN),
            Card(Suit.HEARTS, Rank.KING),
        ]
        feats = enc.encode_state(
            hand=hand, player=0, dealer=0,
            current_bid_rank=None, holder=None,
            consecutive_passes=0, history=[],
        )
        # Hearts (suit index 0): count=3, high=3, has_ace=1, has_ten=1
        hearts_off = _SUIT_STR_OFF + 0 * 4
        assert feats[hearts_off + 0] == 3.0 / 8.0  # count
        assert feats[hearts_off + 1] == 3.0 / 3.0  # high cards
        assert feats[hearts_off + 2] == 1.0  # has ace
        assert feats[hearts_off + 3] == 1.0  # has ten


# ---------------------------------------------------------------------------
#  MCTS integration (smoke test)
# ---------------------------------------------------------------------------


class TestMCTSIntegration:
    def test_mcts_choose_returns_legal_action(self):
        """Verify MCTS returns a legal action (minimal sims)."""
        from trickster.mcts import MCTSConfig, alpha_mcts_choose
        from trickster.models.alpha_net import create_shared_alpha_net

        game = UltiGame()
        state = game.new_game(seed=100)

        net = create_shared_alpha_net(
            state_dim=game.state_dim,
            action_space_size=game.action_space_size,
            body_units=32,
            body_layers=1,
            head_units=16,
            seed=0,
        )
        config = MCTSConfig(
            simulations=10,
            determinizations=2,
            use_value_head=False,
            use_policy_priors=False,
        )
        rng = random.Random(0)

        cp = game.current_player(state)
        action = alpha_mcts_choose(state, game, net, cp, config, rng)
        legal = game.legal_actions(state)
        assert action in legal
