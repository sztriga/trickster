"""Tests for the Ulti game engine (dealing, trick play, scoring)."""

import random

from trickster.games.ulti.cards import (
    Card,
    HAND_SIZE,
    LAST_TRICK_BONUS,
    NUM_PLAYERS,
    Rank,
    Suit,
    TALON_SIZE,
    TOTAL_POINTS,
    TRICKS_PER_GAME,
    make_deck,
)
from trickster.games.ulti.game import (
    GameState,
    current_player,
    deal,
    declare_all_marriages,
    defender_points,
    discard_talon,
    is_terminal,
    last_trick_ulti_check,
    legal_actions,
    next_player,
    pickup_talon,
    play_card,
    set_contract,
    soloist_lost_betli,
    soloist_points,
    soloist_won_durchmars,
    soloist_won_simple,
)

H = Suit.HEARTS
B = Suit.BELLS
L = Suit.LEAVES
A = Suit.ACORNS


def C(suit, rank):
    return Card(suit, rank)


# === Turn order ===


class TestNextPlayer:
    def test_cycles_counterclockwise_0_1_2(self):
        assert next_player(0) == 1
        assert next_player(1) == 2
        assert next_player(2) == 0


# === Dealing ===


class TestDeal:
    def test_three_hands_of_10(self):
        state, talon = deal(seed=42)
        for i in range(3):
            assert len(state.hands[i]) == HAND_SIZE, f"Player {i} should have {HAND_SIZE} cards"

    def test_talon_has_2_cards(self):
        _, talon = deal(seed=42)
        assert len(talon) == TALON_SIZE

    def test_all_32_cards_accounted_for(self):
        state, talon = deal(seed=42)
        all_cards = []
        for h in state.hands:
            all_cards.extend(h)
        all_cards.extend(talon)
        assert len(all_cards) == 32
        assert len(set(all_cards)) == 32

    def test_different_seeds_give_different_deals(self):
        s1, t1 = deal(seed=1)
        s2, t2 = deal(seed=2)
        h1 = set(s1.hands[0])
        h2 = set(s2.hands[0])
        assert h1 != h2

    def test_first_bidder_is_next_after_dealer(self):
        state, _ = deal(seed=42, dealer=0)
        assert state.leader == 1  # dealer=0, next = 1

        state2, _ = deal(seed=42, dealer=2)
        assert state2.leader == 0  # dealer=2, next = 0

    def test_initial_state_is_clean(self):
        state, _ = deal(seed=42)
        assert state.trick_no == 0
        assert state.scores == [0, 0, 0]
        assert all(len(c) == 0 for c in state.captured)
        assert state.last_trick is None
        assert state.trick_cards == []


# === Talon pickup & discard ===


class TestTalonExchange:
    def test_pickup_gives_12_cards(self):
        state, talon = deal(seed=42)
        pickup_talon(state, soloist=1, talon=talon)
        assert len(state.hands[1]) == 12

    def test_discard_back_to_10(self):
        state, talon = deal(seed=42)
        pickup_talon(state, soloist=1, talon=talon)
        # Discard the first 2 cards from hand
        discards = state.hands[1][:2]
        discard_talon(state, discards)
        assert len(state.hands[1]) == 10

    def test_discards_go_to_captured(self):
        state, talon = deal(seed=42)
        pickup_talon(state, soloist=1, talon=talon)
        discards = state.hands[1][:2]
        discard_talon(state, discards)
        assert all(c in state.captured[1] for c in discards)

    def test_discard_points_added_to_score(self):
        state, talon = deal(seed=42)
        pickup_talon(state, soloist=1, talon=talon)
        # Find cards with known points to discard
        ace = C(H, Rank.ACE)
        # Use first 2 cards from hand (might be 0-point cards)
        discards = state.hands[1][:2]
        expected_pts = sum(c.points() for c in discards)
        discard_talon(state, discards)
        assert state.scores[1] == expected_pts


# === Set contract ===


class TestSetContract:
    def test_set_simple(self):
        state, _ = deal(seed=42)
        set_contract(state, soloist=0, trump=H)
        assert state.soloist == 0
        assert state.trump == H
        assert state.betli is False

    def test_set_betli(self):
        state, _ = deal(seed=42)
        set_contract(state, soloist=2, trump=None, betli=True)
        assert state.soloist == 2
        assert state.trump is None
        assert state.betli is True


# === Play phase ===


def _setup_game(seed=42, dealer=0, trump=H, betli=False, soloist=None):
    """Helper: deal, pickup talon, discard 2 cards, set contract, declare marriages."""
    state, talon = deal(seed=seed, dealer=dealer)
    if soloist is None:
        soloist = next_player(dealer)
    pickup_talon(state, soloist=soloist, talon=talon)
    # Discard the first 2 cards (doesn't matter which for these tests)
    discards = state.hands[soloist][:2]
    discard_talon(state, discards)
    set_contract(state, soloist=soloist, trump=trump, betli=betli)
    declare_all_marriages(state)
    return state


class TestCurrentPlayer:
    def test_leader_plays_first(self):
        state = _setup_game()
        assert current_player(state) == state.leader

    def test_second_player_after_one_card(self):
        state = _setup_game()
        leader = state.leader
        card = legal_actions(state)[0]
        play_card(state, card)
        assert current_player(state) == next_player(leader)

    def test_third_player_after_two_cards(self):
        state = _setup_game()
        leader = state.leader
        # Play two cards
        play_card(state, legal_actions(state)[0])
        play_card(state, legal_actions(state)[0])
        assert current_player(state) == next_player(next_player(leader))


class TestLegalActions:
    def test_leader_can_play_any_card(self):
        state = _setup_game()
        player = current_player(state)
        actions = legal_actions(state)
        assert set(actions) == set(state.hands[player])


class TestPlayCard:
    def test_card_removed_from_hand(self):
        state = _setup_game()
        player = current_player(state)
        card = legal_actions(state)[0]
        hand_before = len(state.hands[player])
        play_card(state, card)
        assert len(state.hands[player]) == hand_before - 1
        assert card not in state.hands[player]

    def test_trick_not_complete_after_one_card(self):
        state = _setup_game()
        result = play_card(state, legal_actions(state)[0])
        assert result is None
        assert len(state.trick_cards) == 1

    def test_trick_not_complete_after_two_cards(self):
        state = _setup_game()
        play_card(state, legal_actions(state)[0])
        result = play_card(state, legal_actions(state)[0])
        assert result is None
        assert len(state.trick_cards) == 2

    def test_trick_complete_after_three_cards(self):
        state = _setup_game()
        play_card(state, legal_actions(state)[0])
        play_card(state, legal_actions(state)[0])
        result = play_card(state, legal_actions(state)[0])
        assert result is not None
        assert state.trick_no == 1
        assert state.trick_cards == []  # cleared

    def test_winner_captures_cards(self):
        state = _setup_game()
        play_card(state, legal_actions(state)[0])
        play_card(state, legal_actions(state)[0])
        result = play_card(state, legal_actions(state)[0])
        winner = result.winner
        # Winner captured exactly 3 cards from this trick
        # (plus any from talon discard if winner == soloist)
        assert len(state.captured[winner]) >= 3

    def test_winner_becomes_leader(self):
        state = _setup_game()
        play_card(state, legal_actions(state)[0])
        play_card(state, legal_actions(state)[0])
        result = play_card(state, legal_actions(state)[0])
        assert state.leader == result.winner


# === Full game ===


class TestFullGame:
    def _play_full_game(self, seed=42):
        """Play a full game with random legal moves."""
        state = _setup_game(seed=seed)
        rng = random.Random(seed)
        while not is_terminal(state):
            actions = legal_actions(state)
            card = rng.choice(actions)
            play_card(state, card)
        return state

    def test_10_tricks_played(self):
        state = self._play_full_game()
        assert state.trick_no == TRICKS_PER_GAME
        assert is_terminal(state)

    def test_all_hands_empty(self):
        state = self._play_full_game()
        for i in range(NUM_PLAYERS):
            assert len(state.hands[i]) == 0, f"Player {i} should have no cards left"

    def test_all_cards_captured(self):
        state = self._play_full_game()
        total_captured = sum(len(state.captured[i]) for i in range(NUM_PLAYERS))
        # 30 from tricks + 2 from talon discard = 32
        assert total_captured == 32

    def test_points_sum_to_90_plus_marriages(self):
        state = self._play_full_game()
        total_marriage_pts = sum(pts for _, _, pts in state.marriages)
        total = sum(state.scores)
        expected = TOTAL_POINTS + total_marriage_pts
        assert total == expected, f"Total points should be {expected}, got {total}"

    def test_last_trick_bonus_awarded(self):
        """The last trick winner gets +10 bonus (included in TOTAL_POINTS)."""
        state = self._play_full_game()
        total_marriage_pts = sum(pts for _, _, pts in state.marriages)
        assert sum(state.scores) == TOTAL_POINTS + total_marriage_pts

    def test_soloist_plus_defenders_equals_total(self):
        state = self._play_full_game()
        total_marriage_pts = sum(pts for _, _, pts in state.marriages)
        assert soloist_points(state) + defender_points(state) == TOTAL_POINTS + total_marriage_pts

    def test_multiple_seeds_all_valid(self):
        """Run 20 random games and verify invariants."""
        for seed in range(20):
            state = self._play_full_game(seed=seed)
            assert is_terminal(state)
            assert state.trick_no == TRICKS_PER_GAME
            total_marriage_pts = sum(pts for _, _, pts in state.marriages)
            assert sum(state.scores) == TOTAL_POINTS + total_marriage_pts
            total_cap = sum(len(state.captured[i]) for i in range(NUM_PLAYERS))
            assert total_cap == 32


# === Scoring queries ===


class TestScoringQueries:
    def test_soloist_won_simple(self):
        """Soloist wins if their total > defenders' total (including marriages)."""
        state = _setup_game()
        sol = state.soloist
        defs = [i for i in range(3) if i != sol]

        # Reset scores to test cleanly.
        state.scores = [0, 0, 0]

        # Soloist 50, defenders 40 → wins.
        state.scores[sol] = 50
        state.scores[defs[0]] = 20
        state.scores[defs[1]] = 20
        assert soloist_won_simple(state) is True

        # Equal totals → defenders win (strict inequality).
        state.scores[sol] = 45
        state.scores[defs[0]] = 25
        state.scores[defs[1]] = 20
        assert soloist_won_simple(state) is False

        # Soloist 46, defenders 44 → wins.
        state.scores[sol] = 46
        state.scores[defs[0]] = 24
        state.scores[defs[1]] = 20
        assert soloist_won_simple(state) is True

        # Soloist lower → loses.
        state.scores[sol] = 30
        state.scores[defs[0]] = 40
        state.scores[defs[1]] = 20
        assert soloist_won_simple(state) is False

    def test_soloist_won_durchmars(self):
        state = _setup_game()
        # Give soloist 30 captured cards (10 tricks × 3 cards)
        state.captured[state.soloist] = list(range(30))  # mock
        assert soloist_won_durchmars(state) is True
        state.captured[state.soloist] = list(range(27))  # 9 tricks
        assert soloist_won_durchmars(state) is False

    def test_soloist_lost_betli(self):
        state = _setup_game(betli=True, trump=None)
        state.captured[state.soloist] = []
        assert soloist_lost_betli(state) is False
        # Soloist won at least 1 trick (3 cards)
        state.captured[state.soloist] = list(range(3))
        assert soloist_lost_betli(state) is True


# === 7esre tartás ===


class TestHetesreTartas:
    """The soloist must hold the trump 7 for the last trick in Ulti games."""

    def test_trump_7_filtered_before_last_trick(self):
        """Soloist cannot voluntarily play trump 7 before trick 10."""
        state = _setup_game(trump=H)
        state.has_ulti = True
        state.trick_no = 5  # not the last trick

        # Give soloist hand with trump 7 and other cards.
        trump_7 = Card(H, Rank.SEVEN)
        state.hands[state.soloist] = [
            trump_7,
            Card(H, Rank.ACE),
            Card(B, Rank.ACE),
        ]
        state.leader = state.soloist

        actions = legal_actions(state)
        assert trump_7 not in actions
        assert len(actions) == 2

    def test_trump_7_allowed_on_last_trick(self):
        """Soloist CAN play trump 7 on the last trick (trick 10)."""
        state = _setup_game(trump=H)
        state.has_ulti = True
        state.trick_no = 9  # last trick (0-indexed, TRICKS_PER_GAME - 1)

        trump_7 = Card(H, Rank.SEVEN)
        state.hands[state.soloist] = [trump_7]
        state.leader = state.soloist

        actions = legal_actions(state)
        assert trump_7 in actions

    def test_trump_7_forced_when_only_option(self):
        """If trump 7 is the only legal card, soloist is forced to play it."""
        state = _setup_game(trump=H)
        state.has_ulti = True
        state.trick_no = 5

        trump_7 = Card(H, Rank.SEVEN)
        state.hands[state.soloist] = [trump_7]
        state.leader = state.soloist

        actions = legal_actions(state)
        assert actions == [trump_7]

    def test_non_soloist_can_play_trump_7(self):
        """Defenders have no 7esre tartás restriction."""
        state = _setup_game(trump=H)
        state.has_ulti = True
        state.trick_no = 5

        defender = next_player(state.soloist)
        trump_7 = Card(H, Rank.SEVEN)
        state.hands[defender] = [
            trump_7,
            Card(H, Rank.ACE),
        ]
        state.leader = defender

        actions = legal_actions(state)
        assert trump_7 in actions

    def test_no_restriction_without_ulti(self):
        """Without has_ulti, the soloist can play trump 7 freely."""
        state = _setup_game(trump=H)
        state.has_ulti = False
        state.trick_no = 5

        trump_7 = Card(H, Rank.SEVEN)
        state.hands[state.soloist] = [
            trump_7,
            Card(H, Rank.ACE),
        ]
        state.leader = state.soloist

        actions = legal_actions(state)
        assert trump_7 in actions
