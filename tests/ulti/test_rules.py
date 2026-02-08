"""Tests for Ulti trick-taking rules (3-player)."""

from trickster.games.ulti.cards import Card, Rank, Suit
from trickster.games.ulti.rules import legal_response, resolve_trick


# === Helpers ===

H = Suit.HEARTS
B = Suit.BELLS
L = Suit.LEAVES
A = Suit.ACORNS

def C(suit, rank):
    """Shorthand card constructor."""
    return Card(suit, rank)


# === legal_response tests (normal / trump rules) ===


class TestMustFollowSuit:
    def test_must_follow_when_has_suit(self):
        hand = [C(H, Rank.SEVEN), C(H, Rank.ACE), C(B, Rank.KING)]
        led_suit = H
        played = [C(H, Rank.NINE)]
        result = legal_response(hand, led_suit, played, trump=B)
        # Must follow Hearts; only Hearts cards allowed
        assert all(c.suit == H for c in result)
        assert len(result) > 0

    def test_can_play_anything_when_no_suit_and_no_trump(self):
        hand = [C(B, Rank.SEVEN), C(L, Rank.KING), C(A, Rank.ACE)]
        led_suit = H
        played = [C(H, Rank.NINE)]
        # No Hearts, no trump (trump = None → Betli-like no trump)
        result = legal_response(hand, led_suit, played, trump=None)
        assert set(result) == set(hand)


class TestMustBeat:
    def test_must_beat_higher_same_suit(self):
        hand = [C(H, Rank.SEVEN), C(H, Rank.ACE), C(H, Rank.NINE)]
        led_suit = H
        played = [C(H, Rank.KING)]  # strength 5
        result = legal_response(hand, led_suit, played, trump=B)
        # Must beat King: only Ace (strength 7) qualifies
        # (SEVEN=0, NINE=2 can't beat KING=5)
        assert result == [C(H, Rank.ACE)]

    def test_must_play_same_suit_when_cant_beat(self):
        hand = [C(H, Rank.SEVEN), C(H, Rank.EIGHT), C(B, Rank.ACE)]
        led_suit = H
        played = [C(H, Rank.ACE)]  # strength 7 — can't beat
        result = legal_response(hand, led_suit, played, trump=B)
        # Must follow Hearts but can't beat Ace
        assert set(result) == {C(H, Rank.SEVEN), C(H, Rank.EIGHT)}

    def test_third_player_must_beat_second_players_card(self):
        """Third player must beat the highest same-suit card, not just leader's."""
        hand = [C(H, Rank.SEVEN), C(H, Rank.TEN), C(B, Rank.NINE)]
        led_suit = H
        # Leader played H7, second player played HK
        played = [C(H, Rank.SEVEN), C(H, Rank.KING)]
        result = legal_response(hand, led_suit, played, trump=B)
        # Must beat HK (strength 5): only H10 (strength 6) qualifies
        assert result == [C(H, Rank.TEN)]


class TestMustTrump:
    def test_must_trump_when_cant_follow_suit(self):
        hand = [C(B, Rank.SEVEN), C(H, Rank.KING), C(H, Rank.NINE)]
        led_suit = L
        played = [C(L, Rank.ACE)]
        # Can't follow Leaves; trump is Hearts → must play Hearts
        result = legal_response(hand, led_suit, played, trump=H)
        assert all(c.suit == H for c in result)

    def test_must_beat_existing_trump(self):
        hand = [C(H, Rank.SEVEN), C(H, Rank.ACE), C(B, Rank.KING)]
        led_suit = L
        # Leader played Leaves, second player trumped with H9
        played = [C(L, Rank.ACE), C(H, Rank.NINE)]
        result = legal_response(hand, led_suit, played, trump=H)
        # Must trump AND beat H9 (strength 2): HA (7) beats it, H7 (0) doesn't
        assert result == [C(H, Rank.ACE)]

    def test_must_play_any_trump_when_cant_beat_trump(self):
        hand = [C(H, Rank.SEVEN), C(H, Rank.EIGHT), C(B, Rank.KING)]
        led_suit = L
        played = [C(L, Rank.ACE), C(H, Rank.ACE)]  # trumped with HA
        result = legal_response(hand, led_suit, played, trump=H)
        # Must trump but can't beat HA → play any trump
        assert set(result) == {C(H, Rank.SEVEN), C(H, Rank.EIGHT)}

    def test_play_anything_when_no_suit_and_no_trump(self):
        hand = [C(B, Rank.SEVEN), C(A, Rank.KING)]
        led_suit = L
        played = [C(L, Rank.ACE)]
        # Can't follow Leaves, trump is Hearts, no Hearts in hand
        result = legal_response(hand, led_suit, played, trump=H)
        assert set(result) == set(hand)


class TestBetli:
    def test_must_follow_suit_in_betli(self):
        hand = [C(H, Rank.SEVEN), C(H, Rank.ACE), C(B, Rank.KING)]
        led_suit = H
        played = [C(H, Rank.NINE)]
        result = legal_response(hand, led_suit, played, trump=None, betli=True)
        assert all(c.suit == H for c in result)

    def test_no_must_beat_in_betli(self):
        """In Betli, you can play any card of the led suit — no obligation to beat."""
        hand = [C(H, Rank.SEVEN), C(H, Rank.ACE), C(B, Rank.KING)]
        led_suit = H
        played = [C(H, Rank.NINE)]
        result = legal_response(hand, led_suit, played, trump=None, betli=True)
        # Both H7 and HA are legal (no must-beat)
        assert set(result) == {C(H, Rank.SEVEN), C(H, Rank.ACE)}

    def test_no_trump_obligation_in_betli(self):
        hand = [C(B, Rank.SEVEN), C(A, Rank.KING)]
        led_suit = H
        played = [C(H, Rank.NINE)]
        result = legal_response(hand, led_suit, played, trump=None, betli=True)
        # Can't follow suit, no trump → play anything
        assert set(result) == set(hand)


# === resolve_trick tests ===


class TestResolveTrick:
    def test_highest_of_led_suit_wins(self):
        plays = [(0, C(H, Rank.SEVEN)), (1, C(H, Rank.KING)), (2, C(H, Rank.NINE))]
        result = resolve_trick(plays, trump=B)
        assert result.winner == 1  # HK is highest Hearts

    def test_trump_beats_non_trump(self):
        plays = [(0, C(H, Rank.ACE)), (1, C(B, Rank.SEVEN)), (2, C(H, Rank.KING))]
        result = resolve_trick(plays, trump=B)
        assert result.winner == 1  # B7 trumps

    def test_highest_trump_wins(self):
        plays = [(0, C(H, Rank.ACE)), (1, C(B, Rank.SEVEN)), (2, C(B, Rank.TEN))]
        result = resolve_trick(plays, trump=B)
        assert result.winner == 2  # B10 > B7

    def test_off_suit_non_trump_loses(self):
        plays = [(0, C(H, Rank.SEVEN)), (1, C(L, Rank.ACE)), (2, C(A, Rank.ACE))]
        result = resolve_trick(plays, trump=B)
        # Neither L nor A is led suit (H) or trump (B) → leader wins
        assert result.winner == 0

    def test_leader_wins_when_all_off_suit(self):
        """If followers throw off non-led, non-trump, leader wins."""
        plays = [(0, C(H, Rank.SEVEN)), (1, C(L, Rank.ACE)), (2, C(A, Rank.ACE))]
        result = resolve_trick(plays, trump=B)
        assert result.winner == 0

    def test_betli_ten_below_jack(self):
        """In Betli, 10 is weaker than J."""
        plays = [(0, C(H, Rank.TEN)), (1, C(H, Rank.JACK)), (2, C(H, Rank.NINE))]
        result = resolve_trick(plays, trump=None, betli=True)
        assert result.winner == 1  # HJ beats H10 in Betli

    def test_betli_no_trump(self):
        """In Betli there's no trump — off-suit cards never win."""
        plays = [(0, C(H, Rank.SEVEN)), (1, C(B, Rank.ACE)), (2, C(H, Rank.EIGHT))]
        result = resolve_trick(plays, trump=None, betli=True)
        # B is not led suit, no trump → only H cards compete
        assert result.winner == 2  # H8 > H7

    def test_trick_result_stores_cards_and_players(self):
        plays = [(2, C(H, Rank.ACE)), (0, C(H, Rank.SEVEN)), (1, C(H, Rank.KING))]
        result = resolve_trick(plays, trump=B)
        assert result.cards == (C(H, Rank.ACE), C(H, Rank.SEVEN), C(H, Rank.KING))
        assert result.players == (2, 0, 1)
        assert result.winner == 2

    def test_ten_beats_king_in_normal(self):
        plays = [(0, C(H, Rank.KING)), (1, C(H, Rank.TEN)), (2, C(H, Rank.NINE))]
        result = resolve_trick(plays, trump=B)
        assert result.winner == 1  # H10 > HK in normal
