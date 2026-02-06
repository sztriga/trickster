from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque
from typing import Optional, Tuple

from trickster.games.snapszer.cards import HAND_SIZE, TRICKS_PER_GAME, Card, Color, make_deck
from trickster.games.snapszer.rules import TrickResult, legal_response_cards, resolve_trick


@dataclass(slots=True)
class GameState:
    hands: list[list[Card]]  # [2][N]
    draw_pile: Deque[Card]
    trump_card: Optional[Card]  # visible upcard while talon not exhausted
    trump_color: Color
    trump_upcard_visible: bool
    talon_closed: bool
    talon_closed_by: Optional[int]
    talon_close_any_zero_tricks: bool
    # Marriage declaration for the upcoming trick (leader action).
    pending_marriage: Optional[tuple[int, Color, int]]  # (player, suit, points)
    declared_marriages: list[tuple[int, Color, int]]  # public announcements
    captured: list[list[Card]]  # [2][N] public captured cards by winner
    scores: list[int]  # [2]
    leader: int  # 0 or 1
    trick_no: int  # 0..(TRICKS_PER_GAME-1)
    last_trick: Optional[TrickResult] = None

    def clone(self) -> GameState:
        """Lightweight copy — only mutable containers are duplicated.

        Card, Color, TrickResult and tuples are frozen/immutable and shared.
        This is much faster than ``copy.deepcopy()``.
        """
        return GameState(
            hands=[list(h) for h in self.hands],
            draw_pile=deque(self.draw_pile),
            trump_card=self.trump_card,
            trump_color=self.trump_color,
            trump_upcard_visible=self.trump_upcard_visible,
            talon_closed=self.talon_closed,
            talon_closed_by=self.talon_closed_by,
            talon_close_any_zero_tricks=self.talon_close_any_zero_tricks,
            pending_marriage=self.pending_marriage,
            declared_marriages=list(self.declared_marriages),
            captured=[list(c) for c in self.captured],
            scores=list(self.scores),
            leader=self.leader,
            trick_no=self.trick_no,
            last_trick=self.last_trick,
        )


def deal(seed: int = 0, starting_leader: int = 0) -> GameState:
    rng = random.Random(seed)
    deck = make_deck()
    rng.shuffle(deck)
    hands = [deck[:HAND_SIZE], deck[HAND_SIZE : 2 * HAND_SIZE]]
    # Schnapsen/Snapszer style: bottom card is face-up and defines trump.
    trump_card = deck[-1]
    trump_color = trump_card.color
    # Face-down part of the talon (top draws first); excludes the visible trump card.
    draw_pile: Deque[Card] = deque(deck[2 * HAND_SIZE : -1])
    return GameState(
        hands=hands,
        draw_pile=draw_pile,
        trump_card=trump_card,
        trump_color=trump_color,
        trump_upcard_visible=True,
        talon_closed=False,
        talon_closed_by=None,
        talon_close_any_zero_tricks=False,
        pending_marriage=None,
        declared_marriages=[],
        captured=[[], []],
        scores=[0, 0],
        leader=starting_leader,
        trick_no=0,
        last_trick=None,
    )


def is_terminal(state: GameState) -> bool:
    # Snapszer-like early finish: first to 66 ends the game immediately.
    if state.scores[0] >= 66 or state.scores[1] >= 66:
        return True
    return state.trick_no >= TRICKS_PER_GAME or (len(state.hands[0]) == 0 and len(state.hands[1]) == 0)


def deal_winner(state: GameState) -> Optional[int]:
    """
    Determine who won the deal.

    - If someone reached 66: they win immediately.
    - If talon was closed and the closer did not win: the non-closer wins.
    - If no one reached 66 and all tricks were played in normal (open) play: last trick winner wins.
    """
    if state.scores[0] >= 66 and state.scores[1] < 66:
        return 0
    if state.scores[1] >= 66 and state.scores[0] < 66:
        return 1
    # If both are >=66 (rare edge case), treat higher score as winner.
    if state.scores[0] >= 66 and state.scores[1] >= 66:
        return 0 if state.scores[0] >= state.scores[1] else 1

    if not is_terminal(state):
        return None

    if state.talon_closed and state.talon_closed_by is not None:
        # Closer failed to win; opponent wins.
        return 1 - int(state.talon_closed_by)

    # Normal play to the end: last trick wins if nobody reached 66.
    if state.trick_no >= TRICKS_PER_GAME and state.last_trick is not None:
        return int(state.last_trick.winner)

    # Fallback: compare trick points.
    return 0 if state.scores[0] >= state.scores[1] else 1


def deal_awarded_game_points(state: GameState) -> tuple[int, int, str]:
    """
    Return (winner_idx, game_points_awarded, reason).

    Implements Hungarian snapszer 'írás' semantics:
    - Normal win: 1/2/3 based on loser trick points and whether loser took any trick.
    - Takarás failure: winner gets 2 or 3 based on whether at close time any player had 0 tricks.
    """
    w = deal_winner(state)
    if w is None:
        raise ValueError("Deal is not over yet.")
    w = int(w)
    l = 1 - w

    def _loser_has_trick() -> bool:
        return len(state.captured[l]) > 0

    # If talon was closed and winner is not the closer, it is a takarás failure for the closer.
    if state.talon_closed and state.talon_closed_by is not None and w != int(state.talon_closed_by):
        pts = 3 if state.talon_close_any_zero_tricks else 2
        return w, pts, "takaras_failed"

    # Normal scoring (also used for successful takarás).
    if not _loser_has_trick():
        return w, 3, "schwarz"
    if int(state.scores[l]) < 33:
        return w, 2, "under_33"
    return w, 1, "at_least_33"


def legal_actions(state: GameState, player: int, lead_card: Optional[Card]) -> list[Card]:
    hand = state.hands[player]
    if lead_card is None:
        return list(hand)
    # Strict rules apply after talon is exhausted OR closed (takarás).
    must_follow = state.talon_closed or (len(state.draw_pile) == 0 and state.trump_card is None)
    return legal_response_cards(hand, lead_card, must_follow=must_follow, trump=state.trump_color)


def talon_size(state: GameState) -> int:
    return int(len(state.draw_pile) + (1 if state.trump_card is not None else 0))


def can_close_talon(state: GameState, player: int) -> bool:
    """
    Takarás / talon closing (Hungarian snapszer):
    - Only the leader may close
    - Only while the talon is still open
    - Only if there are at least 3 cards above the upcard (>=4 total in talon)
    """
    if state.talon_closed:
        return False
    if state.trump_card is None:
        return False
    if state.leader != player:
        return False
    # Require full hands (close happens between tricks, after drawing).
    if len(state.hands[0]) != HAND_SIZE or len(state.hands[1]) != HAND_SIZE:
        return False
    return talon_size(state) >= 4


def close_talon(state: GameState, player: int) -> bool:
    """
    Close the talon (takarás). Returns True if closed.
    """
    if not can_close_talon(state, player):
        return False
    state.talon_closed = True
    state.talon_closed_by = int(player)
    # For takarás failure scoring, we need whether at closing time any player had 0 tricks.
    # A player has taken a trick iff they have captured any cards.
    p0_has = len(state.captured[0]) > 0
    p1_has = len(state.captured[1]) > 0
    state.talon_close_any_zero_tricks = not (p0_has and p1_has)
    # Upcard becomes hidden after closing (trump suit still known).
    state.trump_upcard_visible = False
    return True


def can_declare_marriage(state: GameState, player: int, suit: Color) -> bool:
    """
    Declare 20/40 (King+Queen) when leading a trick.
    """
    if state.pending_marriage is not None:
        return False
    if state.leader != player:
        return False
    k = Card(suit, 4)
    q = Card(suit, 3)
    hand = state.hands[player]
    return k in hand and q in hand


def declare_marriage(state: GameState, player: int, suit: Color) -> int:
    """
    Declare a marriage:
    - 40 in trump suit, 20 otherwise
    - Must then lead either the King or Queen of that suit this trick
    Returns the points awarded (20/40).
    """
    if not can_declare_marriage(state, player, suit):
        raise ValueError("Marriage declaration not allowed in this state.")
    pts = 40 if suit == state.trump_color else 20
    state.scores[player] += pts
    state.pending_marriage = (int(player), suit, pts)
    state.declared_marriages.append((int(player), suit, pts))
    return pts


def can_exchange_trump_jack(state: GameState, player: int) -> bool:
    """
    Trump jack exchange (Snapszer/Schnapsen-like):
    - Only meaningful while the trump upcard still exists.
    - Allowed only while the talon is still open (upcard not yet picked up / no takarás).
    - Disallowed once the upcard has been picked up (state.trump_card becomes None).
    """
    if state.talon_closed:
        return False
    if state.trump_card is None or not state.trump_upcard_visible:
        return False
    if state.leader != player:
        return False
    # Require full hands (exchange happens between tricks, after drawing).
    if len(state.hands[0]) != HAND_SIZE or len(state.hands[1]) != HAND_SIZE:
        return False
    # Must not be the last remaining card (upcard-only). In classic rules this corresponds
    # to "at least 2 cards remain in the talon (including the upcard)".
    if talon_size(state) < 2:
        return False
    trump_jack = Card(state.trump_color, 2)  # J/Unter/Alsó has point-value 2 in this project
    return trump_jack in state.hands[player]


def exchange_trump_jack(state: GameState, player: int) -> bool:
    """
    Swap the trump jack from hand with the visible trump upcard.
    Returns True if an exchange happened.
    """
    if not can_exchange_trump_jack(state, player):
        return False
    assert state.trump_card is not None
    trump_jack = Card(state.trump_color, 2)
    hand = state.hands[player]
    # remove jack, add upcard
    hand.remove(trump_jack)
    hand.append(state.trump_card)
    # upcard becomes the jack
    state.trump_card = trump_jack
    return True


def _draw_one(state: GameState, player: int) -> None:
    if state.talon_closed:
        return
    if len(state.hands[player]) >= HAND_SIZE:
        return
    if state.draw_pile:
        state.hands[player].append(state.draw_pile.popleft())
        return
    if state.trump_card is not None:
        state.hands[player].append(state.trump_card)
        state.trump_card = None


def _draw_to_five(state: GameState, player: int) -> None:
    # Draw until player has HAND_SIZE cards or talon empty.
    if state.talon_closed:
        return
    while len(state.hands[player]) < HAND_SIZE and (state.draw_pile or state.trump_card is not None):
        _draw_one(state, player)


def play_trick(
    state: GameState,
    leader_card: Card,
    responder_card: Card,
    *,
    exchange_trump: bool = False,
) -> Tuple[GameState, TrickResult]:
    leader = state.leader
    responder = 1 - leader

    # Optional: exchange trump jack before leading.
    if exchange_trump:
        exchanged = exchange_trump_jack(state, leader)
        if not exchanged:
            raise ValueError("exchange_trump=True but exchange is not legal in this state.")

    # Enforce pending marriage: leader must lead one of the marriage cards.
    if state.pending_marriage is not None:
        p, suit, _pts = state.pending_marriage
        if p != leader:
            raise ValueError("Pending marriage belongs to a different leader.")
        if leader_card not in (Card(suit, 4), Card(suit, 3)):
            raise ValueError("After declaring a marriage you must lead the King or Queen of that suit.")

    # remove cards
    state.hands[leader].remove(leader_card)
    state.hands[responder].remove(responder_card)

    result = resolve_trick(leader_idx=leader, leader_card=leader_card, responder_card=responder_card, trump=state.trump_color)
    # Schnapsen-style scoring: winner takes the points of both cards.
    state.scores[result.winner] += int(leader_card.points()) + int(responder_card.points())
    # Public captured cards (who won which cards is visible).
    state.captured[result.winner].extend([leader_card, responder_card])
    state.leader = result.winner
    state.trick_no += 1
    state.last_trick = result
    state.pending_marriage = None

    # Variant A draw rule: winner draws first, then the other player.
    winner = result.winner
    loser = 1 - winner
    _draw_to_five(state, winner)
    _draw_to_five(state, loser)

    return state, result

