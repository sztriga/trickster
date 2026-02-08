"""Game state and mechanics for the Ulti play phase.

This module handles dealing, trick play, and scoring for a 3-player
Ulti deal once the contract has been determined.  Bidding logic is
not implemented here — the caller sets the soloist, trump, and
contract type before play begins.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

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
from trickster.games.ulti.rules import TrickResult, legal_response, resolve_trick


# ---------------------------------------------------------------------------
#  Turn order
# ---------------------------------------------------------------------------


def next_player(player: int) -> int:
    """Next player in counter-clockwise play order (0 → 1 → 2 → 0).

    Player 1 sits to player 0's right; play passes to the right.
    """
    return (player + 1) % NUM_PLAYERS


# ---------------------------------------------------------------------------
#  Game state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GameState:
    """Mutable state of an Ulti deal during the play phase."""

    hands: list[list[Card]]               # [3], 10 cards each after discard
    trump: Optional[Suit]                  # None for Betli
    betli: bool                            # True for Betli contracts
    soloist: int                           # player index (0, 1, or 2)
    dealer: int                            # who dealt (for rotation tracking)
    captured: list[list[Card]]             # [3], won cards per player
    scores: list[int]                      # [3], accumulated card points
    leader: int                            # who leads the current trick
    trick_no: int                          # 0 .. TRICKS_PER_GAME
    trick_cards: list[tuple[int, Card]]    # cards played so far in current trick
    last_trick: Optional[TrickResult]
    contract_type: Optional[int] = None    # ContractType int value (avoids circular import)
    kontra_level: int = 0                  # 0=none, 1=kontra, 2=rekontra, …
    has_ulti: bool = False                 # True when COMP_ULTI is in the bid (for 7esre tartás)
    marriages: list[tuple[int, Suit, int]] = None  # (player, suit, points) declared marriages
    marriages_declared: bool = False  # True once declare_all_marriages() has run
    trick_history: list[tuple[int, int]] = None  # (leader, winner) for each completed trick
    training_mode: Optional[str] = None  # curriculum flag: "simple"|"betli"|"ulti"|None (all)

    def __post_init__(self):
        if self.marriages is None:
            self.marriages = []
        if self.trick_history is None:
            self.trick_history = []

    def clone(self) -> GameState:
        """Lightweight copy — only mutable containers are duplicated."""
        gs = GameState(
            hands=[list(h) for h in self.hands],
            trump=self.trump,
            betli=self.betli,
            soloist=self.soloist,
            dealer=self.dealer,
            captured=[list(c) for c in self.captured],
            scores=list(self.scores),
            leader=self.leader,
            trick_no=self.trick_no,
            trick_cards=list(self.trick_cards),
            last_trick=self.last_trick,
            contract_type=self.contract_type,
            kontra_level=self.kontra_level,
            has_ulti=self.has_ulti,
            marriages=list(self.marriages),
            marriages_declared=self.marriages_declared,
            trick_history=list(self.trick_history),
            training_mode=self.training_mode,
        )
        return gs


# ---------------------------------------------------------------------------
#  Dealing
# ---------------------------------------------------------------------------


def deal(seed: int = 0, dealer: int = 0) -> tuple[GameState, list[Card]]:
    """Deal cards for an Ulti game.

    Returns ``(state, talon)`` where:
    - ``state`` has 3 hands of 10 cards each
    - ``talon`` is the 2 extra cards (to be picked up by the soloist)

    Deal order (counter-clockwise from dealer):
      5 to each player → 2 to talon → 5 to each player.
    """
    rng = random.Random(seed)
    deck = make_deck()
    rng.shuffle(deck)

    # Play order: first bidder = to dealer's right (next in CCW)
    first = next_player(dealer)
    second = next_player(first)
    third = next_player(second)  # == dealer
    order = [first, second, third]

    hands: list[list[Card]] = [[], [], []]
    pos = 0

    # Round 1: 5 cards each
    for p in order:
        hands[p] = deck[pos : pos + 5]
        pos += 5

    # Talon: 2 cards
    talon = deck[pos : pos + TALON_SIZE]
    pos += TALON_SIZE

    # Round 2: 5 cards each
    for p in order:
        hands[p].extend(deck[pos : pos + 5])
        pos += 5

    assert pos == 32

    state = GameState(
        hands=hands,
        trump=None,
        betli=False,
        soloist=first,      # default; overridden by set_contract
        dealer=dealer,
        captured=[[], [], []],
        scores=[0, 0, 0],
        leader=first,       # first bidder leads trick 1
        trick_no=0,
        trick_cards=[],
        last_trick=None,
    )
    return state, talon


# ---------------------------------------------------------------------------
#  Pre-play setup (soloist picks up talon, discards, contract is set)
# ---------------------------------------------------------------------------


def pickup_talon(state: GameState, soloist: int, talon: list[Card]) -> None:
    """Soloist picks up the 2-card talon (hand goes 10 → 12)."""
    assert len(talon) == TALON_SIZE
    state.hands[soloist].extend(talon)
    state.soloist = soloist


def discard_talon(state: GameState, discards: list[Card]) -> None:
    """Soloist discards 2 cards back down to 10.

    Discarded cards count toward the soloist's captured pile for
    scoring (Aces/Tens in the discard still earn points).
    """
    assert len(discards) == TALON_SIZE
    soloist = state.soloist
    for c in discards:
        state.hands[soloist].remove(c)
    state.captured[soloist].extend(discards)
    state.scores[soloist] += sum(c.points() for c in discards)


def set_contract(
    state: GameState,
    soloist: int,
    trump: Optional[Suit],
    betli: bool = False,
) -> None:
    """Configure the contract for the play phase.

    Parameters
    ----------
    soloist : player index who declared the contract
    trump : trump suit, or None for Betli
    betli : whether Betli rules apply (no must-beat, no trump)
    """
    state.soloist = soloist
    state.trump = trump
    state.betli = betli


# ---------------------------------------------------------------------------
#  Play phase
# ---------------------------------------------------------------------------


def current_player(state: GameState) -> int:
    """Who plays next in the current trick."""
    p = state.leader
    for _ in range(len(state.trick_cards)):
        p = next_player(p)
    return p


def legal_actions(state: GameState) -> List[Card]:
    """Legal cards for the current player.

    Includes the **7esre tartás** rule: in an announced Ulti game,
    the soloist cannot voluntarily play the trump 7 before the last
    trick (trick 10).  If the trump 7 is the only legal card the
    soloist is forced to play it.
    """
    player = current_player(state)
    hand = state.hands[player]

    if len(state.trick_cards) == 0:
        # Leader may play any card
        actions = list(hand)
    else:
        led_suit = state.trick_cards[0][1].suit
        played = [c for _, c in state.trick_cards]
        actions = legal_response(
            hand, led_suit, played,
            trump=state.trump, betli=state.betli,
        )

    # --- 7esre tartás ---
    # Soloist must hold trump 7 for the last trick in Ulti contracts.
    if (
        state.has_ulti
        and player == state.soloist
        and state.trump is not None
        and state.trick_no < TRICKS_PER_GAME - 1   # not the last trick
        and len(actions) > 1                         # has other options
    ):
        trump_seven = Card(state.trump, Rank.SEVEN)
        if trump_seven in actions:
            actions = [c for c in actions if c != trump_seven]

    return actions


def declare_all_marriages(
    state: GameState,
    soloist_marriage_restrict: str | None = None,
) -> None:
    """Declare marriages for all players at the start of the play phase.

    Tournament rules: before the first trick, each player declares all
    King+Queen pairs they hold.  Points are added immediately:
      - Trump K+Q = 40 ("Negyven")
      - Non-trump K+Q = 20 ("Húsz")

    Other players see the *value* but not the *suit*.

    Marriage restriction (Wikipedia rules):
      - In 40-100 contracts: the soloist can only declare the 40
        (trump K+Q).  Additional 20 declarations are not allowed
        and do not count toward scoring.
      - In 20-100 contracts: the soloist can only declare one 20
        (non-trump K+Q).  Additional 40 or 20 declarations are not
        allowed and do not count toward scoring.
      - Defenders always declare all their marriages.

    Parameters
    ----------
    soloist_marriage_restrict : str or None
        ``"40"`` → soloist only declares trump K+Q (for 40-100 contracts)
        ``"20"`` → soloist only declares one non-trump K+Q (for 20-100)
        ``None`` → soloist declares all marriages (default)

    Must be called exactly once, after set_contract() and discard_talon().
    Skipped for Betli (no trump → no marriages).
    """
    if state.marriages_declared:
        return  # idempotent
    state.marriages_declared = True

    if state.betli or state.trump is None:
        return

    for player in range(NUM_PLAYERS):
        hand_set = set(state.hands[player])
        is_soloist = (player == state.soloist)
        declared_20 = False  # track for 20-100 restriction

        for suit in Suit:
            k = Card(suit, Rank.KING)
            q = Card(suit, Rank.QUEEN)
            if k in hand_set and q in hand_set:
                pts = 40 if suit == state.trump else 20

                # Apply soloist marriage restriction
                if is_soloist and soloist_marriage_restrict is not None:
                    if soloist_marriage_restrict == "40":
                        # Only declare the trump marriage (40)
                        if pts != 40:
                            continue
                    elif soloist_marriage_restrict == "20":
                        # Only declare one non-trump marriage (20)
                        if pts == 40:
                            continue  # skip trump marriage
                        if declared_20:
                            continue  # skip additional 20s
                        declared_20 = True

                state.marriages.append((player, suit, pts))
                state.scores[player] += pts


def play_card(state: GameState, card: Card) -> Optional[TrickResult]:
    """Play a card into the current trick.

    The card is removed from the current player's hand and appended
    to the trick.  If the trick is now complete (3 cards), it is
    resolved: the winner captures the cards, points are tallied,
    and the leader is updated.

    Returns
    -------
    TrickResult if the trick completed, else None.
    """
    player = current_player(state)
    assert card in state.hands[player], (
        f"{card} not in player {player}'s hand"
    )

    state.hands[player].remove(card)
    state.trick_cards.append((player, card))

    if len(state.trick_cards) < NUM_PLAYERS:
        return None

    # --- Trick complete: resolve ---
    result = resolve_trick(
        state.trick_cards,
        trump=state.trump,
        betli=state.betli,
    )

    # Winner captures cards and earns points
    trick_points = sum(c.points() for _, c in state.trick_cards)
    state.scores[result.winner] += trick_points
    for _, c in state.trick_cards:
        state.captured[result.winner].append(c)

    # Record trick history (leader, winner) before advancing
    state.trick_history.append((state.leader, result.winner))

    state.trick_no += 1

    # Last trick bonus
    if state.trick_no == TRICKS_PER_GAME:
        state.scores[result.winner] += LAST_TRICK_BONUS

    state.leader = result.winner
    state.last_trick = result
    state.trick_cards = []
    return result


# ---------------------------------------------------------------------------
#  Terminal / scoring queries
# ---------------------------------------------------------------------------


def is_terminal(state: GameState) -> bool:
    """Is the deal over (all 10 tricks played)?"""
    return state.trick_no >= TRICKS_PER_GAME


def soloist_points(state: GameState) -> int:
    """Total card points accumulated by the soloist (including discards)."""
    return state.scores[state.soloist]


def defender_points(state: GameState) -> int:
    """Total card points accumulated by both defenders combined."""
    return sum(s for i, s in enumerate(state.scores) if i != state.soloist)


def soloist_won_simple(state: GameState) -> bool:
    """Did the soloist win Simple? (more total points than defenders combined).

    Both sides' totals include declared marriage points (already in scores).
    Strict inequality: ties go to the defenders.
    """
    return soloist_points(state) > defender_points(state)


def soloist_tricks(state: GameState) -> int:
    """Number of tricks won by the soloist."""
    return len(state.captured[state.soloist]) // NUM_PLAYERS


def soloist_won_durchmars(state: GameState) -> bool:
    """Did the soloist win all 10 tricks?"""
    return soloist_tricks(state) == TRICKS_PER_GAME


def soloist_lost_betli(state: GameState) -> bool:
    """Did the soloist lose Betli (won at least 1 trick)?"""
    return soloist_tricks(state) > 0


def marriage_points(state: GameState, player: int) -> int:
    """Total marriage points declared by *player*."""
    return sum(pts for p, _, pts in state.marriages if p == player)


def soloist_has_40(state: GameState) -> bool:
    """Did the soloist declare a 40 (trump K+Q)?"""
    return any(
        p == state.soloist and s == state.trump and pts == 40
        for p, s, pts in state.marriages
    )


def soloist_has_20(state: GameState) -> bool:
    """Did the soloist declare at least one 20 (non-trump K+Q)?"""
    return any(
        p == state.soloist and pts == 20
        for p, s, pts in state.marriages
    )


def defender_has_40(state: GameState) -> bool:
    """Did any defender declare a 40 (trump K+Q)?"""
    return any(
        p != state.soloist and s == state.trump and pts == 40
        for p, s, pts in state.marriages
    )


def defender_has_20(state: GameState) -> bool:
    """Did any defender declare a 20 (non-trump K+Q)?"""
    return any(
        p != state.soloist and pts == 20
        for p, s, pts in state.marriages
    )


def defender_tricks(state: GameState) -> int:
    """Number of tricks won by the defenders combined."""
    return TRICKS_PER_GAME - soloist_tricks(state)


def defender_won_durchmars(state: GameState) -> bool:
    """Did the defenders win all 10 tricks?"""
    return defender_tricks(state) == TRICKS_PER_GAME


# ---------------------------------------------------------------------------
#  Last-trick analysis (for silent/fallen ulti)
# ---------------------------------------------------------------------------


def last_trick_ulti_check(state: GameState) -> tuple[str, bool]:
    """Check the last trick for ulti / fallen ulti.

    Returns ``(side, won)`` where:
      - ``side`` is ``"soloist"`` or ``"defender"`` or ``"none"``
      - ``won`` is True if they won the trick with trump 7, False if
        they played trump 7 but lost (bukott / fallen).

    Returns ``("none", False)`` if nobody played trump 7 on the last
    trick or the game has no trump.
    """
    if state.last_trick is None or state.trump is None:
        return ("none", False)
    if state.trick_no < TRICKS_PER_GAME:
        return ("none", False)  # game not over

    trump_seven = Card(state.trump, Rank.SEVEN)
    if trump_seven not in state.last_trick.cards:
        return ("none", False)

    # Who played the trump 7?
    idx = state.last_trick.cards.index(trump_seven)
    player_who_played = state.last_trick.players[idx]
    side = "soloist" if player_who_played == state.soloist else "defender"
    won = state.last_trick.winner == player_who_played
    return (side, won)
