"""SnapszerGame — implements GameInterface for AlphaZero training / MCTS.

Wraps the existing game logic + feature encoding behind the generic
:class:`~trickster.games.interface.GameInterface` protocol so that
MCTS and the training loop are fully game-agnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from trickster.games.snapszer.cards import ALL_COLORS, Card, Color, HAND_SIZE, RANK_VALUES, TRICKS_PER_GAME, make_deck

# Pre-compute the full deck once (frozen set for fast set operations in determinize)
_ALL_CARDS: frozenset[Card] = frozenset(make_deck())
from trickster.games.snapszer.features import get_alpha_encoder
from trickster.games.snapszer.game import (
    GameState,
    can_close_talon,
    can_declare_marriage,
    can_exchange_trump_jack,
    close_talon,
    deal,
    deal_awarded_game_points,
    deal_winner,
    declare_marriage,
    exchange_trump_jack,
    is_terminal as _is_terminal,
    legal_actions as _legal_actions,
    play_trick,
)


# ---------------------------------------------------------------------------
# State wrapper
# ---------------------------------------------------------------------------

Action = Union[Card, str]  # Card or "close_talon"


@dataclass(slots=True)
class SnapszerNode:
    """Immutable-ish game node: underlying GameState + pending lead card.

    ``known_voids`` tracks suits each player is *provably* missing.
    Updated when a follower doesn't follow suit in the closed phase
    (must-follow rules), which is a definitive inference.
    """

    gs: GameState
    pending_lead: Card | None  # None → leader's turn; Card → follower's turn
    known_voids: tuple[frozenset[Color], frozenset[Color]] = (
        frozenset(),
        frozenset(),
    )

    def clone(self) -> SnapszerNode:
        return SnapszerNode(
            gs=self.gs.clone(),
            pending_lead=self.pending_lead,
            known_voids=self.known_voids,
        )


# ---------------------------------------------------------------------------
# GameInterface implementation
# ---------------------------------------------------------------------------


# Card → index mapping: 20 cards + 1 close_talon + 4 marriages = 25
_CARD_TO_IDX: dict[Card, int] = {}
_IDX_TO_CARD: dict[int, Card] = {}
_idx = 0
for _col in ALL_COLORS:
    for _rk in RANK_VALUES:
        c = Card(_col, _rk)
        _CARD_TO_IDX[c] = _idx
        _IDX_TO_CARD[_idx] = c
        _idx += 1
_CLOSE_TALON_IDX = 20

# Marriage declaration actions: one per suit (indices 21-24)
_MARRY_START_IDX = 21
_MARRY_ACTIONS: dict[Color, str] = {}
_MARRY_SUIT: dict[str, Color] = {}
for _mi, _mc in enumerate(ALL_COLORS):
    _ms = f"marry_{_mc.value}"
    _MARRY_ACTIONS[_mc] = _ms
    _MARRY_SUIT[_ms] = _mc

_ACTION_SPACE = 25

# Combined action→index dict (avoids branch + method-call overhead)
_ACTION_IDX: dict[Action, int] = {
    **_CARD_TO_IDX,
    "close_talon": _CLOSE_TALON_IDX,
}
for _mi, _mc in enumerate(ALL_COLORS):
    _ACTION_IDX[f"marry_{_mc.value}"] = _MARRY_START_IDX + _mi


class SnapszerGame:
    """GameInterface implementation for Snapszer."""

    def __init__(self) -> None:
        self._enc = get_alpha_encoder()

    # -- game rules --------------------------------------------------------

    @property
    def num_players(self) -> int:
        return 2

    def current_player(self, state: SnapszerNode) -> int:
        if state.pending_lead is None:
            return state.gs.leader
        return 1 - state.gs.leader

    def legal_actions(self, state: SnapszerNode) -> list[Action]:
        gs = state.gs
        if state.pending_lead is not None:
            # Follower's turn
            return _legal_actions(gs, 1 - gs.leader, state.pending_lead)
        leader = gs.leader
        # If a marriage was declared, leader must lead K or Q of that suit.
        if gs.pending_marriage is not None:
            _p, suit, _pts = gs.pending_marriage
            return [c for c in gs.hands[leader]
                    if c.color == suit and c.number in (3, 4)]
        # Normal leader turn: cards + optionally close_talon + marriages
        actions: list[Action] = list(_legal_actions(gs, leader, None))
        if can_close_talon(gs, leader):
            actions.append("close_talon")
        for suit in ALL_COLORS:
            if can_declare_marriage(gs, leader, suit):
                actions.append(_MARRY_ACTIONS[suit])
        return actions

    def apply(self, state: SnapszerNode, action: Action) -> SnapszerNode:
        gs = state.gs
        voids = state.known_voids

        if action == "close_talon":
            new_gs = gs.clone()
            close_talon(new_gs, new_gs.leader)
            return SnapszerNode(gs=new_gs, pending_lead=None, known_voids=voids)

        # Marriage declaration
        if isinstance(action, str) and action in _MARRY_SUIT:
            suit = _MARRY_SUIT[action]
            new_gs = gs.clone()
            declare_marriage(new_gs, new_gs.leader, suit)
            return SnapszerNode(gs=new_gs, pending_lead=None, known_voids=voids)

        card: Card = action  # type: ignore[assignment]
        if state.pending_lead is None:
            # Leader plays — card becomes pending, state unchanged
            return SnapszerNode(gs=gs.clone(), pending_lead=card, known_voids=voids)

        # Follower plays — resolve trick.
        # Bayesian inference: in closed phase (must-follow), not following
        # suit proves the follower lacks that suit.  If they also didn't
        # play trump, they lack trump too.
        follower = 1 - gs.leader
        must_follow = gs.talon_closed or (
            len(gs.draw_pile) == 0 and gs.trump_card is None
        )
        if must_follow and card.color != state.pending_lead.color:
            v = set(voids[follower])
            v.add(state.pending_lead.color)
            if card.color != gs.trump_color:
                # Didn't play trump either → void in trump too
                v.add(gs.trump_color)
            voids = (
                (frozenset(v), voids[1]) if follower == 0
                else (voids[0], frozenset(v))
            )

        new_gs = gs.clone()
        new_gs, _ = play_trick(new_gs, state.pending_lead, card)
        # Auto-exchange trump jack for the new leader (always beneficial)
        if can_exchange_trump_jack(new_gs, new_gs.leader):
            exchange_trump_jack(new_gs, new_gs.leader)
        return SnapszerNode(gs=new_gs, pending_lead=None, known_voids=voids)

    def is_terminal(self, state: SnapszerNode) -> bool:
        return _is_terminal(state.gs)

    def outcome(self, state: SnapszerNode, player: int) -> float:
        """Normalised outcome in [-1, +1] based on game points."""
        gs = state.gs
        winner, pts, _reason = deal_awarded_game_points(gs)
        if winner == player:
            return float(pts) / 3.0
        return -float(pts) / 3.0

    def same_team(self, state: SnapszerNode, player_a: int, player_b: int) -> bool:
        return player_a == player_b  # 2-player, no teams

    # -- imperfect information ---------------------------------------------

    def determinize(
        self, state: SnapszerNode, player: int, rng: random.Random,
    ) -> SnapszerNode:
        gs = state.gs
        det = gs.clone()
        opponent = 1 - player
        all_cards = _ALL_CARDS

        # Cards known to this player
        known: set[Card] = set(det.hands[player])
        known.update(det.captured[0])
        known.update(det.captured[1])
        if det.trump_card is not None and det.trump_upcard_visible:
            known.add(det.trump_card)
        if state.pending_lead is not None and gs.leader != player:
            known.add(state.pending_lead)

        unknown = list(all_cards - known)
        opp_size = len(det.hands[opponent])
        pile_size = len(det.draw_pile)

        # Pin pending lead in opponent hand if applicable
        pinned: list[Card] = []
        if state.pending_lead is not None and gs.leader == opponent:
            pinned = [state.pending_lead]
            unknown = [c for c in unknown if c != state.pending_lead]
            opp_size -= 1  # one slot already taken

        # --- Bayesian constraint: respect known voids ---
        opp_voids = state.known_voids[opponent]
        if opp_voids:
            eligible = [c for c in unknown if c.color not in opp_voids]
            ineligible = [c for c in unknown if c.color in opp_voids]
            rng.shuffle(eligible)
            rng.shuffle(ineligible)

            if len(eligible) >= opp_size:
                opp_cards = eligible[:opp_size]
                leftover = eligible[opp_size:] + ineligible
            else:
                # Not enough eligible (rare edge case) — fill what we can
                opp_cards = eligible + ineligible[:opp_size - len(eligible)]
                leftover = ineligible[opp_size - len(eligible):]
            rng.shuffle(leftover)
        else:
            rng.shuffle(unknown)
            opp_cards = unknown[:opp_size]
            leftover = unknown[opp_size:]

        from collections import deque
        det.hands[opponent] = pinned + opp_cards
        det.draw_pile = deque(leftover[:pile_size])

        if det.trump_card is not None and not det.trump_upcard_visible:
            if pile_size < len(leftover):
                det.trump_card = leftover[pile_size]

        return SnapszerNode(
            gs=det, pending_lead=state.pending_lead,
            known_voids=state.known_voids,
        )

    # -- neural-network encoding -------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._enc.state_dim

    def encode_state(self, state: SnapszerNode, player: int) -> np.ndarray:
        gs = state.gs
        trump_up = gs.trump_card if gs.trump_upcard_visible else None
        return self._enc.encode_state(
            hand=gs.hands[player],
            pending_lead=state.pending_lead,
            draw_pile_size=len(gs.draw_pile),
            captured_self=gs.captured[player],
            captured_opp=gs.captured[1 - player],
            trump_color=gs.trump_color,
            trump_upcard=trump_up,
        )

    # -- fixed action space ------------------------------------------------

    @property
    def action_space_size(self) -> int:
        return _ACTION_SPACE

    def action_to_index(self, action: Action) -> int:
        return _ACTION_IDX[action]

    def legal_action_mask(self, state: SnapszerNode) -> np.ndarray:
        mask = np.zeros(_ACTION_SPACE, dtype=bool)
        _idx = _ACTION_IDX
        for a in self.legal_actions(state):
            mask[_idx[a]] = True
        return mask

    # -- fast random rollout (mutable, no per-move cloning) ---------------

    def fast_rollout(
        self, state: SnapszerNode, perspective: int, rng: random.Random,
    ) -> float:
        """Play random moves on a single mutable copy — no per-move cloning.

        This replaces the generic ``_random_rollout`` loop for Snapszer,
        eliminating ~1 M ``GameState.clone()`` calls per training iteration.
        Trick resolution and draw logic are inlined to cut Python call overhead.
        Close-talon is never chosen (random agents skip it).
        """
        gs = state.gs.clone()          # single clone for the entire rollout
        pending = state.pending_lead
        _hands = gs.hands
        _scores = gs.scores
        _captured = gs.captured
        _trump = gs.trump_color
        _draw_pile = gs.draw_pile
        _choice = rng.choice
        _HAND = HAND_SIZE

        last_winner = gs.leader

        while True:
            # --- inline is_terminal ---
            if _scores[0] >= 66 or _scores[1] >= 66:
                break
            if gs.trick_no >= TRICKS_PER_GAME or (
                len(_hands[0]) == 0 and len(_hands[1]) == 0
            ):
                break

            leader = gs.leader
            follower = 1 - leader

            if pending is None:
                hand_l = _hands[leader]
                # Always declare marriage if K+Q available (free points)
                if gs.pending_marriage is None:
                    for suit in ALL_COLORS:
                        k = Card(suit, 4)
                        q = Card(suit, 3)
                        if k in hand_l and q in hand_l:
                            pts = 40 if suit == _trump else 20
                            _scores[leader] += pts
                            gs.pending_marriage = (leader, suit, pts)
                            # Terminal after marriage?
                            if _scores[0] >= 66 or _scores[1] >= 66:
                                break
                            pending = _choice([k, q])
                            break
                if _scores[0] >= 66 or _scores[1] >= 66:
                    break  # terminal after marriage
                if pending is None:
                    # Leader picks a random card (skip close-talon)
                    pending = _choice(hand_l)
            else:
                # Follower responds
                must_follow = gs.talon_closed or (
                    len(_draw_pile) == 0 and gs.trump_card is None
                )
                if must_follow:
                    # --- inline legal_response_cards ---
                    hand_f = _hands[follower]
                    same = [c for c in hand_f if c.color == pending.color]
                    if not same:
                        trumps = [c for c in hand_f if c.color == _trump]
                        legal = trumps if trumps else hand_f
                    else:
                        higher = [c for c in same if c.number > pending.number]
                        legal = higher if higher else same
                else:
                    legal = _hands[follower]

                resp = _choice(legal)

                # Remove cards from hands
                _hands[leader].remove(pending)
                _hands[follower].remove(resp)

                # --- inline resolve_trick ---
                lead_trump = pending.color == _trump
                resp_trump = resp.color == _trump
                if resp_trump and not lead_trump:
                    winner = follower
                elif lead_trump and not resp_trump:
                    winner = leader
                elif resp.color != pending.color:
                    winner = leader
                elif resp.number > pending.number:
                    winner = follower
                else:
                    winner = leader

                _scores[winner] += pending.number + resp.number
                _captured[winner].append(pending)
                _captured[winner].append(resp)
                gs.leader = winner
                gs.trick_no += 1
                gs.pending_marriage = None
                last_winner = winner

                # --- inline draw-to-five (winner then loser) ---
                if not gs.talon_closed:
                    for p in (winner, 1 - winner):
                        h = _hands[p]
                        while len(h) < _HAND:
                            if _draw_pile:
                                h.append(_draw_pile.popleft())
                            elif gs.trump_card is not None:
                                h.append(gs.trump_card)
                                gs.trump_card = None
                            else:
                                break

                    # --- inline exchange trump jack for new leader ---
                    if (gs.trump_card is not None
                            and gs.trump_upcard_visible
                            and len(_draw_pile) >= 1):  # talon_size >= 2
                        trump_jack = Card(_trump, 2)
                        w_hand = _hands[winner]
                        if trump_jack in w_hand:
                            w_hand.remove(trump_jack)
                            w_hand.append(gs.trump_card)
                            gs.trump_card = trump_jack

                pending = None

        # --- inline deal_awarded_game_points / deal_winner ---
        s0, s1 = _scores
        if s0 >= 66 and s1 < 66:
            w = 0
        elif s1 >= 66 and s0 < 66:
            w = 1
        elif s0 >= 66 and s1 >= 66:
            w = 0 if s0 >= s1 else 1
        elif gs.talon_closed and gs.talon_closed_by is not None:
            w = 1 - gs.talon_closed_by          # closer failed
        else:
            w = last_winner                      # last trick winner wins

        l = 1 - w
        if gs.talon_closed and gs.talon_closed_by is not None and w != gs.talon_closed_by:
            pts = 3 if gs.talon_close_any_zero_tricks else 2
        elif len(_captured[l]) == 0:
            pts = 3                              # schwarz
        elif _scores[l] < 33:
            pts = 2                              # under 33
        else:
            pts = 1                              # at least 33

        return float(pts) / 3.0 if w == perspective else -float(pts) / 3.0

    # -- new game ----------------------------------------------------------

    def new_game(self, seed: int, **kwargs: Any) -> SnapszerNode:
        starting_leader = kwargs.get("starting_leader", seed % 2)
        gs = deal(seed=seed, starting_leader=starting_leader)
        return SnapszerNode(gs=gs, pending_lead=None,
                            known_voids=(frozenset(), frozenset()))
