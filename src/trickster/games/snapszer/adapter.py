"""SnapszerGame — implements GameInterface for AlphaZero training / MCTS.

Wraps the existing game logic + feature encoding behind the generic
:class:`~trickster.games.interface.GameInterface` protocol so that
MCTS and the training loop are fully game-agnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence, Union

import numpy as np

from trickster.games.snapszer.cards import ALL_COLORS, Card, Color, RANK_VALUES, make_deck

# Pre-compute the full deck once (frozen set for fast set operations in determinize)
_ALL_CARDS: frozenset[Card] = frozenset(make_deck())
from trickster.games.snapszer.features import get_alpha_encoder
from trickster.games.snapszer.game import (
    GameState,
    can_close_talon,
    close_talon,
    deal,
    deal_awarded_game_points,
    deal_winner,
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
    """Immutable-ish game node: underlying GameState + pending lead card."""

    gs: GameState
    pending_lead: Card | None  # None → leader's turn; Card → follower's turn

    def clone(self) -> SnapszerNode:
        return SnapszerNode(gs=self.gs.clone(), pending_lead=self.pending_lead)


# ---------------------------------------------------------------------------
# GameInterface implementation
# ---------------------------------------------------------------------------


# Card → index mapping: 20 cards (4 colors * 5 ranks) + 1 close_talon = 21
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
_ACTION_SPACE = 21


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
        # Leader's turn — legal cards + optionally close_talon
        actions: list[Action] = list(_legal_actions(gs, gs.leader, None))
        if can_close_talon(gs, gs.leader):
            actions.append("close_talon")
        return actions

    def apply(self, state: SnapszerNode, action: Action) -> SnapszerNode:
        gs = state.gs
        if action == "close_talon":
            new_gs = gs.clone()
            close_talon(new_gs, new_gs.leader)
            return SnapszerNode(gs=new_gs, pending_lead=None)
        card: Card = action  # type: ignore[assignment]
        if state.pending_lead is None:
            # Leader plays — card becomes pending, state unchanged
            return SnapszerNode(gs=gs.clone(), pending_lead=card)
        # Follower plays — resolve trick
        new_gs = gs.clone()
        new_gs, _ = play_trick(new_gs, state.pending_lead, card)
        return SnapszerNode(gs=new_gs, pending_lead=None)

    def is_terminal(self, state: SnapszerNode) -> bool:
        return _is_terminal(state.gs)

    def outcome(self, state: SnapszerNode, player: int) -> float:
        """Normalised outcome in [-1, +1] based on game points."""
        gs = state.gs
        winner, pts, _reason = deal_awarded_game_points(gs)
        if winner == player:
            return float(pts) / 3.0
        return -float(pts) / 3.0

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
        rng.shuffle(unknown)

        opp_size = len(det.hands[opponent])
        pile_size = len(det.draw_pile)
        idx = 0

        # Pin pending lead in opponent hand if applicable
        if state.pending_lead is not None and gs.leader == opponent:
            opp_hand = [state.pending_lead]
            opp_hand.extend(unknown[idx: idx + opp_size - 1])
            idx += opp_size - 1
        else:
            opp_hand = unknown[idx: idx + opp_size]
            idx += opp_size

        from collections import deque
        det.hands[opponent] = opp_hand
        det.draw_pile = deque(unknown[idx: idx + pile_size])
        idx += pile_size

        if det.trump_card is not None and not det.trump_upcard_visible:
            if idx < len(unknown):
                det.trump_card = unknown[idx]

        return SnapszerNode(gs=det, pending_lead=state.pending_lead)

    # -- neural-network encoding -------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._enc.state_dim

    @property
    def action_dims(self) -> dict[str, int]:
        return {
            "lead": self._enc.lead_policy_dim,
            "follow": self._enc.follow_policy_dim,
        }

    def decision_type(self, state: SnapszerNode) -> str:
        return "follow" if state.pending_lead is not None else "lead"

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

    def encode_actions(
        self,
        state: SnapszerNode,
        player: int,
        actions: Sequence[Action],
    ) -> np.ndarray:
        gs = state.gs
        trump_up = gs.trump_card if gs.trump_upcard_visible else None
        cap_self = gs.captured[player]
        cap_opp = gs.captured[1 - player]
        if state.pending_lead is None:
            return self._enc.encode_lead_policy(
                hand=gs.hands[player],
                actions=list(actions),
                draw_pile_size=len(gs.draw_pile),
                captured_self=cap_self,
                captured_opp=cap_opp,
                trump_color=gs.trump_color,
                trump_upcard=trump_up,
            )
        return self._enc.encode_follow_policy(
            hand=gs.hands[player],
            lead_card=state.pending_lead,
            actions=list(actions),
            draw_pile_size=len(gs.draw_pile),
            captured_self=cap_self,
            captured_opp=cap_opp,
            trump_color=gs.trump_color,
            trump_upcard=trump_up,
        )

    # -- fixed action space ------------------------------------------------

    @property
    def action_space_size(self) -> int:
        return _ACTION_SPACE

    def action_to_index(self, action: Action) -> int:
        if action == "close_talon":
            return _CLOSE_TALON_IDX
        return _CARD_TO_IDX[action]

    def legal_action_mask(self, state: SnapszerNode) -> np.ndarray:
        mask = np.zeros(_ACTION_SPACE, dtype=bool)
        for a in self.legal_actions(state):
            mask[self.action_to_index(a)] = True
        return mask

    # -- new game ----------------------------------------------------------

    def new_game(self, seed: int, **kwargs: Any) -> SnapszerNode:
        starting_leader = kwargs.get("starting_leader", seed % 2)
        gs = deal(seed=seed, starting_leader=starting_leader)
        return SnapszerNode(gs=gs, pending_lead=None)
