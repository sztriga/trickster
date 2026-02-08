"""Feature encoding for AlphaZero state/action representation.

Provides ``AlphaEncoder`` — a numpy-native encoder that produces:

* ``encode_state``  → ``(state_dim,)``  for the value head
* ``encode_lead_policy``  → ``(N, state_dim + lead_act_dim)``  for lead policy
* ``encode_follow_policy`` → ``(N, state_dim + follow_act_dim)`` for follow policy

All encoding uses integer array indexing at runtime — no dicts, no string keys.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from trickster.games.snapszer.cards import ALL_COLORS, HAND_SIZE, MAX_RANK, RANK_VALUES, Card, Color


# ---------------------------------------------------------------------------
#  Feature key lists (define the canonical feature ordering)
# ---------------------------------------------------------------------------


def state_feature_keys() -> list[str]:
    """State-only features for the value head (no action info).

    Shared between lead and follow states.  When leading, the lead-card
    slots are zero; when following, they describe the face-up lead card.
    """
    keys: list[str] = []
    # Hand
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            keys.append(f"hand.has.{col.value}.{n}")
        keys.append(f"hand.count.{col.value}")
        keys.append(f"hand.max.{col.value}")
    keys.append("hand.size")
    # Public
    keys.append("pub.draw_pile_frac")
    keys.append("pub.phase.closed")
    for col in ALL_COLORS:
        keys.append(f"pub.trump.color.{col.value}")
    keys.append("pub.trump_up.present")
    keys.append("pub.trump_up.number")
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            keys.append(f"pub.cap.self.has.{col.value}.{n}")
            keys.append(f"pub.cap.opp.has.{col.value}.{n}")
    keys.append("pub.cap.self.count")
    keys.append("pub.cap.opp.count")
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            keys.append(f"pub.unseen.has.{col.value}.{n}")
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            keys.append(f"opp.known.has.{col.value}.{n}")
    # Phase
    keys.append("phase.is_following")
    # Lead card (zeros when leading)
    for col in ALL_COLORS:
        keys.append(f"lead.color.{col.value}")
    keys.append("lead.number")
    keys.append("lead.is_trump")
    return keys


def lead_action_only_keys() -> list[str]:
    """Action-specific features for the lead policy head."""
    keys: list[str] = []
    for col in ALL_COLORS:
        keys.append(f"a.color.{col.value}")
    keys.append("a.kind.card")
    keys.append("a.kind.close_talon")
    keys.append("a.number")
    keys.append("a.is_high")
    keys.append("a.is_trump")
    keys.append("ctx.exchanged_trump")
    keys.append("ctx.can_close_talon")
    return keys


def follow_action_only_keys() -> list[str]:
    """Action-specific features for the follow policy head."""
    keys: list[str] = []
    for col in ALL_COLORS:
        keys.append(f"a.color.{col.value}")
    keys.append("a.number")
    keys.append("a.is_trump")
    keys.append("a.follows")
    keys.append("a.beats")
    keys.append("a.diff")
    keys.append("ctx.can_follow")
    keys.append("ctx.can_beat")
    return keys


# ---------------------------------------------------------------------------
#  Precomputed lookup tables (module-level, avoids runtime overhead)
# ---------------------------------------------------------------------------

_N_COLORS = len(ALL_COLORS)
_N_RANKS = len(RANK_VALUES)
_INV_MAX_RANK = 1.0 / float(MAX_RANK)

_COLOR_IDX: dict[Color, int] = {c: i for i, c in enumerate(ALL_COLORS)}

# Direct Card → flat index (avoids tuple creation in hot loops)
_CARD_FLAT_OBJ: dict[Card, int] = {
    Card(c, n): ci * _N_RANKS + ri
    for ci, c in enumerate(ALL_COLORS)
    for ri, n in enumerate(RANK_VALUES)
}


# ---------------------------------------------------------------------------
#  AlphaEncoder (state / action separation for value + policy heads)
# ---------------------------------------------------------------------------


class AlphaEncoder:
    """Numpy-native encoder producing separate state and action features.

    * ``encode_state``  → ``(state_dim,)``  for the value head
    * ``encode_lead_policy``  → ``(N, state_dim + lead_act_dim)``  for lead policy
    * ``encode_follow_policy`` → ``(N, state_dim + follow_act_dim)`` for follow policy

    Precomputes integer index arrays once so that encoding is pure integer
    indexing at runtime — no dicts, no string keys.
    """

    def __init__(self) -> None:
        sk = {k: i for i, k in enumerate(state_feature_keys())}
        lak = {k: i for i, k in enumerate(lead_action_only_keys())}
        fak = {k: i for i, k in enumerate(follow_action_only_keys())}

        self.state_dim: int = len(sk)
        self.lead_action_dim: int = len(lak)
        self.follow_action_dim: int = len(fak)
        self.lead_policy_dim: int = self.state_dim + self.lead_action_dim
        self.follow_policy_dim: int = self.state_dim + self.follow_action_dim

        cols = ALL_COLORS
        ranks = RANK_VALUES

        # ---- state index arrays (positions in state vector) ----
        self._s_hand_has = np.array(
            [sk[f"hand.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp,
        )
        self._s_hand_count = np.array([sk[f"hand.count.{c.value}"] for c in cols], dtype=np.intp)
        self._s_hand_max = np.array([sk[f"hand.max.{c.value}"] for c in cols], dtype=np.intp)
        self._s_hand_size: int = sk["hand.size"]

        self._s_pub_draw: int = sk["pub.draw_pile_frac"]
        self._s_pub_closed: int = sk["pub.phase.closed"]
        self._s_trump_color = np.array([sk[f"pub.trump.color.{c.value}"] for c in cols], dtype=np.intp)
        self._s_trump_up_present: int = sk["pub.trump_up.present"]
        self._s_trump_up_number: int = sk["pub.trump_up.number"]
        self._s_cap_self_has = np.array(
            [sk[f"pub.cap.self.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp,
        )
        self._s_cap_opp_has = np.array(
            [sk[f"pub.cap.opp.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp,
        )
        self._s_cap_self_count: int = sk["pub.cap.self.count"]
        self._s_cap_opp_count: int = sk["pub.cap.opp.count"]
        self._s_unseen_has = np.array(
            [sk[f"pub.unseen.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp,
        )
        self._s_opp_known_has = np.array(
            [sk[f"opp.known.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp,
        )

        self._s_phase_following: int = sk["phase.is_following"]
        self._s_lead_color = np.array([sk[f"lead.color.{c.value}"] for c in cols], dtype=np.intp)
        self._s_lead_number: int = sk["lead.number"]
        self._s_lead_is_trump: int = sk["lead.is_trump"]

        # ---- lead action index arrays (offset by state_dim) ----
        off = self.state_dim
        self._la_color = np.array([off + lak[f"a.color.{c.value}"] for c in cols], dtype=np.intp)
        self._la_kind_card: int = off + lak["a.kind.card"]
        self._la_kind_close: int = off + lak["a.kind.close_talon"]
        self._la_number: int = off + lak["a.number"]
        self._la_is_high: int = off + lak["a.is_high"]
        self._la_is_trump: int = off + lak["a.is_trump"]
        self._la_ctx_exch: int = off + lak["ctx.exchanged_trump"]
        self._la_ctx_close: int = off + lak["ctx.can_close_talon"]

        # ---- follow action index arrays (offset by state_dim) ----
        off = self.state_dim
        self._fa_color = np.array([off + fak[f"a.color.{c.value}"] for c in cols], dtype=np.intp)
        self._fa_number: int = off + fak["a.number"]
        self._fa_is_trump: int = off + fak["a.is_trump"]
        self._fa_follows: int = off + fak["a.follows"]
        self._fa_beats: int = off + fak["a.beats"]
        self._fa_diff: int = off + fak["a.diff"]
        self._fa_ctx_follow: int = off + fak["ctx.can_follow"]
        self._fa_ctx_beat: int = off + fak["ctx.can_beat"]

    # ---- helpers ----

    def _fill_state(
        self,
        x: np.ndarray,
        hand: Sequence[Card],
        pending_lead: Card | None,
        draw_pile_size: int,
        captured_self: Sequence[Card],
        captured_opp: Sequence[Card],
        trump_color: Color,
        trump_upcard: Card | None,
    ) -> None:
        """Write state features into *x* (1-D or a single row of a 2-D array)."""
        _cf = _CARD_FLAT_OBJ
        # Hand
        counts = [0] * _N_COLORS
        maxns = [0] * _N_COLORS
        for c in hand:
            ci = _COLOR_IDX[c.color]
            counts[ci] += 1
            if c.number > maxns[ci]:
                maxns[ci] = c.number
            x[self._s_hand_has[_cf[c]]] = 1.0
        for ci in range(_N_COLORS):
            x[self._s_hand_count[ci]] = float(counts[ci])
            x[self._s_hand_max[ci]] = float(maxns[ci]) * _INV_MAX_RANK
        x[self._s_hand_size] = float(len(hand)) / float(HAND_SIZE)

        # Public
        closed = 1.0 if draw_pile_size == 0 else 0.0
        x[self._s_pub_draw] = float(draw_pile_size) / 10.0
        x[self._s_pub_closed] = closed
        x[self._s_trump_color[_COLOR_IDX[trump_color]]] = 1.0
        x[self._s_trump_up_present] = 1.0 if trump_upcard is not None else 0.0
        x[self._s_trump_up_number] = 0.0 if trump_upcard is None else float(trump_upcard.number) * _INV_MAX_RANK
        for c in captured_self:
            x[self._s_cap_self_has[_cf[c]]] = 1.0
        x[self._s_cap_self_count] = float(len(captured_self)) / 20.0
        for c in captured_opp:
            x[self._s_cap_opp_has[_cf[c]]] = 1.0
        x[self._s_cap_opp_count] = float(len(captured_opp)) / 20.0

        # Unseen
        seen = np.zeros(_N_COLORS * _N_RANKS, dtype=np.float64)
        for c in captured_self:
            seen[_cf[c]] = 1.0
        for c in captured_opp:
            seen[_cf[c]] = 1.0
        for c in hand:
            seen[_cf[c]] = 1.0
        if pending_lead is not None:
            seen[_cf[pending_lead]] = 1.0
        unseen = 1.0 - seen
        x[self._s_unseen_has] = unseen
        if closed:
            x[self._s_opp_known_has] = unseen

        # Phase + lead card
        if pending_lead is not None:
            x[self._s_phase_following] = 1.0
            lci = _COLOR_IDX[pending_lead.color]
            x[self._s_lead_color[lci]] = 1.0
            x[self._s_lead_number] = float(pending_lead.number) * _INV_MAX_RANK
            x[self._s_lead_is_trump] = 1.0 if lci == _COLOR_IDX[trump_color] else 0.0

    # ---- public API ----

    def encode_state(
        self,
        hand: Sequence[Card],
        pending_lead: Card | None,
        draw_pile_size: int,
        captured_self: Sequence[Card],
        captured_opp: Sequence[Card],
        trump_color: Color,
        trump_upcard: Card | None,
    ) -> np.ndarray:
        """Encode state for the value head → ``(state_dim,)``."""
        x = np.zeros(self.state_dim, dtype=np.float64)
        self._fill_state(x, hand, pending_lead, draw_pile_size,
                         captured_self, captured_opp, trump_color, trump_upcard)
        return x

    def encode_lead_policy(
        self,
        hand: Sequence[Card],
        actions: List[Card | str],
        draw_pile_size: int,
        captured_self: Sequence[Card],
        captured_opp: Sequence[Card],
        trump_color: Color,
        trump_upcard: Card | None,
        can_close: bool = False,
    ) -> np.ndarray:
        """Encode lead actions → ``(N, lead_policy_dim)``."""
        N = len(actions)
        X = np.zeros((N, self.lead_policy_dim), dtype=np.float64)
        # State features in row 0
        self._fill_state(X[0], hand, None, draw_pile_size,
                         captured_self, captured_opp, trump_color, trump_upcard)
        if N > 1:
            X[1:, :self.state_dim] = X[0, :self.state_dim]

        tc_idx = _COLOR_IDX[trump_color]
        for i, a in enumerate(actions):
            row = X[i]
            if a == "close_talon":
                row[self._la_kind_close] = 1.0
                row[self._la_ctx_close] = 1.0
            else:
                ci = _COLOR_IDX[a.color]
                row[self._la_kind_card] = 1.0
                row[self._la_color[ci]] = 1.0
                row[self._la_number] = float(a.number) * _INV_MAX_RANK
                row[self._la_is_high] = 1.0 if a.number >= 10 else 0.0
                row[self._la_is_trump] = 1.0 if ci == tc_idx else 0.0
        return X

    def encode_follow_policy(
        self,
        hand: Sequence[Card],
        lead_card: Card,
        actions: List[Card],
        draw_pile_size: int,
        captured_self: Sequence[Card],
        captured_opp: Sequence[Card],
        trump_color: Color,
        trump_upcard: Card | None,
    ) -> np.ndarray:
        """Encode follow actions → ``(N, follow_policy_dim)``."""
        N = len(actions)
        X = np.zeros((N, self.follow_policy_dim), dtype=np.float64)
        # State features in row 0 (lead_card is part of state for follower)
        self._fill_state(X[0], hand, lead_card, draw_pile_size,
                         captured_self, captured_opp, trump_color, trump_upcard)
        if N > 1:
            X[1:, :self.state_dim] = X[0, :self.state_dim]

        tc_idx = _COLOR_IDX[trump_color]
        lc_ci = _COLOR_IDX[lead_card.color]
        same_color = [c for c in hand if _COLOR_IDX[c.color] == lc_ci]
        # Shared context features (same for all actions)
        can_follow = 1.0 if same_color else 0.0
        can_beat = 1.0 if any(c.number > lead_card.number for c in same_color) else 0.0

        for i, a in enumerate(actions):
            row = X[i]
            ci = _COLOR_IDX[a.color]
            row[self._fa_color[ci]] = 1.0
            row[self._fa_number] = float(a.number) * _INV_MAX_RANK
            row[self._fa_is_trump] = 1.0 if ci == tc_idx else 0.0
            follows = a.color == lead_card.color
            row[self._fa_follows] = 1.0 if follows else 0.0
            row[self._fa_beats] = 1.0 if follows and a.number > lead_card.number else 0.0
            row[self._fa_diff] = float(a.number - lead_card.number) * _INV_MAX_RANK
            row[self._fa_ctx_follow] = can_follow
            row[self._fa_ctx_beat] = can_beat
        return X


# Module-level singleton
_alpha_encoder: AlphaEncoder | None = None


def get_alpha_encoder() -> AlphaEncoder:
    """Return the module-level AlphaEncoder singleton."""
    global _alpha_encoder
    if _alpha_encoder is None:
        _alpha_encoder = AlphaEncoder()
    return _alpha_encoder
