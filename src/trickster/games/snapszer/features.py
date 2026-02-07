from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from trickster.games.snapszer.cards import ALL_COLORS, HAND_SIZE, MAX_RANK, RANK_VALUES, Card, Color


def _one_hot_color(prefix: str, color: Color) -> Dict[str, float]:
    return {f"{prefix}.color.{c.value}": 1.0 if c == color else 0.0 for c in ALL_COLORS}


def _hand_features(hand: Sequence[Card]) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    # Presence for each (color, rank) pair: len(ALL_COLORS) * len(RANK_VALUES) binary features
    present = {(c.color, c.number) for c in hand}
    for col in ALL_COLORS:
        cnt = 0
        maxn = 0
        for n in RANK_VALUES:
            if (col, n) in present:
                feats[f"hand.has.{col.value}.{n}"] = 1.0
                cnt += 1
                if n > maxn:
                    maxn = n
            else:
                feats[f"hand.has.{col.value}.{n}"] = 0.0
        feats[f"hand.count.{col.value}"] = float(cnt)
        feats[f"hand.max.{col.value}"] = float(maxn) / float(MAX_RANK)

    # Variant A: hand size is typically HAND_SIZE (except near the end).
    feats["hand.size"] = float(len(hand)) / float(HAND_SIZE)
    return feats


def _card_key(c: Card) -> tuple[Color, int]:
    return (c.color, int(c.number))


def _public_features(
    *,
    hand: Sequence[Card],
    draw_pile_size: int,
    captured_self: Sequence[Card],
    captured_opp: Sequence[Card],
    public_seen_extra: Sequence[Card],
    trump_color: Color,
    trump_upcard: Card | None,
) -> Dict[str, float]:
    """
    Public / realistic information:
    - draw pile size (but not order)
    - captured cards for both players (who captured what)
    - any currently face-up seen cards (e.g. the lead card while responding)
    """
    feats: Dict[str, float] = {}

    # Phase flag: once the draw pile is empty, strict follow/beat rules kick in.
    closed = 1.0 if int(draw_pile_size) == 0 else 0.0
    # Maximum draw pile size in this 20-card, 5-hand game is 10.
    feats["pub.draw_pile_frac"] = float(draw_pile_size) / 10.0
    feats["pub.phase.closed"] = closed

    # Trump is public information.
    for col in ALL_COLORS:
        feats[f"pub.trump.color.{col.value}"] = 1.0 if col == trump_color else 0.0
    feats["pub.trump_up.present"] = 1.0 if trump_upcard is not None else 0.0
    feats["pub.trump_up.number"] = 0.0 if trump_upcard is None else (float(trump_upcard.number) / float(MAX_RANK))

    self_set = {_card_key(c) for c in captured_self}
    opp_set = {_card_key(c) for c in captured_opp}

    # Captured-card visibility (public information).
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            feats[f"pub.cap.self.has.{col.value}.{n}"] = 1.0 if (col, n) in self_set else 0.0
            feats[f"pub.cap.opp.has.{col.value}.{n}"] = 1.0 if (col, n) in opp_set else 0.0
    feats["pub.cap.self.count"] = float(len(captured_self)) / 20.0
    feats["pub.cap.opp.count"] = float(len(captured_opp)) / 20.0

    # Unseen pool: cards not in my hand and not already captured (and not currently face-up).
    # This is realistic to track and does NOT reveal draw order or the opponent's hidden hand.
    full = {(col, n) for col in ALL_COLORS for n in RANK_VALUES}
    my_hand = {_card_key(c) for c in hand}
    seen_extra = {_card_key(c) for c in public_seen_extra}
    unseen = full - self_set - opp_set - my_hand - seen_extra
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            feats[f"pub.unseen.has.{col.value}.{n}"] = 1.0 if (col, n) in unseen else 0.0

    # In the closed phase (draw pile empty), the opponent's remaining hand is fully determined
    # by public information (deck composition + captures + my hand + face-up lead card).
    if closed:
        for col in ALL_COLORS:
            for n in RANK_VALUES:
                feats[f"opp.known.has.{col.value}.{n}"] = 1.0 if (col, n) in unseen else 0.0
    else:
        for col in ALL_COLORS:
            for n in RANK_VALUES:
                feats[f"opp.known.has.{col.value}.{n}"] = 0.0

    return feats


def lead_action_features(
    hand: Sequence[Card],
    action: Card,
    *,
    draw_pile_size: int,
    captured_self: Sequence[Card],
    captured_opp: Sequence[Card],
    trump_color: Color,
    trump_upcard: Card | None,
    exchanged_trump: bool = False,
) -> Dict[str, float]:
    feats = {}
    feats.update(_hand_features(hand))
    feats.update(
        _public_features(
            hand=hand,
            draw_pile_size=draw_pile_size,
            captured_self=captured_self,
            captured_opp=captured_opp,
            public_seen_extra=(),
            trump_color=trump_color,
            trump_upcard=trump_upcard,
        )
    )
    feats["a.kind.card"] = 1.0
    feats["a.kind.close_talon"] = 0.0
    feats.update(_one_hot_color("a", action.color))
    feats["a.number"] = float(action.number) / float(MAX_RANK)
    # "High" roughly corresponds to 10/A in Schnapsen.
    feats["a.is_high"] = 1.0 if action.number >= 10 else 0.0
    feats["a.is_trump"] = 1.0 if action.color == trump_color else 0.0
    feats["ctx.exchanged_trump"] = 1.0 if exchanged_trump else 0.0
    feats["ctx.can_close_talon"] = 0.0
    return feats


def lead_close_talon_features(
    hand: Sequence[Card],
    *,
    draw_pile_size: int,
    captured_self: Sequence[Card],
    captured_opp: Sequence[Card],
    trump_color: Color,
    trump_upcard: Card | None,
) -> Dict[str, float]:
    """
    Feature vector for the "close talon (takarás)" lead-side action.
    Uses the same lead model as card actions, but marks the action kind and leaves
    card-specific fields at neutral/zero values.
    """
    feats: Dict[str, float] = {}
    feats.update(_hand_features(hand))
    feats.update(
        _public_features(
            hand=hand,
            draw_pile_size=draw_pile_size,
            captured_self=captured_self,
            captured_opp=captured_opp,
            public_seen_extra=(),
            trump_color=trump_color,
            trump_upcard=trump_upcard,
        )
    )
    feats["a.kind.card"] = 0.0
    feats["a.kind.close_talon"] = 1.0
    for col in ALL_COLORS:
        feats[f"a.color.{col.value}"] = 0.0
    feats["a.number"] = 0.0
    feats["a.is_high"] = 0.0
    feats["a.is_trump"] = 0.0
    feats["ctx.exchanged_trump"] = 0.0
    feats["ctx.can_close_talon"] = 1.0
    return feats


def follow_action_features(
    hand: Sequence[Card],
    lead: Card,
    action: Card,
    *,
    draw_pile_size: int,
    captured_self: Sequence[Card],
    captured_opp: Sequence[Card],
    trump_color: Color,
    trump_upcard: Card | None,
) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats.update(_hand_features(hand))
    # While responding, the lead card is face-up and therefore public information.
    feats.update(
        _public_features(
            hand=hand,
            draw_pile_size=draw_pile_size,
            captured_self=captured_self,
            captured_opp=captured_opp,
            public_seen_extra=(lead,),
            trump_color=trump_color,
            trump_upcard=trump_upcard,
        )
    )

    feats.update(_one_hot_color("lead", lead.color))
    feats["lead.number"] = float(lead.number) / float(MAX_RANK)
    feats["lead.is_trump"] = 1.0 if lead.color == trump_color else 0.0

    feats.update(_one_hot_color("a", action.color))
    feats["a.number"] = float(action.number) / float(MAX_RANK)
    feats["a.is_trump"] = 1.0 if action.color == trump_color else 0.0

    follows = action.color == lead.color
    beats = follows and (action.number > lead.number)
    feats["a.follows"] = 1.0 if follows else 0.0
    feats["a.beats"] = 1.0 if beats else 0.0
    feats["a.diff"] = float(action.number - lead.number) / float(MAX_RANK)

    # "Can I follow / can I beat" from this hand?
    same = [c for c in hand if c.color == lead.color]
    feats["ctx.can_follow"] = 1.0 if same else 0.0
    feats["ctx.can_beat"] = 1.0 if any(c.number > lead.number for c in same) else 0.0

    return feats


def lead_feature_keys() -> list[str]:
    keys: list[str] = []
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            keys.append(f"hand.has.{col.value}.{n}")
        keys.append(f"hand.count.{col.value}")
        keys.append(f"hand.max.{col.value}")
    keys.append("hand.size")

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


def follow_feature_keys() -> list[str]:
    keys: list[str] = []
    for col in ALL_COLORS:
        for n in RANK_VALUES:
            keys.append(f"hand.has.{col.value}.{n}")
        keys.append(f"hand.count.{col.value}")
        keys.append(f"hand.max.{col.value}")
    keys.append("hand.size")

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

    for col in ALL_COLORS:
        keys.append(f"lead.color.{col.value}")
    keys.append("lead.number")
    keys.append("lead.is_trump")

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
#  AlphaZero feature key functions (state / action separation)
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
#  Fast numpy-native encoder (bypasses dicts and string keys entirely)
# ---------------------------------------------------------------------------

_N_COLORS = len(ALL_COLORS)
_N_RANKS = len(RANK_VALUES)
_INV_MAX_RANK = 1.0 / float(MAX_RANK)

# Precompute mappings at module level (avoids enum.value at runtime)
_COLOR_IDX: dict[Color, int] = {c: i for i, c in enumerate(ALL_COLORS)}
_RANK_IDX: dict[int, int] = {n: i for i, n in enumerate(RANK_VALUES)}
_CARD_FLAT: dict[tuple[Color, int], int] = {
    (c, n): ci * _N_RANKS + ri
    for ci, c in enumerate(ALL_COLORS)
    for ri, n in enumerate(RANK_VALUES)
}


class FastEncoder:
    """Numpy-native batch encoder for lead / follow features.

    Precomputes integer index arrays at construction time (from the
    canonical key lists, so it's always in sync).  At runtime every feature
    is written via integer array indexing — no dicts, no string keys, no
    ``enum.value`` calls.

    Usage::

        enc = FastEncoder()
        X = enc.encode_lead_batch(hand, actions, ...)
        probs = model.forward_raw(X)
    """

    __slots__ = (
        "lead_dim", "follow_dim",
        # hand feature indices (shared lead/follow — same offsets)
        "_hand_has", "_hand_count", "_hand_max", "_hand_size",
        # public feature indices (lead)
        "_l_pub_draw", "_l_pub_closed",
        "_l_trump_color", "_l_trump_up_present", "_l_trump_up_number",
        "_l_cap_self_has", "_l_cap_opp_has",
        "_l_cap_self_count", "_l_cap_opp_count",
        "_l_unseen_has", "_l_opp_known_has",
        # lead action indices
        "_l_a_color", "_l_a_kind_card", "_l_a_kind_close",
        "_l_a_number", "_l_a_is_high", "_l_a_is_trump",
        "_l_ctx_exch", "_l_ctx_close",
        # follow public indices (same structure, different offsets)
        "_f_pub_draw", "_f_pub_closed",
        "_f_trump_color", "_f_trump_up_present", "_f_trump_up_number",
        "_f_cap_self_has", "_f_cap_opp_has",
        "_f_cap_self_count", "_f_cap_opp_count",
        "_f_unseen_has", "_f_opp_known_has",
        # follow-specific: hand indices
        "_f_hand_has", "_f_hand_count", "_f_hand_max", "_f_hand_size",
        # follow lead-card indices (shared across batch)
        "_f_lead_color", "_f_lead_number", "_f_lead_is_trump",
        # follow action indices
        "_f_a_color", "_f_a_number", "_f_a_is_trump",
        "_f_a_follows", "_f_a_beats", "_f_a_diff",
        "_f_ctx_can_follow", "_f_ctx_can_beat",
    )

    def __init__(self) -> None:
        lk = {k: i for i, k in enumerate(lead_feature_keys())}
        fk = {k: i for i, k in enumerate(follow_feature_keys())}
        self.lead_dim = len(lk)
        self.follow_dim = len(fk)

        cols = ALL_COLORS
        ranks = RANK_VALUES

        # --- lead indices ---
        self._hand_has = np.array(
            [lk[f"hand.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._hand_count = np.array([lk[f"hand.count.{c.value}"] for c in cols], dtype=np.intp)
        self._hand_max = np.array([lk[f"hand.max.{c.value}"] for c in cols], dtype=np.intp)
        self._hand_size = lk["hand.size"]

        self._l_pub_draw = lk["pub.draw_pile_frac"]
        self._l_pub_closed = lk["pub.phase.closed"]
        self._l_trump_color = np.array([lk[f"pub.trump.color.{c.value}"] for c in cols], dtype=np.intp)
        self._l_trump_up_present = lk["pub.trump_up.present"]
        self._l_trump_up_number = lk["pub.trump_up.number"]
        self._l_cap_self_has = np.array(
            [lk[f"pub.cap.self.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._l_cap_opp_has = np.array(
            [lk[f"pub.cap.opp.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._l_cap_self_count = lk["pub.cap.self.count"]
        self._l_cap_opp_count = lk["pub.cap.opp.count"]
        self._l_unseen_has = np.array(
            [lk[f"pub.unseen.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._l_opp_known_has = np.array(
            [lk[f"opp.known.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._l_a_color = np.array([lk[f"a.color.{c.value}"] for c in cols], dtype=np.intp)
        self._l_a_kind_card = lk["a.kind.card"]
        self._l_a_kind_close = lk["a.kind.close_talon"]
        self._l_a_number = lk["a.number"]
        self._l_a_is_high = lk["a.is_high"]
        self._l_a_is_trump = lk["a.is_trump"]
        self._l_ctx_exch = lk["ctx.exchanged_trump"]
        self._l_ctx_close = lk["ctx.can_close_talon"]

        # --- follow indices ---
        self._f_hand_has = np.array(
            [fk[f"hand.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._f_hand_count = np.array([fk[f"hand.count.{c.value}"] for c in cols], dtype=np.intp)
        self._f_hand_max = np.array([fk[f"hand.max.{c.value}"] for c in cols], dtype=np.intp)
        self._f_hand_size = fk["hand.size"]

        self._f_pub_draw = fk["pub.draw_pile_frac"]
        self._f_pub_closed = fk["pub.phase.closed"]
        self._f_trump_color = np.array([fk[f"pub.trump.color.{c.value}"] for c in cols], dtype=np.intp)
        self._f_trump_up_present = fk["pub.trump_up.present"]
        self._f_trump_up_number = fk["pub.trump_up.number"]
        self._f_cap_self_has = np.array(
            [fk[f"pub.cap.self.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._f_cap_opp_has = np.array(
            [fk[f"pub.cap.opp.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._f_cap_self_count = fk["pub.cap.self.count"]
        self._f_cap_opp_count = fk["pub.cap.opp.count"]
        self._f_unseen_has = np.array(
            [fk[f"pub.unseen.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._f_opp_known_has = np.array(
            [fk[f"opp.known.has.{c.value}.{n}"] for c in cols for n in ranks], dtype=np.intp
        )
        self._f_lead_color = np.array([fk[f"lead.color.{c.value}"] for c in cols], dtype=np.intp)
        self._f_lead_number = fk["lead.number"]
        self._f_lead_is_trump = fk["lead.is_trump"]
        self._f_a_color = np.array([fk[f"a.color.{c.value}"] for c in cols], dtype=np.intp)
        self._f_a_number = fk["a.number"]
        self._f_a_is_trump = fk["a.is_trump"]
        self._f_a_follows = fk["a.follows"]
        self._f_a_beats = fk["a.beats"]
        self._f_a_diff = fk["a.diff"]
        self._f_ctx_can_follow = fk["ctx.can_follow"]
        self._f_ctx_can_beat = fk["ctx.can_beat"]

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _card_flat_indices(cards: Sequence[Card]) -> list[int]:
        return [_CARD_FLAT[(c.color, c.number)] for c in cards]

    def _fill_hand(self, x: np.ndarray, hand: Sequence[Card],
                   has_idx: np.ndarray, count_idx: np.ndarray,
                   max_idx: np.ndarray, size_idx: int) -> None:
        counts = [0] * _N_COLORS
        maxns = [0] * _N_COLORS
        for c in hand:
            ci = _COLOR_IDX[c.color]
            counts[ci] += 1
            if c.number > maxns[ci]:
                maxns[ci] = c.number
            x[has_idx[_CARD_FLAT[(c.color, c.number)]]] = 1.0
        for ci in range(_N_COLORS):
            x[count_idx[ci]] = float(counts[ci])
            x[max_idx[ci]] = float(maxns[ci]) * _INV_MAX_RANK
        x[size_idx] = float(len(hand)) / float(HAND_SIZE)

    def _fill_public(
        self, x: np.ndarray,
        hand: Sequence[Card],
        draw_pile_size: int,
        captured_self: Sequence[Card],
        captured_opp: Sequence[Card],
        seen_extra: Sequence[Card],
        trump_color: Color,
        trump_upcard,
        # index arrays
        pub_draw: int, pub_closed: int,
        trump_color_idx: np.ndarray, trump_up_present: int, trump_up_number: int,
        cap_self_has: np.ndarray, cap_opp_has: np.ndarray,
        cap_self_count: int, cap_opp_count: int,
        unseen_has: np.ndarray, opp_known_has: np.ndarray,
    ) -> None:
        closed = 1.0 if draw_pile_size == 0 else 0.0
        x[pub_draw] = float(draw_pile_size) / 10.0
        x[pub_closed] = closed
        x[trump_color_idx[_COLOR_IDX[trump_color]]] = 1.0
        x[trump_up_present] = 1.0 if trump_upcard is not None else 0.0
        x[trump_up_number] = 0.0 if trump_upcard is None else float(trump_upcard.number) * _INV_MAX_RANK

        for c in captured_self:
            x[cap_self_has[_CARD_FLAT[(c.color, c.number)]]] = 1.0
        x[cap_self_count] = float(len(captured_self)) / 20.0
        for c in captured_opp:
            x[cap_opp_has[_CARD_FLAT[(c.color, c.number)]]] = 1.0
        x[cap_opp_count] = float(len(captured_opp)) / 20.0

        # Unseen = full deck - self_captured - opp_captured - hand - seen_extra
        seen = np.zeros(_N_COLORS * _N_RANKS, dtype=np.float64)
        for c in captured_self:
            seen[_CARD_FLAT[(c.color, c.number)]] = 1.0
        for c in captured_opp:
            seen[_CARD_FLAT[(c.color, c.number)]] = 1.0
        for c in hand:
            seen[_CARD_FLAT[(c.color, c.number)]] = 1.0
        for c in seen_extra:
            seen[_CARD_FLAT[(c.color, c.number)]] = 1.0
        unseen = 1.0 - seen
        x[unseen_has] = unseen
        if closed:
            x[opp_known_has] = unseen

    # ----- public batch encoders ------------------------------------------

    def encode_lead_batch(
        self,
        hand: Sequence[Card],
        actions: List[Card],
        draw_pile_size: int,
        captured_self: Sequence[Card],
        captured_opp: Sequence[Card],
        trump_color: Color,
        trump_upcard,
    ) -> np.ndarray:
        """Encode N lead card actions from one state → (N, lead_dim) array."""
        N = len(actions)
        X = np.zeros((N, self.lead_dim), dtype=np.float64)

        # State features — compute in row 0
        self._fill_hand(X[0], hand,
                        self._hand_has, self._hand_count, self._hand_max, self._hand_size)
        self._fill_public(X[0], hand, draw_pile_size, captured_self, captured_opp,
                          (), trump_color, trump_upcard,
                          self._l_pub_draw, self._l_pub_closed,
                          self._l_trump_color, self._l_trump_up_present, self._l_trump_up_number,
                          self._l_cap_self_has, self._l_cap_opp_has,
                          self._l_cap_self_count, self._l_cap_opp_count,
                          self._l_unseen_has, self._l_opp_known_has)
        # Broadcast state to all rows
        if N > 1:
            X[1:] = X[0]

        # Action features (per row)
        tc_idx = _COLOR_IDX[trump_color]
        for i, a in enumerate(actions):
            row = X[i]
            row[self._l_a_kind_card] = 1.0
            ci = _COLOR_IDX[a.color]
            row[self._l_a_color[ci]] = 1.0
            row[self._l_a_number] = float(a.number) * _INV_MAX_RANK
            row[self._l_a_is_high] = 1.0 if a.number >= 10 else 0.0
            row[self._l_a_is_trump] = 1.0 if ci == tc_idx else 0.0
        return X

    def encode_follow_batch(
        self,
        hand: Sequence[Card],
        lead_card: Card,
        actions: List[Card],
        draw_pile_size: int,
        captured_self: Sequence[Card],
        captured_opp: Sequence[Card],
        trump_color: Color,
        trump_upcard,
    ) -> np.ndarray:
        """Encode N follow card actions from one state → (N, follow_dim) array."""
        N = len(actions)
        X = np.zeros((N, self.follow_dim), dtype=np.float64)

        # State features — row 0
        self._fill_hand(X[0], hand,
                        self._f_hand_has, self._f_hand_count, self._f_hand_max, self._f_hand_size)
        self._fill_public(X[0], hand, draw_pile_size, captured_self, captured_opp,
                          (lead_card,), trump_color, trump_upcard,
                          self._f_pub_draw, self._f_pub_closed,
                          self._f_trump_color, self._f_trump_up_present, self._f_trump_up_number,
                          self._f_cap_self_has, self._f_cap_opp_has,
                          self._f_cap_self_count, self._f_cap_opp_count,
                          self._f_unseen_has, self._f_opp_known_has)
        # Lead card features (same for all rows)
        lc_ci = _COLOR_IDX[lead_card.color]
        tc_idx = _COLOR_IDX[trump_color]
        X[0, self._f_lead_color[lc_ci]] = 1.0
        X[0, self._f_lead_number] = float(lead_card.number) * _INV_MAX_RANK
        X[0, self._f_lead_is_trump] = 1.0 if lc_ci == tc_idx else 0.0
        # Contextual (shared): can_follow, can_beat
        same_color = [c for c in hand if _COLOR_IDX[c.color] == lc_ci]
        can_follow = 1.0 if same_color else 0.0
        can_beat = 1.0 if any(c.number > lead_card.number for c in same_color) else 0.0
        X[0, self._f_ctx_can_follow] = can_follow
        X[0, self._f_ctx_can_beat] = can_beat

        # Broadcast state to all rows
        if N > 1:
            X[1:] = X[0]

        # Action features (per row)
        for i, a in enumerate(actions):
            row = X[i]
            ci = _COLOR_IDX[a.color]
            row[self._f_a_color[ci]] = 1.0
            row[self._f_a_number] = float(a.number) * _INV_MAX_RANK
            row[self._f_a_is_trump] = 1.0 if ci == tc_idx else 0.0
            follows = a.color == lead_card.color
            row[self._f_a_follows] = 1.0 if follows else 0.0
            row[self._f_a_beats] = 1.0 if follows and a.number > lead_card.number else 0.0
            row[self._f_a_diff] = float(a.number - lead_card.number) * _INV_MAX_RANK
        return X


# Module-level singleton (constructed once at import time)
_fast_encoder: FastEncoder | None = None


def get_fast_encoder() -> FastEncoder:
    """Return the module-level FastEncoder singleton."""
    global _fast_encoder
    if _fast_encoder is None:
        _fast_encoder = FastEncoder()
    return _fast_encoder


# ---------------------------------------------------------------------------
#  AlphaZero encoder (state / action separation for value + policy heads)
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

    # ---- helpers (same logic as FastEncoder._fill_*) ----

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
        # Hand
        counts = [0] * _N_COLORS
        maxns = [0] * _N_COLORS
        for c in hand:
            ci = _COLOR_IDX[c.color]
            counts[ci] += 1
            if c.number > maxns[ci]:
                maxns[ci] = c.number
            x[self._s_hand_has[_CARD_FLAT[(c.color, c.number)]]] = 1.0
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
            x[self._s_cap_self_has[_CARD_FLAT[(c.color, c.number)]]] = 1.0
        x[self._s_cap_self_count] = float(len(captured_self)) / 20.0
        for c in captured_opp:
            x[self._s_cap_opp_has[_CARD_FLAT[(c.color, c.number)]]] = 1.0
        x[self._s_cap_opp_count] = float(len(captured_opp)) / 20.0

        # Unseen
        seen = np.zeros(_N_COLORS * _N_RANKS, dtype=np.float64)
        for c in captured_self:
            seen[_CARD_FLAT[(c.color, c.number)]] = 1.0
        for c in captured_opp:
            seen[_CARD_FLAT[(c.color, c.number)]] = 1.0
        for c in hand:
            seen[_CARD_FLAT[(c.color, c.number)]] = 1.0
        if pending_lead is not None:
            seen[_CARD_FLAT[(pending_lead.color, pending_lead.number)]] = 1.0
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

