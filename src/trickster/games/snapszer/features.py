from __future__ import annotations

from typing import Dict, Sequence

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

