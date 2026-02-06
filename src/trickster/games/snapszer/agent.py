from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from trickster.games.snapszer.features import follow_action_features, lead_action_features, lead_close_talon_features
from trickster.models.linear import LinearBinaryModel
from trickster.games.snapszer.cards import Card, Color


@dataclass(slots=True)
class LearnedAgent:
    lead_model: LinearBinaryModel
    follow_model: LinearBinaryModel
    rng: random.Random
    epsilon: float = 0.05

    def choose_lead(
        self,
        hand: Sequence[Card],
        legal: Sequence[Card],
        *,
        draw_pile_size: int = 0,
        captured_self: Sequence[Card] = (),
        captured_opp: Sequence[Card] = (),
        trump_color: Optional[Color] = None,
        trump_upcard: Optional[Card] = None,
    ) -> Card:
        if not legal:
            raise ValueError("No legal actions for lead")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(legal))

        # Break ties randomly to avoid "first legal card" bias when probabilities match
        # (common for untrained models where all logits are 0 => p=0.5).
        best: list[Card] = []
        best_p = float("-inf")
        tol = 1e-12
        for a in legal:
            p = self.lead_model.predict_proba(
                lead_action_features(
                    hand,
                    a,
                    draw_pile_size=draw_pile_size,
                    captured_self=captured_self,
                    captured_opp=captured_opp,
                    trump_color=trump_color or a.color,
                    trump_upcard=trump_upcard,
                    exchanged_trump=False,
                )
            )
            if p > best_p + tol:
                best_p = p
                best = [a]
            elif abs(p - best_p) <= tol:
                best.append(a)
        return self.rng.choice(best) if best else self.rng.choice(list(legal))

    def choose_lead_or_close_talon(
        self,
        hand: Sequence[Card],
        legal: Sequence[Card],
        *,
        can_close_talon: bool,
        draw_pile_size: int = 0,
        captured_self: Sequence[Card] = (),
        captured_opp: Sequence[Card] = (),
        trump_color: Optional[Color] = None,
        trump_upcard: Optional[Card] = None,
    ) -> tuple[bool, Card]:
        """
        Choose between:
        - closing the talon (takarás) right now (if legal), OR
        - leading a card as usual.

        The decision is learned: we score a synthetic "close" action with the same
        lead_model and compare it to the best lead-card score.
        """
        if not legal:
            raise ValueError("No legal actions for lead")

        # Exploration across the *combined* action set.
        if self.rng.random() < self.epsilon:
            if can_close_talon and self.rng.random() < (1.0 / (len(legal) + 1)):
                return True, self.rng.choice(list(legal))
            return False, self.rng.choice(list(legal))

        tc = trump_color or legal[0].color

        # Score best card.
        best_cards: list[Card] = []
        best_p = float("-inf")
        tol = 1e-12
        for a in legal:
            p = self.lead_model.predict_proba(
                lead_action_features(
                    hand,
                    a,
                    draw_pile_size=draw_pile_size,
                    captured_self=captured_self,
                    captured_opp=captured_opp,
                    trump_color=tc,
                    trump_upcard=trump_upcard,
                    exchanged_trump=False,
                )
            )
            if p > best_p + tol:
                best_p = p
                best_cards = [a]
            elif abs(p - best_p) <= tol:
                best_cards.append(a)
        best_card = self.rng.choice(best_cards) if best_cards else self.rng.choice(list(legal))

        if not can_close_talon:
            return False, best_card

        # Score "close talon" action.
        p_close = self.lead_model.predict_proba(
            lead_close_talon_features(
                hand,
                draw_pile_size=draw_pile_size,
                captured_self=captured_self,
                captured_opp=captured_opp,
                trump_color=tc,
                trump_upcard=trump_upcard,
            )
        )

        return (p_close > best_p + tol), best_card

    def choose_follow(
        self,
        hand: Sequence[Card],
        lead: Card,
        legal: Sequence[Card],
        *,
        draw_pile_size: int = 0,
        captured_self: Sequence[Card] = (),
        captured_opp: Sequence[Card] = (),
        trump_color: Optional[Color] = None,
        trump_upcard: Optional[Card] = None,
    ) -> Card:
        if not legal:
            raise ValueError("No legal actions for follow")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(legal))

        best: list[Card] = []
        best_p = float("-inf")
        tol = 1e-12
        for a in legal:
            p = self.follow_model.predict_proba(
                follow_action_features(
                    hand,
                    lead,
                    a,
                    draw_pile_size=draw_pile_size,
                    captured_self=captured_self,
                    captured_opp=captured_opp,
                    trump_color=trump_color or lead.color,
                    trump_upcard=trump_upcard,
                )
            )
            if p > best_p + tol:
                best_p = p
                best = [a]
            elif abs(p - best_p) <= tol:
                best.append(a)
        return self.rng.choice(best) if best else self.rng.choice(list(legal))

