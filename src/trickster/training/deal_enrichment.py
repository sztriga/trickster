"""Deal enrichment and neural talon discard for training.

  - ``_value_enriched_new_game``: re-deal until value head approves
  - ``_new_game_with_neural_discard``: value-head-driven talon discard
  - ``_enrichment_threshold``: anneal minimum soloist value over training
"""
from __future__ import annotations

import copy
import random
from itertools import combinations

import numpy as np

from trickster.games.ulti.adapter import UltiGame, UltiNode, build_auction_constraints
from trickster.games.ulti.cards import Card, Rank, Suit
from trickster.games.ulti.game import (
    NUM_PLAYERS,
    deal,
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
)
from trickster.model import UltiNetWrapper


# ---------------------------------------------------------------------------
#  Deal enrichment via value head
# ---------------------------------------------------------------------------


def _value_enriched_new_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    seed: int,
    min_value: float,
    training_mode: str = "simple",
    max_attempts: int = 20,
) -> tuple[UltiNode, float]:
    """Deal games until the value head rates the soloist above *min_value*.

    Returns (state, soloist_value) — the best deal found, or the last
    attempt if none pass the threshold.
    """
    best_state = None
    best_val = -1.0
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt * 100_000
        state = game.new_game(
            seed=attempt_seed,
            training_mode=training_mode,
            starting_leader=seed % 3,
        )
        sol = state.gs.soloist
        feats = game.encode_state(state, sol)
        val = wrapper.predict_value(feats)
        if val > best_val:
            best_state, best_val = state, val
        if val >= min_value:
            return state, val
    return best_state, best_val


def _enrichment_threshold(step: int, total_steps: int) -> float:
    """Return the minimum soloist value threshold for deal filtering.

    Currently disabled — always returns -999 (accept all deals).
    The value head must see the full distribution of hands (including
    bad ones) to produce accurate predictions for bidding.
    """
    return -999.0


# ---------------------------------------------------------------------------
#  Neural talon discard for training
# ---------------------------------------------------------------------------


def _new_game_with_neural_discard(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    seed: int,
    training_mode: str,
    dealer: int,
) -> UltiNode:
    """Deal a new training game with value-head-driven talon discard.

    For adu contracts (parti, ulti, 40-100): picks up the talon,
    evaluates up to 20 discard pairs via the value head, and keeps
    the best.  For betli: same but with betli flag.

    Falls back to the standard ``game.new_game()`` for unknown modes.
    """
    rng = random.Random(seed)
    gs, talon = deal(seed=seed, dealer=dealer)
    gs.training_mode = training_mode
    soloist = next_player(dealer)
    gs.soloist = soloist

    pickup_talon(gs, soloist, talon)

    if training_mode == "betli":
        trump = None
        betli = True
        set_contract(gs, soloist, trump=None, betli=True)
    else:
        betli = False
        suits_in_hand = list(set(c.suit for c in gs.hands[soloist]))
        trump = rng.choice(suits_in_hand)

        if training_mode == "40-100":
            trump_k = Card(trump, Rank.KING)
            trump_q = Card(trump, Rank.QUEEN)
            hand = gs.hands[soloist]
            for needed in (trump_k, trump_q):
                if needed not in hand:
                    all_cards = []
                    for p in range(NUM_PLAYERS):
                        if p != soloist:
                            all_cards.extend((p, c) for c in gs.hands[p])
                    for owner, card in all_cards:
                        if card == needed:
                            swappable = [c for c in hand
                                         if c != trump_k and c != trump_q]
                            give = rng.choice(swappable)
                            hand.remove(give)
                            hand.append(needed)
                            gs.hands[owner].remove(needed)
                            gs.hands[owner].append(give)
                            break

        set_contract(gs, soloist, trump=trump)

    # ── Neural discard: evaluate up to 20 best discard pairs ──
    hand = gs.hands[soloist]
    all_pairs = list(combinations(range(len(hand)), 2))

    max_pairs = 20
    if not betli and len(all_pairs) > max_pairs:
        def _discard_score(pair):
            score = 0.0
            for idx in pair:
                c = hand[idx]
                if c.suit == trump:
                    score += 100
                score -= c.points()
                if training_mode == "40-100":
                    if c == Card(trump, Rank.KING) or c == Card(trump, Rank.QUEEN):
                        score += 1000
                if training_mode == "ulti":
                    if trump and c == Card(trump, Rank.SEVEN):
                        score += 1000
            return score
        all_pairs.sort(key=_discard_score)
        all_pairs = all_pairs[:max_pairs]

    if betli:
        comps = frozenset({"betli"})
    else:
        comps = frozenset({"parti"})
        if training_mode == "ulti":
            comps = frozenset({"parti", "ulti"})
        elif training_mode == "40-100":
            comps = frozenset({"parti", "40", "100"})

    is_red = (trump is not None and trump == Suit.HEARTS)
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())

    feats_list: list[np.ndarray] = []
    valid_pairs: list[tuple[int, int]] = []

    for i, j in all_pairs:
        d0, d1 = hand[i], hand[j]

        gs_copy = copy.deepcopy(gs)
        discard_talon(gs_copy, [d0, d1])

        if training_mode == "ulti":
            gs_copy.has_ulti = True

        marriage_restrict = "40" if training_mode == "40-100" else None
        declare_all_marriages(gs_copy, soloist_marriage_restrict=marriage_restrict)

        constraints = build_auction_constraints(gs_copy, comps)

        node = UltiNode(
            gs=gs_copy,
            known_voids=empty_voids,
            bid_rank=1,
            is_red=is_red,
            contract_components=comps,
            dealer=dealer,
            must_have=constraints,
        )
        feats = game.encode_state(node, soloist)
        feats_list.append(feats)
        valid_pairs.append((i, j))

    if feats_list:
        batch = np.stack(feats_list)
        values = wrapper.batch_value(batch)
        best_idx = int(np.argmax(values))
        bi, bj = valid_pairs[best_idx]
        best_discards = [hand[bi], hand[bj]]
    else:
        best_discards = hand[:2]

    discard_talon(gs, best_discards)

    if training_mode == "ulti":
        gs.has_ulti = True

    marriage_restrict = "40" if training_mode == "40-100" else None
    declare_all_marriages(gs, soloist_marriage_restrict=marriage_restrict)
    constraints = build_auction_constraints(gs, comps)

    return UltiNode(
        gs=gs,
        known_voids=empty_voids,
        bid_rank=1,
        is_red=is_red,
        contract_components=comps,
        dealer=dealer,
        must_have=constraints,
    )
