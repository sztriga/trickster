"""UltiGame — implements GameInterface for AlphaZero training / MCTS.

Wraps the Ulti game engine behind the generic GameInterface protocol
so that MCTS, training, and evaluation code remain game-agnostic.

Key design points for Ulti (vs. Snapszer):
  - 3 players (1 soloist vs. 2 defenders)
  - Defenders form a *coalition*: ``same_team`` returns True for both.
  - Action space = 32 (one per card in the deck).
  - ``new_game`` automatically performs the pre-play setup
    (deal → talon pickup → random discard → random contract) so that
    training focuses on the play phase.

v2 upgrades:
  - UltiNode carries auction context (bid, kontras, dealer)
  - Inference-enhanced PIMC: determinize() respects auction constraints
    (marriage declarations, bid-implied card holdings)
  - Expanded encoder (259-dim) with card counting, contract DNA, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trickster.games.ulti.cards import (
    ALL_SUITS,
    Card,
    NUM_PLAYERS,
    Rank,
    Suit,
    make_deck,
)
from trickster.games.ulti.encoder import UltiEncoder, _CARD_IDX, NUM_CARDS, STATE_DIM
from trickster.games.ulti.game import (
    GameState,
    current_player as _current_player,
    deal,
    declare_all_marriages,
    discard_talon,
    is_terminal as _is_terminal,
    legal_actions as _legal_actions,
    next_player,
    pickup_talon,
    play_card,
    set_contract,
    soloist_lost_betli,
    soloist_won_simple,
)

# ---------------------------------------------------------------------------
#  Pre-computed card sets / indices
# ---------------------------------------------------------------------------

_ALL_CARDS: frozenset[Card] = frozenset(make_deck())

_IDX_TO_CARD: dict[int, Card] = {idx: card for card, idx in _CARD_IDX.items()}


# ---------------------------------------------------------------------------
#  State wrapper (v2 — carries auction context)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UltiNode:
    """Game node for MCTS: underlying GameState + inference context.

    ``known_voids[p]`` is the set of suits that player *p* is provably
    missing (inferred when they fail to follow suit).

    v2 additions:
    - ``bid_rank``: winning bid rank (1–38) for auction-context encoding
    - ``is_red``, ``is_open``: contract flags
    - ``contract_components``: frozenset of component labels
    - ``dealer``: dealer index (for seat-position encoding)
    - ``component_kontras``: per-component kontra levels
    - ``must_have``: auction-derived card constraints per player
      (e.g. soloist must hold trump K+Q if they bid 40-100)
    """

    gs: GameState
    known_voids: tuple[frozenset[Suit], frozenset[Suit], frozenset[Suit]]
    bid_rank: int = 0
    is_red: bool = False
    is_open: bool = False
    contract_components: frozenset[str] | None = None
    dealer: int = 0
    component_kontras: dict[str, int] = field(default_factory=dict)
    must_have: dict[int, frozenset[Card]] = field(default_factory=dict)

    def clone(self) -> UltiNode:
        return UltiNode(
            gs=self.gs.clone(),
            known_voids=self.known_voids,
            bid_rank=self.bid_rank,
            is_red=self.is_red,
            is_open=self.is_open,
            contract_components=self.contract_components,
            dealer=self.dealer,
            component_kontras=dict(self.component_kontras),
            must_have=dict(self.must_have),
        )


def build_auction_constraints(
    gs: GameState,
    contract_components: frozenset[str] | None,
) -> dict[int, frozenset[Card]]:
    """Derive must-have card constraints from auction/contract info.

    These constraints are used by ``determinize()`` to avoid generating
    impossible worlds during PIMC simulations.

    Rules:
    - If bid includes '40' (40-100): soloist must have had trump K+Q
      at the start of play (they declared it). If those cards haven't
      been captured yet and aren't in the observer's hand, they must
      be in the soloist's hand.
    - If bid includes '20' (20-100): soloist declared a non-trump K+Q.
      We know they hold at least one such pair (exact suit unknown).
    - If bid includes 'ulti': soloist likely holds trump 7.
    - Marriage declarations: any player who declared a marriage held
      K+Q of that suit at play start.
    """
    constraints: dict[int, set[Card]] = {i: set() for i in range(NUM_PLAYERS)}

    if contract_components is None or gs.trump is None:
        return {i: frozenset(s) for i, s in constraints.items()}

    # 40-100: soloist declared trump K+Q marriage
    if "40" in contract_components and gs.trump is not None:
        constraints[gs.soloist].add(Card(gs.trump, Rank.KING))
        constraints[gs.soloist].add(Card(gs.trump, Rank.QUEEN))

    # Ulti: soloist should hold trump 7 (not strictly required at start,
    # but extremely likely — and enforced by 7esre tartás during play)
    if "ulti" in contract_components and gs.trump is not None:
        constraints[gs.soloist].add(Card(gs.trump, Rank.SEVEN))

    # Marriage declarations: player held K+Q of the declared suit
    for player, suit, _pts in gs.marriages:
        constraints[player].add(Card(suit, Rank.KING))
        constraints[player].add(Card(suit, Rank.QUEEN))

    # Filter out cards that have already been captured or are in the
    # trick — these are no longer in anyone's hand.
    played: set[Card] = set()
    for pile in gs.captured:
        played.update(pile)
    for _, c in gs.trick_cards:
        played.add(c)

    for p in constraints:
        constraints[p] -= played

    return {i: frozenset(s) for i, s in constraints.items()}


# ---------------------------------------------------------------------------
#  GameInterface implementation
# ---------------------------------------------------------------------------


class UltiGame:
    """GameInterface implementation for the Ulti play phase (v2)."""

    def __init__(self) -> None:
        self._enc = UltiEncoder()

    # -- game rules --------------------------------------------------------

    @property
    def num_players(self) -> int:
        return NUM_PLAYERS

    def current_player(self, state: UltiNode) -> int:
        return _current_player(state.gs)

    def legal_actions(self, state: UltiNode) -> list[Card]:
        return _legal_actions(state.gs)

    def apply(self, state: UltiNode, action: Card) -> UltiNode:
        gs = state.gs.clone()
        voids = list(state.known_voids)

        player = _current_player(gs)

        # --- Void inference ---
        if gs.trick_cards:
            led_suit = gs.trick_cards[0][1].suit
            if action.suit != led_suit:
                v = set(voids[player])
                v.add(led_suit)
                if gs.trump is not None and action.suit != gs.trump:
                    v.add(gs.trump)
                voids[player] = frozenset(v)

        play_card(gs, action)
        return UltiNode(
            gs=gs,
            known_voids=tuple(voids),
            bid_rank=state.bid_rank,
            is_red=state.is_red,
            is_open=state.is_open,
            contract_components=state.contract_components,
            dealer=state.dealer,
            component_kontras=state.component_kontras,
            must_have=state.must_have,
        )

    def is_terminal(self, state: UltiNode) -> bool:
        return _is_terminal(state.gs)

    def outcome(self, state: UltiNode, player: int) -> float:
        """Outcome in [-1, +1]: +1 for the winning side, -1 for the losing."""
        gs = state.gs
        if gs.betli:
            soloist_wins = not soloist_lost_betli(gs)
        else:
            soloist_wins = soloist_won_simple(gs)

        if player == gs.soloist:
            return 1.0 if soloist_wins else -1.0
        return -1.0 if soloist_wins else 1.0

    # -- coalition / team --------------------------------------------------

    def same_team(
        self, state: UltiNode, player_a: int, player_b: int,
    ) -> bool:
        """Defenders are on the same team; soloist is alone."""
        if player_a == player_b:
            return True
        sol = state.gs.soloist
        return player_a != sol and player_b != sol

    # -- imperfect information (inference-enhanced PIMC) --------------------

    def determinize(
        self, state: UltiNode, player: int, rng: random.Random,
    ) -> UltiNode:
        """Sample a world consistent with *player*'s observations.

        v2 upgrades over v1:
        1. **Auction constraints**: Cards that a player MUST hold
           (inferred from bids/marriages) are assigned first.
        2. **Void constraints**: As before — don't give voided suits.
        3. **Terített visibility**: If the game is open (terített),
           the soloist's hand is visible to all.

        The result is a world where opponents' hands are plausible
        given all public information.
        """
        gs = state.gs.clone()

        # 1. Identify all cards visible to the observer
        known: set[Card] = set(gs.hands[player])
        for i in range(NUM_PLAYERS):
            known.update(gs.captured[i])
        for _, c in gs.trick_cards:
            known.add(c)

        # Terített: soloist's hand is visible to everyone
        if state.is_open and state.gs.soloist != player:
            known.update(gs.hands[state.gs.soloist])

        # 2. Unknown cards = full deck minus all known
        unknown = [c for c in _ALL_CARDS if c not in known]

        # 3. Opponents whose hands need shuffling
        if state.is_open:
            # In terített, only the non-soloist opponents need shuffling
            opps = [i for i in range(NUM_PLAYERS)
                    if i != player and i != state.gs.soloist]
        else:
            opps = [i for i in range(NUM_PLAYERS) if i != player]

        opp_sizes = [len(gs.hands[opp]) for opp in opps]
        opp_voids = [state.known_voids[opp] for opp in opps]

        # 4. Auction constraints — must-have cards for opponents
        #    Assign these FIRST before random shuffling.
        must_have = state.must_have
        pre_assigned: dict[int, list[Card]] = {opp: [] for opp in opps}
        remaining_unknown = list(unknown)

        for opp in opps:
            required = must_have.get(opp, frozenset())
            for card in required:
                if card in remaining_unknown:
                    pre_assigned[opp].append(card)
                    remaining_unknown.remove(card)

        # 5. Random assignment respecting void constraints
        pool = list(remaining_unknown)
        rng.shuffle(pool)

        for oi, opp in enumerate(opps):
            voids = opp_voids[oi]
            pre = pre_assigned[opp]
            need = opp_sizes[oi] - len(pre)

            if need <= 0:
                gs.hands[opp] = pre[:opp_sizes[oi]]
                # Return unused pre-assigned cards to pool
                for c in pre[opp_sizes[oi]:]:
                    pool.append(c)
                continue

            if voids:
                eligible = [c for c in pool if c.suit not in voids]
                ineligible = [c for c in pool if c.suit in voids]
                rng.shuffle(eligible)
                rng.shuffle(ineligible)

                if len(eligible) >= need:
                    assigned = eligible[:need]
                    pool = eligible[need:] + ineligible
                else:
                    # Rare edge case: fill with whatever is left
                    assigned = eligible + ineligible[: need - len(eligible)]
                    pool = ineligible[need - len(eligible):]
            else:
                assigned = pool[:need]
                pool = pool[need:]

            gs.hands[opp] = pre + assigned

        return UltiNode(
            gs=gs,
            known_voids=state.known_voids,
            bid_rank=state.bid_rank,
            is_red=state.is_red,
            is_open=state.is_open,
            contract_components=state.contract_components,
            dealer=state.dealer,
            component_kontras=state.component_kontras,
            must_have=state.must_have,
        )

    # -- neural-network encoding (v2 — 259-dim) ----------------------------

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    def encode_state(self, state: UltiNode, player: int) -> np.ndarray:
        gs = state.gs
        return self._enc.encode_state(
            hand=gs.hands[player],
            captured=gs.captured,
            trick_cards=gs.trick_cards,
            trump=gs.trump,
            betli=gs.betli,
            soloist=gs.soloist,
            player=player,
            trick_no=gs.trick_no,
            scores=gs.scores,
            known_voids=state.known_voids,
            marriages=gs.marriages,
            # v2 additions
            trick_history=gs.trick_history,
            contract_components=state.contract_components,
            is_red=state.is_red,
            is_open=state.is_open,
            bid_rank=state.bid_rank,
            soloist_saw_talon=True,
            dealer=state.dealer,
            component_kontras=state.component_kontras if state.component_kontras else None,
        )

    # -- fixed action space ------------------------------------------------

    @property
    def action_space_size(self) -> int:
        return NUM_CARDS

    def action_to_index(self, action: Card) -> int:
        return _CARD_IDX[action]

    def legal_action_mask(self, state: UltiNode) -> np.ndarray:
        mask = np.zeros(NUM_CARDS, dtype=bool)
        for a in self.legal_actions(state):
            mask[_CARD_IDX[a]] = True
        return mask

    # -- new game (auto-setup for training) --------------------------------

    def new_game(self, seed: int, **kwargs: Any) -> UltiNode:
        """Deal and auto-setup a game for the play phase.

        Performs the full pre-play sequence automatically:
        1. Deal 10 cards each + 2-card talon.
        2. First bidder becomes soloist (simplified — no bidding).
        3. Soloist picks up talon, randomly discards 2 cards.
        4. Trump is chosen randomly from soloist's suits
           (10 % chance of Betli for variety).

        The ``starting_leader`` kwarg is interpreted as dealer index.
        ``training_mode`` can filter to specific contract types.
        """
        rng = random.Random(seed)
        dealer = kwargs.get("starting_leader", 0) % NUM_PLAYERS
        training_mode = kwargs.get("training_mode", None)

        gs, talon = deal(seed=seed, dealer=dealer)
        gs.training_mode = training_mode
        soloist = next_player(dealer)  # first bidder

        # Soloist picks up talon
        pickup_talon(gs, soloist, talon)

        # Choose contract based on training_mode
        if training_mode == "betli":
            betli = True
        elif training_mode in ("simple", "ulti"):
            betli = False
        else:
            betli = rng.random() < 0.1

        if betli:
            set_contract(gs, soloist, trump=None, betli=True)
        else:
            suits_in_hand = list(set(c.suit for c in gs.hands[soloist]))
            trump = rng.choice(suits_in_hand)
            set_contract(gs, soloist, trump=trump)

        # Discard 2 cards (random)
        discards = rng.sample(gs.hands[soloist], 2)
        discard_talon(gs, discards)

        # Declare all marriages before the play phase begins.
        declare_all_marriages(gs)

        # Build contract components for training (simplified)
        if betli:
            comps = frozenset({"betli"})
        else:
            comps = frozenset({"parti"})
            if training_mode == "ulti":
                comps = frozenset({"parti", "ulti"})
                gs.has_ulti = True

        # Build auction constraints from marriages/contract
        constraints = build_auction_constraints(gs, comps)

        empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())
        return UltiNode(
            gs=gs,
            known_voids=empty_voids,
            bid_rank=1,  # Passz rank as default for training
            contract_components=comps,
            dealer=dealer,
            must_have=constraints,
        )
