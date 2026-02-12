"""Oracle Hand Evaluator — estimates win probability per contract type.

Given a 12-card hand (after talon pickup, before discard), the Oracle
tries every contract type by:

1. For each contract class (4 trump suits + betli = 5 options):
   a. Use the neural discard (value-head evaluation of all 66 combos).
   b. Set up the game state for that contract.
   c. Run shallow MCTS rollouts to estimate soloist win probability.

2. Return a "Contract Heatmap": win rates over the 5 contract classes.

The Oracle is used as a *teacher* for the Auction Head:
  - Target = one-hot of the Oracle's best contract
  - Loss = CrossEntropy(AuctionHead prediction, Oracle target)

This avoids the chicken-and-egg problem: the play AI learns to
play well first, and the Oracle uses that play ability to evaluate
which contracts are viable.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from trickster.games.ulti.adapter import (
    UltiGame,
    UltiNode,
    build_auction_constraints,
)
from trickster.games.ulti.cards import (
    ALL_SUITS,
    Card,
    NUM_PLAYERS,
    Rank,
    Suit,
    make_deck,
)
from trickster.games.ulti.game import (
    GameState,
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
    soloist_lost_betli,
    soloist_won_simple,
)
from trickster.mcts import MCTSConfig, alpha_mcts_policy
from trickster.model import CONTRACT_CLASSES, NUM_CONTRACTS, UltiNetWrapper

# ---------------------------------------------------------------------------
#  Oracle configuration
# ---------------------------------------------------------------------------

# Shallow MCTS for oracle game rollouts
ORACLE_MCTS = MCTSConfig(
    simulations=10,
    determinizations=1,
    c_puct=1.5,
    dirichlet_alpha=0.0,
    dirichlet_weight=0.0,
    use_value_head=True,
    use_policy_priors=True,
    visit_temp=0.1,
)

ORACLE_ROLLOUTS = 3  # games per contract class


# ---------------------------------------------------------------------------
#  Neural discard: evaluate all C(12,2) = 66 possible discards
# ---------------------------------------------------------------------------


def neural_discard(
    hand: list[Card],
    wrapper: UltiNetWrapper,
    game: UltiGame,
    soloist: int,
    betli: bool,
    trump: Suit | None,
    dealer: int,
) -> list[Card]:
    """Pick the best 2 cards to discard using the value head.

    Evaluates all 66 possible discards by encoding the resulting
    10-card hand as an initial game state and asking the value head
    to predict the win probability.  Returns the discard that
    yields the highest predicted value for the soloist.
    """
    assert len(hand) == 12

    combos = list(combinations(range(len(hand)), 2))
    if not combos:
        return hand[:2]

    # Build a batch of encoded states — one per discard option
    comps = frozenset({"betli"}) if betli else frozenset({"parti"})
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())
    states_batch: list[np.ndarray] = []
    discard_pairs: list[tuple[Card, Card]] = []

    for i, j in combos:
        d0, d1 = hand[i], hand[j]
        kept = [c for k, c in enumerate(hand) if k != i and k != j]
        discard_pairs.append((d0, d1))

        feats = game._enc.encode_state(
            hand=kept,
            captured=[[], [], []],
            trick_cards=[],
            trump=trump,
            betli=betli,
            soloist=soloist,
            player=soloist,
            trick_no=0,
            scores=[0, 0, 0],
            known_voids=empty_voids,
            contract_components=comps,
            dealer=dealer,
        )
        states_batch.append(feats)

    # Batch evaluation through value head
    batch = np.stack(states_batch)
    values = wrapper.batch_value(batch)

    best_idx = int(np.argmax(values))
    return list(discard_pairs[best_idx])


# ---------------------------------------------------------------------------
#  Build a playable game state from hand + contract
# ---------------------------------------------------------------------------


def _build_state(
    hand12: list[Card],
    soloist: int,
    dealer: int,
    betli: bool,
    trump: Suit | None,
    discards: list[Card],
    seed: int,
) -> UltiNode:
    """Create a UltiNode ready for play from a 12-card soloist hand.

    Opponents get random cards from the remaining pool.
    """
    rng = random.Random(seed)

    # 10-card hand after discard
    kept = [c for c in hand12 if c not in discards]
    assert len(kept) == 10

    # Remaining cards go to opponents
    all_cards = set(make_deck())
    pool = list(all_cards - set(hand12))
    rng.shuffle(pool)
    assert len(pool) == 20  # 32 - 12

    # Deal 10 to each opponent
    opps = [p for p in range(NUM_PLAYERS) if p != soloist]
    hands: list[list[Card]] = [[] for _ in range(NUM_PLAYERS)]
    hands[soloist] = kept
    hands[opps[0]] = pool[:10]
    hands[opps[1]] = pool[10:20]

    # Calculate discard points
    discard_points = sum(c.points() for c in discards)

    gs = GameState(
        hands=hands,
        trump=trump,
        betli=betli,
        soloist=soloist,
        dealer=dealer,
        captured=[[],  [], []],
        scores=[0, 0, 0],
        leader=soloist,  # soloist leads first trick
        trick_no=0,
        trick_cards=[],
        last_trick=None,
        training_mode="betli" if betli else "simple",
    )
    # Store talon discards — their points count for the defenders
    gs.talon_discards = list(discards)

    # Declare marriages
    declare_all_marriages(gs)

    comps = frozenset({"betli"}) if betli else frozenset({"parti"})
    constraints = build_auction_constraints(gs, comps)
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())

    return UltiNode(
        gs=gs,
        known_voids=empty_voids,
        bid_rank=1,
        contract_components=comps,
        dealer=dealer,
        must_have=constraints,
    )


# ---------------------------------------------------------------------------
#  Play out a single game, return win/loss for soloist
# ---------------------------------------------------------------------------


def _play_one_oracle_game(
    state: UltiNode,
    game: UltiGame,
    wrapper: UltiNetWrapper,
    soloist: int,
    rng: random.Random,
) -> bool:
    """Play a game using MCTS for soloist, random for defenders.
    Returns True if soloist won."""
    rng_rand = random.Random(rng.randrange(1 << 30))

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        if player == soloist:
            _pi, action = alpha_mcts_policy(
                state, game, wrapper, player, ORACLE_MCTS, rng,
            )
        else:
            action = rng_rand.choice(actions)

        state = game.apply(state, action)

    gs = state.gs
    if gs.betli:
        return not soloist_lost_betli(gs)
    return soloist_won_simple(gs)


# ---------------------------------------------------------------------------
#  Oracle: estimate win probability per contract class
# ---------------------------------------------------------------------------


def oracle_evaluate(
    hand12: list[Card],
    soloist: int,
    dealer: int,
    wrapper: UltiNetWrapper,
    game: UltiGame,
    seed: int,
    num_rollouts: int = ORACLE_ROLLOUTS,
) -> np.ndarray:
    """Evaluate a 12-card hand across all 5 contract classes.

    For each of the 5 contract types (4 suits simple + betli):
    1. Choose best discard via neural evaluation.
    2. Play ``num_rollouts`` games with shallow MCTS.
    3. Record win rate.

    Parameters
    ----------
    hand12 : list of 12 Cards (after talon pickup)
    soloist : player index of the soloist
    dealer : dealer index
    wrapper : trained UltiNetWrapper for MCTS and value evaluation
    game : UltiGame instance
    seed : random seed base
    num_rollouts : games per contract class (default 3)

    Returns
    -------
    heatmap : (NUM_CONTRACTS,) numpy array of win rates [0, 1]
              Values of -1 indicate an impossible contract
              (e.g., trump suit not in hand).
    """
    suits = list(ALL_SUITS)
    heatmap = np.zeros(NUM_CONTRACTS, dtype=np.float64)

    for ci, (mode, suit_idx) in enumerate(CONTRACT_CLASSES):
        betli = (mode == "betli")
        trump = suits[suit_idx] if suit_idx is not None else None

        # Feasibility check: soloist must hold at least one trump card
        if not betli and trump is not None:
            has_suit = any(c.suit == trump for c in hand12)
            if not has_suit:
                heatmap[ci] = -1.0
                continue

        # Find best discard for this contract
        discards = neural_discard(
            hand12, wrapper, game, soloist, betli, trump, dealer,
        )

        # Play rollouts and measure win rate
        wins = 0
        for r in range(num_rollouts):
            game_seed = seed + ci * 10000 + r
            rng = random.Random(game_seed)

            state = _build_state(
                hand12, soloist, dealer, betli, trump, discards, game_seed,
            )

            if _play_one_oracle_game(state, game, wrapper, soloist, rng):
                wins += 1

        heatmap[ci] = wins / max(1, num_rollouts)

    return heatmap


def oracle_best_contract(heatmap: np.ndarray) -> int:
    """Return the contract class index with the highest win rate.

    Ignores impossible contracts (value == -1).
    """
    valid = heatmap.copy()
    valid[valid < 0] = -float("inf")
    return int(np.argmax(valid))


def oracle_target_distribution(heatmap: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert heatmap to a soft probability distribution for training.

    Uses softmax with temperature.  Impossible contracts (value -1)
    are set to zero probability.

    Parameters
    ----------
    heatmap : (NUM_CONTRACTS,) win rates
    temperature : softmax temperature (lower = sharper)

    Returns
    -------
    probs : (NUM_CONTRACTS,) probability distribution
    """
    valid = heatmap.copy()
    mask = valid >= 0
    if not mask.any():
        # Fallback: uniform over all
        return np.ones(NUM_CONTRACTS) / NUM_CONTRACTS

    logits = np.full(NUM_CONTRACTS, -1e9)
    logits[mask] = valid[mask] / max(temperature, 1e-8)

    # Softmax
    logits -= logits.max()
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    return probs
