#!/usr/bin/env python3
"""Ulti baseline training — curriculum with PyTorch.

Supports four curriculum modes:
  --mode simple   Train only on Simple (Parti) contracts.
  --mode betli    Train only on Betli contracts.
  --mode mixed    Train on a 50/50 mix of Simple and Betli.
  --mode auto     Soloist chooses contract via Auction Head + Oracle.

Self-play:
  All 3 players use the same UltiNet.  Defenders use a lower MCTS
  budget (--def-sims, default 8) for speed, while the soloist uses
  --sims (default 20).  This replaces the old "vs Random" self-play.

Neural discard (--neural-discard):
  Evaluates all C(12,2)=66 possible discards through the value head.

Auto mode (Level 3 curriculum):
  The soloist uses the Auction Head to choose a contract.  The Oracle
  provides the training target via cross-entropy loss.

Elite Benchmark (--elite-interval):
  Periodically evaluates the Neural Net (policy-only, 0 MCTS sims)
  vs the Oracle (MCTS 50) to measure how close intuition is to search.

Usage:
    python scripts/train_baseline.py [--mode simple] [--steps 200]
    python scripts/train_baseline.py --mode mixed --steps 200 --self-play
    python scripts/train_baseline.py --mode auto --steps 200 --neural-discard
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import ALL_SUITS, Card, Rank, Suit, BETLI_STRENGTH
from trickster.games.ulti.game import (
    discard_talon,
    soloist_lost_betli,
    soloist_won_simple,
)
from trickster.mcts import MCTSConfig, alpha_mcts_policy
from trickster.model import (
    CONTRACT_CLASSES,
    NUM_CONTRACTS,
    NUM_FLAGS,
    TRUMP_CLASSES,
    UltiNet,
    UltiNetWrapper,
    legacy_to_auction_targets,
)
from trickster.train_utils import ReplayBuffer, outcome_for_player, simple_outcome, calculate_reward


# ---------------------------------------------------------------------------
#  Mode helpers
# ---------------------------------------------------------------------------

MODES = ("simple", "betli", "mixed", "auto")


def _pick_training_mode(mode: str, rng: random.Random) -> str:
    """Return the training_mode string for a single game."""
    if mode == "mixed":
        return rng.choice(["simple", "betli"])
    if mode == "auto":
        return "auto"
    return mode


def _soloist_won(state: UltiNode) -> bool:
    """Determine if soloist won, respecting betli vs simple."""
    gs = state.gs
    if gs.betli:
        return not soloist_lost_betli(gs)
    return soloist_won_simple(gs)


# ---------------------------------------------------------------------------
#  Greedy talon discard heuristics (fallback for early training)
# ---------------------------------------------------------------------------


def _greedy_discard(hand: list[Card], betli: bool, trump: Suit | None) -> list[Card]:
    """Pick 2 cards to discard from a 12-card hand using a heuristic.

    Betli mode:  Discard the two highest-strength cards (A, K, ...).
    Simple mode: Discard the two weakest non-trump cards.
    """
    candidates = list(hand)
    if betli:
        candidates.sort(key=lambda c: -BETLI_STRENGTH[c.rank])
    else:
        def simple_key(c: Card) -> tuple[int, int]:
            is_trump = 1 if (trump is not None and c.suit == trump) else 0
            return (is_trump, c.rank.value)
        candidates.sort(key=simple_key)
    return candidates[:2]


# ---------------------------------------------------------------------------
#  Neural discard wrapper (for use as _discard_fn callback)
# ---------------------------------------------------------------------------


def make_neural_discard_fn(
    wrapper: UltiNetWrapper,
    game: UltiGame,
    soloist: int,
    dealer: int,
):
    """Create a discard callback that uses the value head."""
    from trickster.evaluator import neural_discard

    def _fn(hand: list[Card], betli: bool, trump: Suit | None) -> list[Card]:
        return neural_discard(hand, wrapper, game, soloist, betli, trump, dealer)
    return _fn


# ---------------------------------------------------------------------------
#  Auto-mode: auction head picks contract + neural discard
# ---------------------------------------------------------------------------


def _auto_play_one_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    config: MCTSConfig,
    def_config: MCTSConfig,
    seed: int,
    use_oracle: bool = True,
    oracle_rollouts: int = 3,
    self_play_defenders: bool = True,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]],
    np.ndarray | None,
    int | None,
    np.ndarray | None,
]:
    """Play one game in auto mode with full self-play."""
    from trickster.evaluator import (
        neural_discard,
        oracle_evaluate,
        oracle_best_contract,
    )

    rng = random.Random(seed)
    dealer = seed % 3
    _gs, _talon, soloist = game.get_hand12(seed=seed, dealer=dealer)
    hand12 = list(_gs.hands[soloist])

    # --- Auction Head: pick contract ---
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())
    auction_feats = game._enc.encode_state(
        hand=hand12,
        captured=[[], [], []],
        trick_cards=[],
        trump=None,
        betli=False,
        soloist=soloist,
        player=soloist,
        trick_no=0,
        scores=[0, 0, 0],
        known_voids=empty_voids,
        dealer=dealer,
    )

    auction_probs = wrapper.predict_auction(auction_feats)
    suits = list(ALL_SUITS)
    for ci, (mode, suit_idx) in enumerate(CONTRACT_CLASSES):
        if mode != "betli" and suit_idx is not None:
            if not any(c.suit == suits[suit_idx] for c in hand12):
                auction_probs[ci] = 0.0
    p_sum = auction_probs.sum()
    if p_sum > 0:
        auction_probs /= p_sum
    else:
        auction_probs = np.ones(NUM_CONTRACTS) / NUM_CONTRACTS

    chosen_ci = rng.choices(range(NUM_CONTRACTS), weights=auction_probs, k=1)[0]
    mode_str, suit_idx = CONTRACT_CLASSES[chosen_ci]
    betli = (mode_str == "betli")
    trump = suits[suit_idx] if suit_idx is not None else None

    discards = neural_discard(hand12, wrapper, game, soloist, betli, trump, dealer)

    oracle_heatmap = None
    oracle_target = None
    if use_oracle:
        oracle_heatmap = oracle_evaluate(
            hand12, soloist, dealer, wrapper, game,
            seed=seed + 100000,
            num_rollouts=oracle_rollouts,
        )
        oracle_target = oracle_best_contract(oracle_heatmap)

    discard_fn = lambda h, b, t: discards
    state = game.new_game(
        seed=seed,
        training_mode="auto",
        starting_leader=dealer,
        _contract_idx=chosen_ci,
        _discard_fn=discard_fn,
    )
    soloist_idx = state.gs.soloist

    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        # Self-play: soloist uses full MCTS, defenders use lower budget
        if player == soloist_idx:
            pi, action = alpha_mcts_policy(
                state, game, wrapper, player, config, rng,
            )
        elif self_play_defenders:
            pi, action = alpha_mcts_policy(
                state, game, wrapper, player, def_config, rng,
            )
        else:
            action = rng.choice(actions)
            pi = np.zeros(game.action_space_size, dtype=np.float32)
            pi[game.action_to_index(action)] = 1.0

        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append((
            state_feats.copy(),
            mask.copy(),
            pi.copy(),
            player,
        ))

        state = game.apply(state, action)

    won = _soloist_won(state)

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for state_feats, mask, pi, player in trajectory:
        # Use simple ±1 rewards that match game.outcome() / MCTS scale.
        # The compound scaling (outcome_for_player) is only meaningful
        # when compound contracts are actually being trained.
        reward = simple_outcome(state, player)
        is_sol = (player == soloist_idx)
        samples.append((state_feats, mask, pi, reward, is_sol))

    return samples, auction_feats, oracle_target, oracle_heatmap


# ---------------------------------------------------------------------------
#  Self-play: one full game → training samples
# ---------------------------------------------------------------------------


def play_one_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    config: MCTSConfig,
    seed: int,
    mode: str = "simple",
    use_neural_discard: bool = False,
    def_config: MCTSConfig | None = None,
    self_play_defenders: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Play one self-play game (simple/betli/mixed modes).

    With ``self_play_defenders=True``, defenders use the same UltiNet
    with ``def_config`` (lower MCTS budget) instead of random play.
    """
    rng = random.Random(seed)
    training_mode = _pick_training_mode(mode, rng)

    if use_neural_discard:
        dealer = seed % 3
        from trickster.games.ulti.game import next_player
        soloist = next_player(dealer)
        discard_fn = make_neural_discard_fn(wrapper, game, soloist, dealer)
    else:
        discard_fn = _greedy_discard

    state = game.new_game(
        seed=seed,
        training_mode=training_mode,
        _discard_fn=discard_fn,
    )
    soloist_idx = state.gs.soloist
    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        # Soloist always uses full MCTS
        if player == soloist_idx:
            pi, action = alpha_mcts_policy(
                state, game, wrapper, player, config, rng,
            )
        elif self_play_defenders and def_config is not None:
            # Defenders use same net with fewer sims
            pi, action = alpha_mcts_policy(
                state, game, wrapper, player, def_config, rng,
            )
        else:
            # Fallback: random play for defenders
            action = rng.choice(actions)
            pi = np.zeros(game.action_space_size, dtype=np.float32)
            pi[game.action_to_index(action)] = 1.0

        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append((
            state_feats.copy(),
            mask.copy(),
            pi.copy(),
            player,
        ))

        state = game.apply(state, action)

    won = _soloist_won(state)

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for state_feats, mask, pi, player in trajectory:
        # Use simple ±1 rewards that match game.outcome() / MCTS scale.
        # This gives a strong learning signal and ensures the value head
        # predictions are on the same scale as MCTS backpropagation.
        reward = simple_outcome(state, player)
        is_sol = (player == soloist_idx)
        samples.append((state_feats, mask, pi, reward, is_sol))

    return samples


# ---------------------------------------------------------------------------
#  Evaluation: trained model vs random (standard benchmark)
# ---------------------------------------------------------------------------


def eval_vs_random(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    config: MCTSConfig,
    mode: str = "simple",
    num_games: int = 20,
    seed: int = 99999,
    use_neural_discard: bool = False,
) -> dict[str, float]:
    """Play trained agent (player 0) vs random opponents."""
    wins: dict[str, int] = {}
    counts: dict[str, int] = {}
    rng_mode = random.Random(seed)

    def _inc(key: str, won: bool) -> None:
        counts[key] = counts.get(key, 0) + 1
        if won:
            wins[key] = wins.get(key, 0) + 1

    eval_mode = mode if mode != "auto" else "mixed"

    for g in range(num_games):
        rng = random.Random(seed + g)
        rng_rand = random.Random(seed + g + 50000)

        training_mode = _pick_training_mode(eval_mode, rng_mode)
        dealer = g % 3

        if use_neural_discard:
            from trickster.games.ulti.game import next_player
            soloist = next_player(dealer)
            discard_fn = make_neural_discard_fn(wrapper, game, soloist, dealer)
        else:
            discard_fn = _greedy_discard

        state = game.new_game(
            seed=seed + g,
            training_mode=training_mode,
            starting_leader=dealer,
            _discard_fn=discard_fn,
        )
        is_soloist = (state.gs.soloist == 0)
        role = "sol" if is_soloist else "def"

        while not game.is_terminal(state):
            player = game.current_player(state)
            actions = game.legal_actions(state)

            if len(actions) <= 1:
                state = game.apply(state, actions[0])
                continue

            if player == 0:
                pi, action = alpha_mcts_policy(
                    state, game, wrapper, player, config, rng,
                )
            else:
                action = rng_rand.choice(actions)

            state = game.apply(state, action)

        p0_won = game.outcome(state, 0) > 0

        _inc(training_mode, p0_won)
        _inc(f"{training_mode}_{role}", p0_won)
        _inc("all", p0_won)
        _inc(f"all_{role}", p0_won)

    result: dict[str, float] = {}
    for key in sorted(counts):
        result[key] = wins.get(key, 0) / max(1, counts[key])
    return result


# ---------------------------------------------------------------------------
#  Elite Benchmark: Neural Net (0 MCTS) vs Oracle (MCTS 50)
# ---------------------------------------------------------------------------


def eval_elite(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    mode: str = "simple",
    num_games: int = 10,
    seed: int = 88888,
    oracle_sims: int = 50,
) -> dict[str, float]:
    """Compare 'Intuition' (policy-only, no MCTS) vs 'Search' (MCTS 50).

    Player 0 = Neural Net (uses predict_policy directly, 0 sims).
    Player 1,2 = Oracle (full MCTS with ``oracle_sims`` simulations).

    This measures how close the NN's raw intuition is to proper search.
    """
    oracle_config = MCTSConfig(
        simulations=oracle_sims,
        determinizations=1,
        c_puct=1.5,
        dirichlet_alpha=0.0,
        dirichlet_weight=0.0,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.1,
    )

    wins: dict[str, int] = {}
    counts: dict[str, int] = {}
    rng_mode = random.Random(seed)

    def _inc(key: str, won: bool) -> None:
        counts[key] = counts.get(key, 0) + 1
        if won:
            wins[key] = wins.get(key, 0) + 1

    eval_mode = mode if mode != "auto" else "mixed"

    for g in range(num_games):
        rng = random.Random(seed + g)
        training_mode = _pick_training_mode(eval_mode, rng_mode)
        dealer = g % 3

        state = game.new_game(
            seed=seed + g,
            training_mode=training_mode,
            starting_leader=dealer,
            _discard_fn=_greedy_discard,
        )
        is_soloist = (state.gs.soloist == 0)
        role = "sol" if is_soloist else "def"

        while not game.is_terminal(state):
            player = game.current_player(state)
            actions = game.legal_actions(state)

            if len(actions) <= 1:
                state = game.apply(state, actions[0])
                continue

            if player == 0:
                # Pure intuition: pick the highest-prob legal action
                state_feats = game.encode_state(state, player)
                mask = game.legal_action_mask(state)
                probs = wrapper.predict_policy(state_feats, mask)
                best_idx = int(np.argmax(probs))
                # Convert card index → Card action
                from trickster.games.ulti.adapter import _IDX_TO_CARD
                action = _IDX_TO_CARD[best_idx]
            else:
                # Oracle: full MCTS search
                _pi, action = alpha_mcts_policy(
                    state, game, wrapper, player, oracle_config, rng,
                )

            state = game.apply(state, action)

        p0_won = game.outcome(state, 0) > 0

        _inc(training_mode, p0_won)
        _inc(f"{training_mode}_{role}", p0_won)
        _inc("all", p0_won)
        _inc(f"all_{role}", p0_won)

    result: dict[str, float] = {}
    for key in sorted(counts):
        result[key] = wins.get(key, 0) / max(1, counts[key])
    return result


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ulti Curriculum Training — simple / betli / mixed / auto",
    )
    parser.add_argument("--mode", type=str, default="simple",
                        choices=MODES,
                        help="Curriculum mode (default simple)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training iterations (default 200)")
    parser.add_argument("--games-per-step", type=int, default=4,
                        help="Self-play games per training step (default 4)")
    parser.add_argument("--sims", type=int, default=20,
                        help="MCTS simulations per move for soloist (default 20)")
    parser.add_argument("--def-sims", type=int, default=8,
                        help="MCTS simulations per move for defenders (default 8)")
    parser.add_argument("--dets", type=int, default=1,
                        help="MCTS determinizations per move (default 1)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default 1e-3)")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N steps (default 10)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per evaluation (default 20)")
    parser.add_argument("--elite-interval", type=int, default=0,
                        help="Elite benchmark every N steps (0=disabled)")
    parser.add_argument("--elite-games", type=int, default=10,
                        help="Games per elite benchmark (default 10)")
    parser.add_argument("--elite-sims", type=int, default=50,
                        help="MCTS sims for Oracle in elite benchmark (default 50)")
    parser.add_argument("--buffer-size", type=int, default=50000,
                        help="Replay buffer capacity (default 50000)")
    parser.add_argument("--body-units", type=int, default=256,
                        help="Backbone hidden units (default 256)")
    parser.add_argument("--body-layers", type=int, default=4,
                        help="Backbone layers (default 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (default cpu)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the model (default models/ulti_{mode}.pt)")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to load a pre-trained model to continue training")
    parser.add_argument("--neural-discard", action="store_true",
                        help="Use neural discard (value head) instead of greedy heuristic")
    parser.add_argument("--oracle-rollouts", type=int, default=3,
                        help="Rollouts per contract for Oracle (auto mode, default 3)")
    parser.add_argument("--no-self-play", action="store_true",
                        help="Use random defenders instead of self-play (legacy mode)")
    parser.add_argument("--checkpoint-interval", type=int, default=0,
                        help="Save checkpoint every N steps (0=disabled, e.g. 50)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: models/checkpoints/{mode})")
    args = parser.parse_args()

    if args.save is None:
        args.save = f"models/ulti_{args.mode}.pt"

    # ---- Setup ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    mode = args.mode
    use_neural = args.neural_discard or mode == "auto"
    self_play = not args.no_self_play

    game = UltiGame()
    net = UltiNet(
        input_dim=game.state_dim,
        body_units=args.body_units,
        body_layers=args.body_layers,
        action_dim=game.action_space_size,
        num_contracts=NUM_CONTRACTS,
    )

    # Optionally load a pre-trained model
    if args.load:
        checkpoint = torch.load(args.load, weights_only=True)
        model_dict = net.state_dict()
        pretrained = {k: v for k, v in checkpoint["model_state_dict"].items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        net.load_state_dict(model_dict)
        skipped = set(checkpoint["model_state_dict"]) - set(pretrained)
        new_keys = set(model_dict) - set(checkpoint["model_state_dict"])
        if skipped:
            print(f"  Skipped incompatible keys: {skipped}")
        if new_keys:
            print(f"  Randomly initialized new keys: {new_keys}")
        print(f"  Loaded pre-trained model from {args.load}")

    wrapper = UltiNetWrapper(net, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    buffer = ReplayBuffer(capacity=args.buffer_size)
    np_rng = np.random.default_rng(args.seed)

    # Auction training buffer (for auto mode)
    # Stores (feats, trump_target, flags_target) for multi-component training
    auction_buffer: list[tuple[np.ndarray, int, np.ndarray]] = []
    AUCTION_BUF_MAX = 5000

    # MCTS config for soloist self-play (exploration on)
    train_config = MCTSConfig(
        simulations=args.sims,
        determinizations=args.dets,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=1.0,
    )

    # MCTS config for defender self-play (lower budget, less exploration)
    def_config = MCTSConfig(
        simulations=args.def_sims,
        determinizations=args.dets,
        c_puct=1.5,
        dirichlet_alpha=0.1,
        dirichlet_weight=0.15,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.5,
    )

    # MCTS config for evaluation (exploitation)
    eval_config = MCTSConfig(
        simulations=args.sims,
        determinizations=args.dets,
        c_puct=1.5,
        dirichlet_alpha=0.0,
        dirichlet_weight=0.0,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.1,
    )

    param_count = sum(p.numel() for p in net.parameters())

    mode_label = {
        "simple": "Simple (Parti)",
        "betli": "Betli",
        "mixed": "Mixed (Simple + Betli)",
        "auto": "Auto (Auction Head + Oracle)",
    }

    print("=" * 64)
    print(f"  Ulti Training — {mode_label[mode]} Curriculum")
    print("=" * 64)
    print(f"  Mode: {mode}")
    print(f"  Model: UltiNet {args.body_units}x{args.body_layers} "
          f"({param_count:,} params)")
    print(f"  MCTS: soloist={args.sims} sims, "
          f"defenders={'self-play ' + str(args.def_sims) + ' sims' if self_play else 'random'}")
    print(f"  Steps: {args.steps} x {args.games_per_step} games/step "
          f"= {args.steps * args.games_per_step} games")
    print(f"  LR: {args.lr}  Batch: {args.batch_size}  "
          f"Buffer: {args.buffer_size}")
    print(f"  Device: {device}")
    print(f"  Discard: {'neural (value head)' if use_neural else 'greedy heuristic'}")
    if mode == "auto":
        print(f"  Oracle: {args.oracle_rollouts} rollouts/contract")
    print(f"  Eval: every {args.eval_interval} steps, "
          f"{args.eval_games} games vs Random")
    if args.elite_interval > 0:
        print(f"  Elite: every {args.elite_interval} steps, "
              f"{args.elite_games} games — NN(0) vs Oracle({args.elite_sims})")
    ckpt_dir: Path | None = None
    if args.checkpoint_interval > 0:
        ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
            Path("models") / "checkpoints" / mode
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Checkpoints: every {args.checkpoint_interval} steps → {ckpt_dir}/")
    print()

    t0 = time.perf_counter()
    total_games = 0
    total_samples = 0
    best_wr = 0.0

    for step in range(1, args.steps + 1):
        step_t = time.perf_counter()

        # ---- 1. Self-play: collect samples ----
        step_samples = 0
        step_auction_samples = 0
        for g in range(args.games_per_step):
            game_seed = args.seed + step * 1000 + g

            if mode == "auto":
                samples, auc_feats, oracle_target, _ = _auto_play_one_game(
                    game, wrapper, train_config, def_config, game_seed,
                    use_oracle=True,
                    oracle_rollouts=args.oracle_rollouts,
                    self_play_defenders=self_play,
                )
                # Store multi-component auction training sample
                if auc_feats is not None and oracle_target is not None:
                    trump_t, flags_t = legacy_to_auction_targets(oracle_target)
                    auction_buffer.append((auc_feats.copy(), trump_t, flags_t.copy()))
                    if len(auction_buffer) > AUCTION_BUF_MAX:
                        auction_buffer = auction_buffer[-AUCTION_BUF_MAX:]
                    step_auction_samples += 1
            else:
                samples = play_one_game(
                    game, wrapper, train_config, game_seed,
                    mode=mode, use_neural_discard=use_neural,
                    def_config=def_config,
                    self_play_defenders=self_play,
                )

            for s, m, p, r, is_sol in samples:
                buffer.push(s, m, p, r, is_soloist=is_sol)
            step_samples += len(samples)
            total_games += 1

        total_samples += step_samples

        # ---- 2. Train on replay buffer ----
        avg_vloss = 0.0
        avg_ploss = 0.0
        avg_aloss = 0.0

        if len(buffer) >= args.batch_size:
            net.train()
            train_steps = max(1, step_samples // args.batch_size)
            total_vloss = 0.0
            total_ploss = 0.0
            total_aloss = 0.0
            auction_train_count = 0

            for _ in range(train_steps):
                states, masks, policies, rewards = buffer.sample(
                    args.batch_size, np_rng,
                )

                s_t = torch.from_numpy(states).float().to(device)
                m_t = torch.from_numpy(masks).bool().to(device)
                pi_t = torch.from_numpy(policies).float().to(device)
                z_t = torch.from_numpy(rewards).float().to(device)

                log_probs, values = net(s_t, m_t)

                value_loss = F.mse_loss(values, z_t)
                policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
                loss = value_loss + policy_loss

                # ---- Multi-component auction head training (auto mode) ----
                if mode == "auto" and len(auction_buffer) >= 8:
                    auc_batch_size = min(16, len(auction_buffer))
                    auc_indices = np_rng.choice(
                        len(auction_buffer), size=auc_batch_size, replace=False,
                    )
                    auc_states = np.stack([auction_buffer[i][0] for i in auc_indices])
                    auc_trump_targets = np.array(
                        [auction_buffer[i][1] for i in auc_indices], dtype=np.int64,
                    )
                    auc_flag_targets = np.stack(
                        [auction_buffer[i][2] for i in auc_indices],
                    )

                    auc_s = torch.from_numpy(auc_states).float().to(device)
                    auc_tt = torch.from_numpy(auc_trump_targets).long().to(device)
                    auc_ft = torch.from_numpy(auc_flag_targets).float().to(device)

                    trump_lp, flag_logits = net.forward_auction(auc_s)

                    # Trump loss: cross-entropy
                    trump_loss = F.nll_loss(trump_lp, auc_tt)
                    # Flags loss: binary cross-entropy
                    flags_loss = F.binary_cross_entropy_with_logits(
                        flag_logits, auc_ft,
                    )
                    auction_loss = trump_loss + flags_loss

                    loss = loss + auction_loss
                    total_aloss += auction_loss.item()
                    auction_train_count += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()

                total_vloss += value_loss.item()
                total_ploss += policy_loss.item()

            avg_vloss = total_vloss / train_steps
            avg_ploss = total_ploss / train_steps
            if auction_train_count > 0:
                avg_aloss = total_aloss / auction_train_count

        elapsed = time.perf_counter() - t0
        step_time = time.perf_counter() - step_t

        # ---- 3. Progress ----
        aloss_str = f"  aloss={avg_aloss:.4f}" if mode == "auto" else ""
        print(
            f"\r  step {step:3d}/{args.steps}  "
            f"games={total_games:4d}  "
            f"samples={total_samples:5d}  "
            f"buf={len(buffer):5d}  "
            f"vloss={avg_vloss:.4f}  "
            f"ploss={avg_ploss:.4f}"
            f"{aloss_str}  "
            f"[{step_time:.1f}s / {elapsed:.0f}s]",
            end="", flush=True,
        )

        # ---- 4. Evaluate vs Random ----
        if step % args.eval_interval == 0 or step == args.steps:
            wr_dict = eval_vs_random(
                game, wrapper, eval_config,
                mode=mode,
                num_games=args.eval_games,
                seed=step * 7777,
                use_neural_discard=use_neural,
            )
            wr = wr_dict.get("all", 0.0)
            tag = " *BEST*" if wr > best_wr else ""
            if wr > best_wr:
                best_wr = wr

            contract_types = sorted(
                k for k in wr_dict
                if k not in ("all", "all_sol", "all_def") and "_" not in k
            )
            parts: list[str] = []
            for ct in contract_types:
                sol_key = f"{ct}_sol"
                def_key = f"{ct}_def"
                sol_wr = wr_dict.get(sol_key)
                def_wr = wr_dict.get(def_key)
                sol_str = f"{sol_wr:.0%}" if sol_wr is not None else "n/a"
                def_str = f"{def_wr:.0%}" if def_wr is not None else "n/a"
                parts.append(
                    f"{ct}: {wr_dict[ct]:.0%} "
                    f"(sol={sol_str}, def={def_str})"
                )

            all_sol = wr_dict.get("all_sol")
            all_def = wr_dict.get("all_def")
            sol_str = f"{all_sol:.0%}" if all_sol is not None else "n/a"
            def_str = f"{all_def:.0%}" if all_def is not None else "n/a"

            print(
                f"\n  >>> EVAL step {step}: "
                f"WR={wr:.0%} (sol={sol_str}, def={def_str})"
                f"  {' | '.join(parts)}"
                f"  (best: {best_wr:.0%}){tag}"
            )

        # ---- 5. Elite Benchmark: NN(0) vs Oracle(MCTS) ----
        if args.elite_interval > 0 and (
            step % args.elite_interval == 0 or step == args.steps
        ):
            elite_t = time.perf_counter()
            elite_dict = eval_elite(
                game, wrapper,
                mode=mode,
                num_games=args.elite_games,
                seed=step * 3333,
                oracle_sims=args.elite_sims,
            )
            elite_wr = elite_dict.get("all", 0.0)
            elite_sol = elite_dict.get("all_sol")
            elite_def = elite_dict.get("all_def")
            e_sol = f"{elite_sol:.0%}" if elite_sol is not None else "n/a"
            e_def = f"{elite_def:.0%}" if elite_def is not None else "n/a"
            elite_time = time.perf_counter() - elite_t

            print(
                f"  >>> ELITE step {step}: "
                f"NN(0) vs Oracle({args.elite_sims}) — "
                f"WR={elite_wr:.0%} (sol={e_sol}, def={e_def})  "
                f"[{elite_time:.1f}s]"
            )

        # ---- 6. Checkpoint ----
        if ckpt_dir is not None and (
            step % args.checkpoint_interval == 0 or step == args.steps
        ):
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            torch.save({
                "model_state_dict": net.state_dict(),
                "body_units": args.body_units,
                "body_layers": args.body_layers,
                "input_dim": game.state_dim,
                "action_dim": game.action_space_size,
                "num_contracts": NUM_CONTRACTS,
                "training_mode": mode,
                "step": step,
                "total_games": total_games,
                "total_samples": total_samples,
                "best_win_rate": best_wr,
            }, ckpt_path)
            print(f"  >>> CHECKPOINT saved: {ckpt_path}")

    # ---- Save ----
    total_time = time.perf_counter() - t0
    print()
    print("=" * 64)
    print("  Training Complete")
    print("=" * 64)
    print(f"  Games: {total_games}  Samples: {total_samples}")
    print(f"  Time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Best win rate: {best_wr:.0%}")
    if mode == "auto":
        print(f"  Auction buffer: {len(auction_buffer)} samples")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": args.body_units,
        "body_layers": args.body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "num_contracts": NUM_CONTRACTS,
        "training_mode": mode,
        "total_games": total_games,
        "best_win_rate": best_wr,
    }, save_path)
    print(f"  Model saved to {save_path}")


if __name__ == "__main__":
    main()
