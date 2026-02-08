#!/usr/bin/env python3
"""Ulti baseline training — curriculum with PyTorch.

Supports three curriculum modes:
  --mode simple   Train only on Simple (Parti) contracts.
  --mode betli    Train only on Betli contracts.
  --mode mixed    Train on a 50/50 mix of Simple and Betli.

Self-play loop:
  1. Deal a game via ``UltiGame.new_game(training_mode=...)``.
  2. All 3 players use MCTS (N=20 sims, 1 determinization) to pick cards.
  3. Collect (state, mask, mcts_policy, reward) tuples in a ReplayBuffer.
  4. Train the ``UltiNet`` on mini-batches of 64 with Adam.
  5. Every 10 training steps, evaluate vs Random over 20 games.

The "mixed" mode is the true test of the shared backbone: the Is_Betli
bit in the Contract DNA must flip the AI's entire strategy from
"win tricks with high cards" to "dump high cards to lose tricks".

Usage:
    python scripts/train_baseline.py [--mode simple] [--steps 200]
    python scripts/train_baseline.py --mode betli --steps 100
    python scripts/train_baseline.py --mode mixed --steps 200
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
from trickster.games.ulti.cards import Card, Rank, Suit, BETLI_STRENGTH
from trickster.games.ulti.game import (
    discard_talon,
    soloist_lost_betli,
    soloist_won_simple,
)
from trickster.mcts import MCTSConfig, alpha_mcts_policy
from trickster.model import UltiNet, UltiNetWrapper
from trickster.train_utils import ReplayBuffer, outcome_for_player, simple_outcome


# ---------------------------------------------------------------------------
#  Mode helpers
# ---------------------------------------------------------------------------

MODES = ("simple", "betli", "mixed")


def _pick_training_mode(mode: str, rng: random.Random) -> str:
    """Return the training_mode string for a single game."""
    if mode == "mixed":
        return rng.choice(["simple", "betli"])
    return mode


def _soloist_won(state: UltiNode) -> bool:
    """Determine if soloist won, respecting betli vs simple."""
    gs = state.gs
    if gs.betli:
        return not soloist_lost_betli(gs)
    return soloist_won_simple(gs)


# ---------------------------------------------------------------------------
#  Greedy talon discard heuristics
# ---------------------------------------------------------------------------


def _betli_discard_key(card: Card) -> int:
    """Sort key for Betli discard: highest Betli-strength first.

    In Betli, the ranking is A(low) < K < Q < J < 10 < 9 < 8 < 7(high).
    Discarding the highest-strength cards (7s, 8s, 9s) gives the soloist
    a fighting chance by removing dangerous trick-winners.
    """
    return -BETLI_STRENGTH[card.rank]


def _simple_discard_key(card: Card) -> int:
    """Sort key for Simple discard: lowest-value / weakest first.

    Discard low-value non-trump cards to tighten the hand.
    We prefer keeping Aces and Tens (high point value).
    Sort by rank ascending — first two will be discarded.
    """
    return card.rank.value


def _greedy_discard(hand: list[Card], betli: bool, trump: Suit | None) -> list[Card]:
    """Pick 2 cards to discard from a 12-card hand using a heuristic.

    Betli mode:  Discard the two highest-strength cards (7s, 8s, 9s).
                 These are the most dangerous in Betli because they win tricks.
    Simple mode: Discard the two weakest non-trump cards.
                 Keeps strong trump cards and point cards (A, 10).
    """
    candidates = list(hand)

    if betli:
        candidates.sort(key=_betli_discard_key)
    else:
        # Prefer discarding non-trump, then by weakness
        def simple_key(c: Card) -> tuple[int, int]:
            is_trump = 1 if (trump is not None and c.suit == trump) else 0
            return (is_trump, c.rank.value)
        candidates.sort(key=simple_key)

    return candidates[:2]


# ---------------------------------------------------------------------------
#  Self-play: one full game → training samples
# ---------------------------------------------------------------------------


def play_one_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    config: MCTSConfig,
    seed: int,
    mode: str = "simple",
    use_greedy_discard: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Play one self-play game.

    Returns list of ``(state_feats, mask, pi, reward, is_soloist)``.
    The ``is_soloist`` flag enables weighted replay sampling.

    When ``use_greedy_discard`` is True (default), the soloist uses a
    heuristic discard instead of random:
    - Betli: discard highest-strength cards (7s, 8s, 9s)
    - Simple: discard weakest non-trump cards
    """
    rng = random.Random(seed)
    training_mode = _pick_training_mode(mode, rng)
    state = game.new_game(
        seed=seed,
        training_mode=training_mode,
        _discard_fn=_greedy_discard if use_greedy_discard else None,
    )
    soloist_idx = state.gs.soloist
    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        # MCTS search
        pi, action = alpha_mcts_policy(
            state, game, wrapper, player, config, rng,
        )

        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append((
            state_feats.copy(),
            mask.copy(),
            pi.copy(),
            player,
        ))

        state = game.apply(state, action)

    # Determine outcome (handles both simple and betli)
    won = _soloist_won(state)

    # Fill in rewards, tagging each sample with is_soloist
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for state_feats, mask, pi, player in trajectory:
        reward = outcome_for_player(state, player, won)
        is_sol = (player == soloist_idx)
        samples.append((state_feats, mask, pi, reward, is_sol))

    return samples


# ---------------------------------------------------------------------------
#  Evaluation: trained model vs random
# ---------------------------------------------------------------------------


def eval_vs_random(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    config: MCTSConfig,
    mode: str = "simple",
    num_games: int = 20,
    seed: int = 99999,
    use_greedy_discard: bool = True,
) -> dict[str, float]:
    """Play trained agent (player 0) vs random opponents.

    Returns a dict with per-role, per-contract-type win rates::

        {
            "simple": 0.70,         # combined WR for simple
            "simple_sol": 0.45,     # soloist WR for simple
            "simple_def": 0.82,     # defender WR for simple
            "betli": 0.73,
            "betli_sol": 0.20,
            "betli_def": 0.97,
            "all": 0.71,
            "all_sol": 0.33,
            "all_def": 0.90,
        }
    """
    # Track wins and counts per (mode, role)
    wins: dict[str, int] = {}
    counts: dict[str, int] = {}
    rng_mode = random.Random(seed)

    def _inc(key: str, won: bool) -> None:
        counts[key] = counts.get(key, 0) + 1
        if won:
            wins[key] = wins.get(key, 0) + 1

    for g in range(num_games):
        rng = random.Random(seed + g)
        rng_rand = random.Random(seed + g + 50000)

        training_mode = _pick_training_mode(mode, rng_mode)
        state = game.new_game(
            seed=seed + g,
            training_mode=training_mode,
            starting_leader=g % 3,
            _discard_fn=_greedy_discard if use_greedy_discard else None,
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

        # Track per (contract_type), per (contract_type + role), and global
        _inc(training_mode, p0_won)
        _inc(f"{training_mode}_{role}", p0_won)
        _inc("all", p0_won)
        _inc(f"all_{role}", p0_won)

    # Build result dict
    result: dict[str, float] = {}
    for key in sorted(counts):
        result[key] = wins.get(key, 0) / max(1, counts[key])
    return result


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ulti Curriculum Training — simple / betli / mixed",
    )
    parser.add_argument("--mode", type=str, default="simple",
                        choices=MODES,
                        help="Curriculum mode: simple, betli, or mixed (default simple)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training iterations (default 200)")
    parser.add_argument("--games-per-step", type=int, default=4,
                        help="Self-play games per training step (default 4)")
    parser.add_argument("--sims", type=int, default=20,
                        help="MCTS simulations per move (default 20)")
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
    args = parser.parse_args()

    if args.save is None:
        args.save = f"models/ulti_{args.mode}.pt"

    # ---- Setup ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    mode = args.mode

    game = UltiGame()
    net = UltiNet(
        input_dim=game.state_dim,
        body_units=args.body_units,
        body_layers=args.body_layers,
        action_dim=game.action_space_size,
    )

    # Optionally load a pre-trained model (e.g. simple-trained → fine-tune on mixed)
    if args.load:
        checkpoint = torch.load(args.load, weights_only=True)
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded pre-trained model from {args.load}")

    wrapper = UltiNetWrapper(net, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    buffer = ReplayBuffer(capacity=args.buffer_size)
    np_rng = np.random.default_rng(args.seed)

    # MCTS config for self-play (exploration on)
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

    # MCTS config for evaluation (exploitation)
    eval_config = MCTSConfig(
        simulations=args.sims,
        determinizations=args.dets,
        c_puct=1.5,
        dirichlet_alpha=0.0,   # no noise
        dirichlet_weight=0.0,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.1,        # near-greedy
    )

    param_count = sum(p.numel() for p in net.parameters())

    mode_label = {"simple": "Simple (Parti)", "betli": "Betli", "mixed": "Mixed (Simple + Betli)"}

    print("=" * 64)
    print(f"  Ulti Training — {mode_label[mode]} Curriculum")
    print("=" * 64)
    print(f"  Mode: {mode}")
    print(f"  Model: UltiNet {args.body_units}x{args.body_layers} "
          f"({param_count:,} params)")
    print(f"  MCTS: {args.sims} sims x {args.dets} dets")
    print(f"  Steps: {args.steps} x {args.games_per_step} games/step "
          f"= {args.steps * args.games_per_step} games")
    print(f"  LR: {args.lr}  Batch: {args.batch_size}  "
          f"Buffer: {args.buffer_size}")
    print(f"  Device: {device}")
    print(f"  Eval: every {args.eval_interval} steps, "
          f"{args.eval_games} games vs Random")
    print()

    t0 = time.perf_counter()
    total_games = 0
    total_samples = 0
    best_wr = 0.0

    for step in range(1, args.steps + 1):
        step_t = time.perf_counter()

        # ---- 1. Self-play: collect samples ----
        step_samples = 0
        for g in range(args.games_per_step):
            game_seed = args.seed + step * 1000 + g
            samples = play_one_game(game, wrapper, train_config, game_seed, mode=mode)
            for s, m, p, r, is_sol in samples:
                buffer.push(s, m, p, r, is_soloist=is_sol)
            step_samples += len(samples)
            total_games += 1

        total_samples += step_samples

        # ---- 2. Train on replay buffer ----
        if len(buffer) >= args.batch_size:
            net.train()
            # Do multiple gradient steps per self-play batch
            train_steps = max(1, step_samples // args.batch_size)
            total_vloss = 0.0
            total_ploss = 0.0

            for _ in range(train_steps):
                states, masks, policies, rewards = buffer.sample(
                    args.batch_size, np_rng,
                )

                s_t = torch.from_numpy(states).float().to(device)
                m_t = torch.from_numpy(masks).bool().to(device)
                pi_t = torch.from_numpy(policies).float().to(device)
                z_t = torch.from_numpy(rewards).float().to(device)

                log_probs, values = net(s_t, m_t)

                # Value loss: MSE(predicted_value, reward)
                value_loss = F.mse_loss(values, z_t)

                # Policy loss: Cross-entropy(MCTS_policy, predicted_policy)
                # = -sum(pi * log_probs)
                policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()

                loss = value_loss + policy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()

                total_vloss += value_loss.item()
                total_ploss += policy_loss.item()

            avg_vloss = total_vloss / train_steps
            avg_ploss = total_ploss / train_steps
        else:
            avg_vloss = 0.0
            avg_ploss = 0.0

        elapsed = time.perf_counter() - t0
        step_time = time.perf_counter() - step_t

        # ---- 3. Progress ----
        print(
            f"\r  step {step:3d}/{args.steps}  "
            f"games={total_games:4d}  "
            f"samples={total_samples:5d}  "
            f"buf={len(buffer):5d}  "
            f"vloss={avg_vloss:.4f}  "
            f"ploss={avg_ploss:.4f}  "
            f"[{step_time:.1f}s / {elapsed:.0f}s]",
            end="", flush=True,
        )

        # ---- 4. Evaluate ----
        if step % args.eval_interval == 0 or step == args.steps:
            wr_dict = eval_vs_random(
                game, wrapper, eval_config,
                mode=mode,
                num_games=args.eval_games,
                seed=step * 7777,
            )
            wr = wr_dict.get("all", 0.0)
            tag = " *BEST*" if wr > best_wr else ""
            if wr > best_wr:
                best_wr = wr

            # Format per-mode + per-role breakdown
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

    # ---- Save ----
    total_time = time.perf_counter() - t0
    print()
    print("=" * 64)
    print("  Training Complete")
    print("=" * 64)
    print(f"  Games: {total_games}  Samples: {total_samples}")
    print(f"  Time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Best win rate: {best_wr:.0%}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": args.body_units,
        "body_layers": args.body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "training_mode": mode,
        "total_games": total_games,
        "best_win_rate": best_wr,
    }, save_path)
    print(f"  Model saved to {save_path}")


if __name__ == "__main__":
    main()
