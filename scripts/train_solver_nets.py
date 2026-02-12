#!/usr/bin/env python3
"""Train neural networks from solver-generated data (Parti only).

Two networks are trained on data produced by SolverPIMC self-play:

  1. **Hand Evaluator** — Given the soloist's 10 cards + trump suit,
     predict the probability of winning Parti.  Trained with binary
     cross-entropy; validated via calibration buckets.

  2. **Card Play Policy** — Given the full 259-dim game state, predict
     the card the solver would play.  Trained with cross-entropy over
     the 32-card action space (masked to legal moves during inference).

The script has three phases:
  Phase 1: Generate data (solver vs solver self-play)
  Phase 2: Train hand evaluator
  Phase 3: Train policy network

Usage:
    # Generate data only (inspect before training)
    python scripts/train_solver_nets.py --phase data --num-games 500

    # Generate + train both networks
    python scripts/train_solver_nets.py --phase all --num-games 2000

    # Train hand evaluator from existing data
    python scripts/train_solver_nets.py --phase hand --data-dir data/solver_parti

    # Train policy from existing data
    python scripts/train_solver_nets.py --phase policy --data-dir data/solver_parti
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from trickster.games.ulti.adapter import UltiGame, UltiNode, _CARD_IDX
from trickster.games.ulti.cards import (
    ALL_SUITS,
    Card,
    NUM_PLAYERS,
    Rank,
    Suit,
    TRICKS_PER_GAME,
)
from trickster.games.ulti.encoder import STATE_DIM, NUM_CARDS
from trickster.games.ulti.game import (
    current_player,
    is_terminal,
    legal_actions,
    play_card,
    soloist_won_simple,
)
from trickster.solver import SolverPIMC


# ---------------------------------------------------------------------------
#  Greedy discard heuristic (reused from train_baseline)
# ---------------------------------------------------------------------------


def _greedy_discard(hand: list[Card], betli: bool, trump: Suit | None) -> list[Card]:
    """Pick 2 cards to discard: weakest non-trump cards first."""
    candidates = list(hand)

    def simple_key(c: Card) -> tuple[int, int]:
        is_trump = 1 if (trump is not None and c.suit == trump) else 0
        return (is_trump, c.rank.value)

    candidates.sort(key=simple_key)
    return candidates[:2]


# ---------------------------------------------------------------------------
#  Hand encoding for the Hand Evaluator (36-dim)
# ---------------------------------------------------------------------------

_SUIT_IDX: dict[Suit, int] = {s: i for i, s in enumerate(ALL_SUITS)}


def encode_hand(hand: list[Card], trump: Suit) -> np.ndarray:
    """Encode a 10-card hand + trump suit into a 36-dim feature vector.

    Layout:
      [0:32]  Card presence bitmap (1.0 if card is in hand)
      [32:36] Trump suit one-hot (4 values)
    """
    x = np.zeros(36, dtype=np.float32)
    for c in hand:
        x[_CARD_IDX[c]] = 1.0
    x[32 + _SUIT_IDX[trump]] = 1.0
    return x


HAND_DIM = 36


# ===========================================================================
#  Phase 1: Data Generation
# ===========================================================================


def generate_data(
    num_games: int,
    soloist_dets: int,
    defender_dets: int,
    solver_depth: int,
    data_dir: Path,
    seed_offset: int = 0,
) -> dict[str, int]:
    """Play solver-vs-solver games and save training data.

    Produces two datasets saved as .npz files:
      hand_data.npz  — {X: (N, 36), y: (N,)}
      policy_data.npz — {X: (M, 259), y: (M,), masks: (M, 32)}

    Returns summary statistics.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    game = UltiGame()
    sol_solver = SolverPIMC(
        num_determinizations=soloist_dets,
        max_exact_tricks=solver_depth,
        game=game,
    )
    def_solver = SolverPIMC(
        num_determinizations=defender_dets,
        max_exact_tricks=solver_depth,
        game=game,
    )

    # Accumulators
    hand_X: list[np.ndarray] = []
    hand_y: list[float] = []

    policy_X: list[np.ndarray] = []
    policy_y: list[int] = []
    policy_masks: list[np.ndarray] = []

    stats = {
        "games": 0,
        "soloist_wins": 0,
        "total_moves": 0,
        "forced_moves": 0,  # only 1 legal card
    }

    t_start = time.perf_counter()

    for gi in range(num_games):
        seed = seed_offset + gi
        rng = random.Random(seed)

        # Set up a Parti game with greedy discard
        state = game.new_game(
            seed=seed,
            training_mode="simple",
            _discard_fn=_greedy_discard,
        )
        gs = state.gs
        soloist = gs.soloist
        trump = gs.trump

        # Record hand evaluator sample (soloist's hand after discard)
        hand_feat = encode_hand(gs.hands[soloist], trump)

        # Play the full game, recording policy data
        move_count = 0
        while not game.is_terminal(state):
            player = game.current_player(state)
            legal = game.legal_actions(state)

            # Choose card via solver
            if player == soloist:
                card = sol_solver.choose_action(state, player, rng)
            else:
                card = def_solver.choose_action(state, player, rng)

            # Record policy training example (skip forced moves — no signal)
            if len(legal) > 1:
                feats = game.encode_state(state, player)
                mask = game.legal_action_mask(state)
                target_idx = _CARD_IDX[card]

                policy_X.append(feats.astype(np.float32))
                policy_y.append(target_idx)
                policy_masks.append(mask.astype(np.float32))
            else:
                stats["forced_moves"] += 1

            state = game.apply(state, card)
            move_count += 1

        # Determine outcome
        won = soloist_won_simple(state.gs)
        hand_X.append(hand_feat)
        hand_y.append(1.0 if won else 0.0)

        stats["games"] += 1
        stats["soloist_wins"] += int(won)
        stats["total_moves"] += move_count

        # Progress
        elapsed = time.perf_counter() - t_start
        if (gi + 1) % 50 == 0 or gi == 0 or (gi + 1) == num_games:
            rate = (gi + 1) / elapsed
            eta = (num_games - gi - 1) / rate if rate > 0 else 0
            wr = stats["soloist_wins"] / stats["games"] * 100
            print(
                f"  [{gi+1:>5}/{num_games}] "
                f"soloist WR={wr:.1f}%  "
                f"moves={stats['total_moves']}  "
                f"rate={rate:.1f} games/s  "
                f"ETA={eta:.0f}s"
            )

    # Convert and save
    hand_X_np = np.stack(hand_X)
    hand_y_np = np.array(hand_y, dtype=np.float32)
    policy_X_np = np.stack(policy_X)
    policy_y_np = np.array(policy_y, dtype=np.int64)
    policy_masks_np = np.stack(policy_masks)

    np.savez_compressed(
        data_dir / "hand_data.npz",
        X=hand_X_np, y=hand_y_np,
    )
    np.savez_compressed(
        data_dir / "policy_data.npz",
        X=policy_X_np, y=policy_y_np, masks=policy_masks_np,
    )

    stats["hand_samples"] = len(hand_y)
    stats["policy_samples"] = len(policy_y)

    # Save stats
    with open(data_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.perf_counter() - t_start
    print(f"\nData generation complete in {elapsed:.1f}s")
    print(f"  Hand evaluator samples: {stats['hand_samples']}")
    print(f"  Policy samples:         {stats['policy_samples']} "
          f"(+{stats['forced_moves']} forced moves skipped)")
    print(f"  Soloist win rate:       {stats['soloist_wins']}/{stats['games']} "
          f"({stats['soloist_wins']/max(1,stats['games'])*100:.1f}%)")
    print(f"  Saved to: {data_dir}")

    # Print sample data for inspection
    _print_sample_data(hand_X_np, hand_y_np, policy_X_np, policy_y_np, policy_masks_np)

    return stats


def _print_sample_data(
    hand_X: np.ndarray,
    hand_y: np.ndarray,
    policy_X: np.ndarray,
    policy_y: np.ndarray,
    policy_masks: np.ndarray,
) -> None:
    """Print a few examples from each dataset for sanity checking."""
    from trickster.games.ulti.encoder import _CARD_IDX

    idx_to_card = {v: k for k, v in _CARD_IDX.items()}
    suit_names = {0: "HEARTS", 1: "BELLS", 2: "LEAVES", 3: "ACORNS"}

    print("\n--- Sample hand evaluator data (first 5) ---")
    for i in range(min(5, len(hand_y))):
        cards_in_hand = [idx_to_card[j] for j in range(32) if hand_X[i, j] > 0.5]
        trump_idx = int(np.argmax(hand_X[i, 32:36]))
        trump_name = suit_names[trump_idx]
        outcome = "WIN" if hand_y[i] > 0.5 else "LOSS"
        hand_str = ", ".join(c.short() for c in sorted(cards_in_hand, key=lambda c: (c.suit.value, c.rank.value)))
        print(f"  [{i}] trump={trump_name:<7s} {outcome:<4s} | {hand_str}")

    print(f"\n--- Sample policy data (first 5) ---")
    for i in range(min(5, len(policy_y))):
        target_card = idx_to_card[policy_y[i]]
        n_legal = int(policy_masks[i].sum())
        trick_no = policy_X[i, 196]  # scalar offset for trick_no (normalised)
        print(
            f"  [{i}] target={target_card.short():<5s}  "
            f"legal={n_legal}  trick_frac={trick_no:.2f}  "
            f"feat_norm={np.linalg.norm(policy_X[i]):.1f}"
        )


# ===========================================================================
#  Phase 2: Hand Evaluator
# ===========================================================================


class HandEvaluator(nn.Module):
    """MLP that predicts soloist win probability from hand + trump.

    Input:  36-dim (32 card bits + 4 trump one-hot)
    Output: scalar logit (use sigmoid for probability)
    """

    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = HAND_DIM
        for h in layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_hand_evaluator(
    data_dir: Path,
    layer_sizes: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    model_dir: Path,
) -> None:
    """Train the hand evaluator from saved data."""
    data = np.load(data_dir / "hand_data.npz")
    X = torch.from_numpy(data["X"])
    y = torch.from_numpy(data["y"])

    n = len(y)
    print(f"\nHand evaluator: {n} samples, {sum(y > 0.5).item()} wins "
          f"({sum(y > 0.5).item()/n*100:.1f}%)")
    print(f"  Architecture: {HAND_DIM} -> {' -> '.join(map(str, layer_sizes))} -> 1")

    # Train / validation split (80/20)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = HandEvaluator(layer_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 10
    best_state = None

    print(f"  Training for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_dl:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_n += len(yb)

        # Validate
        model.eval()
        val_loss = 0.0
        val_n = 0
        all_probs: list[float] = []
        all_labels: list[float] = []
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(yb)
                val_n += len(yb)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.tolist())
                all_labels.extend(yb.tolist())

        train_loss /= max(1, train_n)
        val_loss /= max(1, val_n)

        # Accuracy
        preds = np.array(all_probs) > 0.5
        labels = np.array(all_labels) > 0.5
        acc = (preds == labels).mean() * 100

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:>3d}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={acc:.1f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibration analysis
    print("\n--- Calibration (validation set) ---")
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_dl:
            probs = torch.sigmoid(model(xb))
            all_probs.extend(probs.tolist())
            all_labels.extend(yb.tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)

    # 5 calibration buckets
    bucket_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    print(f"  {'Bucket':<12s} {'Count':>6s} {'Pred%':>7s} {'Actual%':>9s}")
    print(f"  {'-'*36}")
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (probs_arr >= lo) & (probs_arr < hi)
        count = mask.sum()
        if count > 0:
            pred_mean = probs_arr[mask].mean() * 100
            actual_mean = labels_arr[mask].mean() * 100
        else:
            pred_mean = actual_mean = 0.0
        print(f"  {lo:.0%}-{min(hi,1.0):.0%}       {count:>5d}  "
              f"{pred_mean:>6.1f}%  {actual_mean:>8.1f}%")

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "hand_evaluator.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "layer_sizes": layer_sizes,
        "input_dim": HAND_DIM,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, save_path)
    print(f"\n  Model saved to {save_path}")


# ===========================================================================
#  Phase 3: Policy Network
# ===========================================================================


class PolicyNetwork(nn.Module):
    """MLP that predicts the solver's card choice.

    Input:  259-dim game state (from UltiEncoder)
    Output: 32 logits (one per card in the deck)
    """

    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = STATE_DIM
        for h in layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, NUM_CARDS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_policy_network(
    data_dir: Path,
    layer_sizes: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    model_dir: Path,
) -> None:
    """Train the policy network from saved data."""
    data = np.load(data_dir / "policy_data.npz")
    X = torch.from_numpy(data["X"])
    y = torch.from_numpy(data["y"])
    masks = torch.from_numpy(data["masks"])

    n = len(y)
    print(f"\nPolicy network: {n} samples")
    print(f"  Architecture: {STATE_DIM} -> {' -> '.join(map(str, layer_sizes))} -> {NUM_CARDS}")

    # Train / validation split (80/20)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx], masks[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx], masks[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = PolicyNetwork(layer_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 10
    best_state = None

    print(f"  Training for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        for xb, yb, mb in train_dl:
            logits = model(xb)
            # Mask illegal moves to -inf before loss
            logits_masked = logits + (1.0 - mb) * (-1e9)
            loss = F.cross_entropy(logits_masked, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_correct += (logits_masked.argmax(dim=1) == yb).sum().item()
            train_n += len(yb)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for xb, yb, mb in val_dl:
                logits = model(xb)
                logits_masked = logits + (1.0 - mb) * (-1e9)
                loss = F.cross_entropy(logits_masked, yb)
                val_loss += loss.item() * len(yb)
                val_correct += (logits_masked.argmax(dim=1) == yb).sum().item()
                val_n += len(yb)

        train_loss /= max(1, train_n)
        val_loss /= max(1, val_n)
        train_acc = train_correct / max(1, train_n) * 100
        val_acc = val_correct / max(1, val_n) * 100

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:>3d}  "
                  f"train_loss={train_loss:.4f} acc={train_acc:.1f}%  "
                  f"val_loss={val_loss:.4f} acc={val_acc:.1f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Per-position accuracy breakdown
    print("\n--- Per-position accuracy (validation set) ---")
    model.eval()

    # Bucket by trick number (encoded in scalar section)
    trick_correct: dict[int, int] = {}
    trick_total: dict[int, int] = {}

    with torch.no_grad():
        for xb, yb, mb in val_dl:
            logits = model(xb)
            logits_masked = logits + (1.0 - mb) * (-1e9)
            preds = logits_masked.argmax(dim=1)
            correct = (preds == yb)

            # Extract trick number from the state vector (normalised)
            # Scalar offset + 0 = trick_no / TRICKS_PER_GAME
            trick_fracs = xb[:, 196].numpy()  # _SCALAR_OFF + 0
            trick_nos = (trick_fracs * TRICKS_PER_GAME).round().astype(int)

            for t, c in zip(trick_nos, correct.numpy()):
                trick_correct[t] = trick_correct.get(t, 0) + int(c)
                trick_total[t] = trick_total.get(t, 0) + 1

    print(f"  {'Trick':>6s} {'Count':>6s} {'Accuracy':>9s}")
    print(f"  {'-'*23}")
    for t in sorted(trick_total.keys()):
        acc = trick_correct[t] / trick_total[t] * 100
        print(f"  {t:>5d}  {trick_total[t]:>5d}   {acc:>7.1f}%")

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "policy_network.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "layer_sizes": layer_sizes,
        "input_dim": STATE_DIM,
        "output_dim": NUM_CARDS,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, save_path)
    print(f"\n  Model saved to {save_path}")


# ===========================================================================
#  CLI
# ===========================================================================


def parse_layer_sizes(s: str) -> list[int]:
    """Parse a comma-separated string of integers."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train neural networks from solver-generated data (Parti).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Phase selection
    parser.add_argument(
        "--phase", choices=["data", "hand", "policy", "all"], default="all",
        help="Which phase(s) to run (default: all)",
    )

    # Data generation
    parser.add_argument("--num-games", type=int, default=2000,
                        help="Number of games to generate (default: 2000)")
    parser.add_argument("--soloist-dets", type=int, default=15,
                        help="Determinizations for soloist solver (default: 15)")
    parser.add_argument("--defender-dets", type=int, default=5,
                        help="Determinizations for defender solver (default: 5)")
    parser.add_argument("--solver-depth", type=int, default=6,
                        help="Exact solver depth in tricks (default: 6)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Starting seed for data generation (default: 0)")

    # Architecture
    parser.add_argument("--hand-layers", type=str, default="128,64",
                        help="Hand evaluator hidden layers (default: 128,64)")
    parser.add_argument("--policy-layers", type=str, default="256,256,256",
                        help="Policy network hidden layers (default: 256,256,256)")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")

    # Paths
    parser.add_argument("--data-dir", type=str, default="data/solver_parti",
                        help="Directory for generated data (default: data/solver_parti)")
    parser.add_argument("--model-dir", type=str, default="models/solver_parti",
                        help="Directory for trained models (default: models/solver_parti)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    hand_layers = parse_layer_sizes(args.hand_layers)
    policy_layers = parse_layer_sizes(args.policy_layers)

    phases = set()
    if args.phase == "all":
        phases = {"data", "hand", "policy"}
    else:
        phases.add(args.phase)

    print("=" * 60)
    print("  Solver-to-NN Training Pipeline (Parti)")
    print("=" * 60)

    # Phase 1: Data generation
    if "data" in phases:
        print(f"\n{'='*60}")
        print(f"  Phase 1: Data Generation")
        print(f"{'='*60}")
        print(f"  Games:         {args.num_games}")
        print(f"  Soloist dets:  {args.soloist_dets}")
        print(f"  Defender dets: {args.defender_dets}")
        print(f"  Solver depth:  {args.solver_depth}")
        print()

        generate_data(
            num_games=args.num_games,
            soloist_dets=args.soloist_dets,
            defender_dets=args.defender_dets,
            solver_depth=args.solver_depth,
            data_dir=data_dir,
            seed_offset=args.seed_offset,
        )

    # Phase 2: Hand evaluator
    if "hand" in phases:
        if not (data_dir / "hand_data.npz").exists():
            print(f"\nERROR: {data_dir / 'hand_data.npz'} not found. "
                  f"Run --phase data first.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  Phase 2: Hand Evaluator Training")
        print(f"{'='*60}")

        train_hand_evaluator(
            data_dir=data_dir,
            layer_sizes=hand_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_dir=model_dir,
        )

    # Phase 3: Policy network
    if "policy" in phases:
        if not (data_dir / "policy_data.npz").exists():
            print(f"\nERROR: {data_dir / 'policy_data.npz'} not found. "
                  f"Run --phase data first.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  Phase 3: Policy Network Training")
        print(f"{'='*60}")

        train_policy_network(
            data_dir=data_dir,
            layer_sizes=policy_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_dir=model_dir,
        )

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
