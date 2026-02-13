#!/usr/bin/env python3
"""Train Ulti Parti models at 3 strength tiers and evaluate head-to-head.

Mirrors the Snapszer hybrid training workflow:
  1. Train 3 tiers (Scout / Knight / Bishop) with increasing budgets.
  2. Round-robin evaluation: every model plays every other model.
  3. Print a ranking table so you can confirm learning is happening.

Each tier uses the HybridPlayer (MCTS + Cython alpha-beta solver).
Models are saved to ``models/ulti/<tier-name>/``.

Usage:
    # Train all 3 tiers + eval (default)
    python scripts/train_ulti.py

    # Train only, skip eval
    python scripts/train_ulti.py --no-eval

    # Eval only (models must already exist)
    python scripts/train_ulti.py --eval-only

    # Single tier
    python scripts/train_ulti.py --tiers scout
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import Suit
from trickster.games.ulti.game import soloist_won_simple
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, UltiNetWrapper
from trickster.train_utils import ReplayBuffer, simple_outcome


# ---------------------------------------------------------------------------
#  Tier definitions
# ---------------------------------------------------------------------------

@dataclass
class UltiTier:
    name: str
    steps: int
    games_per_step: int
    train_steps: int          # SGD steps per iteration (decoupled from games)
    sims: int
    def_sims: int
    endgame_tricks: int
    pimc_dets: int
    solver_temp: float
    body_units: int
    body_layers: int
    lr_start: float
    lr_end: float             # cosine decay target
    batch_size: int
    buffer_size: int
    description: str

    @property
    def total_games(self) -> int:
        return self.steps * self.games_per_step


TIERS: dict[str, UltiTier] = {
    "scout": UltiTier(
        name="U1-Scout",
        steps=500, games_per_step=8, train_steps=50,
        sims=20, def_sims=8,
        endgame_tricks=6, pimc_dets=20, solver_temp=0.5,
        body_units=256, body_layers=4,
        lr_start=1e-3, lr_end=2e-4,
        batch_size=64, buffer_size=50_000,
        description="Scout (500 steps, 4k games) — baseline",
    ),
    "knight": UltiTier(
        name="U2-Knight",
        steps=2000, games_per_step=8, train_steps=80,
        sims=30, def_sims=12,
        endgame_tricks=6, pimc_dets=20, solver_temp=0.5,
        body_units=256, body_layers=4,
        lr_start=1e-3, lr_end=1e-4,
        batch_size=64, buffer_size=50_000,
        description="Knight (2000 steps, 16k games) — medium",
    ),
    "bishop": UltiTier(
        name="U3-Bishop",
        steps=8000, games_per_step=8, train_steps=100,
        sims=30, def_sims=12,
        endgame_tricks=6, pimc_dets=20, solver_temp=0.5,
        body_units=256, body_layers=4,
        lr_start=1e-3, lr_end=5e-5,
        batch_size=64, buffer_size=50_000,
        description="Bishop (8000 steps, 64k games) — strong",
    ),
}

TIER_ORDER = ["scout", "knight", "bishop"]


# ---------------------------------------------------------------------------
#  Self-play: one game → training samples
# ---------------------------------------------------------------------------


def play_one_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    sol_cfg: MCTSConfig,
    def_cfg: MCTSConfig,
    seed: int,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
    solver_temp: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Play one hybrid self-play game (Parti, random trump).

    Returns list of (state, mask, policy, reward, is_soloist).
    """
    rng = random.Random(seed)
    state = game.new_game(
        seed=seed,
        training_mode="simple",
        starting_leader=seed % 3,
    )
    soloist_idx = state.gs.soloist

    sol_hybrid = HybridPlayer(
        game, wrapper,
        mcts_config=sol_cfg,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )
    def_hybrid = HybridPlayer(
        game, wrapper,
        mcts_config=def_cfg,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )

    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        if player == soloist_idx:
            pi, action = sol_hybrid.choose_action_with_policy(state, player, rng)
        else:
            pi, action = def_hybrid.choose_action_with_policy(state, player, rng)

        state_feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        trajectory.append((
            state_feats.copy(),
            mask.copy(),
            np.asarray(pi, dtype=np.float32).copy(),
            player,
        ))

        state = game.apply(state, action)

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for state_feats, mask, pi, player in trajectory:
        reward = simple_outcome(state, player)
        is_sol = (player == soloist_idx)
        samples.append((state_feats, mask, pi, reward, is_sol))

    return samples


# ---------------------------------------------------------------------------
#  Training loop for one tier
# ---------------------------------------------------------------------------


def _cosine_lr(step: int, total_steps: int, lr_start: float, lr_end: float) -> float:
    """Cosine annealing from lr_start to lr_end."""
    if total_steps <= 1:
        return lr_start
    frac = step / total_steps
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * frac))


def train_tier(
    tier: UltiTier,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[UltiNet, Path]:
    """Train one tier. Returns (net, model_dir)."""

    model_dir = Path("models") / "ulti" / tier.name
    model_dir.mkdir(parents=True, exist_ok=True)

    game = UltiGame()
    net = UltiNet(
        input_dim=game.state_dim,
        body_units=tier.body_units,
        body_layers=tier.body_layers,
        action_dim=game.action_space_size,
    )
    wrapper = UltiNetWrapper(net, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=tier.lr_start, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=tier.buffer_size, seed=seed + 1)
    np_rng = np.random.default_rng(seed)

    sol_cfg = MCTSConfig(
        simulations=tier.sims, determinizations=1,
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_weight=0.25,
        use_value_head=True, use_policy_priors=True, visit_temp=1.0,
    )
    def_cfg = MCTSConfig(
        simulations=tier.def_sims, determinizations=1,
        c_puct=1.5, dirichlet_alpha=0.1, dirichlet_weight=0.15,
        use_value_head=True, use_policy_priors=True, visit_temp=0.5,
    )

    param_count = sum(p.numel() for p in net.parameters())
    report_interval = max(1, tier.steps // 20)

    print(f"    Net: {tier.body_units}x{tier.body_layers} ({param_count:,} params)")
    print(f"    MCTS: sol={tier.sims}s def={tier.def_sims}s | "
          f"Solver: {SOLVER_ENGINE} (endgame={tier.endgame_tricks}t, "
          f"PIMC={tier.pimc_dets}d)")
    print(f"    LR: {tier.lr_start} → {tier.lr_end} (cosine)  "
          f"batch={tier.batch_size}  SGD/step={tier.train_steps}  "
          f"buffer={tier.buffer_size:,}")
    print()

    t0 = time.perf_counter()
    total_games = 0
    total_samples = 0
    total_sgd_steps = 0

    # Track metrics over time
    history_vloss: list[float] = []
    history_ploss: list[float] = []
    history_pacc: list[float] = []
    history_sol_vloss: list[float] = []
    history_def_vloss: list[float] = []

    for step in range(1, tier.steps + 1):
        # ---- 0. Update learning rate (cosine schedule) ----
        lr = _cosine_lr(step, tier.steps, tier.lr_start, tier.lr_end)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- 1. Self-play ----
        step_samples = 0
        sp_wins = 0
        sp_total = 0

        for g in range(tier.games_per_step):
            game_seed = seed + step * 1000 + g
            samples = play_one_game(
                game, wrapper, sol_cfg, def_cfg, game_seed,
                endgame_tricks=tier.endgame_tricks,
                pimc_dets=tier.pimc_dets,
                solver_temp=tier.solver_temp,
            )

            # Track soloist self-play win rate
            sol_rewards = [r for _, _, _, r, is_sol in samples if is_sol]
            if sol_rewards:
                sp_total += 1
                if sol_rewards[0] > 0:
                    sp_wins += 1

            for s, m, p, r, is_sol in samples:
                buffer.push(s, m, p, r, is_soloist=is_sol)
            step_samples += len(samples)
            total_games += 1

        total_samples += step_samples

        # ---- 2. Train (decoupled SGD steps) ----
        avg_vloss = 0.0
        avg_ploss = 0.0
        avg_pacc = 0.0
        avg_sol_vloss = 0.0
        avg_def_vloss = 0.0

        if len(buffer) >= tier.batch_size:
            net.train()
            n_train = tier.train_steps
            total_vloss = 0.0
            total_ploss = 0.0
            total_pacc = 0.0
            total_sol_vloss = 0.0
            total_def_vloss = 0.0
            sol_count = 0
            def_count = 0

            for _ in range(n_train):
                states, masks, policies, rewards, is_sol = buffer.sample(
                    tier.batch_size, np_rng,
                )

                s_t = torch.from_numpy(states).float().to(device)
                m_t = torch.from_numpy(masks).bool().to(device)
                pi_t = torch.from_numpy(policies).float().to(device)
                z_t = torch.from_numpy(rewards).float().to(device)
                is_sol_t = torch.from_numpy(is_sol).bool().to(device)

                log_probs, values = net.forward_dual(s_t, m_t, is_sol_t)

                value_loss = F.mse_loss(values, z_t)
                policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
                loss = value_loss + policy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()

                total_vloss += value_loss.item()
                total_ploss += policy_loss.item()

                # Policy accuracy: does top-1 prediction match target's argmax?
                with torch.no_grad():
                    pred_top1 = log_probs.argmax(dim=-1)
                    target_top1 = pi_t.argmax(dim=-1)
                    total_pacc += (pred_top1 == target_top1).float().mean().item()

                # Per-role value loss
                with torch.no_grad():
                    sol_mask = is_sol_t
                    def_mask = ~is_sol_t
                    if sol_mask.any():
                        sv = F.mse_loss(values[sol_mask], z_t[sol_mask]).item()
                        total_sol_vloss += sv
                        sol_count += 1
                    if def_mask.any():
                        dv = F.mse_loss(values[def_mask], z_t[def_mask]).item()
                        total_def_vloss += dv
                        def_count += 1

            total_sgd_steps += n_train
            avg_vloss = total_vloss / n_train
            avg_ploss = total_ploss / n_train
            avg_pacc = total_pacc / n_train
            avg_sol_vloss = total_sol_vloss / max(1, sol_count)
            avg_def_vloss = total_def_vloss / max(1, def_count)

        history_vloss.append(avg_vloss)
        history_ploss.append(avg_ploss)
        history_pacc.append(avg_pacc)
        history_sol_vloss.append(avg_sol_vloss)
        history_def_vloss.append(avg_def_vloss)
        sp_wr = sp_wins / max(1, sp_total)

        # ---- 3. Report ----
        if step % report_interval == 0 or step == 1 or step == tier.steps:
            elapsed = time.perf_counter() - t0
            pct = step / tier.steps * 100
            rate = total_games / elapsed if elapsed > 0 else 0

            # Trend: compare current window to previous window
            window = max(1, report_interval)
            recent_v = np.mean(history_vloss[-window:])
            prev_v = np.mean(history_vloss[-2*window:-window]) if len(history_vloss) > window else recent_v
            recent_p = np.mean(history_ploss[-window:])
            prev_p = np.mean(history_ploss[-2*window:-window]) if len(history_ploss) > window else recent_p
            recent_acc = np.mean(history_pacc[-window:])

            v_arrow = "↓" if recent_v < prev_v - 0.005 else ("↑" if recent_v > prev_v + 0.005 else "→")
            p_arrow = "↓" if recent_p < prev_p - 0.01 else ("↑" if recent_p > prev_p + 0.01 else "→")

            buf_stats = buffer.stats()
            buf_swr = buf_stats.get("sol_win_rate", 0.0) if buf_stats else 0.0

            print(
                f"    step {step:>5d}/{tier.steps} ({pct:4.0f}%)  "
                f"games={total_games:>5d}  "
                f"vloss={avg_vloss:.4f}{v_arrow} "
                f"(sol={avg_sol_vloss:.3f} def={avg_def_vloss:.3f})  "
                f"ploss={avg_ploss:.4f}{p_arrow}  "
                f"pacc={recent_acc:.0%}  "
                f"lr={lr:.1e}  "
                f"[{elapsed:.0f}s]",
                flush=True,
            )

    train_time = time.perf_counter() - t0

    # ---- Summary ----
    final_v = np.mean(history_vloss[-report_interval:])
    final_p = np.mean(history_ploss[-report_interval:])
    final_acc = np.mean(history_pacc[-report_interval:])
    start_v = np.mean(history_vloss[:report_interval]) if len(history_vloss) > report_interval else final_v
    start_p = np.mean(history_ploss[:report_interval]) if len(history_ploss) > report_interval else final_p
    start_acc = np.mean(history_pacc[:report_interval]) if len(history_pacc) > report_interval else final_acc

    print()
    print(f"    ┌─ TRAINING SUMMARY ──────────────────────────────────────")
    print(f"    │  Games: {total_games:,}  Samples: {total_samples:,}  "
          f"SGD steps: {total_sgd_steps:,}  Time: {train_time:.0f}s")
    print(f"    │  Value head:   {start_v:.4f} → {final_v:.4f}  "
          f"({'improved' if final_v < start_v - 0.001 else 'plateau'})")
    print(f"    │  Policy head:  {start_p:.4f} → {final_p:.4f}  "
          f"({'improved' if final_p < start_p - 0.01 else 'plateau'})")
    print(f"    │  Policy acc:   {start_acc:.1%} → {final_acc:.1%}  "
          f"({'improved' if final_acc > start_acc + 0.01 else 'plateau'})")
    print(f"    └──────────────────────────────────────────────────────────")

    # ---- Save ----
    save_path = model_dir / "model.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": tier.body_units,
        "body_layers": tier.body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "training_mode": "simple",
        "method": "hybrid",
        "endgame_tricks": tier.endgame_tricks,
        "pimc_dets": tier.pimc_dets,
        "total_games": total_games,
        "total_samples": total_samples,
        "total_sgd_steps": total_sgd_steps,
        "train_time_s": round(train_time, 1),
        "final_vloss": round(final_v, 6),
        "final_ploss": round(final_p, 6),
        "final_pacc": round(final_acc, 4),
    }, save_path)

    info = {
        "tier": tier.name,
        "steps": tier.steps,
        "games_per_step": tier.games_per_step,
        "train_steps_per_iter": tier.train_steps,
        "total_games": total_games,
        "total_samples": total_samples,
        "total_sgd_steps": total_sgd_steps,
        "sims": tier.sims,
        "def_sims": tier.def_sims,
        "endgame_tricks": tier.endgame_tricks,
        "pimc_dets": tier.pimc_dets,
        "body_units": tier.body_units,
        "body_layers": tier.body_layers,
        "lr_start": tier.lr_start,
        "lr_end": tier.lr_end,
        "solver": SOLVER_ENGINE,
        "train_time_s": round(train_time, 1),
        "final_vloss": round(final_v, 6),
        "final_ploss": round(final_p, 6),
        "final_pacc": round(final_acc, 4),
    }
    (model_dir / "train_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8",
    )

    print(f"    Saved to {save_path}")
    return net, model_dir


# ---------------------------------------------------------------------------
#  Head-to-head evaluation
# ---------------------------------------------------------------------------


def load_model(model_dir: Path, device: str = "cpu") -> tuple[UltiNet, UltiNetWrapper]:
    """Load a saved UltiNet from a tier directory."""
    cp = torch.load(model_dir / "model.pt", weights_only=False, map_location=device)
    net = UltiNet(
        input_dim=cp.get("input_dim", 291),
        body_units=cp.get("body_units", 256),
        body_layers=cp.get("body_layers", 4),
        action_dim=cp.get("action_dim", 32),
    )
    net.load_state_dict(cp["model_state_dict"])
    wrapper = UltiNetWrapper(net, device=device)
    return net, wrapper


def play_eval_game(
    game: UltiGame,
    sol_wrapper: UltiNetWrapper | None,
    def_wrapper: UltiNetWrapper | None,
    mcts_cfg: MCTSConfig,
    seed: int,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
) -> tuple[bool, str | None]:
    """Play one eval game. Returns (soloist_won, trump_name)."""
    rng = random.Random(seed)
    state = game.new_game(
        seed=seed,
        training_mode="simple",
        starting_leader=seed % 3,
    )
    soloist_idx = state.gs.soloist
    trump_name = state.gs.trump.value if state.gs.trump else None

    sol_hybrid = None
    if sol_wrapper is not None:
        sol_hybrid = HybridPlayer(
            game, sol_wrapper,
            mcts_config=mcts_cfg,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,
        )

    def_hybrid = None
    if def_wrapper is not None:
        def_hybrid = HybridPlayer(
            game, def_wrapper,
            mcts_config=mcts_cfg,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,
        )

    rng_rand = random.Random(seed + 50000)

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            state = game.apply(state, actions[0])
            continue

        is_soloist = (player == soloist_idx)
        hybrid = sol_hybrid if is_soloist else def_hybrid

        if hybrid is not None:
            action = hybrid.choose_action(state, player, rng)
        else:
            action = rng_rand.choice(actions)

        state = game.apply(state, action)

    won = soloist_won_simple(state.gs)
    return won, trump_name


def play_match(
    game: UltiGame,
    wrapper_a: UltiNetWrapper | None,
    wrapper_b: UltiNetWrapper | None,
    deals: int = 200,
    seed: int = 0,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
) -> dict:
    """Play a match: A as soloist vs B as defenders, then swap.

    Returns detailed results dict.
    """

    mcts_cfg = MCTSConfig(
        simulations=20, determinizations=1,
        c_puct=1.5, dirichlet_alpha=0.0, dirichlet_weight=0.0,
        use_value_head=True, use_policy_priors=True, visit_temp=0.1,
    )

    # A as soloist, B defends
    a_sol_wins = 0
    a_sol_total = deals // 2
    for g in range(a_sol_total):
        won, _ = play_eval_game(
            game, wrapper_a, wrapper_b, mcts_cfg,
            seed=seed + g,
            endgame_tricks=endgame_tricks,
            pimc_dets=pimc_dets,
        )
        if won:
            a_sol_wins += 1

    # B as soloist, A defends
    b_sol_wins = 0
    b_sol_total = deals - a_sol_total
    for g in range(b_sol_total):
        won, _ = play_eval_game(
            game, wrapper_b, wrapper_a, mcts_cfg,
            seed=seed + a_sol_total + g,
            endgame_tricks=endgame_tricks,
            pimc_dets=pimc_dets,
        )
        if won:
            b_sol_wins += 1

    # A wins when: A is soloist and wins, OR B is soloist and loses
    a_wins = a_sol_wins + (b_sol_total - b_sol_wins)
    b_wins = deals - a_wins

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "deals": deals,
        "a_wr": a_wins / deals,
        "a_sol_wr": a_sol_wins / max(1, a_sol_total),
        "b_sol_wr": b_sol_wins / max(1, b_sol_total),
    }


# ---------------------------------------------------------------------------
#  Round-robin evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    tier_names: list[str],
    eval_deals: int = 200,
    device: str = "cpu",
) -> None:
    """Round-robin evaluation of all trained tiers + random baseline."""
    game = UltiGame()

    # Build agent list: (name, wrapper_or_none)
    agents: list[tuple[str, UltiNetWrapper | None]] = []

    # Random baseline
    agents.append(("Random", None))

    # Load each tier's model
    for tier_key in tier_names:
        tier = TIERS[tier_key]
        model_dir = Path("models") / "ulti" / tier.name
        if not (model_dir / "model.pt").exists():
            print(f"  ⚠ Skipping {tier.name} — no model found at {model_dir}")
            continue
        _, wrapper = load_model(model_dir, device)

        # Read training info for the label
        info_path = model_dir / "train_info.json"
        label = tier.name
        if info_path.exists():
            info = json.loads(info_path.read_text())
            label = f"{tier.name} ({info.get('total_games', '?')}g)"

        agents.append((label, wrapper))

    n = len(agents)
    if n < 2:
        print("  Need at least 2 agents for evaluation.")
        return

    print(f"  Round-robin: {n} agents, {eval_deals} deals per matchup")
    print(f"  Solver: {SOLVER_ENGINE}")
    print()

    total_wins = [0] * n
    total_deals = [0] * n
    margin_table: list[list[str]] = [["-"] * n for _ in range(n)]
    matchup_idx = 0
    total_matchups = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            matchup_idx += 1
            name_a = agents[i][0]
            name_b = agents[j][0]
            print(
                f"  [{matchup_idx}/{total_matchups}] {name_a} vs {name_b}...",
                end="", flush=True,
            )
            t0 = time.perf_counter()
            result = play_match(
                game,
                agents[i][1], agents[j][1],
                deals=eval_deals,
                seed=i * 10000 + j * 100,
            )
            elapsed = time.perf_counter() - t0

            a_wr = result["a_wr"]
            b_wr = 1 - a_wr
            total_wins[i] += result["a_wins"]
            total_wins[j] += result["b_wins"]
            total_deals[i] += result["deals"]
            total_deals[j] += result["deals"]

            margin = a_wr - 0.5  # positive = A stronger
            margin_table[i][j] = f"{a_wr:.0%}"
            margin_table[j][i] = f"{b_wr:.0%}"

            winner = name_a if a_wr > 0.5 else name_b if a_wr < 0.5 else "DRAW"
            print(
                f"  A={a_wr:.0%} "
                f"(sol: A={result['a_sol_wr']:.0%} B={result['b_sol_wr']:.0%})  "
                f"→ {winner}  [{elapsed:.1f}s]"
            )

    # ── Ranking ───────────────────────────────────────────────────────
    print()
    print("  ┌─ RANKING " + "─" * 55)
    print("  │")

    ranking = sorted(range(n), key=lambda k: total_wins[k] / max(1, total_deals[k]), reverse=True)
    max_name = max(len(agents[k][0]) for k in ranking)

    print(f"  │  {'Rank':<5} {'Agent':<{max_name+2}} {'Wins':>6} {'/ Deals':>8}  {'Win rate':>8}")
    print(f"  │  {'─'*5} {'─'*(max_name+2)} {'─'*6} {'─'*8}  {'─'*8}")

    for rank, k in enumerate(ranking, 1):
        name = agents[k][0]
        wins = total_wins[k]
        deals = total_deals[k]
        wr = wins / max(1, deals)
        medal = ">> " if rank == 1 else "   "
        print(f"  │  {medal}{rank:<3} {name:<{max_name+2}} {wins:>5} / {deals:<5}  {wr:>8.1%}")

    print("  │")

    # Margin table
    print("  │  Matchup table (cell = row's win rate vs column):")
    header = "  │  " + " " * (max_name + 2)
    for k in ranking:
        short = agents[k][0][:12]
        header += f" {short:>12}"
    print(header)
    for k in ranking:
        row = f"  │  {agents[k][0]:<{max_name+2}}"
        for j in ranking:
            row += f" {margin_table[k][j]:>12}"
        print(row)

    print("  │")
    print("  └" + "─" * 65)
    print()

    # ── What to look for ──────────────────────────────────────────────
    print("  WHAT TO LOOK FOR:")
    print("  • Each tier should beat the one below it (Bishop > Knight > Scout > Random)")
    print("  • Soloist win rates should be well above 50% vs Random (soloist has position advantage)")
    print("  • Value loss (vloss) should decrease during training → the value head is learning")
    print("  • Policy loss (ploss) should decrease during training → the policy head is learning")
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Ulti Parti models at 3 tiers + head-to-head eval",
    )
    parser.add_argument("--tiers", type=str, nargs="+", default=TIER_ORDER,
                        choices=list(TIERS.keys()),
                        help="Which tiers to train (default: all 3)")
    parser.add_argument("--eval-deals", type=int, default=200,
                        help="Deals per evaluation matchup (default 200)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing models")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation (training only)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (default cpu)")
    args = parser.parse_args()

    tier_keys = args.tiers

    # ── Training ──────────────────────────────────────────────────────
    if not args.eval_only:
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║         ULTI PARTI — TIERED TRAINING                            ║")
        print("║         Hybrid self-play: MCTS + Cython alpha-beta solver       ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        total_games = sum(TIERS[k].total_games for k in tier_keys)
        print(f"  Tiers: {', '.join(TIERS[k].name for k in tier_keys)}")
        print(f"  Total budget: {total_games:,} games")
        print(f"  Solver: {SOLVER_ENGINE}")
        print()

        for tier_key in tier_keys:
            tier = TIERS[tier_key]
            print(f"  ┌─ {tier.name}: {tier.description} " + "─" * 20)
            print(f"  │  Budget: {tier.steps} steps × {tier.games_per_step} gpi = {tier.total_games:,} games, "
                  f"{tier.train_steps} SGD/step")

            train_tier(tier, seed=args.seed, device=args.device)

            print(f"  └─ {tier.name} complete")
            print()

    # ── Evaluation ────────────────────────────────────────────────────
    if not args.no_eval:
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║         HEAD-TO-HEAD EVALUATION                                 ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

        run_evaluation(
            tier_keys,
            eval_deals=args.eval_deals,
            device=args.device,
        )


if __name__ == "__main__":
    main()
