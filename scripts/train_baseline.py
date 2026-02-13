#!/usr/bin/env python3
"""Ulti hybrid training — MCTS + alpha-beta solver, Parti curriculum.

Self-play uses the HybridPlayer: neural MCTS for the opening tricks,
then PIMC + exact alpha-beta for the endgame.  This gives the network
clean, exact training labels for endgame positions while MCTS handles
the imperfect-information opening where exact solving is infeasible.

Trump is chosen randomly from the soloist's hand (all 4 suits equally
likely, including Hearts / red).

Usage:
    # Quick test (200 steps ~ 2 min)
    python scripts/train_baseline.py --steps 200

    # Longer run with checkpoints
    python scripts/train_baseline.py --steps 2000 --checkpoint-interval 200

    # Continue from checkpoint
    python scripts/train_baseline.py --steps 2000 --load models/checkpoints/simple/step_01000.pt

    # Parallel self-play (4 workers)
    python scripts/train_baseline.py --workers 4 --games-per-step 16

    # Custom solver settings (more exact tricks, more PIMC samples)
    python scripts/train_baseline.py --endgame-tricks 7 --pimc-dets 30
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

from trickster.games.ulti.game import soloist_won_simple
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
from trickster.mcts import MCTSConfig, alpha_mcts_policy
from trickster.model import UltiNet, UltiNetWrapper, NUM_CONTRACTS
from trickster.train_utils import ReplayBuffer, simple_outcome


# ---------------------------------------------------------------------------
#  Multiprocessing helpers (must be at module level for pickling)
# ---------------------------------------------------------------------------

_WORKER_GAME: UltiGame | None = None
_WORKER_NET: UltiNet | None = None
_WORKER_WRAPPER: UltiNetWrapper | None = None
_WORKER_OPP_NET: UltiNet | None = None
_WORKER_OPP_WRAPPER: UltiNetWrapper | None = None


def _init_worker(net_kwargs: dict, device: str) -> None:
    """Called once per worker process to create game + networks."""
    global _WORKER_GAME, _WORKER_NET, _WORKER_WRAPPER
    global _WORKER_OPP_NET, _WORKER_OPP_WRAPPER
    _WORKER_GAME = UltiGame()
    _WORKER_NET = UltiNet(**net_kwargs)
    _WORKER_WRAPPER = UltiNetWrapper(_WORKER_NET, device=device)
    _WORKER_OPP_NET = UltiNet(**net_kwargs)
    _WORKER_OPP_WRAPPER = UltiNetWrapper(_WORKER_OPP_NET, device=device)


def _play_game_in_worker(
    args: tuple,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Worker entry-point for parallel self-play."""
    (state_dict, opp_state_dict, sol_mcts_cfg, def_mcts_cfg,
     seed, endgame_tricks, pimc_dets, solver_temp, enrich_thresh) = args
    _WORKER_NET.load_state_dict(state_dict)
    opp_wrapper = None
    if opp_state_dict is not None:
        _WORKER_OPP_NET.load_state_dict(opp_state_dict)
        opp_wrapper = _WORKER_OPP_WRAPPER
    init_state = None
    if enrich_thresh > -999.0:
        init_state, _ = value_enriched_new_game(
            _WORKER_GAME, _WORKER_WRAPPER, seed, min_value=enrich_thresh,
        )
    return play_one_game(
        _WORKER_GAME, _WORKER_WRAPPER, sol_mcts_cfg, def_mcts_cfg, seed,
        endgame_tricks=endgame_tricks,
        pimc_dets=pimc_dets,
        solver_temp=solver_temp,
        opponent_wrapper=opp_wrapper,
        initial_state=init_state,
    )


# ---------------------------------------------------------------------------
#  Deal enrichment via value head
# ---------------------------------------------------------------------------


def value_enriched_new_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    seed: int,
    min_value: float,
    max_attempts: int = 20,
) -> tuple[UltiNode, float]:
    """Deal games until the value head rates the soloist above *min_value*.

    Returns (state, soloist_value) — the best deal found, or the last
    attempt if none pass the threshold.  Each attempt uses a different
    seed offset so deals are diverse.
    """
    best_state = None
    best_val = -1.0
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt * 100_000
        state = game.new_game(
            seed=attempt_seed,
            training_mode="simple",
            starting_leader=seed % 3,
        )
        sol = state.gs.soloist
        feats = game.encode_state(state, sol)
        val = wrapper.predict_value(feats)
        if val > best_val:
            best_state, best_val = state, val
        if val >= min_value:
            return state, val
    return best_state, best_val  # best we found


def enrichment_threshold(step: int, total_steps: int) -> float:
    """Anneal the minimum soloist value from 0.0 → −1.0 over training.

    Phase 1 (first 25%):   v >= 0.0   (only "even-ish or better" deals)
    Phase 2 (25%-60%):     v >= -0.25  (slightly losing included)
    Phase 3 (60%+):        disabled    (all deals accepted)

    Returns the threshold, or -999 to signal "accept everything".
    """
    frac = step / max(1, total_steps)
    if frac < 0.25:
        return 0.0
    if frac < 0.60:
        return -0.25
    return -999.0  # effectively disabled


# ---------------------------------------------------------------------------
#  Self-play: one full game → training samples
# ---------------------------------------------------------------------------


def play_one_game(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    sol_mcts_config: MCTSConfig,
    def_mcts_config: MCTSConfig,
    seed: int,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
    solver_temp: float = 1.0,
    opponent_wrapper: UltiNetWrapper | None = None,
    initial_state: UltiNode | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
    """Play one hybrid self-play game (Parti, any trump suit).

    Uses HybridPlayer for all seats: neural MCTS for opening tricks,
    PIMC + exact alpha-beta for the endgame.  The soloist gets a
    higher MCTS budget than the defenders.

    If *initial_state* is provided (e.g. from deal enrichment),
    it is used directly instead of dealing a new game.
    """
    rng = random.Random(seed)

    if initial_state is not None:
        state = initial_state
    else:
        state = game.new_game(
            seed=seed,
            training_mode="simple",
            starting_leader=seed % 3,
        )
    soloist_idx = state.gs.soloist

    # Build hybrid players for soloist and defenders
    sol_hybrid = HybridPlayer(
        game, wrapper,
        mcts_config=sol_mcts_config,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )
    def_wrapper = opponent_wrapper if opponent_wrapper is not None else wrapper
    def_hybrid = HybridPlayer(
        game, def_wrapper,
        mcts_config=def_mcts_config,
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

        # Choose action using hybrid player (MCTS early, solver late)
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

    # Compute rewards
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for state_feats, mask, pi, player in trajectory:
        reward = simple_outcome(state, player)
        is_sol = (player == soloist_idx)
        samples.append((state_feats, mask, pi, reward, is_sol))

    return samples


# ---------------------------------------------------------------------------
#  Evaluation: hybrid agent vs random opponents
# ---------------------------------------------------------------------------


def eval_vs_random(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    eval_mcts_config: MCTSConfig,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
    num_games: int = 20,
    seed: int = 99999,
) -> dict[str, float]:
    """Play hybrid agent vs random opponents, reporting per-role WR.

    Player 0 is always the hybrid agent.  The soloist is determined
    by the deal (random), so roughly 1/3 of games have the agent as
    soloist and 2/3 as defender.
    """
    sol_wins = 0
    sol_total = 0
    def_wins = 0
    def_total = 0

    hybrid = HybridPlayer(
        game, wrapper,
        mcts_config=eval_mcts_config,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=0.1,  # near-greedy for evaluation
    )

    for g in range(num_games):
        rng = random.Random(seed + g)
        rng_rand = random.Random(seed + g + 50000)
        dealer = g % 3

        state = game.new_game(
            seed=seed + g,
            training_mode="simple",
            starting_leader=dealer,
        )
        soloist_idx = state.gs.soloist
        agent_is_soloist = (soloist_idx == 0)

        while not game.is_terminal(state):
            player = game.current_player(state)
            actions = game.legal_actions(state)

            if len(actions) <= 1:
                state = game.apply(state, actions[0])
                continue

            if player == 0:
                action = hybrid.choose_action(state, player, rng)
            else:
                action = rng_rand.choice(actions)

            state = game.apply(state, action)

        agent_won = game.outcome(state, 0) > 0
        if agent_is_soloist:
            sol_total += 1
            if agent_won:
                sol_wins += 1
        else:
            def_total += 1
            if agent_won:
                def_wins += 1

    total_wins = sol_wins + def_wins
    wr = total_wins / max(1, num_games)
    sol_wr = sol_wins / max(1, sol_total)
    def_wr = def_wins / max(1, def_total)
    return {
        "all": wr,
        "wins": total_wins,
        "games": num_games,
        "sol_wr": sol_wr,
        "sol_wins": sol_wins,
        "sol_games": sol_total,
        "def_wr": def_wr,
        "def_wins": def_wins,
        "def_games": def_total,
    }


# ---------------------------------------------------------------------------
#  Head-to-head: checkpoint A vs checkpoint B (both hybrid)
# ---------------------------------------------------------------------------


def eval_head_to_head(
    game: UltiGame,
    wrapper_a: UltiNetWrapper,
    wrapper_b: UltiNetWrapper,
    mcts_config: MCTSConfig,
    def_mcts_config: MCTSConfig,
    endgame_tricks: int = 6,
    pimc_dets: int = 20,
    num_games: int = 100,
    seed: int = 55555,
) -> dict[str, float]:
    """Play wrapper_a vs wrapper_b with swapped roles.

    Both players use the hybrid engine.  Returns win rates for A.
    """
    wins_a = 0
    total = 0

    for g in range(num_games):
        rng = random.Random(seed + g)
        dealer = g % 3

        state = game.new_game(
            seed=seed + g,
            training_mode="simple",
            starting_leader=dealer,
        )
        soloist_idx = state.gs.soloist

        # Alternate roles
        a_is_soloist = (g % 2 == 0)
        sol_w = wrapper_a if a_is_soloist else wrapper_b
        def_w = wrapper_b if a_is_soloist else wrapper_a

        sol_hybrid = HybridPlayer(
            game, sol_w,
            mcts_config=mcts_config,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,
        )
        def_hybrid = HybridPlayer(
            game, def_w,
            mcts_config=def_mcts_config,
            endgame_tricks=endgame_tricks,
            pimc_determinizations=pimc_dets,
            solver_temperature=0.1,
        )

        while not game.is_terminal(state):
            player = game.current_player(state)
            actions = game.legal_actions(state)

            if len(actions) <= 1:
                state = game.apply(state, actions[0])
                continue

            if player == soloist_idx:
                action = sol_hybrid.choose_action(state, player, rng)
            else:
                action = def_hybrid.choose_action(state, player, rng)

            state = game.apply(state, action)

        soloist_won = soloist_won_simple(state.gs)
        a_won = (soloist_won and a_is_soloist) or (not soloist_won and not a_is_soloist)
        total += 1
        if a_won:
            wins_a += 1

    return {
        "a_wr": wins_a / max(1, total),
        "games": total,
    }


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ulti Hybrid Training — MCTS + Alpha-Beta Solver (Parti)",
    )
    # Training
    parser.add_argument("--steps", type=int, default=200,
                        help="Training iterations (default 200)")
    parser.add_argument("--games-per-step", type=int, default=4,
                        help="Self-play games per training step (default 4)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default 1e-3)")
    parser.add_argument("--buffer-size", type=int, default=50000,
                        help="Replay buffer capacity (default 50000)")

    # MCTS parameters
    parser.add_argument("--sims", type=int, default=20,
                        help="MCTS simulations for soloist (default 20)")
    parser.add_argument("--def-sims", type=int, default=8,
                        help="MCTS simulations for defenders (default 8)")
    parser.add_argument("--dets", type=int, default=1,
                        help="MCTS determinizations (default 1)")

    # Solver / hybrid parameters
    parser.add_argument("--endgame-tricks", type=int, default=6,
                        help="Switch to exact solver when this many tricks remain (default 6)")
    parser.add_argument("--pimc-dets", type=int, default=20,
                        help="PIMC determinizations for the solver (default 20)")
    parser.add_argument("--solver-temp", type=float, default=0.5,
                        help="Solver policy temperature for training (default 0.5)")

    # Network
    parser.add_argument("--body-units", type=int, default=256,
                        help="Backbone hidden units (default 256)")
    parser.add_argument("--body-layers", type=int, default=4,
                        help="Backbone layers (default 4)")

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N steps (default 10)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per evaluation (default 20)")

    # General
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (default cpu)")
    parser.add_argument("--save", type=str, default="models/ulti/model.pt",
                        help="Path to save the model (default models/ulti/model.pt)")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to load a pre-trained model")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel self-play processes (default 1)")

    # Deal enrichment (curriculum via value head)
    parser.add_argument("--no-enrichment", action="store_true",
                        help="Disable deal enrichment (use fully random deals)")
    parser.add_argument("--enrich-random-frac", type=float, default=0.3,
                        help="Fraction of games that are always random (default 0.3)")
    parser.add_argument("--enrich-warmup", type=int, default=20,
                        help="Skip enrichment for first N steps (value head untrained, default 20)")

    args = parser.parse_args()

    # ---- Setup ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

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
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded pre-trained model from {args.load}")

    wrapper = UltiNetWrapper(net, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    buffer = ReplayBuffer(capacity=args.buffer_size, seed=args.seed + 1)
    np_rng = np.random.default_rng(args.seed)

    # ---- MCTS configs ----
    sol_mcts_config = MCTSConfig(
        simulations=args.sims,
        determinizations=args.dets,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=1.0,
    )

    def_mcts_config = MCTSConfig(
        simulations=args.def_sims,
        determinizations=args.dets,
        c_puct=1.5,
        dirichlet_alpha=0.1,
        dirichlet_weight=0.15,
        use_value_head=True,
        use_policy_priors=True,
        visit_temp=0.5,
    )

    eval_mcts_config = MCTSConfig(
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

    enrichment_on = not args.no_enrichment

    # ---- Banner ----
    print("=" * 64)
    print("  Ulti Hybrid Training — Parti (all trump suits)")
    print("=" * 64)
    print(f"  Solver: {SOLVER_ENGINE} "
          f"(endgame={args.endgame_tricks} tricks, "
          f"PIMC={args.pimc_dets} dets)")
    print(f"  Model: UltiNet {args.body_units}x{args.body_layers} "
          f"({param_count:,} params)")
    print(f"  MCTS: soloist={args.sims} sims, "
          f"defenders={args.def_sims} sims")
    print(f"  Steps: {args.steps} x {args.games_per_step} games/step "
          f"= {args.steps * args.games_per_step} games")
    print(f"  LR: {args.lr}  Batch: {args.batch_size}  "
          f"Buffer: {args.buffer_size}")
    print(f"  Device: {device}")
    if args.workers > 1:
        print(f"  Workers: {args.workers} (parallel self-play)")
    print(f"  Talon: random 2 cards → defender points (no discard step)")
    if enrichment_on:
        print(f"  Deal enrichment: ON (value head, "
              f"warmup={args.enrich_warmup} steps, "
              f"{args.enrich_random_frac:.0%} always random)")
    else:
        print(f"  Deal enrichment: OFF (fully random deals)")
    print(f"  Solver temperature: {args.solver_temp}")
    print(f"  Eval: every {args.eval_interval} steps, "
          f"{args.eval_games} games vs Random")
    print()

    # ---- Parallel self-play pool ----
    executor = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        net_kwargs = {
            "input_dim": game.state_dim,
            "body_units": args.body_units,
            "body_layers": args.body_layers,
            "action_dim": game.action_space_size,
            "num_contracts": NUM_CONTRACTS,
        }
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(net_kwargs, "cpu"),
        )

    t0 = time.perf_counter()
    total_games = 0
    total_samples = 0
    best_wr = 0.0
    solver_decisions = 0
    mcts_decisions = 0

    for step in range(1, args.steps + 1):
        step_t = time.perf_counter()

        # ---- 1. Self-play: collect samples ----
        step_samples = 0
        sp_wins = 0
        sp_total = 0

        def _tally_soloist(game_samples):
            nonlocal sp_wins, sp_total
            sol_r = [r for _, _, _, r, is_sol in game_samples if is_sol]
            if sol_r:
                sp_total += 1
                if sol_r[0] > 0:
                    sp_wins += 1

        if executor is not None:
            # Parallel self-play
            state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
            past_warmup = enrichment_on and step > args.enrich_warmup
            thresh = enrichment_threshold(step, args.steps) if past_warmup else -999.0
            tasks = []
            for g in range(args.games_per_step):
                use_enrich = (
                    past_warmup
                    and thresh > -999.0
                    and (g / args.games_per_step) >= args.enrich_random_frac
                )
                tasks.append((
                    state_dict, None, sol_mcts_config, def_mcts_config,
                    args.seed + step * 1000 + g,
                    args.endgame_tricks, args.pimc_dets, args.solver_temp,
                    thresh if use_enrich else -999.0,
                ))
            for samples in executor.map(_play_game_in_worker, tasks):
                _tally_soloist(samples)
                for s, m, p, r, is_sol in samples:
                    buffer.push(s, m, p, r, is_soloist=is_sol)
                step_samples += len(samples)
                total_games += 1
        else:
            # Sequential self-play
            past_warmup = enrichment_on and step > args.enrich_warmup
            thresh = enrichment_threshold(step, args.steps) if past_warmup else -999.0
            for g in range(args.games_per_step):
                game_seed = args.seed + step * 1000 + g

                # Deal enrichment: some games always random, rest enriched
                use_enrichment = (
                    past_warmup
                    and thresh > -999.0
                    and (g / args.games_per_step) >= args.enrich_random_frac
                )
                if use_enrichment:
                    init_state, _ = value_enriched_new_game(
                        game, wrapper, game_seed, min_value=thresh,
                    )
                else:
                    init_state = None

                samples = play_one_game(
                    game, wrapper, sol_mcts_config, def_mcts_config, game_seed,
                    endgame_tricks=args.endgame_tricks,
                    pimc_dets=args.pimc_dets,
                    solver_temp=args.solver_temp,
                    initial_state=init_state,
                )

                _tally_soloist(samples)
                for s, m, p, r, is_sol in samples:
                    buffer.push(s, m, p, r, is_soloist=is_sol)
                step_samples += len(samples)
                total_games += 1

        total_samples += step_samples

        # ---- 2. Train on replay buffer ----
        avg_vloss = 0.0
        avg_ploss = 0.0

        if len(buffer) >= args.batch_size:
            net.train()
            train_steps = max(1, step_samples // args.batch_size)
            total_vloss = 0.0
            total_ploss = 0.0

            for _ in range(train_steps):
                states, masks, policies, rewards, is_sol = buffer.sample(
                    args.batch_size, np_rng,
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

            avg_vloss = total_vloss / train_steps
            avg_ploss = total_ploss / train_steps

        elapsed = time.perf_counter() - t0
        step_time = time.perf_counter() - step_t

        # ---- 3. Progress logging ----
        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            sp_str = f"{sp_wins}/{sp_total}" if sp_total > 0 else "-"
            buf_stats = buffer.stats()
            buf_swr = buf_stats.get("sol_win_rate", 0.0)
            buf_str = f"  buf_swr={buf_swr:.0%}" if buf_stats else ""
            log_past_warmup = enrichment_on and step > args.enrich_warmup
            cur_thresh = enrichment_threshold(step, args.steps) if log_past_warmup else -999.0
            if cur_thresh > -999.0:
                enrich_str = f"  v>={cur_thresh:+.2f}"
            elif enrichment_on and step <= args.enrich_warmup:
                enrich_str = "  warmup"
            else:
                enrich_str = ""
            print(
                f"  step {step:3d}/{args.steps}  "
                f"games={total_games:4d}  "
                f"samples={total_samples:5d}  "
                f"sp={sp_str:>5s}  "
                f"vloss={avg_vloss:.4f}  "
                f"ploss={avg_ploss:.4f}"
                f"{buf_str}{enrich_str}  "
                f"[{step_time:.1f}s / {elapsed:.0f}s]"
            )

        # ---- 4. Evaluate vs Random ----
        if step % args.eval_interval == 0 or step == args.steps:
            eval_t = time.perf_counter()
            wr_dict = eval_vs_random(
                game, wrapper, eval_mcts_config,
                endgame_tricks=args.endgame_tricks,
                pimc_dets=args.pimc_dets,
                num_games=args.eval_games,
                seed=step * 7777,
            )
            eval_time = time.perf_counter() - eval_t
            wr = wr_dict["all"]
            tag = " *BEST*" if wr > best_wr else ""
            if wr > best_wr:
                best_wr = wr

            sol_str = (f"sol={wr_dict['sol_wins']}/{wr_dict['sol_games']}"
                       if wr_dict['sol_games'] > 0 else "sol=-")
            def_str = (f"def={wr_dict['def_wins']}/{wr_dict['def_games']}"
                       if wr_dict['def_games'] > 0 else "def=-")
            print(
                f"  >>> EVAL step {step}: "
                f"WR={wr:.0%} ({wr_dict['wins']}/{wr_dict['games']})  "
                f"{sol_str}  {def_str}  "
                f"best={best_wr:.0%}{tag}  "
                f"[{eval_time:.1f}s]"
            )

    # ---- Shutdown ----
    if executor is not None:
        executor.shutdown(wait=False)

    total_time = time.perf_counter() - t0
    print()
    print("=" * 64)
    print("  Training Complete")
    print("=" * 64)
    print(f"  Games: {total_games}  Samples: {total_samples}")
    print(f"  Time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Best win rate: {best_wr:.0%}")
    print(f"  Solver engine: {SOLVER_ENGINE}")

    # Save final model
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": args.body_units,
        "body_layers": args.body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "num_contracts": NUM_CONTRACTS,
        "training_mode": "simple",
        "method": "hybrid",
        "endgame_tricks": args.endgame_tricks,
        "pimc_dets": args.pimc_dets,
        "total_games": total_games,
        "best_win_rate": best_wr,
    }, save_path)
    print(f"  Model saved to {save_path}")


if __name__ == "__main__":
    main()
