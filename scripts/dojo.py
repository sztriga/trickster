#!/usr/bin/env python3
"""Dojo — focused contract training with biased dealing.

Forces specific contract games with biased dealing that gives the soloist
hands resembling real contract hands.  The model learns both what good
hands look like and how to play them.

Usage:
    python scripts/dojo.py scout --contract betli --steps 200 --games-per-step 8
    python scripts/dojo.py knight --contract betli --alpha 0.5 --suit-sigma 1.0
    python scripts/dojo.py scout --contract betli --save-as scout --workers 4
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.bidding.evaluator import _make_eval_state
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import (
    ALL_SUITS,
    BETLI_STRENGTH,
    Card,
    Rank,
    Suit,
    make_deck,
)
from trickster.games.ulti.game import (
    GameState,
    deal,
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
)
from trickster.games.ulti.hybrid import HybridPlayer
from trickster.games.ulti.rewards import simple_outcome
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, make_wrapper
from trickster.train_utils import ReplayBuffer
from trickster.training.model_io import auto_device, load_net
from trickster.training.tiers import TIERS


# ---------------------------------------------------------------------------
#  Biased dealing
# ---------------------------------------------------------------------------

def biased_betli_deal(
    rng: random.Random,
    alpha: float = 0.5,
    suit_sigma: float = 1.0,
) -> tuple[list[list[Card]], list[Card]]:
    """Deal 10 biased cards to soloist (player 0), rest randomly.

    Returns (hands[3], talon[2]) where hands[0] is the biased soloist hand.
    """
    deck = make_deck()

    # Per-suit multipliers for suit concentration
    suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in ALL_SUITS}

    # Compute sampling weights
    weights = []
    for card in deck:
        rank_strength = BETLI_STRENGTH[card.rank]
        w = math.exp(-alpha * rank_strength) * suit_mult[card.suit]
        weights.append(w)

    # Weighted sampling without replacement for soloist (10 cards)
    remaining = list(range(len(deck)))
    remaining_weights = list(weights)
    soloist_indices: list[int] = []

    for _ in range(10):
        total = sum(remaining_weights)
        r = rng.random() * total
        cumulative = 0.0
        chosen = 0
        for i, w in enumerate(remaining_weights):
            cumulative += w
            if cumulative >= r:
                chosen = i
                break
        soloist_indices.append(remaining[chosen])
        remaining.pop(chosen)
        remaining_weights.pop(chosen)

    soloist_hand = [deck[i] for i in soloist_indices]
    rest = [deck[i] for i in remaining]
    rng.shuffle(rest)

    # Remaining 22 cards: 10 to def1, 10 to def2, 2 to talon
    hands: list[list[Card]] = [soloist_hand, rest[:10], rest[10:20]]
    talon = rest[20:22]

    return hands, talon


def hand_quality(hand: list[Card]) -> float:
    """Betli hand quality: 0.0 (all Aces) to 1.0 (all 7s).

    quality = sum(7 - BETLI_STRENGTH[rank]) / 70.0
    """
    return sum(7 - BETLI_STRENGTH[c.rank] for c in hand) / 70.0


# ---------------------------------------------------------------------------
#  Smart discard via evaluator
# ---------------------------------------------------------------------------

def best_betli_discard(
    gs: GameState,
    soloist: int,
    wrapper,
    game: UltiGame,
) -> list[Card]:
    """Find the best 2 cards to discard for betli using the value head.

    Evaluates all C(12,2)=66 discard pairs and picks the one that
    maximises the soloist value prediction.
    """
    hand = gs.hands[soloist]
    assert len(hand) == 12

    cdef = CONTRACT_DEFS["betli"]
    best_val = -float("inf")
    best_pair: tuple[Card, Card] = (hand[0], hand[1])

    # Build evaluation states for all discard pairs
    for c1, c2 in combinations(hand, 2):
        node = _make_eval_state(
            gs, soloist,
            trump=None,
            discards=(c1, c2),
            contract_def=cdef,
            is_piros=False,
            dealer=gs.dealer,
        )
        feats = game.encode_state(node, soloist)
        val = wrapper.predict_value(feats)
        if val > best_val:
            best_val = val
            best_pair = (c1, c2)

    return list(best_pair)


# ---------------------------------------------------------------------------
#  Play one dojo game
# ---------------------------------------------------------------------------

def play_one_dojo_game(
    game: UltiGame,
    wrapper,
    sol_player: HybridPlayer,
    def_player: HybridPlayer,
    rng: random.Random,
    alpha: float,
    suit_sigma: float,
    kontra: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]], float]:
    """Play one biased betli game, return (samples, quality).

    Each sample: (state_feats, action_mask, policy, reward, is_soloist).
    """
    # 1. Biased deal
    hands, talon = biased_betli_deal(rng, alpha, suit_sigma)
    dealer = 2  # arbitrary; soloist is player 0
    soloist = 0

    quality = hand_quality(hands[soloist])

    # Build GameState with 10-card hands
    gs = GameState(
        hands=hands,
        trump=None,
        betli=False,
        soloist=soloist,
        dealer=dealer,
        captured=[[], [], []],
        scores=[0, 0, 0],
        leader=next_player(dealer),
        trick_no=0,
        trick_cards=[],
        last_trick=None,
    )

    # 2. Pickup talon (hand goes 10 → 12) and smart discard
    pickup_talon(gs, soloist, talon)
    discards = best_betli_discard(gs, soloist, wrapper, game)
    discard_talon(gs, discards)

    # 3. Set contract
    set_contract(gs, soloist, trump=None, betli=True)
    gs.training_mode = "betli"
    declare_all_marriages(gs, soloist_marriage_restrict=None)

    # Build UltiNode
    node = UltiNode(
        gs=gs,
        known_voids=(frozenset(), frozenset(), frozenset()),
        bid_rank=0,
        is_red=False,
        contract_components=frozenset({"betli"}),
        dealer=dealer,
    )

    # Kontra decision (simple: defenders kontra if their value > 0.4)
    if kontra:
        for def_p in range(3):
            if def_p == soloist:
                continue
            feats = game.encode_state(node, def_p)
            def_val = wrapper.predict_value(feats)
            if def_val > 0.4:
                node.component_kontras["betli"] = 1
                node.gs.kontra_level = 1
                break

    # 4. Play the game
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []

    state = node
    while not game.is_terminal(state):
        player = game.current_player(state)
        is_sol = (player == soloist)
        hp = sol_player if is_sol else def_player

        # Get policy + action
        pi, action, _sv = hp.choose_action_with_policy(state, player, rng)

        # Encode state
        feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)

        samples.append((feats, mask, pi, 0.0, is_sol))  # reward filled later

        state = game.apply(state, action)

    # 5. Label with terminal outcome
    labeled: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
    for feats, mask, pi, _, is_sol in samples:
        player_for_reward = soloist if is_sol else (1 if soloist != 1 else 2)
        reward = simple_outcome(state, player_for_reward)
        labeled.append((feats, mask, pi, reward, is_sol))

    return labeled, quality


# ---------------------------------------------------------------------------
#  Worker process support
# ---------------------------------------------------------------------------

# Per-worker globals (initialised once per process)
_W_GAME: UltiGame | None = None
_W_NET: UltiNet | None = None
_W_WRAPPER = None


def _init_worker(net_kwargs: dict) -> None:
    """Called once per worker process to create game + network."""
    global _W_GAME, _W_NET, _W_WRAPPER
    _W_GAME = UltiGame()
    _W_NET = UltiNet(**net_kwargs)
    _W_NET.eval()
    _W_WRAPPER = make_wrapper(_W_NET, device="cpu")


def _play_game_in_worker(
    args: tuple,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]], float]:
    """Worker entry-point for parallel dojo self-play."""
    (weights, sol_cfg, def_cfg, game_seed, alpha, suit_sigma,
     kontra, endgame_tricks, pimc_dets, solver_temp) = args

    global _W_NET, _W_WRAPPER, _W_GAME

    # Update weights from main process
    _W_NET.load_state_dict(weights, strict=False)
    _W_NET.eval()
    _W_WRAPPER = make_wrapper(_W_NET, device="cpu")

    rng = random.Random(game_seed)

    sol_player = HybridPlayer(
        _W_GAME, _W_WRAPPER, mcts_config=sol_cfg,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )
    def_player = HybridPlayer(
        _W_GAME, _W_WRAPPER, mcts_config=def_cfg,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )

    return play_one_dojo_game(
        _W_GAME, _W_WRAPPER, sol_player, def_player, rng,
        alpha, suit_sigma, kontra,
    )


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

@dataclass
class DojoConfig:
    # Source / target
    source: str = "scout"
    save_as: str | None = None
    contract: str = "betli"

    # Training budget
    steps: int = 200
    games_per_step: int = 8
    train_steps: int = 50
    batch_size: int = 64
    buffer_size: int = 30_000

    # Learning rate
    lr_start: float = 5e-4
    lr_end: float = 1e-4

    # Dealing bias
    alpha: float = 0.5
    suit_sigma: float = 1.0

    # MCTS
    sol_sims: int = 40
    sol_dets: int = 2
    def_sims: int = 20
    def_dets: int = 2
    endgame_tricks: int = 6
    pimc_dets: int = 20
    solver_temp: float = 0.5

    # Kontra
    kontra: bool = True

    # Workers
    num_workers: int = 1

    # Device
    device: str = "cpu"
    seed: int = 42


def train_dojo(cfg: DojoConfig) -> None:
    """Run focused dojo training."""

    # ── Load model ────────────────────────────────────────────────
    model_dir = Path(f"models/ulti/{cfg.source}/final/{cfg.contract}")
    model_pt = model_dir / "model.pt"
    if not model_pt.exists():
        print(f"Error: no model found at {model_dir}")
        sys.exit(1)

    # Load with strict=False to handle legacy models missing bid_value_fc
    # or carrying old single policy_head/value_fc keys.
    cp = torch.load(model_pt, weights_only=False, map_location=cfg.device)
    game = UltiGame()
    net = UltiNet(
        input_dim=cp.get("input_dim", game.state_dim),
        body_units=cp.get("body_units", 256),
        body_layers=cp.get("body_layers", 4),
        action_dim=cp.get("action_dim", game.action_space_size),
    )
    net.load_state_dict(cp["model_state_dict"], strict=False)

    net.to(cfg.device)
    wrapper = make_wrapper(net, device=cfg.device)

    # ── Setup ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr_start)
    buf = ReplayBuffer(capacity=cfg.buffer_size, seed=cfg.seed)
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    sol_mcts = MCTSConfig(
        simulations=cfg.sol_sims,
        determinizations=cfg.sol_dets,
        use_value_head=True,
        use_policy_priors=True,
    )
    def_mcts = MCTSConfig(
        simulations=cfg.def_sims,
        determinizations=cfg.def_dets,
        use_value_head=True,
        use_policy_priors=True,
    )

    # Quality tier tracking
    TIERS_N = 5
    tier_wins = [0] * TIERS_N
    tier_total = [0] * TIERS_N
    tier_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    save_as = cfg.save_as or cfg.source

    # ── Header ────────────────────────────────────────────────────
    print(f"  Dojo: {cfg.contract} training")
    print(f"  Source: {cfg.source} → Save: {save_as}")
    print(f"  Steps: {cfg.steps} × {cfg.games_per_step} games")
    print(f"  Alpha: {cfg.alpha}, Suit σ: {cfg.suit_sigma}")
    print(f"  LR: {cfg.lr_start:.1e} → {cfg.lr_end:.1e}")
    if cfg.num_workers > 1:
        print(f"  Self-play: {cfg.num_workers} workers (process pool)")
    else:
        print(f"  Self-play: sequential")
    print(f"  Device: {cfg.device}")
    print()

    total_games = 0
    total_wins = 0
    total_kontras = 0
    t0 = time.perf_counter()

    # ── Worker pool ───────────────────────────────────────────────
    executor = None
    body_layers = len([m for m in net.backbone if isinstance(m, torch.nn.Linear)])
    net_kwargs = {
        "input_dim": net.input_dim,
        "body_units": net.body_units,
        "body_layers": body_layers,
        "action_dim": net.action_dim,
    }

    if cfg.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            initializer=_init_worker,
            initargs=(net_kwargs,),
        )

    try:
        for step in range(1, cfg.steps + 1):
            # ── Learning rate schedule ───────────────────────────────
            frac = (step - 1) / max(cfg.steps - 1, 1)
            lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Generate games ───────────────────────────────────────
            net.eval()

            step_wins = 0
            step_qualities: list[float] = []
            step_kontras = 0
            step_sol_vals: list[float] = []
            step_samples = 0

            def _collect_result(
                samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]],
                quality: float,
            ) -> None:
                nonlocal step_wins, step_kontras, step_samples

                step_qualities.append(quality)

                # Determine win (soloist reward > 0)
                sol_reward = next(r for _, _, _, r, is_sol in samples if is_sol)
                won = sol_reward > 0
                if won:
                    step_wins += 1

                # Quality tier tracking
                tier_idx = min(int(quality * TIERS_N), TIERS_N - 1)
                tier_total[tier_idx] += 1
                if won:
                    tier_wins[tier_idx] += 1

                # Check kontra (kontra doubles stakes → reward magnitude > 1.1)
                if abs(sol_reward) > 1.1:
                    step_kontras += 1

                # Push to buffer
                for feats, mask, pi, reward, is_sol in samples:
                    buf.push(feats, mask, pi, reward, is_sol)
                    step_samples += 1

            if executor is not None:
                # --- Parallel self-play ---
                all_weights = {k: v.cpu() for k, v in net.state_dict().items()}
                tasks = []
                for g in range(cfg.games_per_step):
                    game_seed = cfg.seed + step * 1000 + g
                    tasks.append((
                        all_weights, sol_mcts, def_mcts, game_seed,
                        cfg.alpha, cfg.suit_sigma, cfg.kontra,
                        cfg.endgame_tricks, cfg.pimc_dets, cfg.solver_temp,
                    ))
                for samples, quality in executor.map(_play_game_in_worker, tasks):
                    _collect_result(samples, quality)
            else:
                # --- Sequential self-play ---
                wrapper = make_wrapper(net, device=cfg.device)
                sol_player = HybridPlayer(
                    game, wrapper, mcts_config=sol_mcts,
                    endgame_tricks=cfg.endgame_tricks,
                    pimc_determinizations=cfg.pimc_dets,
                    solver_temperature=cfg.solver_temp,
                )
                def_player = HybridPlayer(
                    game, wrapper, mcts_config=def_mcts,
                    endgame_tricks=cfg.endgame_tricks,
                    pimc_determinizations=cfg.pimc_dets,
                    solver_temperature=cfg.solver_temp,
                )
                for g in range(cfg.games_per_step):
                    samples, quality = play_one_dojo_game(
                        game, wrapper, sol_player, def_player, rng,
                        cfg.alpha, cfg.suit_sigma, cfg.kontra,
                    )
                    _collect_result(samples, quality)

            total_games += cfg.games_per_step
            total_wins += step_wins
            total_kontras += step_kontras

            # ── SGD ──────────────────────────────────────────────────
            v_loss_avg = 0.0
            p_loss_avg = 0.0
            sgd_count = 0

            if len(buf) >= cfg.batch_size:
                net.train()
                for _ in range(cfg.train_steps):
                    states, masks, policies, rewards, is_sol, _on_pol = buf.sample(
                        cfg.batch_size, np_rng,
                    )
                    s_t = torch.from_numpy(states).float().to(cfg.device)
                    m_t = torch.from_numpy(masks).bool().to(cfg.device)
                    pi_t = torch.from_numpy(policies).float().to(cfg.device)
                    z_t = torch.from_numpy(rewards).float().to(cfg.device)
                    is_sol_t = torch.from_numpy(is_sol).bool().to(cfg.device)

                    log_probs, values = net.forward_dual(s_t, m_t, is_sol_t)
                    value_loss = F.huber_loss(values, z_t, delta=1.0)
                    policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
                    loss = value_loss + policy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

                    v_loss_avg += value_loss.item()
                    p_loss_avg += policy_loss.item()
                    sgd_count += 1

            if sgd_count > 0:
                v_loss_avg /= sgd_count
                p_loss_avg /= sgd_count

            # ── Logging ──────────────────────────────────────────────
            win_rate = step_wins / cfg.games_per_step
            mean_q = np.mean(step_qualities) if step_qualities else 0.0
            elapsed = time.perf_counter() - t0
            gps = total_games / elapsed if elapsed > 0 else 0
            overall_wr = total_wins / total_games if total_games > 0 else 0

            if step % 10 == 0 or step == cfg.steps:
                print(
                    f"  [{step:4d}/{cfg.steps}]  "
                    f"win={overall_wr:.0%}  "
                    f"q={mean_q:.2f}  "
                    f"v_loss={v_loss_avg:.4f}  "
                    f"p_loss={p_loss_avg:.4f}  "
                    f"buf={len(buf):,}  "
                    f"lr={lr:.1e}  "
                    f"{gps:.1f} g/s"
                )
            else:
                print(
                    f"\r  [{step:4d}/{cfg.steps}] {gps:.1f} g/s",
                    end="", flush=True,
                )

            # Periodic quality tier breakdown
            if step % 25 == 0 or step == cfg.steps:
                print("  ── Quality tiers ──")
                for i in range(TIERS_N):
                    if tier_total[i] > 0:
                        tw = tier_wins[i] / tier_total[i]
                        print(f"    {tier_labels[i]}: {tw:.0%} ({tier_wins[i]}/{tier_total[i]})")
                    else:
                        print(f"    {tier_labels[i]}: n/a")
                print()

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    # ── Save ──────────────────────────────────────────────────────
    out_dir = Path(f"models/ulti/{save_as}/final/{cfg.contract}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": net.body_units,
        "body_layers": body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "training_mode": "betli",
        "method": "dojo",
        "dojo_contract": cfg.contract,
        "dojo_alpha": cfg.alpha,
        "dojo_suit_sigma": cfg.suit_sigma,
        "dojo_steps": cfg.steps,
        "dojo_total_games": total_games,
    }, out_dir / "model.pt")

    elapsed = time.perf_counter() - t0
    overall_wr = total_wins / total_games if total_games > 0 else 0

    print()
    print(f"  ┌─ DOJO COMPLETE ─────────────────────────────────")
    print(f"  │  Contract:   {cfg.contract}")
    print(f"  │  Games:      {total_games:,}")
    print(f"  │  Win rate:   {overall_wr:.1%}")
    print(f"  │  Kontras:    {total_kontras}")
    print(f"  │  Time:       {elapsed:.0f}s ({total_games / elapsed:.1f} g/s)")
    print(f"  │  Saved:      {out_dir}/model.pt")
    print(f"  └─────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dojo — focused contract training with biased dealing",
    )
    parser.add_argument("model", help="Source model (e.g. scout, knight, bronze)")
    parser.add_argument("--contract", default="betli", help="Contract to train (default: betli)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--games-per-step", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--alpha", type=float, default=0.5, help="Betli rank bias strength")
    parser.add_argument("--suit-sigma", type=float, default=1.0, help="Suit concentration variance")
    parser.add_argument("--lr-start", type=float, default=5e-4)
    parser.add_argument("--lr-end", type=float, default=1e-4)
    parser.add_argument("--sol-sims", type=int, default=40)
    parser.add_argument("--sol-dets", type=int, default=2)
    parser.add_argument("--def-sims", type=int, default=20)
    parser.add_argument("--def-dets", type=int, default=2)
    parser.add_argument("--endgame-tricks", type=int, default=6)
    parser.add_argument("--pimc-dets", type=int, default=20)
    parser.add_argument("--solver-temp", type=float, default=0.5)
    parser.add_argument("--kontra", action="store_true", default=True)
    parser.add_argument("--no-kontra", dest="kontra", action="store_false")
    parser.add_argument("--save-as", default=None, help="Target model name (default: same as source)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Detect device
    if args.device:
        device = args.device
    else:
        tier = TIERS.get(args.model)
        body_units = tier.body_units if tier else 256
        body_layers = tier.body_layers if tier else 4
        device = auto_device(body_units, body_layers)

    cfg = DojoConfig(
        source=args.model,
        save_as=args.save_as,
        contract=args.contract,
        steps=args.steps,
        games_per_step=args.games_per_step,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        alpha=args.alpha,
        suit_sigma=args.suit_sigma,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
        sol_sims=args.sol_sims,
        sol_dets=args.sol_dets,
        def_sims=args.def_sims,
        def_dets=args.def_dets,
        endgame_tricks=args.endgame_tricks,
        pimc_dets=args.pimc_dets,
        solver_temp=args.solver_temp,
        kontra=args.kontra,
        num_workers=args.workers,
        device=device,
        seed=args.seed,
    )

    train_dojo(cfg)


if __name__ == "__main__":
    main()
