"""Single-contract training loop.

Trains one contract at one tier using hybrid self-play (MCTS + solver).
Used by train_e2e.py for Phase 1, or can be called standalone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trickster.games.ulti.adapter import UltiGame
from trickster.hybrid import SOLVER_ENGINE
from trickster.model import UltiNet, _ort_available
from trickster.train_utils import _GAME_PTS_MAX
from trickster.training.model_io import auto_device
from trickster.training.tiers import ContractSpec, Tier
from trickster.training.ulti_hybrid import (
    UltiTrainConfig,
    UltiTrainStats,
    train_ulti_hybrid,
)


# ---------------------------------------------------------------------------
#  Config builder
# ---------------------------------------------------------------------------

def _build_config(
    tier: Tier,
    spec: ContractSpec,
    *,
    seed: int,
    device: str,
    num_workers: int,
    enrichment: bool,
    leaf_batch_size: int = 8,
) -> UltiTrainConfig:
    # GPU tiers: use cross-game batching, disable process pool
    if tier.gpu and device not in ("cpu",):
        concurrent = tier.games_per_step
        workers = 1
    else:
        concurrent = 1
        workers = num_workers

    return UltiTrainConfig(
        steps=tier.steps,
        games_per_step=tier.games_per_step,
        train_steps=tier.train_steps,
        batch_size=tier.batch_size,
        buffer_size=tier.buffer_size,
        lr_start=tier.lr_start,
        lr_end=tier.lr_end,
        sol_sims=tier.sol_sims,
        sol_dets=tier.sol_dets,
        def_sims=tier.def_sims,
        def_dets=tier.def_dets,
        endgame_tricks=tier.endgame_tricks,
        pimc_dets=tier.pimc_dets,
        solver_temp=tier.solver_temp,
        body_units=tier.body_units,
        body_layers=tier.body_layers,
        leaf_batch_size=leaf_batch_size,
        num_workers=workers,
        concurrent_games=concurrent,
        enrichment=enrichment,
        seed=seed,
        device=device,
        training_mode=spec.training_mode,
    )


# ---------------------------------------------------------------------------
#  Progress callbacks
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def progress_bar(tier: Tier):
    """Single updating line with progress bar and ETA."""
    bar_width = 30

    def on_progress(stats: UltiTrainStats) -> None:
        step = stats.step
        frac = step / stats.total_steps
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = stats.train_time_s
        eta = elapsed / frac * (1 - frac) if frac > 0 else 0

        deals_s = stats.total_games / max(0.1, elapsed)

        print(
            f"\r    {bar} {frac*100:5.1f}%  "
            f"{step}/{stats.total_steps}  "
            f"games={stats.total_games:,}  "
            f"{deals_s:.1f} g/s  "
            f"elapsed {_fmt_time(elapsed)}  "
            f"ETA {_fmt_time(eta)}   ",
            end="", flush=True,
        )

        if step == stats.total_steps:
            print()

    return on_progress, 1


def progress_verbose(tier: Tier):
    """Per-step details every 5%."""
    interval = max(1, tier.steps // 20)

    def on_progress(stats: UltiTrainStats) -> None:
        step = stats.step
        if step % interval != 0 and step != 1 and step != stats.total_steps:
            return

        pct = step / stats.total_steps * 100
        if stats.sp_sol_total > 0:
            avg_pts = (stats.sp_sol_pts_sum / stats.sp_sol_total) * _GAME_PTS_MAX / 2
        else:
            avg_pts = 0.0

        print(
            f"    step {step:>5d}/{stats.total_steps} ({pct:4.0f}%)  "
            f"games={stats.total_games:>5d}  "
            f"vloss={stats.vloss:.4f}  ploss={stats.ploss:.4f}  "
            f"sol_pts={avg_pts:+.2f}  "
            f"lr={stats.lr:.1e}  "
            f"[{stats.train_time_s:.0f}s]",
            flush=True,
        )

    return on_progress, interval


# ---------------------------------------------------------------------------
#  Core: train one contract at one tier
# ---------------------------------------------------------------------------

def train_one_tier(
    tier: Tier,
    spec: ContractSpec,
    *,
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 1,
    enrichment: bool = True,
    verbose: bool = False,
) -> tuple[UltiNet, Path]:
    """Train one tier for one contract.  Returns (net, model_dir)."""
    tier_name = tier.tier_name(spec.name_prefix)
    model_dir = Path("models") / spec.model_dir / tier_name
    model_dir.mkdir(parents=True, exist_ok=True)

    game = UltiGame()
    cfg = _build_config(
        tier, spec,
        seed=seed, device=device, num_workers=num_workers,
        enrichment=enrichment,
    )

    # Config summary
    param_count = sum(
        p.numel() for p in UltiNet(
            input_dim=game.state_dim, body_units=cfg.body_units,
            body_layers=cfg.body_layers, action_dim=game.action_space_size,
        ).parameters()
    )
    resolved = auto_device(cfg.body_units, cfg.body_layers, force=cfg.device,
                           gpu_tier=tier.gpu)

    # Inference backend
    if resolved == "cpu":
        inf_backend = "ONNX (CPU)" if _ort_available() else "PyTorch (CPU)"
    else:
        inf_backend = f"PyTorch ({resolved.upper()})"

    print(f"    Net: {cfg.body_units}x{cfg.body_layers} ({param_count:,} params)")
    print(f"    Inference: {inf_backend}  |  Solver: {SOLVER_ENGINE}")
    print(f"    MCTS: sol={cfg.sol_sims}s def={cfg.def_sims}s  "
          f"endgame={cfg.endgame_tricks}t  PIMC={cfg.pimc_dets}d")
    print(f"    LR: {cfg.lr_start} → {cfg.lr_end} (cosine)  "
          f"batch={cfg.batch_size}  SGD/step={cfg.train_steps}  "
          f"buffer={cfg.buffer_size:,}")
    if cfg.concurrent_games > 1:
        print(f"    Self-play: {cfg.concurrent_games} concurrent games (GPU batching)")
    elif cfg.num_workers > 1:
        print(f"    Self-play: {cfg.num_workers} workers (process pool)")
    else:
        print(f"    Self-play: sequential")
    print()

    # Train
    on_progress, _interval = (progress_verbose if verbose else progress_bar)(tier)
    net, stats = train_ulti_hybrid(cfg, on_progress=on_progress)

    # Summary
    ri = max(1, tier.steps // 20)
    final_v = np.mean(stats.history_vloss[-ri:]) if stats.history_vloss else 0
    final_p = np.mean(stats.history_ploss[-ri:]) if stats.history_ploss else 0
    final_acc = np.mean(stats.history_pacc[-ri:]) if stats.history_pacc else 0
    start_v = np.mean(stats.history_vloss[:ri]) if len(stats.history_vloss) > ri else final_v
    start_p = np.mean(stats.history_ploss[:ri]) if len(stats.history_ploss) > ri else final_p
    start_acc = np.mean(stats.history_pacc[:ri]) if len(stats.history_pacc) > ri else final_acc

    print()
    print(f"    ┌─ TRAINING SUMMARY ──────────────────────────────────────")
    print(f"    │  Games: {stats.total_games:,}  Samples: {stats.total_samples:,}  "
          f"SGD steps: {stats.total_sgd_steps:,}  Time: {stats.train_time_s:.0f}s")
    print(f"    │  Value head:   {start_v:.4f} → {final_v:.4f}  "
          f"({'improved' if final_v < start_v - 0.001 else 'plateau'})")
    print(f"    │  Policy head:  {start_p:.4f} → {final_p:.4f}  "
          f"({'improved' if final_p < start_p - 0.01 else 'plateau'})")
    print(f"    │  Policy acc:   {start_acc:.1%} → {final_acc:.1%}  "
          f"({'improved' if final_acc > start_acc + 0.01 else 'plateau'})")
    print(f"    └──────────────────────────────────────────────────────────")

    # Save
    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": cfg.body_units,
        "body_layers": cfg.body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "training_mode": cfg.training_mode,
        "method": "hybrid",
        "total_games": stats.total_games,
        "total_samples": stats.total_samples,
        "total_sgd_steps": stats.total_sgd_steps,
        "train_time_s": round(stats.train_time_s, 1),
        "final_vloss": round(float(final_v), 6),
        "final_ploss": round(float(final_p), 6),
        "final_pacc": round(float(final_acc), 4),
    }, model_dir / "model.pt")

    (model_dir / "train_info.json").write_text(json.dumps({
        "tier": tier_name,
        "contract": spec.training_mode,
        "steps": tier.steps,
        "games_per_step": tier.games_per_step,
        "total_games": stats.total_games,
        "sol_sims": tier.sol_sims,
        "def_sims": tier.def_sims,
        "endgame_tricks": cfg.endgame_tricks,
        "pimc_dets": cfg.pimc_dets,
        "body_units": tier.body_units,
        "body_layers": tier.body_layers,
        "train_time_s": round(stats.train_time_s, 1),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"    Saved to {model_dir / 'model.pt'}")
    return net, model_dir
