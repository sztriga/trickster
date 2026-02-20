"""Tier and contract definitions for Ulti training.

All training hyperparameters live here — scripts just orchestrate them.

Buffer reuse ≈ (steps × train_steps × batch_size) / buffer_size.
Target: 100-150x.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
#  Contract definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractSpec:
    """Contract-specific settings for training."""
    display_name: str       # "PARTI", "BETLI", etc.
    training_mode: str      # passed to UltiGame.new_game()


CONTRACTS: dict[str, ContractSpec] = {
    "parti": ContractSpec("PARTI", "simple"),
    "ulti": ContractSpec("ULTI", "ulti"),
    "40-100": ContractSpec("40-100", "40-100"),
    "betli": ContractSpec("BETLI", "betli"),
}

CONTRACT_KEYS = list(CONTRACTS.keys())


# ---------------------------------------------------------------------------
#  Tier definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    """Hyperparameters for one training strength level.

    All training is bidding-based from scratch (no Phase 1).
    """
    label: str
    description: str

    # Training budget
    steps: int
    games_per_step: int
    train_steps: int
    buffer_size: int
    lr_start: float
    lr_end: float

    # MCTS
    sol_sims: int
    def_sims: int
    sol_dets: int = 2
    def_dets: int = 2

    # Solver / PIMC
    endgame_tricks: int = 6
    pimc_dets: int = 20
    solver_temp: float = 0.5

    # Network
    body_units: int = 256
    body_layers: int = 4
    batch_size: int = 64

    # GPU flag — when True the training loop uses CUDA + cross-game
    # inference batching instead of CPU + ONNX.
    gpu: bool = False

    @property
    def total_games(self) -> int:
        return self.steps * self.games_per_step

    @property
    def reuse(self) -> float:
        return (self.steps * self.train_steps * self.batch_size) / self.buffer_size



TIERS: dict[str, Tier] = {
    # ── Capacity branch: bigger nets, moderate data ──
    "scout": Tier(
        label="Scout",
        description="Scout — quick iteration (256×4, 16k deals)",
        steps=2000, games_per_step=8, train_steps=50, buffer_size=50_000,
        sol_sims=40, def_sims=20,
        lr_start=1e-3, lr_end=2e-4,
    ),
    "knight": Tier(
        label="Knight",
        description="Knight — standard (256×4, 48k deals)",
        steps=6000, games_per_step=8, train_steps=50, buffer_size=80_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "bishop": Tier(
        label="Bishop",
        description="Bishop — larger net (384×4, 48k deals)",
        steps=4000, games_per_step=12, train_steps=50, buffer_size=80_000,
        sol_sims=80, def_sims=30,
        body_units=384, body_layers=4,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "rook": Tier(
        label="Rook",
        description="Rook — wide+deep (512×6, 48k deals)",
        steps=4000, games_per_step=12, train_steps=50, buffer_size=80_000,
        sol_sims=80, def_sims=30, pimc_dets=40,
        body_units=512, body_layers=6,
        lr_start=1e-3, lr_end=5e-5,
    ),
    "captain": Tier(
        label="Captain",
        description="Captain — max capacity (1024×6, 72k deals, GPU)",
        steps=6000, games_per_step=12, train_steps=50, buffer_size=120_000,
        sol_sims=120, def_sims=60, pimc_dets=60,
        body_units=1024, body_layers=6,
        lr_start=5e-4, lr_end=5e-5,
    ),
    # ── Volume branch: same 256×4 net, more data ──
    "bronze": Tier(
        label="Bronze",
        description="Bronze — 2× Knight data (256×4, 96k deals)",
        steps=6000, games_per_step=16, train_steps=50, buffer_size=100_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "silver": Tier(
        label="Silver",
        description="Silver — 6× Knight data (256×4, 288k deals)",
        steps=12000, games_per_step=24, train_steps=50, buffer_size=200_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=5e-5,
    ),
    "gold": Tier(
        label="Gold",
        description="Gold — 16× Knight data (256×4, 768k deals)",
        steps=24000, games_per_step=32, train_steps=50, buffer_size=400_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=5e-5,
    ),
    # ── Search branch: same 256×4 net, deeper MCTS + PIMC ──
    "hawk": Tier(
        label="Hawk",
        description="Hawk — 3× search depth (256×4, 24k deals)",
        steps=3000, games_per_step=8, train_steps=50, buffer_size=60_000,
        sol_sims=120, def_sims=50, sol_dets=4, def_dets=4,
        pimc_dets=40, endgame_tricks=7,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "eagle": Tier(
        label="Eagle",
        description="Eagle — 6× search depth (256×4, 18k deals)",
        steps=2250, games_per_step=8, train_steps=50, buffer_size=50_000,
        sol_sims=250, def_sims=100, sol_dets=8, def_dets=6,
        pimc_dets=80, endgame_tricks=7,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "falcon": Tier(
        label="Falcon",
        description="Falcon — 12× search depth (256×4, 12k deals)",
        steps=1500, games_per_step=8, train_steps=50, buffer_size=50_000,
        sol_sims=500, def_sims=200, sol_dets=12, def_dets=8,
        pimc_dets=120, endgame_tricks=8,
        lr_start=1e-3, lr_end=5e-5,
    ),
    "oracle": Tier(
        label="Oracle",
        description="Oracle — max search (256×4, 7k deals)",
        steps=900, games_per_step=8, train_steps=50, buffer_size=50_000,
        sol_sims=1000, def_sims=400, sol_dets=16, def_dets=12,
        pimc_dets=200, endgame_tricks=8,
        lr_start=1e-3, lr_end=5e-5,
    ),
    # ── GPU branch: large model, requires CUDA ──
    "king": Tier(
        label="King",
        description="King — GPU large model (2048×8, 128k deals)",
        steps=8000, games_per_step=16, train_steps=100, buffer_size=300_000,
        sol_sims=150, def_sims=60, sol_dets=4, def_dets=3,
        pimc_dets=80, endgame_tricks=7,
        body_units=2048, body_layers=8, batch_size=256,
        lr_start=3e-4, lr_end=2e-5,
        gpu=True,
    ),
}
