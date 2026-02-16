"""Tier and contract definitions for Ulti training.

All training hyperparameters live here — scripts just orchestrate them.

Buffer reuse ≈ (steps × train_steps × batch_size) / buffer_size.
Target: 100-150x.  Knight (128x) is the proven sweet spot.
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
    model_dir: str          # subdirectory under models/
    name_prefix: str        # tier name prefix (P, U, H, B)


CONTRACTS: dict[str, ContractSpec] = {
    "parti": ContractSpec("PARTI", "simple", "parti", "P"),
    "ulti": ContractSpec("ULTI", "ulti", "ulti", "U"),
    "40-100": ContractSpec("40-100", "40-100", "40-100", "H"),
    "betli": ContractSpec("BETLI", "betli", "betli", "B"),
}

CONTRACT_KEYS = list(CONTRACTS.keys())


# ---------------------------------------------------------------------------
#  Tier definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    """Hyperparameters for one training strength level."""
    label: str
    index: int
    description: str

    # Base (single-contract) training
    steps: int
    games_per_step: int
    train_steps: int
    buffer_size: int
    lr_start: float
    lr_end: float

    # E2E (bidding) training
    e2e_steps: int
    e2e_gpi: int
    e2e_train_steps: int
    e2e_buffer_size: int

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

    @property
    def total_games(self) -> int:
        return self.steps * self.games_per_step

    @property
    def e2e_total_games(self) -> int:
        return self.e2e_steps * self.e2e_gpi

    @property
    def base_reuse(self) -> float:
        return (self.steps * self.train_steps * self.batch_size) / self.buffer_size

    @property
    def e2e_reuse(self) -> float:
        return (self.e2e_steps * self.e2e_train_steps * self.batch_size) / self.e2e_buffer_size

    def tier_name(self, prefix: str) -> str:
        return f"{prefix}{self.index}-{self.label}"


TIERS: dict[str, Tier] = {
    # Capacity branch: bigger nets, moderate games
    "scout": Tier(
        label="Scout", index=1,
        description="Scout — fast iteration (256×4, 24k total, 67/33 split)",
        steps=500, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=1000, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=40, def_sims=20,
        lr_start=1e-3, lr_end=2e-4,
    ),
    "knight": Tier(
        label="Knight", index=2,
        description="Knight — proven baseline (256×4, 80k total, 80/20 split)",
        steps=2000, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=2000, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "knight2": Tier(
        label="Knight2", index=14,
        description="Knight2 — rebalanced 50/50 split (256×4, 80k total)",
        steps=1200, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=5200, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=140_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "knight3": Tier(
        label="Knight3", index=15,
        description="Knight3 — e2e-heavy 30/70 split (256×4, 80k total)",
        steps=750, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=7000, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=190_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "knight4": Tier(
        label="Knight4", index=16,
        description="Knight4 — pure e2e from scratch (256×4, 80k total)",
        steps=0, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=10000, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=270_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "bishop": Tier(
        label="Bishop", index=3,
        description="Bishop — larger net (384×4, 120k total)",
        steps=2000, games_per_step=12, train_steps=50, buffer_size=50_000,
        e2e_steps=2000, e2e_gpi=12, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=80, def_sims=30,
        body_units=384, body_layers=4,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "rook": Tier(
        label="Rook", index=4,
        description="Rook — wide+deep (512×6, 120k total)",
        steps=2000, games_per_step=12, train_steps=50, buffer_size=50_000,
        e2e_steps=2000, e2e_gpi=12, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=80, def_sims=30, pimc_dets=40,
        body_units=512, body_layers=6,
        lr_start=1e-3, lr_end=5e-5,
    ),
    "captain": Tier(
        label="Captain", index=5,
        description="Captain — max capacity (1024×6, 180k total, GPU)",
        steps=3000, games_per_step=12, train_steps=50, buffer_size=75_000,
        e2e_steps=3000, e2e_gpi=12, e2e_train_steps=50, e2e_buffer_size=75_000,
        sol_sims=120, def_sims=60, pimc_dets=60,
        body_units=1024, body_layers=6,
        lr_start=5e-4, lr_end=5e-5,
    ),
    # Volume branch: same 256×4 net, more games
    "bronze": Tier(
        label="Bronze", index=7,
        description="Bronze — 2× Knight data (256×4, 160k total)",
        steps=2000, games_per_step=16, train_steps=50, buffer_size=50_000,
        e2e_steps=2000, e2e_gpi=16, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "silver": Tier(
        label="Silver", index=8,
        description="Silver — 6× Knight data (256×4, 480k total)",
        steps=4000, games_per_step=24, train_steps=50, buffer_size=100_000,
        e2e_steps=4000, e2e_gpi=24, e2e_train_steps=50, e2e_buffer_size=100_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=5e-5,
    ),
    "gold": Tier(
        label="Gold", index=9,
        description="Gold — 16× Knight data (256×4, 1.3M total)",
        steps=8000, games_per_step=32, train_steps=50, buffer_size=200_000,
        e2e_steps=8000, e2e_gpi=32, e2e_train_steps=50, e2e_buffer_size=200_000,
        sol_sims=60, def_sims=24,
        lr_start=1e-3, lr_end=5e-5,
    ),
    # Search branch: same 256×4 net, deeper MCTS + PIMC
    "hawk": Tier(
        label="Hawk", index=10,
        description="Hawk — 3× search depth (256×4, 48k total)",
        steps=1000, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=2000, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=120, def_sims=50, sol_dets=4, def_dets=4,
        pimc_dets=40, endgame_tricks=7,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "eagle": Tier(
        label="Eagle", index=11,
        description="Eagle — 6× search depth (256×4, 38k total)",
        steps=800, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=1500, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=250, def_sims=100, sol_dets=8, def_dets=6,
        pimc_dets=80, endgame_tricks=7,
        lr_start=1e-3, lr_end=1e-4,
    ),
    "falcon": Tier(
        label="Falcon", index=12,
        description="Falcon — 12× search depth (256×4, 24k total)",
        steps=500, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=1000, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=500, def_sims=200, sol_dets=12, def_dets=8,
        pimc_dets=120, endgame_tricks=8,
        lr_start=1e-3, lr_end=5e-5,
    ),
    "oracle": Tier(
        label="Oracle", index=13,
        description="Oracle — max search (256×4, 14k total)",
        steps=300, games_per_step=8, train_steps=50, buffer_size=50_000,
        e2e_steps=600, e2e_gpi=8, e2e_train_steps=50, e2e_buffer_size=50_000,
        sol_sims=1000, def_sims=400, sol_dets=16, def_dets=12,
        pimc_dets=200, endgame_tricks=8,
        lr_start=1e-3, lr_end=5e-5,
    ),
}
