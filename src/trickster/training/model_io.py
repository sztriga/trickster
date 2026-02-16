"""Model loading, device selection, and path conventions for Ulti.

Path conventions:
    Base:  models/<contract>/<prefix><index>-<Label>/model.pt
    E2E:   models/e2e/<name>/<contract>/model.pt

Source names:
    "knight"       →  models/e2e/knight/parti/model.pt, ...
    "knight_base"  →  models/parti/P2-Knight/model.pt, ...
"""
from __future__ import annotations

from pathlib import Path

import torch

from trickster.games.ulti.adapter import UltiGame
from trickster.model import UltiNet, UltiNetWrapper, make_wrapper


# ---------------------------------------------------------------------------
#  Device selection
# ---------------------------------------------------------------------------

_GPU_PARAM_THRESHOLD = 3_000_000


def estimate_params(body_units: int, body_layers: int) -> int:
    """Rough parameter estimate (dominant term: L × U²)."""
    return body_layers * body_units * body_units


def auto_device(
    body_units: int = 256,
    body_layers: int = 4,
    force: str | None = None,
) -> str:
    """Pick training device based on model size and available hardware."""
    if force is not None and force != "auto":
        return force

    params = estimate_params(body_units, body_layers)
    if params < _GPU_PARAM_THRESHOLD:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
#  Path conventions
# ---------------------------------------------------------------------------

_CONTRACT_KEYS = ["parti", "ulti", "40-100", "betli"]

# contract → (directory name, base-model prefix letter)
_CONTRACT_META: dict[str, tuple[str, str]] = {
    "parti":  ("parti",  "P"),
    "ulti":   ("ulti",   "U"),
    "40-100": ("40-100", "H"),
    "betli":  ("betli",  "B"),
}

# tier name → index used in base-model directory names
_TIER_INDEX: dict[str, int] = {
    "scout":   1,
    "knight":  2,
    "knight2": 14,
    "knight3": 15,
    "knight4": 16,
    "bishop":  3,
    "rook":    4,
    "captain": 5,
    "bronze":  7,
    "silver":  8,
    "gold":    9,
    "hawk":   10,
    "eagle":  11,
    "falcon": 12,
    "oracle": 13,
}


def _base_path(contract: str, tier: str) -> Path:
    dir_name, prefix = _CONTRACT_META[contract]
    idx = _TIER_INDEX.get(tier, 0)
    return Path("models") / dir_name / f"{prefix}{idx}-{tier.title()}"


def _e2e_path(contract: str, name: str) -> Path:
    dir_name, _ = _CONTRACT_META[contract]
    return Path("models") / "e2e" / name / dir_name


def resolve_paths(source: str) -> dict[str, Path]:
    """Resolve a source name to contract model paths.

    "knight"       → e2e models
    "knight_base"  → base (intermediate) models
    """
    if source.endswith("_base"):
        tier = source.removesuffix("_base")
        if tier not in _TIER_INDEX:
            raise ValueError(
                f"Unknown base tier '{tier}'. "
                f"Known tiers: {', '.join(_TIER_INDEX)}"
            )
        return {c: _base_path(c, tier) for c in _CONTRACT_KEYS}

    return {c: _e2e_path(c, source) for c in _CONTRACT_KEYS}


def list_available_sources(models_root: str | Path = "models") -> list[str]:
    """Scan the filesystem and return source keys that have models."""
    root = Path(models_root)
    sources: list[str] = []

    # Base models
    for tier in _TIER_INDEX:
        paths = {c: _base_path(c, tier) for c in _CONTRACT_KEYS}
        if any((p / "model.pt").exists() for p in paths.values()):
            sources.append(f"{tier}_base")

    # E2E models
    e2e_dir = root / "e2e"
    if e2e_dir.is_dir():
        for tier_dir in sorted(e2e_dir.iterdir()):
            if tier_dir.is_dir():
                name = tier_dir.name
                paths = {c: _e2e_path(c, name) for c in _CONTRACT_KEYS}
                if any((p / "model.pt").exists() for p in paths.values()):
                    sources.append(name)

    return sources


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------

def load_net(model_dir: str | Path, device: str = "cpu") -> UltiNet | None:
    """Load a UltiNet from a model directory.  Returns None if not found."""
    model_pt = Path(model_dir) / "model.pt"
    if not model_pt.exists():
        return None
    cp = torch.load(model_pt, weights_only=False, map_location=device)
    game = UltiGame()
    net = UltiNet(
        input_dim=cp.get("input_dim", game.state_dim),
        body_units=cp.get("body_units", 256),
        body_layers=cp.get("body_layers", 4),
        action_dim=cp.get("action_dim", game.action_space_size),
    )
    net.load_state_dict(cp["model_state_dict"])
    return net


def load_wrappers(
    source: str, device: str = "cpu",
) -> dict[str, UltiNetWrapper]:
    """Load all contract wrappers for a named source.

    Returns {} for unknown sources or missing models.
    """
    paths = resolve_paths(source)
    wrappers: dict[str, UltiNetWrapper] = {}
    for key, path in paths.items():
        net = load_net(path, device)
        if net is not None:
            net.eval()
            wrappers[key] = make_wrapper(net, device=device)
    return wrappers


# ---------------------------------------------------------------------------
#  Display labels (display-key → human label)
# ---------------------------------------------------------------------------

DK_LABELS: dict[str, str] = {
    "p.parti":  "P.Parti",
    "40-100":   "40-100",
    "ulti":     "Ulti",
    "betli":    "Betli",
    "p.40-100": "P.40-100",
    "p.ulti":   "P.Ulti",
    "p.betli":  "P.Betli",
}
