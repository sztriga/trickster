"""Model loading, device selection, and path conventions for Ulti.

Path conventions:
    Base:  models/ulti/<tier>/base/<contract>/model.pt
    Final: models/ulti/<name>/final/<contract>/model.pt

Source names:
    "knight_light"       →  models/ulti/knight_light/final/parti/model.pt, ...
    "knight_light_base"  →  models/ulti/knight_light/base/parti/model.pt, ...
"""
from __future__ import annotations

from pathlib import Path

import torch

from trickster.games.ulti.adapter import UltiGame
from trickster.model import UltiNet, UltiNetWrapper, make_wrapper
from trickster.training.tiers import TIERS


# ---------------------------------------------------------------------------
#  Device selection
# ---------------------------------------------------------------------------

def estimate_params(body_units: int, body_layers: int) -> int:
    """Rough parameter estimate (dominant term: L × U²)."""
    return body_layers * body_units * body_units


def auto_device(
    body_units: int = 256,
    body_layers: int = 4,
    force: str | None = None,
    *,
    gpu_tier: bool = False,
) -> str:
    """Pick training device.

    CPU tiers use ONNX Runtime for fast single-sample inference.
    GPU tiers (``gpu_tier=True``) use PyTorch on CUDA with cross-game
    inference batching.

    Returns ``"cpu"`` unless the tier requests GPU **and** hardware
    is available, or the caller explicitly forces a device.
    """
    if force is not None and force != "auto":
        return force

    if gpu_tier:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
#  Path conventions
# ---------------------------------------------------------------------------

_CONTRACT_KEYS = ["parti", "ulti", "40-100", "betli"]

_ULTI_ROOT = Path("models/ulti")


def _base_path(contract: str, tier: str) -> Path:
    return _ULTI_ROOT / tier / "base" / contract


def _e2e_path(contract: str, name: str) -> Path:
    return _ULTI_ROOT / name / "final" / contract


def resolve_paths(source: str) -> dict[str, Path]:
    """Resolve a source name to contract model paths.

    "knight_light"       → final (e2e) models
    "knight_light_base"  → base (intermediate) models
    """
    if source.endswith("_base"):
        tier = source.removesuffix("_base")
        if tier not in TIERS:
            raise ValueError(
                f"Unknown base tier '{tier}'. "
                f"Known tiers: {', '.join(TIERS)}"
            )
        return {c: _base_path(c, tier) for c in _CONTRACT_KEYS}

    return {c: _e2e_path(c, source) for c in _CONTRACT_KEYS}


def list_available_sources(models_root: str | Path = "models") -> list[str]:
    """Scan the filesystem and return source keys that have models."""
    root = Path(models_root)
    ulti_dir = root / "ulti"
    sources: list[str] = []

    if not ulti_dir.is_dir():
        return sources

    for model_dir in sorted(ulti_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name

        # Final (e2e) models
        final_dir = model_dir / "final"
        if final_dir.is_dir() and any(
            (final_dir / c / "model.pt").exists() for c in _CONTRACT_KEYS
        ):
            sources.append(name)

        # Base models
        base_dir = model_dir / "base"
        if base_dir.is_dir() and any(
            (base_dir / c / "model.pt").exists() for c in _CONTRACT_KEYS
        ):
            sources.append(f"{name}_base")

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
