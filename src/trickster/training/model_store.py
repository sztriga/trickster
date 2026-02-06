from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from trickster.training.policy import TrainedPolicy
from trickster.training.self_play import load_policy, save_policy


@dataclass(frozen=True, slots=True)
class ModelSlots:
    latest: Path
    prev: Path


def default_slots(models_dir: str | Path = "models") -> ModelSlots:
    d = Path(models_dir)
    return ModelSlots(latest=d / "latest.pkl", prev=d / "prev.pkl")


def save_latest_and_prev(policy: TrainedPolicy, *, models_dir: str | Path = "models") -> ModelSlots:
    """
    Keep exactly two rolling slots:
    - <models_dir>/latest.pkl (newest)
    - <models_dir>/prev.pkl   (previous newest)

    On save:
    - if latest exists, it is moved to prev (overwriting prev)
    - then policy is written to latest
    """
    slots = default_slots(models_dir)
    slots.latest.parent.mkdir(parents=True, exist_ok=True)

    if slots.latest.exists():
        if slots.prev.exists():
            slots.prev.unlink()
        slots.latest.replace(slots.prev)

    save_policy(policy, slots.latest)
    return slots


def load_slot(name: str, *, models_dir: str | Path = "models") -> TrainedPolicy:
    slots = default_slots(models_dir)
    if name == "latest":
        return load_policy(slots.latest)
    if name == "prev":
        return load_policy(slots.prev)
    raise ValueError(f"Unknown slot name: {name!r}. Expected 'latest' or 'prev'.")


def slot_exists(name: str, *, models_dir: str | Path = "models") -> bool:
    slots = default_slots(models_dir)
    if name == "latest":
        return slots.latest.exists()
    if name == "prev":
        return slots.prev.exists()
    return False

