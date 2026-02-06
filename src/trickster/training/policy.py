from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from trickster.games.snapszer.features import follow_feature_keys, lead_feature_keys
from trickster.models.mlp import MLPBinaryModel
from trickster.models.linear import LinearBinaryModel
from trickster.training.model_spec import ModelSpec


@dataclass(slots=True)
class TrainedPolicy:
    spec: ModelSpec
    lead_model: Any
    follow_model: Any


def create_policy(spec: ModelSpec) -> TrainedPolicy:
    kind = spec.canonical()["kind"]
    params: Dict[str, Any] = spec.canonical()["params"]

    if kind == "linear":
        return TrainedPolicy(spec=spec, lead_model=LinearBinaryModel(), follow_model=LinearBinaryModel())

    if kind == "mlp":
        hidden_units = int(params.get("hidden_units", 64))
        hidden_layers = int(params.get("hidden_layers", 1))
        activation = str(params.get("activation", "relu")).lower()
        if activation not in ("relu", "tanh"):
            activation = "relu"
        seed = int(params.get("init_seed", 0))
        return TrainedPolicy(
            spec=spec,
            lead_model=MLPBinaryModel(lead_feature_keys(), hidden_units=hidden_units, hidden_layers=hidden_layers, activation=activation, seed=seed + 1),
            follow_model=MLPBinaryModel(follow_feature_keys(), hidden_units=hidden_units, hidden_layers=hidden_layers, activation=activation, seed=seed + 2),
        )

    raise ValueError(f"Unknown model kind: {kind!r}")

