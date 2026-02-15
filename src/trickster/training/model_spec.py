from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    Defines a model family + its hyperparameters.

    Changing ANY parameter creates a different model_id and therefore a different
    checkpoint folder.
    """

    kind: str  # "linear" | "mlp"
    params: Dict[str, Any]
    game: str = "snapszer"  # which game this model is trained for
    method: str = "direct"  # "direct" | "expert"

    def canonical(self) -> Dict[str, Any]:
        # Normalize common fields for stable hashing
        kind = str(self.kind).strip().lower()
        method = str(self.method).strip().lower()
        params = dict(self.params or {})
        return {"game": str(self.game).strip().lower(), "kind": kind, "method": method, "params": _jsonable(params)}


def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    # fallback: string representation
    return str(x)


def model_id(spec: ModelSpec) -> str:
    canon = spec.canonical()
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    h = hashlib.sha1(blob).hexdigest()  # stable, short enough
    return f"{canon['kind']}-{h[:10]}"


def model_dir(spec: ModelSpec, *, root: str | Path = "models/snapszer") -> Path:
    return Path(root) / model_id(spec)


def write_spec(spec: ModelSpec, *, root: str | Path = "models/snapszer") -> Path:
    d = model_dir(spec, root=root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "spec.json"
    p.write_text(json.dumps(spec.canonical(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_spec(path: str | Path) -> ModelSpec:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelSpec(kind=obj["kind"], params=obj.get("params", {}), game=obj.get("game", "snapszer"), method=obj.get("method", "direct"))


def list_model_dirs(*, root: str | Path = "models/snapszer") -> List[Path]:
    r = Path(root)
    if not r.exists():
        return []
    out: list[Path] = []
    for p in r.iterdir():
        if p.is_dir() and (p / "spec.json").exists():
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def model_label_from_dir(d: Path) -> str:
    """
    Human-friendly label for GUI dropdowns.
    """
    try:
        spec = read_spec(d / "spec.json")
        canon = spec.canonical()
        k = canon["kind"]
        m = canon.get("method", "direct")
        params = canon["params"]
        tag = f" {m}" if m not in ("direct",) else ""
        if k == "linear":
            return f"{d.name}  (linear{tag})"
        if k == "mlp":
            h = params.get("hidden_units", "?")
            layers = params.get("hidden_layers", 1)
            act = params.get("activation", "?")
            layers_str = f" L={layers}" if layers not in (1, "1", None) else ""
            return f"{d.name}  (mlp h={h}{layers_str} act={act}{tag})"
        if k == "alphazero":
            body = params.get("body_units", "?")
            blayers = params.get("body_layers", "?")
            head = params.get("head_units", "?")
            method_tag = " hybrid" if m == "hybrid" else ""
            return f"{d.name}  (alphazero {body}x{blayers} head={head}{method_tag})"
        return f"{d.name}  ({k}{tag})"
    except Exception:
        return d.name

