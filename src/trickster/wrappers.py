"""Inference wrappers for UltiNet — numpy API for MCTS / training.

``UltiNetWrapper`` bridges PyTorch ↔ numpy for MCTS self-play.
``OnnxUltiWrapper`` is a drop-in replacement that uses ONNX Runtime,
eliminating per-layer Python dispatch overhead (~10× faster on CPU
for small models).

``make_wrapper`` auto-selects the best available backend.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trickster.games.ulti.encoder import _SCALAR_OFF
from trickster.model import UltiNet


# ---------------------------------------------------------------------------
#  MCTS-compatible wrapper (numpy ↔ torch bridge)
# ---------------------------------------------------------------------------


class UltiNetWrapper:
    """Wraps ``UltiNet`` to match the numpy API expected by MCTS.

    Auto-detects the player role (soloist vs defender) from the
    ``is_soloist`` scalar in the encoded state features and routes
    through the appropriate role-specific head.

    Play phase:
      - ``predict_value(state_feats) → float``
      - ``predict_policy(state_feats, mask) → np.ndarray``
      - ``predict_both(state_feats, mask) → (float, np.ndarray)``

    Auction phase:
      - ``predict_auction(state_feats) → np.ndarray`` (legacy 5 probs)
      - ``predict_auction_components(state_feats) → (trump_probs, flag_probs)``
      - ``batch_value(state_batch) → np.ndarray``
    """

    # Index of the is_soloist flag in encoded state features
    _IS_SOL_IDX = _SCALAR_OFF + 1

    def __init__(self, net: UltiNet, device: str = "cpu") -> None:
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)

    def _detect_role(self, state_feats: np.ndarray) -> bool:
        """Read the is_soloist flag from encoded features."""
        return bool(state_feats.flat[self._IS_SOL_IDX] > 0.5)

    def predict_value(self, state_feats: np.ndarray) -> float:
        is_sol = self._detect_role(state_feats)
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            _, value = self.net.forward_role(x, is_soloist=is_sol)
        return float(value.item())

    def predict_policy(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        is_sol = self._detect_role(state_feats)
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            m = torch.from_numpy(mask.reshape(1, -1)).bool().to(self.device)
            log_probs, _ = self.net.forward_role(x, m, is_soloist=is_sol)
            probs = log_probs.exp().squeeze(0)
        return probs.cpu().numpy()

    def predict_both(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        is_sol = self._detect_role(state_feats)
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            m = torch.from_numpy(mask.reshape(1, -1)).bool().to(self.device)
            log_probs, value = self.net.forward_role(x, m, is_soloist=is_sol)
            probs = log_probs.exp().squeeze(0)
        return float(value.item()), probs.cpu().numpy()

    def predict_both_batch(
        self,
        feats_list: list[np.ndarray],
        mask_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched predict_both — amortises PyTorch overhead across leaves.

        Auto-detects soloist/defender role per sample from encoded features
        and routes through the appropriate role-specific head via
        ``forward_dual``.

        Parameters
        ----------
        feats_list : list of (input_dim,) arrays
        mask_list  : list of (action_dim,) bool arrays

        Returns
        -------
        values : (B,) float array — value from each sample's current player
        probs  : (B, action_dim) float array — policy probabilities
        """
        B = len(feats_list)
        if B == 0:
            return np.empty(0, dtype=np.float32), np.zeros(
                (0, self.net.action_dim), dtype=np.float32,
            )

        is_sol_list = [self._detect_role(f) for f in feats_list]

        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(feats_list)).float().to(self.device)
            m = torch.from_numpy(np.stack(mask_list)).bool().to(self.device)
            is_sol = torch.tensor(
                is_sol_list, dtype=torch.bool, device=self.device,
            )
            log_probs, values = self.net.forward_dual(x, m, is_sol)
            probs = log_probs.exp()
        return values.cpu().numpy(), probs.cpu().numpy()

    def batch_value(self, states: np.ndarray) -> np.ndarray:
        """Evaluate soloist value head on a batch of states.

        Uses the role-specific soloist head (``value_fc_sol``) which is
        trained by ``forward_dual`` during self-play.  This replaces the
        old shared ``value_fc`` path which was never updated by the
        bidding training loop.

        Callers (bidding evaluator, neural discard) always evaluate from
        the soloist's perspective, so the soloist head is correct here.
        """
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(states).float().to(self.device)
            body = self.net.backbone(x)
            values = self.net.value_fc_sol(body).squeeze(-1)
        return values.cpu().numpy()

    def predict_auction(self, state_feats: np.ndarray) -> np.ndarray:
        """Legacy 5-class auction probabilities."""
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            log_probs = self.net.forward_auction_legacy(x)
            probs = log_probs.exp().squeeze(0)
        return probs.cpu().numpy()

    def predict_auction_components(
        self, state_feats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-component auction prediction.

        Returns
        -------
        trump_probs : (TRUMP_CLASSES,) probabilities over suit / no-trump
        flag_probs  : (NUM_FLAGS,) probabilities for each contract flag
        """
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            trump_lp, flag_logits = self.net.forward_auction(x)
            trump_probs = trump_lp.exp().squeeze(0).cpu().numpy()
            flag_probs = torch.sigmoid(flag_logits).squeeze(0).cpu().numpy()
        return trump_probs, flag_probs


# ---------------------------------------------------------------------------
#  ONNX Runtime wrapper (fast CPU inference — drop-in for UltiNetWrapper)
# ---------------------------------------------------------------------------


def _ort_available() -> bool:
    """Check whether onnxruntime is installed."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _export_role_onnx(net: UltiNet, role: str) -> bytes:
    """Export backbone + one role's heads to an in-memory ONNX model."""
    import io

    class _RoleGraph(nn.Module):
        def __init__(self, backbone, policy_head, value_fc):
            super().__init__()
            self.backbone = backbone
            self.policy_head = policy_head
            self.value_fc = value_fc

        def forward(self, x: torch.Tensor, mask: torch.Tensor):
            body = self.backbone(x)
            logits = self.policy_head(body)
            logits = logits.masked_fill(~mask, -1e9)
            log_probs = F.log_softmax(logits, dim=-1)
            value = self.value_fc(body).squeeze(-1)
            return log_probs, value

    if role == "sol":
        graph = _RoleGraph(net.backbone, net.policy_head_sol, net.value_fc_sol)
    else:
        graph = _RoleGraph(net.backbone, net.policy_head_def, net.value_fc_def)
    graph.eval()

    dummy_x = torch.randn(1, net.input_dim)
    dummy_m = torch.ones(1, net.action_dim, dtype=torch.bool)

    buf = io.BytesIO()
    torch.onnx.export(
        graph, (dummy_x, dummy_m), buf,
        input_names=["x", "mask"],
        output_names=["log_probs", "value"],
        dynamic_axes={
            "x": {0: "B"}, "mask": {0: "B"},
            "log_probs": {0: "B"}, "value": {0: "B"},
        },
    )
    return buf.getvalue()


def _export_value_onnx(net: UltiNet) -> bytes:
    """Export backbone + soloist value head for batch_value()."""
    import io

    class _ValueGraph(nn.Module):
        def __init__(self, backbone, value_fc):
            super().__init__()
            self.backbone = backbone
            self.value_fc = value_fc

        def forward(self, x: torch.Tensor):
            body = self.backbone(x)
            return self.value_fc(body).squeeze(-1)

    graph = _ValueGraph(net.backbone, net.value_fc_sol)
    graph.eval()

    buf = io.BytesIO()
    torch.onnx.export(
        graph, torch.randn(1, net.input_dim), buf,
        input_names=["x"], output_names=["value"],
        dynamic_axes={"x": {0: "B"}, "value": {0: "B"}},
    )
    return buf.getvalue()


class OnnxUltiWrapper:
    """ONNX Runtime inference wrapper — drop-in for ``UltiNetWrapper``.

    Exports the PyTorch model to ONNX and runs all inference through
    ONNX Runtime, eliminating per-layer Python dispatch overhead.
    ~10x faster per NN call on CPU for small (256x4) models.

    Call ``sync_weights(net)`` after SGD steps to re-export with
    the latest weights.  This costs ~50ms and is called once per
    training iteration (amortised over thousands of NN calls).

    The ``net`` attribute is kept for compatibility with code that
    reads ``wrapper.net`` (e.g. to access state_dict for workers).
    """

    _IS_SOL_IDX = _SCALAR_OFF + 1

    def __init__(self, net: UltiNet) -> None:
        import onnxruntime as ort

        self.net = net
        self.device = torch.device("cpu")
        self._action_dim = net.action_dim
        self._ort = ort
        self._opts = ort.SessionOptions()
        self._opts.inter_op_num_threads = 1
        self._opts.intra_op_num_threads = 1
        self._opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._sessions: dict[str, ort.InferenceSession] = {}
        self._value_session: ort.InferenceSession | None = None
        self.sync_weights(net)

    def sync_weights(self, net: UltiNet) -> None:
        """Re-export ONNX sessions from the current PyTorch weights."""
        self.net = net
        was_training = net.training
        net.eval()
        net.cpu()

        ort = self._ort
        for role in ("sol", "def"):
            onnx_bytes = _export_role_onnx(net, role)
            self._sessions[role] = ort.InferenceSession(
                onnx_bytes, self._opts, providers=["CPUExecutionProvider"],
            )

        val_bytes = _export_value_onnx(net)
        self._value_session = ort.InferenceSession(
            val_bytes, self._opts, providers=["CPUExecutionProvider"],
        )

        if was_training:
            net.train()

    def _detect_role(self, state_feats: np.ndarray) -> bool:
        return bool(state_feats.flat[self._IS_SOL_IDX] > 0.5)

    def _run(self, role: str, x: np.ndarray, mask: np.ndarray):
        sess = self._sessions[role]
        return sess.run(None, {
            "x": np.ascontiguousarray(x, dtype=np.float32),
            "mask": np.ascontiguousarray(mask, dtype=np.bool_),
        })

    # -- Play-phase API (matches UltiNetWrapper) --

    def predict_value(self, state_feats: np.ndarray) -> float:
        role = "sol" if self._detect_role(state_feats) else "def"
        x = state_feats.reshape(1, -1)
        mask = np.ones((1, self._action_dim), dtype=np.bool_)
        _, v = self._run(role, x, mask)
        return float(v[0])

    def predict_policy(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        role = "sol" if self._detect_role(state_feats) else "def"
        lp, _ = self._run(role, state_feats.reshape(1, -1), mask.reshape(1, -1))
        return np.exp(lp[0])

    def predict_both(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        role = "sol" if self._detect_role(state_feats) else "def"
        lp, v = self._run(role, state_feats.reshape(1, -1), mask.reshape(1, -1))
        return float(v[0]), np.exp(lp[0])

    def predict_both_batch(
        self,
        feats_list: list[np.ndarray],
        mask_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        B = len(feats_list)
        if B == 0:
            return np.empty(0, dtype=np.float32), np.zeros(
                (0, self._action_dim), dtype=np.float32,
            )

        x = np.stack(feats_list).astype(np.float32)
        m = np.stack(mask_list).astype(np.bool_)
        is_sol = np.array([self._detect_role(f) for f in feats_list])

        values = np.zeros(B, dtype=np.float32)
        probs = np.zeros((B, self._action_dim), dtype=np.float32)

        sol_idx = np.where(is_sol)[0]
        def_idx = np.where(~is_sol)[0]

        if len(sol_idx) > 0:
            lp, v = self._run("sol", x[sol_idx], m[sol_idx])
            values[sol_idx] = v.ravel()
            probs[sol_idx] = np.exp(lp)
        if len(def_idx) > 0:
            lp, v = self._run("def", x[def_idx], m[def_idx])
            values[def_idx] = v.ravel()
            probs[def_idx] = np.exp(lp)

        return values, probs

    # -- Bidding / discard API --

    def batch_value(self, states: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(states, dtype=np.float32)
        (values,) = self._value_session.run(None, {"x": x})
        return values

    def predict_auction(self, state_feats: np.ndarray) -> np.ndarray:
        """Legacy auction — falls back to PyTorch (called rarely)."""
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float()
            log_probs = self.net.forward_auction_legacy(x)
            probs = log_probs.exp().squeeze(0)
        return probs.cpu().numpy()

    def predict_auction_components(
        self, state_feats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Auction components — falls back to PyTorch (called rarely)."""
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float()
            trump_lp, flag_logits = self.net.forward_auction(x)
            trump_probs = trump_lp.exp().squeeze(0).cpu().numpy()
            flag_probs = torch.sigmoid(flag_logits).squeeze(0).cpu().numpy()
        return trump_probs, flag_probs


def make_wrapper(
    net: UltiNet, device: str = "cpu", *, use_onnx: bool | None = None,
) -> UltiNetWrapper | OnnxUltiWrapper:
    """Create the best available inference wrapper.

    Parameters
    ----------
    net : UltiNet
    device : target device (ONNX only supports "cpu")
    use_onnx : True = force ONNX, False = force PyTorch,
               None = auto (ONNX if available and device is cpu)
    """
    if use_onnx is None:
        use_onnx = device == "cpu" and _ort_available()
    if use_onnx:
        try:
            return OnnxUltiWrapper(net)
        except Exception:
            pass
    return UltiNetWrapper(net, device=device)
