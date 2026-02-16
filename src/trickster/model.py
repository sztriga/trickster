"""UltiNet — PyTorch AlphaZero model for Ulti play + auction.

Architecture:
    Input (259-dim) → Shared Backbone (4×256 + LayerNorm + ReLU)
                       ├── Policy Head   → Linear(256, 32)   → LogSoftmax
                       ├── Value Head    → Linear(256, 1)     (unbounded)
                       └── Auction Head (multi-component)
                            ├── Trump sub-head → Linear(128, 5) → LogSoftmax
                            │                    (♥, ♦, ♠, ♣, NoTrump)
                            └── Flags sub-head → Linear(128, 4) → Sigmoid
                                                  (Is_100, Is_Ulti, Is_Betli, Is_Durchmars)

The multi-component auction head predicts trump suit and contract
flags *independently*, allowing compound bids like "Ulti + Parti"
or "40-100 + Parti" to be represented naturally.

Legacy ``NUM_CONTRACTS`` / ``CONTRACT_CLASSES`` are kept for backward
compatibility with the evaluator and auto-mode training loop.  The
new component-based outputs are used for training targets.

The ``UltiNetWrapper`` adapts the PyTorch model to the numpy-based
MCTS interface expected by ``alpha_mcts_policy`` / ``alpha_mcts_choose``.

``OnnxUltiWrapper`` is a drop-in replacement that uses ONNX Runtime
for inference — ~10x faster per call on CPU for small models because
it eliminates PyTorch's per-layer Python dispatch overhead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trickster.games.ulti.encoder import STATE_DIM, NUM_CARDS, _SCALAR_OFF

# ---------------------------------------------------------------------------
#  Auction head constants
# ---------------------------------------------------------------------------

# Trump sub-head: 5 classes (softmax)
TRUMP_CLASSES = 5  # ♥=0  ♦=1  ♠=2  ♣=3  NoTrump=4

# Contract-flag sub-head: 4 independent binary flags (sigmoid)
FLAG_NAMES: list[str] = ["is_100", "is_ulti", "is_betli", "is_durchmars"]
NUM_FLAGS = len(FLAG_NAMES)

# Legacy 5-class mapping kept for evaluator / auto-mode compat
NUM_CONTRACTS = 5
CONTRACT_CLASSES: list[tuple[str, int | None]] = [
    ("simple", 0),   # Parti ♥
    ("simple", 1),   # Parti ♦
    ("simple", 2),   # Parti ♠
    ("simple", 3),   # Parti ♣
    ("betli",  None), # Betli (no trump)
]


def auction_components_to_legacy(trump_idx: int, flags: np.ndarray) -> int:
    """Convert multi-component prediction to a legacy contract index.

    Betli flag overrides trump choice → index 4.
    Otherwise trump_idx 0..3 maps directly.
    """
    if flags[2] > 0.5:  # is_betli
        return 4
    return min(trump_idx, 3)


def legacy_to_auction_targets(
    legacy_idx: int,
) -> tuple[int, np.ndarray]:
    """Convert a legacy contract index to (trump_target, flags_target).

    Returns
    -------
    trump_target : int  (0..4)
    flags_target : (4,) float array of 0/1
    """
    flags = np.zeros(NUM_FLAGS, dtype=np.float32)
    if legacy_idx == 4:  # betli
        flags[2] = 1.0  # is_betli
        return 4, flags
    # simple: trump = legacy_idx, no extra flags
    return legacy_idx, flags


# ---------------------------------------------------------------------------
#  PyTorch model
# ---------------------------------------------------------------------------


class UltiNet(nn.Module):
    """Shared-backbone AlphaZero network for Ulti play + auction.

    Architecture (dual-head with stop-gradient):

        Input → Shared Backbone (4×256)
                  │
                  ├─ [detach] → Soloist Policy Head (256→32) + Soloist Value Head (256→1)
                  ├─ [detach] → Defender Policy Head (256→32) + Defender Value Head (256→1)
                  │
                  └─ [grad] → Shared Policy Head (256→32) + Shared Value Head (256→1)
                               (backbone training signal only)

    During **inference** (MCTS), the correct role head is selected
    based on the ``is_soloist`` flag in the state features.

    During **training**, ``forward_dual()`` routes each sample through
    its role-specific head with ``detach()`` on the backbone output
    (stop-gradient), so heads never push conflicting gradients to the
    backbone.  The backbone learns from a shared head that sees all
    samples with full gradients — purely perceptual features.

    Parameters
    ----------
    input_dim : int
        State feature vector length (default 259).
    body_units : int
        Width of each backbone layer (default 256).
    body_layers : int
        Number of backbone layers (default 4).
    action_dim : int
        Number of possible actions / cards (default 32).
    num_contracts : int
        Legacy auction classes (default 5, kept for compat).
    """

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        body_units: int = 256,
        body_layers: int = 4,
        action_dim: int = NUM_CARDS,
        num_contracts: int = NUM_CONTRACTS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.body_units = body_units
        self.action_dim = action_dim
        self.num_contracts = num_contracts

        hidden = body_units // 2  # 128

        # --- Shared backbone ---
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(body_layers):
            layers.append(nn.Linear(in_dim, body_units))
            layers.append(nn.LayerNorm(body_units))
            layers.append(nn.ReLU())
            in_dim = body_units
        self.backbone = nn.Sequential(*layers)

        # --- Shared heads (backbone training signal) ---
        self.policy_head = nn.Linear(body_units, action_dim)
        self.value_fc = nn.Linear(body_units, 1)

        # --- Role-specific heads (stop-gradient from backbone) ---
        self.policy_head_sol = nn.Linear(body_units, action_dim)
        self.policy_head_def = nn.Linear(body_units, action_dim)
        self.value_fc_sol = nn.Linear(body_units, 1)
        self.value_fc_def = nn.Linear(body_units, 1)

        # --- Auction head: shared hidden layer ---
        self.auction_shared = nn.Sequential(
            nn.Linear(body_units, hidden),
            nn.ReLU(),
        )
        # Trump sub-head: 5 classes (H, D, S, C, NoTrump) — softmax
        self.auction_trump = nn.Linear(hidden, TRUMP_CLASSES)
        # Flags sub-head: 4 binary (is_100, is_ulti, is_betli, is_duri) — sigmoid
        self.auction_flags = nn.Linear(hidden, NUM_FLAGS)

        # Xavier init for all heads
        for fc in [self.policy_head, self.value_fc,
                   self.policy_head_sol, self.policy_head_def,
                   self.value_fc_sol, self.value_fc_def,
                   self.auction_trump, self.auction_flags]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
        for m in self.auction_shared:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy forward — uses shared heads (backward compat).

        Returns (log_probs, value).
        """
        body = self.backbone(x)

        logits = self.policy_head(body)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)

        value = self.value_fc(body).squeeze(-1)
        return log_probs, value

    def forward_role(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_soloist: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward through role-specific head (for MCTS).

        Routes through soloist or defender head based on ``is_soloist``.
        Full gradient flow (no detach) — suitable for inference only.

        Returns (log_probs, value).
        """
        body = self.backbone(x)

        if is_soloist:
            logits = self.policy_head_sol(body)
            value = self.value_fc_sol(body).squeeze(-1)
        else:
            logits = self.policy_head_def(body)
            value = self.value_fc_def(body).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, value

    def forward_dual(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        is_soloist_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward with role-specific heads.

        Each sample is routed through its role-specific head with full
        gradient flow.  The architectural separation ensures that
        soloist gradients only pass through the soloist head and
        defender gradients only through the defender head — no
        cross-contamination.  The backbone receives gradients from
        both roles, learning shared perceptual features.

        Parameters
        ----------
        x : (B, input_dim)
        mask : (B, action_dim) bool
        is_soloist_batch : (B,) bool — True for soloist samples

        Returns
        -------
        log_probs : (B, action_dim) — from role-specific policy heads
        values : (B,) — from role-specific value heads
        """
        body = self.backbone(x)

        B = x.shape[0]
        logits = torch.zeros(B, self.action_dim, device=x.device)
        raw_val = torch.zeros(B, device=x.device)

        sol = is_soloist_batch.bool()
        deff = ~sol

        if sol.any():
            logits[sol] = self.policy_head_sol(body[sol])
            raw_val[sol] = self.value_fc_sol(body[sol]).squeeze(-1)
        if deff.any():
            logits[deff] = self.policy_head_def(body[deff])
            raw_val[deff] = self.value_fc_def(body[deff]).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, raw_val

    def forward_auction(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-component auction forward pass.

        Parameters
        ----------
        x : Tensor (B, input_dim)

        Returns
        -------
        trump_log_probs : Tensor (B, TRUMP_CLASSES)
            Log-probabilities over trump suit choices.
        flag_logits : Tensor (B, NUM_FLAGS)
            Raw logits for contract flags (apply sigmoid for probs).
        """
        body = self.backbone(x)
        h = self.auction_shared(body)
        trump_logits = self.auction_trump(h)
        trump_lp = F.log_softmax(trump_logits, dim=-1)
        flag_logits = self.auction_flags(h)
        return trump_lp, flag_logits

    def forward_auction_legacy(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy 5-class auction forward (for backward compat).

        Constructs 5-class log-probs from the multi-component outputs:
          classes 0-3 = trump log-prob (H,D,S,C) + log(1 - betli_prob)
          class 4     = log(betli_prob) + log(NoTrump_prob)

        This is a differentiable bridge so old training code still works.
        """
        trump_lp, flag_logits = self.forward_auction(x)
        betli_prob = torch.sigmoid(flag_logits[:, 2:3])  # is_betli
        not_betli = (1.0 - betli_prob).clamp(min=1e-7)

        # Suit classes: P(suit_i, not_betli) = P(suit_i) * P(not_betli)
        suit_lp = trump_lp[:, :4] + torch.log(not_betli)
        # Betli class: P(betli) * P(no_trump)
        betli_lp = torch.log(betli_prob.clamp(min=1e-7)) + trump_lp[:, 4:5]

        logits5 = torch.cat([suit_lp, betli_lp], dim=-1)
        return F.log_softmax(logits5, dim=-1)

    def forward_value_only(self, x: torch.Tensor) -> torch.Tensor:
        """Value head only — used for neural discard evaluation."""
        body = self.backbone(x)
        return self.value_fc(body).squeeze(-1)


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
    """Export backbone + one role's heads to an in-memory ONNX model.

    Parameters
    ----------
    net : UltiNet (must be on CPU, eval mode)
    role : "sol" or "def"

    Returns
    -------
    Raw ONNX model bytes.
    """
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
        """Re-export ONNX sessions from the current PyTorch weights.

        Called once per training iteration after SGD updates.
        """
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
