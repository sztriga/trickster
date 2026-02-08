"""AlphaZero-style neural network: SharedAlphaNet with dual heads.

All numpy, no framework dependency.  The ``SharedAlphaNet`` has a shared
body MLP and two output heads:

* **Value head** – ``state_features → V(s) ∈ [-1, +1]`` via *tanh*,
  trained with MSE against the normalised game outcome.
* **Policy head** – ``state_features → logits[action_space]``, masked
  softmax.  Trained with cross-entropy against the MCTS visit distribution.
"""

from __future__ import annotations
from typing import Literal

import numpy as np

# Suppress numpy warnings globally — numerical edge cases (overflow in
# matmul, divide-by-zero in softmax) are handled by gradient clipping
# and careful coding.  Removing per-call ``np.errstate`` context managers
# eliminates ~0.5s of overhead per 100k neural-net forward passes.
np.seterr(all="ignore")

Activation = Literal["relu", "tanh"]

# Stability constants (same as mlp.py)
_ACT_CLIP = 20.0
_LOGIT_CLIP = 15.0
_WEIGHT_CLIP = 5.0
_GRAD_CLIP = 5.0


def _tanh_safe(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -_LOGIT_CLIP, _LOGIT_CLIP))


# ---------------------------------------------------------------------------
#  Single MLP building block
# ---------------------------------------------------------------------------


class _MLP:
    """Bare MLP: hidden layers → scalar output (no output activation).

    Forward returns ``(logits, pre_activations, activations)`` for backprop.
    """

    __slots__ = (
        "input_dim", "hidden_units", "hidden_layers", "activation",
        "Ws", "bs", "Wout", "bout",
    )

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 64,
        hidden_layers: int = 1,
        activation: Activation = "relu",
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.activation = activation
        rng = np.random.default_rng(seed)

        h = hidden_units
        d = input_dim
        L = max(1, hidden_layers)

        def _lim(fan_in: int, fan_out: int) -> float:
            return float(np.sqrt(6.0 / max(1, fan_in + fan_out)))

        self.Ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        for i in range(L):
            fi = d if i == 0 else h
            lim = _lim(fi, h)
            self.Ws.append(rng.uniform(-lim, lim, (h, fi)).astype(np.float64))
            self.bs.append(np.zeros(h, dtype=np.float64))
        lim = _lim(h, 1)
        self.Wout = rng.uniform(-lim, lim, (h,)).astype(np.float64)
        self.bout = 0.0

    # -- activation helpers ------------------------------------------------

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.minimum(np.maximum(z, 0.0), _ACT_CLIP)
        return np.tanh(z)

    def _act_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return ((z > 0) & (z < _ACT_CLIP)).astype(np.float64)
        t = np.tanh(z)
        return (1.0 - t * t).astype(np.float64)

    # -- forward -----------------------------------------------------------

    def forward(self, X: np.ndarray):
        """Forward pass on (N, d) or (d,) input.

        Returns ``(raw_logits, list[pre_act], list[activations])``
        where ``activations[0] = X`` and ``logits`` has no output activation.
        """
        A = X
        Zs: list[np.ndarray] = []
        As: list[np.ndarray] = [A]
        with np.errstate(all="ignore"):
            for W, b in zip(self.Ws, self.bs):
                Z = A @ W.T + b
                Zs.append(Z)
                A = self._act(Z)
                As.append(A)
            logits = As[-1] @ self.Wout + self.bout
        return logits, Zs, As

    # -- backward + update -------------------------------------------------

    def backward_and_update(
        self,
        d_logits: np.ndarray,
        Zs: list[np.ndarray],
        As: list[np.ndarray],
        lr: float,
        l2: float,
    ) -> None:
        """Backward from ``d_logits`` (gradient of loss w.r.t. raw logits).

        ``d_logits`` is (N,) — one scalar per sample.
        """
        N = max(1, d_logits.shape[0]) if d_logits.ndim > 0 else 1
        with np.errstate(all="ignore"):
            dWout = d_logits @ As[-1] / N if d_logits.ndim > 0 else d_logits * As[-1]
            dbout = float(np.mean(d_logits))
            dA = d_logits[:, None] * self.Wout[None, :] if d_logits.ndim > 0 else d_logits * self.Wout

            L = len(self.Ws)
            dWs: list[np.ndarray] = [np.empty(0)] * L
            dbs: list[np.ndarray] = [np.empty(0)] * L
            for i in range(L - 1, -1, -1):
                dZ = dA * self._act_grad(Zs[i])
                if dZ.ndim > 1:
                    dWs[i] = dZ.T @ As[i] / N
                    dbs[i] = np.mean(dZ, axis=0)
                else:
                    dWs[i] = np.outer(dZ, As[i])
                    dbs[i] = dZ
                if i > 0:
                    dA = dZ @ self.Ws[i]

        self._apply_grads(dWout, dbout, dWs, dbs, lr, l2)

    def _apply_grads(self, dWout, dbout, dWs, dbs, lr, l2):
        g2 = float(np.sum(dWout * dWout)) + dbout * dbout
        for dW, db in zip(dWs, dbs):
            g2 += float(np.sum(dW * dW)) + float(np.sum(db * db))
        gnorm = float(np.sqrt(g2))
        if not np.isfinite(gnorm):
            return
        if gnorm > _GRAD_CLIP:
            s = _GRAD_CLIP / (gnorm + 1e-8)
            dWout = dWout * s
            dbout = dbout * s
            dWs = [dW * s for dW in dWs]
            dbs = [db * s for db in dbs]
        decay = 1.0 - lr * l2
        self.Wout = np.clip(self.Wout * decay + lr * dWout, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self.bout = max(-_WEIGHT_CLIP, min(_WEIGHT_CLIP, float(self.bout + lr * dbout)))
        for i in range(len(self.Ws)):
            self.Ws[i] = np.clip(self.Ws[i] * decay + lr * dWs[i], -_WEIGHT_CLIP, _WEIGHT_CLIP)
            self.bs[i] = np.clip(self.bs[i] + lr * dbs[i], -_WEIGHT_CLIP, _WEIGHT_CLIP)


# ---------------------------------------------------------------------------
#  SharedAlphaNet — dual-head with shared body (for AlphaZero training)
# ---------------------------------------------------------------------------


class SharedAlphaNet:
    """Dual-head AlphaZero network with a shared body MLP.

    Architecture::

        state → [shared body MLP] → body_out
                                     ├── [value head MLP]  → tanh → V(s) ∈ [-1,+1]
                                     └── [policy head MLP] → masked softmax → π(a|s)

    Combined loss: ``L = (z − v)² − π·log(p) + c·‖θ‖²``
    """

    __slots__ = (
        "state_dim", "action_space_size", "body_activation",
        # Shared body
        "Ws_body", "bs_body",
        # Value head: body_units → head_units → scalar
        "Wv1", "bv1", "Wv2", "bv2",
        # Policy head: body_units → head_units → action_space_size
        "Wp1", "bp1", "Wp2", "bp2",
    )

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        Ws_body: list[np.ndarray],
        bs_body: list[np.ndarray],
        body_activation: Activation,
        Wv1: np.ndarray,
        bv1: np.ndarray,
        Wv2: np.ndarray,
        bv2: float,
        Wp1: np.ndarray,
        bp1: np.ndarray,
        Wp2: np.ndarray,
        bp2: np.ndarray,
    ) -> None:
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.body_activation = body_activation
        self.Ws_body = Ws_body
        self.bs_body = bs_body
        self.Wv1 = Wv1
        self.bv1 = bv1
        self.Wv2 = Wv2
        self.bv2 = bv2
        self.Wp1 = Wp1
        self.bp1 = bp1
        self.Wp2 = Wp2
        self.bp2 = bp2

    # -- activation helpers ------------------------------------------------

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.body_activation == "relu":
            return np.minimum(np.maximum(z, 0.0), _ACT_CLIP)
        return np.tanh(z)

    def _act_grad(self, z: np.ndarray) -> np.ndarray:
        if self.body_activation == "relu":
            return ((z > 0) & (z < _ACT_CLIP)).astype(np.float64)
        t = np.tanh(z)
        return (1.0 - t * t).astype(np.float64)

    # -- forward helpers ---------------------------------------------------

    def _forward_body(self, X: np.ndarray):
        """Forward through shared body.

        Returns ``(body_output, Zs, As)`` for backprop.
        ``As[0] = X``, ``As[-1]`` is the last hidden activation.
        """
        A = X
        Zs: list[np.ndarray] = []
        As: list[np.ndarray] = [A]
        for W, b in zip(self.Ws_body, self.bs_body):
            Z = A @ W.T + b
            Zs.append(Z)
            A = self._act(Z)
            As.append(A)
        return A, Zs, As

    # -- public inference --------------------------------------------------

    def predict_value(self, state_feats: np.ndarray) -> float:
        """V(s) for a single state.  ``state_feats`` is ``(state_dim,)``."""
        X = state_feats.reshape(1, -1) if state_feats.ndim == 1 else state_feats
        body_out, _, _ = self._forward_body(X)
        body_out = body_out.ravel()
        hv = self._act(body_out @ self.Wv1.T + self.bv1)
        v = _tanh_safe(np.dot(hv, self.Wv2) + self.bv2)
        return float(v)

    def predict_policy(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        """Masked softmax π(a|s).  Returns ``(action_space,)``."""
        X = state_feats.reshape(1, -1) if state_feats.ndim == 1 else state_feats
        body_out, _, _ = self._forward_body(X)
        body_out = body_out.ravel()
        hp = self._act(body_out @ self.Wp1.T + self.bp1)
        logits = hp @ self.Wp2.T + self.bp2
        logits = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
        logits[~mask] = -1e9
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        exp_l[~mask] = 0.0
        return exp_l / (np.sum(exp_l) + 1e-12)

    def predict_both(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Shared-body forward for both heads.  Returns ``(value, probs)``."""
        X = state_feats.reshape(1, -1) if state_feats.ndim == 1 else state_feats
        body_out, _, _ = self._forward_body(X)
        body_flat = body_out.ravel()
        # Value
        hv = self._act(body_flat @ self.Wv1.T + self.bv1)
        v = float(_tanh_safe(np.dot(hv, self.Wv2) + self.bv2))
        # Policy
        hp = self._act(body_flat @ self.Wp1.T + self.bp1)
        logits = hp @ self.Wp2.T + self.bp2
        logits = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
        logits[~mask] = -1e9
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        exp_l[~mask] = 0.0
        probs = exp_l / (np.sum(exp_l) + 1e-12)
        return v, probs

    # -- training ----------------------------------------------------------

    def train_batch(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        pis: np.ndarray,
        zs: np.ndarray,
        lr: float,
        l2: float,
    ) -> tuple[float, float]:
        """One SGD step on a mini-batch.

        Parameters
        ----------
        states : (B, state_dim)
        masks  : (B, action_space) boolean
        pis    : (B, action_space) MCTS visit distributions
        zs     : (B,) game outcomes in [-1, +1]
        lr     : learning rate
        l2     : L2 regularization coefficient

        Returns
        -------
        (mean_value_mse, mean_policy_ce) *before* the update.
        """
        B = states.shape[0]

        with np.errstate(all="ignore"):
            # === Forward ===
            # Body
            body_out, body_Zs, body_As = self._forward_body(states)

            # Value head
            Zv1 = body_out @ self.Wv1.T + self.bv1       # (B, head)
            Hv1 = self._act(Zv1)
            logits_v = Hv1 @ self.Wv2 + self.bv2          # (B,)
            v = _tanh_safe(logits_v)                        # (B,)

            # Policy head
            Zp1 = body_out @ self.Wp1.T + self.bp1        # (B, head)
            Hp1 = self._act(Zp1)
            logits_p = Hp1 @ self.Wp2.T + self.bp2        # (B, action)
            logits_p = np.clip(logits_p, -_LOGIT_CLIP, _LOGIT_CLIP)
            logits_p[~masks] = -1e9
            shifted = logits_p - np.max(logits_p, axis=1, keepdims=True)
            exp_l = np.exp(shifted)
            exp_l[~masks] = 0.0
            sums = np.sum(exp_l, axis=1, keepdims=True) + 1e-12
            p = exp_l / sums                               # (B, action)

            # === Losses ===
            err = zs - v
            vmse = float(np.mean(err * err))
            pce = -float(np.mean(np.sum(pis * np.log(p + 1e-12), axis=1)))

            # === Backward ===
            # Value gradient (gradient ascent = positive direction improves)
            d_logits_v = err * (1.0 - v * v)               # (B,)

            # Value head backward
            dHv1 = d_logits_v[:, None] * self.Wv2[None, :] # (B, head)
            dZv1 = dHv1 * self._act_grad(Zv1)              # (B, head)
            d_body_v = dZv1 @ self.Wv1                      # (B, body)

            dWv2 = d_logits_v @ Hv1 / B                    # (head,)
            dbv2 = float(np.mean(d_logits_v))
            dWv1 = dZv1.T @ body_out / B                   # (head, body)
            dbv1 = np.mean(dZv1, axis=0)                    # (head,)

            # Policy gradient: d(CE)/d(logit) = p - pi  →  ascent = pi - p
            d_logits_p = np.zeros_like(logits_p)
            d_logits_p[masks] = pis[masks] - p[masks]       # (B, action)

            # Policy head backward
            dHp1 = d_logits_p @ self.Wp2                    # (B, head)
            dZp1 = dHp1 * self._act_grad(Zp1)              # (B, head)
            d_body_p = dZp1 @ self.Wp1                      # (B, body)

            dWp2 = d_logits_p.T @ Hp1 / B                  # (action, head)
            dbp2 = np.mean(d_logits_p, axis=0)              # (action,)
            dWp1 = dZp1.T @ body_out / B                   # (head, body)
            dbp1 = np.mean(dZp1, axis=0)                    # (head,)

            # Combined body gradient
            d_body = d_body_v + d_body_p                    # (B, body)

            # Body backward
            body_L = len(self.Ws_body)
            dWs_body: list[np.ndarray] = [np.empty(0)] * body_L
            dbs_body: list[np.ndarray] = [np.empty(0)] * body_L
            dA = d_body
            for i in range(body_L - 1, -1, -1):
                dZ = dA * self._act_grad(body_Zs[i])
                dWs_body[i] = dZ.T @ body_As[i] / B
                dbs_body[i] = np.mean(dZ, axis=0)
                if i > 0:
                    dA = dZ @ self.Ws_body[i]

        # Gradient clipping
        g2 = (
            float(np.sum(dWv2 * dWv2)) + dbv2 * dbv2
            + float(np.sum(dWv1 * dWv1)) + float(np.sum(dbv1 * dbv1))
            + float(np.sum(dWp2 * dWp2)) + float(np.sum(dbp2 * dbp2))
            + float(np.sum(dWp1 * dWp1)) + float(np.sum(dbp1 * dbp1))
        )
        for dW, db in zip(dWs_body, dbs_body):
            g2 += float(np.sum(dW * dW)) + float(np.sum(db * db))
        gnorm = float(np.sqrt(g2))
        if not np.isfinite(gnorm):
            return vmse, pce
        if gnorm > _GRAD_CLIP:
            sc = _GRAD_CLIP / (gnorm + 1e-8)
            dWv2 *= sc; dbv2 *= sc; dWv1 *= sc; dbv1 *= sc
            dWp2 *= sc; dbp2 *= sc; dWp1 *= sc; dbp1 *= sc
            dWs_body = [dW * sc for dW in dWs_body]
            dbs_body = [db * sc for db in dbs_body]

        # Apply gradients with L2 decay
        decay = 1.0 - lr * l2
        self.Wv2 = np.clip(self.Wv2 * decay + lr * dWv2, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self.bv2 = max(-_WEIGHT_CLIP, min(_WEIGHT_CLIP, float(self.bv2 + lr * dbv2)))
        self.Wv1 = np.clip(self.Wv1 * decay + lr * dWv1, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self.bv1 = np.clip(self.bv1 + lr * dbv1, -_WEIGHT_CLIP, _WEIGHT_CLIP)

        self.Wp2 = np.clip(self.Wp2 * decay + lr * dWp2, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self.bp2 = np.clip(self.bp2 + lr * dbp2, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self.Wp1 = np.clip(self.Wp1 * decay + lr * dWp1, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self.bp1 = np.clip(self.bp1 + lr * dbp1, -_WEIGHT_CLIP, _WEIGHT_CLIP)

        for i in range(body_L):
            self.Ws_body[i] = np.clip(
                self.Ws_body[i] * decay + lr * dWs_body[i],
                -_WEIGHT_CLIP, _WEIGHT_CLIP,
            )
            self.bs_body[i] = np.clip(
                self.bs_body[i] + lr * dbs_body[i],
                -_WEIGHT_CLIP, _WEIGHT_CLIP,
            )

        return vmse, pce


def create_shared_alpha_net(
    state_dim: int,
    action_space_size: int,
    body_units: int = 128,
    body_layers: int = 2,
    head_units: int = 64,
    activation: Activation = "relu",
    seed: int = 0,
) -> SharedAlphaNet:
    """Factory: create a fresh SharedAlphaNet with Xavier initialization."""
    rng = np.random.default_rng(seed)

    def _lim(fan_in: int, fan_out: int) -> float:
        return float(np.sqrt(6.0 / max(1, fan_in + fan_out)))

    h = body_units

    # Body
    Ws_body: list[np.ndarray] = []
    bs_body: list[np.ndarray] = []
    for i in range(body_layers):
        fi = state_dim if i == 0 else h
        lim = _lim(fi, h)
        Ws_body.append(rng.uniform(-lim, lim, (h, fi)).astype(np.float64))
        bs_body.append(np.zeros(h, dtype=np.float64))

    # Value head
    lim = _lim(h, head_units)
    Wv1 = rng.uniform(-lim, lim, (head_units, h)).astype(np.float64)
    bv1 = np.zeros(head_units, dtype=np.float64)
    lim = _lim(head_units, 1)
    Wv2 = rng.uniform(-lim, lim, (head_units,)).astype(np.float64)
    bv2 = 0.0

    # Policy head
    lim = _lim(h, head_units)
    Wp1 = rng.uniform(-lim, lim, (head_units, h)).astype(np.float64)
    bp1 = np.zeros(head_units, dtype=np.float64)
    lim = _lim(head_units, action_space_size)
    Wp2 = rng.uniform(-lim, lim, (action_space_size, head_units)).astype(np.float64)
    bp2 = np.zeros(action_space_size, dtype=np.float64)

    return SharedAlphaNet(
        state_dim=state_dim,
        action_space_size=action_space_size,
        Ws_body=Ws_body,
        bs_body=bs_body,
        body_activation=activation,
        Wv1=Wv1, bv1=bv1, Wv2=Wv2, bv2=bv2,
        Wp1=Wp1, bp1=bp1, Wp2=Wp2, bp2=bp2,
    )
