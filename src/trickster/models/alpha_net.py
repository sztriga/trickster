"""AlphaZero-style network: value head + policy scoring.

All numpy, no framework dependency.  The architecture mirrors
:class:`MLPBinaryModel` (same hidden-layer structure, same stability
tricks) but with two distinct output heads:

* **Value head** – ``state_features → V(s) ∈ [-1, +1]`` via *tanh*,
  trained with MSE against the normalised game outcome.
* **Policy head** – ``(state+action)_features → logit`` per legal
  action, softmaxed externally.  Trained with cross-entropy against
  the MCTS visit distribution.

Each head is a separate MLP (no shared backbone) so they can have
different input dimensions (value sees state only; policy sees
state + action).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


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
#  Value head
# ---------------------------------------------------------------------------


class ValueHead:
    """``state_features → V(s) ∈ [-1, +1]``  (tanh output, MSE loss)."""

    __slots__ = ("_mlp",)

    def __init__(
        self, input_dim: int, hidden_units: int = 64,
        hidden_layers: int = 1, activation: Activation = "relu", seed: int = 0,
    ) -> None:
        self._mlp = _MLP(input_dim, hidden_units, hidden_layers, activation, seed)

    @property
    def input_dim(self) -> int:
        return self._mlp.input_dim

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return V(s) in [-1, +1].  X is (N, state_dim) or (state_dim,)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        logits, _, _ = self._mlp.forward(X)
        return _tanh_safe(logits)

    def train_batch(
        self,
        X: np.ndarray,
        z: np.ndarray,
        lr: float = 0.01,
        l2: float = 1e-5,
    ) -> float:
        """One SGD step.  Returns mean squared error before update.

        X: (N, state_dim), z: (N,) targets in [-1, +1].
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        logits, Zs, As = self._mlp.forward(X)
        v = _tanh_safe(logits)  # (N,)
        err = z - v  # (N,)
        mse = float(np.mean(err * err))
        # d(MSE)/d(logit) = -2 * err * (1 - v^2) / N
        # We absorb the 1/N into backward_and_update, and use gradient ascent
        # convention (positive = direction of improvement) like the rest of the
        # codebase, so: d_logits = err * (1 - v^2)   (= negative of loss grad)
        d_logits = err * (1.0 - v * v)
        self._mlp.backward_and_update(d_logits, Zs, As, lr, l2)
        return mse


# ---------------------------------------------------------------------------
#  Policy head
# ---------------------------------------------------------------------------


class PolicyHead:
    """``(state+action)_features → logit``, softmaxed over legal actions.

    Trained with cross-entropy against the MCTS visit distribution π.
    """

    __slots__ = ("_mlp",)

    def __init__(
        self, input_dim: int, hidden_units: int = 64,
        hidden_layers: int = 1, activation: Activation = "relu", seed: int = 0,
    ) -> None:
        self._mlp = _MLP(input_dim, hidden_units, hidden_layers, activation, seed)

    @property
    def input_dim(self) -> int:
        return self._mlp.input_dim

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return policy probabilities.  X is (N, policy_dim).

        Returns (N,) softmax probabilities over the N candidate actions.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        logits, _, _ = self._mlp.forward(X)
        # Stable softmax
        logits = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        return exp_l / (np.sum(exp_l) + 1e-12)

    def forward_logits(self, X: np.ndarray):
        """Return raw logits + intermediate tensors for backprop."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._mlp.forward(X)

    def train_step(
        self,
        X: np.ndarray,
        pi: np.ndarray,
        lr: float = 0.01,
        l2: float = 1e-5,
    ) -> float:
        """One SGD step for a single decision point.

        X: (K, policy_dim) — one row per legal action at this state.
        pi: (K,) — MCTS visit distribution (sums to 1).

        Returns cross-entropy loss before update.
        """
        logits, Zs, As = self._mlp.forward(X)
        logits = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        p = exp_l / (np.sum(exp_l) + 1e-12)  # (K,)

        # Cross-entropy loss: -sum(pi * log(p))
        loss = -float(np.sum(pi * np.log(p + 1e-12)))

        # Gradient of softmax cross-entropy: d_logits = pi - p
        # backward_and_update divides by N=K (mean over action rows),
        # but all K rows are one softmax — scale by K to get the correct
        # sum gradient.
        K = float(len(pi))
        d_logits = (pi - p) * K
        self._mlp.backward_and_update(d_logits, Zs, As, lr, l2)
        return loss


# ---------------------------------------------------------------------------
#  Fixed-output policy head (AlphaZero-style: state → action_space logits)
# ---------------------------------------------------------------------------


class FixedPolicyHead:
    """``state_features → logits[action_space_size]``, masked softmax.

    Unlike PolicyHead, this takes ONLY state features and outputs one logit
    per possible action.  Illegal actions are masked to -inf before softmax.
    This avoids the gradient cancellation issue of (state,action) input architectures.
    """

    __slots__ = (
        "action_dim", "_Ws", "_bs", "_Wout", "_bout",
        "_activation", "_hidden_layers",
    )

    def __init__(
        self, input_dim: int, action_dim: int, hidden_units: int = 64,
        hidden_layers: int = 1, activation: Activation = "relu", seed: int = 0,
    ) -> None:
        self.action_dim = action_dim
        self._Ws: list[np.ndarray] = []
        self._bs: list[np.ndarray] = []
        rng = np.random.default_rng(seed)
        h = hidden_units
        d = input_dim
        L = max(1, hidden_layers)
        self._activation = activation
        self._hidden_layers = L

        def _lim(fi: int, fo: int) -> float:
            return float(np.sqrt(6.0 / max(1, fi + fo)))

        for i in range(L):
            fi = d if i == 0 else h
            lim = _lim(fi, h)
            self._Ws.append(rng.uniform(-lim, lim, (h, fi)).astype(np.float64))
            self._bs.append(np.zeros(h, dtype=np.float64))
        # Output layer: h → action_dim
        lim = _lim(h, action_dim)
        self._Wout = rng.uniform(-lim, lim, (action_dim, h)).astype(np.float64)
        self._bout = np.zeros(action_dim, dtype=np.float64)

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self._activation == "relu":
            return np.minimum(np.maximum(z, 0.0), _ACT_CLIP)
        return np.tanh(z)

    def _act_grad(self, z: np.ndarray) -> np.ndarray:
        if self._activation == "relu":
            return ((z > 0) & (z < _ACT_CLIP)).astype(np.float64)
        t = np.tanh(z)
        return (1.0 - t * t).astype(np.float64)

    def forward(self, state_feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return masked softmax probabilities over action space.

        state_feats: (state_dim,)
        mask: (action_dim,) boolean — True for legal actions.
        Returns: (action_dim,) probabilities (0 for illegal).
        """
        A = state_feats
        for W, b in zip(self._Ws, self._bs):
            A = self._act(A @ W.T + b)
        logits = A @ self._Wout.T + self._bout  # (action_dim,)
        logits = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
        logits[~mask] = -1e9
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        exp_l[~mask] = 0.0
        s = np.sum(exp_l)
        return exp_l / (s + 1e-12)

    def train_step(
        self,
        state_feats: np.ndarray,
        pi: np.ndarray,
        mask: np.ndarray,
        lr: float = 0.01,
        l2: float = 1e-5,
    ) -> float:
        """One SGD step.

        state_feats: (state_dim,)
        pi: (action_dim,) — target distribution (0 for illegal actions).
        mask: (action_dim,) boolean.
        Returns cross-entropy loss before update.
        """
        # Forward
        A = state_feats
        Zs: list[np.ndarray] = []
        As: list[np.ndarray] = [A]
        for W, b in zip(self._Ws, self._bs):
            Z = A @ W.T + b
            Zs.append(Z)
            A = self._act(Z)
            As.append(A)
        logits = A @ self._Wout.T + self._bout  # (action_dim,)
        logits = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
        logits[~mask] = -1e9
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        exp_l[~mask] = 0.0
        s = np.sum(exp_l)
        p = exp_l / (s + 1e-12)  # (action_dim,)

        # Loss
        loss = -float(np.sum(pi * np.log(p + 1e-12)))

        # Gradient: d_logits[i] = p[i] - pi[i] for legal, 0 for illegal
        d_logits = np.zeros(self.action_dim, dtype=np.float64)
        d_logits[mask] = p[mask] - pi[mask]  # loss gradient (we'll negate for ascent)

        # Backward through output layer
        # dWout = outer(d_logits, As[-1])
        dWout = np.outer(-d_logits, As[-1])  # negate for gradient ascent
        dbout = -d_logits.copy()
        dA = (-d_logits) @ self._Wout  # (h,)

        # Backward through hidden layers
        dWs: list[np.ndarray] = [np.empty(0)] * self._hidden_layers
        dbs: list[np.ndarray] = [np.empty(0)] * self._hidden_layers
        for i in range(self._hidden_layers - 1, -1, -1):
            dZ = dA * self._act_grad(Zs[i])
            dWs[i] = np.outer(dZ, As[i])
            dbs[i] = dZ
            if i > 0:
                dA = dZ @ self._Ws[i]

        # Gradient norm and clip
        g2 = float(np.sum(dWout * dWout)) + float(np.sum(dbout * dbout))
        for dW, db in zip(dWs, dbs):
            g2 += float(np.sum(dW * dW)) + float(np.sum(db * db))
        gnorm = float(np.sqrt(g2))
        if not np.isfinite(gnorm):
            return loss
        if gnorm > _GRAD_CLIP:
            sc = _GRAD_CLIP / (gnorm + 1e-8)
            dWout *= sc; dbout *= sc
            dWs = [dW * sc for dW in dWs]; dbs = [db * sc for db in dbs]

        # Apply
        decay = 1.0 - lr * l2
        self._Wout = np.clip(self._Wout * decay + lr * dWout, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self._bout = np.clip(self._bout + lr * dbout, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        for i in range(self._hidden_layers):
            self._Ws[i] = np.clip(self._Ws[i] * decay + lr * dWs[i], -_WEIGHT_CLIP, _WEIGHT_CLIP)
            self._bs[i] = np.clip(self._bs[i] + lr * dbs[i], -_WEIGHT_CLIP, _WEIGHT_CLIP)
        return loss

    def train_batch(
        self,
        states: np.ndarray,
        pis: np.ndarray,
        masks: np.ndarray,
        lr: float = 0.01,
        l2: float = 1e-5,
    ) -> float:
        """Batched SGD step — average gradients over B samples.

        states: (B, state_dim)
        pis: (B, action_dim) — target distributions.
        masks: (B, action_dim) boolean.
        Returns mean cross-entropy loss before update.
        """
        B = states.shape[0]
        total_loss = 0.0
        acc_dWout = np.zeros_like(self._Wout)
        acc_dbout = np.zeros_like(self._bout)
        acc_dWs = [np.zeros_like(W) for W in self._Ws]
        acc_dbs = [np.zeros_like(b) for b in self._bs]

        for i in range(B):
            loss_i, dWout_i, dbout_i, dWs_i, dbs_i = self._compute_gradients(
                states[i], pis[i], masks[i],
            )
            total_loss += loss_i
            acc_dWout += dWout_i
            acc_dbout += dbout_i
            for j in range(self._hidden_layers):
                acc_dWs[j] += dWs_i[j]
                acc_dbs[j] += dbs_i[j]

        # Average
        inv_B = 1.0 / B
        acc_dWout *= inv_B
        acc_dbout *= inv_B
        for j in range(self._hidden_layers):
            acc_dWs[j] *= inv_B
            acc_dbs[j] *= inv_B

        # Clip and apply
        g2 = float(np.sum(acc_dWout ** 2)) + float(np.sum(acc_dbout ** 2))
        for dW, db in zip(acc_dWs, acc_dbs):
            g2 += float(np.sum(dW ** 2)) + float(np.sum(db ** 2))
        gnorm = float(np.sqrt(g2))
        if not np.isfinite(gnorm):
            return total_loss / B
        if gnorm > _GRAD_CLIP:
            sc = _GRAD_CLIP / (gnorm + 1e-8)
            acc_dWout *= sc; acc_dbout *= sc
            acc_dWs = [dW * sc for dW in acc_dWs]
            acc_dbs = [db * sc for db in acc_dbs]

        decay = 1.0 - lr * l2
        self._Wout = np.clip(self._Wout * decay + lr * acc_dWout, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        self._bout = np.clip(self._bout + lr * acc_dbout, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        for j in range(self._hidden_layers):
            self._Ws[j] = np.clip(self._Ws[j] * decay + lr * acc_dWs[j], -_WEIGHT_CLIP, _WEIGHT_CLIP)
            self._bs[j] = np.clip(self._bs[j] + lr * acc_dbs[j], -_WEIGHT_CLIP, _WEIGHT_CLIP)
        return total_loss / B

    def _compute_gradients(self, state_feats, pi, mask):
        """Compute loss and gradients for ONE sample (no update)."""
        A = state_feats
        Zs = []
        As = [A]
        for W, b in zip(self._Ws, self._bs):
            Z = A @ W.T + b
            Zs.append(Z)
            A = self._act(Z)
            As.append(A)
        logits = A @ self._Wout.T + self._bout
        logits = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
        logits[~mask] = -1e9
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        exp_l[~mask] = 0.0
        s = np.sum(exp_l)
        p = exp_l / (s + 1e-12)

        loss = -float(np.sum(pi * np.log(p + 1e-12)))

        d_logits = np.zeros(self.action_dim, dtype=np.float64)
        d_logits[mask] = p[mask] - pi[mask]

        # Backward (gradient ascent = negate loss gradient)
        dWout = np.outer(-d_logits, As[-1])
        dbout = -d_logits.copy()
        dA = (-d_logits) @ self._Wout

        dWs = [np.empty(0)] * self._hidden_layers
        dbs = [np.empty(0)] * self._hidden_layers
        for i in range(self._hidden_layers - 1, -1, -1):
            dZ = dA * self._act_grad(Zs[i])
            dWs[i] = np.outer(dZ, As[i])
            dbs[i] = dZ
            if i > 0:
                dA = dZ @ self._Ws[i]

        return loss, dWout, dbout, dWs, dbs


# ---------------------------------------------------------------------------
#  Combined AlphaNet
# ---------------------------------------------------------------------------


@dataclass
class AlphaNet:
    """Container for one value head + one fixed-output policy head.

    The policy head outputs logits over the entire action space,
    masked to legal actions at each state.
    """

    value: ValueHead
    policy: FixedPolicyHead

    def predict_value(self, state_features: np.ndarray) -> float:
        """V(s) for a single state."""
        v = self.value.forward(state_features)
        return float(v[0]) if v.ndim > 0 else float(v)

    def predict_policy(
        self, state_features: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        """π(a|s) masked softmax over action space.  Returns (action_space,)."""
        return self.policy.forward(state_features, mask)


def create_alpha_net(
    state_dim: int,
    action_space_size: int,
    hidden_units: int = 64,
    hidden_layers: int = 1,
    activation: Activation = "relu",
    seed: int = 0,
) -> AlphaNet:
    """Factory: create a fresh AlphaNet with the given dimensions."""
    value = ValueHead(
        state_dim, hidden_units, hidden_layers, activation, seed=seed + 1,
    )
    policy = FixedPolicyHead(
        state_dim, action_space_size, hidden_units, hidden_layers,
        activation, seed=seed + 10,
    )
    return AlphaNet(value=value, policy=policy)


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
        with np.errstate(all="ignore"):
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
        with np.errstate(all="ignore"):
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
        with np.errstate(all="ignore"):
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
        with np.errstate(all="ignore"):
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
