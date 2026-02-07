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
