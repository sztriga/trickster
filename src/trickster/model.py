"""UltiNet — PyTorch AlphaZero model for Ulti play phase.

Architecture:
    Input (259-dim) → Shared Backbone (4×256 + LayerNorm + ReLU)
                       ├── Policy Head → Linear(256, 32) → LogSoftmax
                       └── Value Head  → Linear(256, 1)  → Tanh

The ``UltiNetWrapper`` adapts the PyTorch model to the numpy-based
MCTS interface expected by ``alpha_mcts_policy`` / ``alpha_mcts_choose``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trickster.games.ulti.encoder import STATE_DIM, NUM_CARDS


# ---------------------------------------------------------------------------
#  PyTorch model
# ---------------------------------------------------------------------------


class UltiNet(nn.Module):
    """Shared-backbone AlphaZero network for the Ulti play phase.

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
    """

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        body_units: int = 256,
        body_layers: int = 4,
        action_dim: int = NUM_CARDS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.body_units = body_units
        self.action_dim = action_dim

        # --- Shared backbone ---
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(body_layers):
            layers.append(nn.Linear(in_dim, body_units))
            layers.append(nn.LayerNorm(body_units))
            layers.append(nn.ReLU())
            in_dim = body_units
        self.backbone = nn.Sequential(*layers)

        # --- Policy head (card selection) ---
        self.policy_head = nn.Linear(body_units, action_dim)

        # --- Value head (win/loss prediction) ---
        self.value_fc = nn.Linear(body_units, 1)

        # Xavier init for heads
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.xavier_uniform_(self.value_fc.weight)
        nn.init.zeros_(self.value_fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, input_dim)
            Encoded state features.
        mask : Tensor (B, action_dim), optional
            Boolean mask — True = legal action.

        Returns
        -------
        log_probs : Tensor (B, action_dim)
            Log-probabilities over actions (masked softmax).
        value : Tensor (B,)
            Predicted value in [-1, +1].
        """
        body = self.backbone(x)

        # Policy: masked log-softmax
        logits = self.policy_head(body)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)

        # Value: tanh squash
        value = torch.tanh(self.value_fc(body)).squeeze(-1)

        return log_probs, value


# ---------------------------------------------------------------------------
#  MCTS-compatible wrapper (numpy ↔ torch bridge)
# ---------------------------------------------------------------------------


class UltiNetWrapper:
    """Wraps ``UltiNet`` to match the numpy API expected by MCTS.

    The MCTS code calls:
      - ``predict_value(state_feats) → float``
      - ``predict_policy(state_feats, mask) → np.ndarray``
      - ``predict_both(state_feats, mask) → (float, np.ndarray)``

    This wrapper converts between numpy arrays and PyTorch tensors,
    running inference in ``torch.no_grad()`` mode.
    """

    def __init__(self, net: UltiNet, device: str = "cpu") -> None:
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)

    def predict_value(self, state_feats: np.ndarray) -> float:
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            _, value = self.net(x)
        return float(value.item())

    def predict_policy(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            m = torch.from_numpy(mask.reshape(1, -1)).bool().to(self.device)
            log_probs, _ = self.net(x, m)
            probs = log_probs.exp().squeeze(0)
        return probs.cpu().numpy()

    def predict_both(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            m = torch.from_numpy(mask.reshape(1, -1)).bool().to(self.device)
            log_probs, value = self.net(x, m)
            probs = log_probs.exp().squeeze(0)
        return float(value.item()), probs.cpu().numpy()
