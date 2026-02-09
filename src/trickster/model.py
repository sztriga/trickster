"""UltiNet — PyTorch AlphaZero model for Ulti play + auction.

Architecture:
    Input (259-dim) → Shared Backbone (4×256 + LayerNorm + ReLU)
                       ├── Policy Head   → Linear(256, 32)   → LogSoftmax
                       ├── Value Head    → Linear(256, 1)    → Tanh
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
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trickster.games.ulti.encoder import STATE_DIM, NUM_CARDS

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

        # --- Policy head (card selection) ---
        self.policy_head = nn.Linear(body_units, action_dim)

        # --- Value head (win/loss prediction) ---
        self.value_fc = nn.Linear(body_units, 1)

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
        """Forward pass (play phase).

        Returns (log_probs, value).
        """
        body = self.backbone(x)

        logits = self.policy_head(body)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)

        value = torch.tanh(self.value_fc(body)).squeeze(-1)
        return log_probs, value

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
        return torch.tanh(self.value_fc(body)).squeeze(-1)


# ---------------------------------------------------------------------------
#  MCTS-compatible wrapper (numpy ↔ torch bridge)
# ---------------------------------------------------------------------------


class UltiNetWrapper:
    """Wraps ``UltiNet`` to match the numpy API expected by MCTS.

    Play phase:
      - ``predict_value(state_feats) → float``
      - ``predict_policy(state_feats, mask) → np.ndarray``
      - ``predict_both(state_feats, mask) → (float, np.ndarray)``

    Auction phase:
      - ``predict_auction(state_feats) → np.ndarray`` (legacy 5 probs)
      - ``predict_auction_components(state_feats) → (trump_probs, flag_probs)``
      - ``batch_value(state_batch) → np.ndarray``
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

    def batch_value(self, states: np.ndarray) -> np.ndarray:
        """Evaluate value head on a batch of states."""
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(states).float().to(self.device)
            values = self.net.forward_value_only(x)
        return values.cpu().numpy()

    def predict_auction(self, state_feats: np.ndarray) -> np.ndarray:
        """Legacy 5-class auction probabilities."""
        self.net.eval()
        with torch.no_grad():
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
        with torch.no_grad():
            x = torch.from_numpy(state_feats.reshape(1, -1)).float().to(self.device)
            trump_lp, flag_logits = self.net.forward_auction(x)
            trump_probs = trump_lp.exp().squeeze(0).cpu().numpy()
            flag_probs = torch.sigmoid(flag_logits).squeeze(0).cpu().numpy()
        return trump_probs, flag_probs
