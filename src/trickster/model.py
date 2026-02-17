"""UltiNet — PyTorch AlphaZero model for Ulti play + auction.

Architecture:
    Input → Shared Backbone (L × U + LayerNorm + ReLU)
            ├── Policy Head (sol/def)  → LogSoftmax
            ├── Value Head  (sol/def)  → unbounded scalar
            └── Auction Head (multi-component)

Inference wrappers (``UltiNetWrapper``, ``OnnxUltiWrapper``,
``make_wrapper``) live in ``trickster.wrappers`` and are re-exported
here for backward compatibility.
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
#  Re-exports from trickster.wrappers (backward compatibility)
# ---------------------------------------------------------------------------

from trickster.wrappers import (  # noqa: F401, E402
    OnnxUltiWrapper,
    UltiNetWrapper,
    _ort_available,
    make_wrapper,
)
