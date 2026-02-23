"""Inference wrappers for UltiNet — numpy API for MCTS / training.

``UltiNetWrapper`` bridges PyTorch ↔ numpy for MCTS self-play.
``make_wrapper`` creates a wrapper for the given device.
"""
from __future__ import annotations

import numpy as np
import torch

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
      - ``batch_bid_value(state_batch) → np.ndarray``
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

    def batch_bid_value(self, states: np.ndarray) -> np.ndarray:
        """Evaluate the dedicated bid value head on a batch of states."""
        self.net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(states).float().to(self.device)
            values = self.net.forward_bid_value(x)
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


def make_wrapper(net: UltiNet, device: str = "cpu") -> UltiNetWrapper:
    """Create an inference wrapper for the given device."""
    return UltiNetWrapper(net, device=device)
