"""Central bidding constants.

All bidding-related thresholds and parameters live here so that
training, evaluation, tournament, and API code stay in sync.
"""

# ---------------------------------------------------------------------------
#  Kontra / Rekontra thresholds (normalised value-head output)
# ---------------------------------------------------------------------------

#: Defender kontras if their value prediction exceeds this.
#: 0.4 normalised ≈ 2 game points.
KONTRA_THRESHOLD: float = 0.4

#: Soloist re-kontras if their value prediction exceeds this.
#: 0.8 normalised ≈ 4 game points.
REKONTRA_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
#  Bid / pickup evaluation
# ---------------------------------------------------------------------------

#: Per-defender penalty (game points) when all three players pass.
PASS_PENALTY: float = 2.0

#: Minimum expected per-defender game points to place a bid.
#: Set to -PASS_PENALTY so the first bidder bids whenever the expected
#: outcome is better than the guaranteed pass penalty.
MIN_BID_PTS: float = -PASS_PENALTY

# ---------------------------------------------------------------------------
#  Softmax temperature for contract selection during training
# ---------------------------------------------------------------------------

#: Starting temperature (high → exploratory).
BID_TEMP_START: float = 2.0

#: Final temperature (low → greedy).
BID_TEMP_END: float = 0.1


# ---------------------------------------------------------------------------
#  Per-contract pickup quantile overrides
# ---------------------------------------------------------------------------

#: Override the global pickup_quantile for specific contracts.
#: The winning contract (by vote across talon samples) is determined first,
#: then the quantile value is recomputed using the override if present.
PICKUP_QUANTILE_OVERRIDES: dict[str, float] = {
    "betli": 0.25,  # More conservative for betli
}

# ---------------------------------------------------------------------------
#  Display key helpers
# ---------------------------------------------------------------------------

def _display_key(contract_key: str, is_piros: bool) -> str:
    """Create a display key that distinguishes red from non-red."""
    return f"p.{contract_key}" if is_piros else contract_key


def _model_key(display_key: str) -> str:
    """Strip the piros prefix to get the model/buffer key."""
    return display_key.removeprefix("p.")
