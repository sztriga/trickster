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

#: Minimum expected per-defender game points to place a bid.
#: Passed as ``min_bid_pts`` to ``run_auction`` and bidding training.
MIN_BID_PTS: float = 0.0

#: Starting bid threshold (annealed down to MIN_BID_PTS during training).
MIN_BID_PTS_START: float = 4.0

#: Per-defender penalty (game points) when all three players pass.
PASS_PENALTY: float = 2.0

# ---------------------------------------------------------------------------
#  Training exploration
# ---------------------------------------------------------------------------

#: Fraction of bidding-training games that use a random contract
#: (instead of the NN's choice) for exploration.
EXPLORATION_FRAC: float = 0.2
