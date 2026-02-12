"""Centralized constants and defaults for Snapszer search & play.

Every tunable default lives here.  Import from this module instead
of hardcoding magic numbers elsewhere.

Usage::

    from trickster.games.snapszer.constants import (
        DEFAULT_PIMC_SAMPLES,
        DEFAULT_LATE_THRESHOLD,
        DEFAULT_MCTS_SIMS,
        DEFAULT_MCTS_DETS,
    )
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#  MCTS defaults (used by the web app and evaluation scripts)
# ---------------------------------------------------------------------------

DEFAULT_MCTS_SIMS: int = 50
"""MCTS simulations per determinization."""

DEFAULT_MCTS_DETS: int = 6
"""MCTS determinizations (number of sampled worlds per move)."""

# ---------------------------------------------------------------------------
#  PIMC / Hybrid search defaults
# ---------------------------------------------------------------------------

DEFAULT_PIMC_SAMPLES: int = 30
"""Number of worlds to sample in PIMC.

At trick 2 (70 possible worlds) 30 samples gives ~43% coverage.
At trick 3 (15 possible worlds) 30 samples is exhaustive.
"""

DEFAULT_LATE_THRESHOLD: int = 8
"""Talon cards remaining at which we switch from MCTS to PIMC.

* talon > 8  (trick 1 only) -> MCTS  — high uncertainty, NN intuition.
* talon <= 8 (tricks 2-5)   -> PIMC  — sampled minimax, ~220ms at trick 2.
* phase 2    (talon gone)    -> Pure Minimax — exact, <0.01ms.

Cython-accelerated alpha-beta benchmarks (~130x faster than Python):
  talon=10 (trick 1):  ~115ms/solve -> PIMC 30s ~ 2.1s
  talon=8  (trick 2):  ~10ms/solve  -> PIMC 30s ~ 220ms
  talon=6  (trick 3):  ~1.5ms/solve -> PIMC 30s ~ 28ms
  talon=4  (trick 4):  ~0.2ms/solve -> PIMC 30s ~ 4.6ms
  talon=2  (trick 5):  ~0.05ms/solve -> PIMC 30s ~ 0.8ms
"""

# ---------------------------------------------------------------------------
#  Evaluation defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL_SIMS: int = 50
"""MCTS simulations per move during evaluation."""

DEFAULT_EVAL_DETS: int = 4
"""MCTS determinizations during evaluation."""

DEFAULT_EVAL_DEALS: int = 200
"""Number of deals per matchup in round-robin evaluation."""
