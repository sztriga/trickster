"""Re-export shim — canonical location is ``trickster.games.ulti.hybrid``."""
from trickster.games.ulti.hybrid import *  # noqa: F401, F403
from trickster.games.ulti.hybrid import (  # noqa: F401 — explicit for type-checkers
    SOLVER_ENGINE,
    HybridPlayer,
)
