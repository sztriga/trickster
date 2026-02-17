"""Re-export shim — canonical location is ``trickster.games.ulti.solver``."""
from trickster.games.ulti.solver import *  # noqa: F401, F403
from trickster.games.ulti.solver import (  # noqa: F401 — explicit for type-checkers
    SolverPIMC,
    SolveStats,
    get_node_count,
    get_solve_stats,
    solve_best,
    solve_root,
)
