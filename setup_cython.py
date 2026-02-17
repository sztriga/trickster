"""Build script for all Cython extensions.

Usage:
    pip install cython                              # one-time
    python setup_cython.py build_ext --inplace      # compile

Extensions built:
    trickster._solver_core              — Ulti alpha-beta endgame solver
    trickster.games.snapszer._fast_minimax — Snapszer alpha-beta + PIMC

After building, verify with:
    python -c "from trickster._solver_core import solve_root; print('OK')"
    python -c "from trickster.games.snapszer._fast_minimax import c_alphabeta; print('OK')"
"""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="trickster._solver_core",
        sources=["src/trickster/_solver_core.pyx"],
    ),
    Extension(
        name="trickster.games.snapszer._fast_minimax",
        sources=["src/trickster/games/snapszer/_fast_minimax.pyx"],
    ),
]

setup(
    name="trickster-cython-ext",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "language_level": "3",
        },
    ),
    package_dir={"": "src"},
    packages=["trickster", "trickster.games.snapszer"],
)
