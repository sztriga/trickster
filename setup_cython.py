"""Build script for the Cython alpha-beta solver extension.

Usage:
    pip install cython                              # one-time
    python setup_cython.py build_ext --inplace      # compile

The compiled extension (_solver_core.pyd on Windows, .so on Linux)
is placed next to the .pyx source in src/trickster/.

After building, verify with:
    python -c "from trickster._solver_core import solve_root; print('OK')"
"""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="trickster._solver_core",
        sources=["src/trickster/_solver_core.pyx"],
    ),
]

setup(
    name="trickster-solver-ext",
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
    packages=["trickster"],
)
