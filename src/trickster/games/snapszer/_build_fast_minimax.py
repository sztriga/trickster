"""Build the Cython-accelerated minimax module.

Usage:
    python src/trickster/games/snapszer/_build_fast_minimax.py
"""
import os
import sys

from Cython.Build import cythonize
from setuptools import Extension, Distribution

src_dir = os.path.dirname(os.path.abspath(__file__))
pyx_path = os.path.join(src_dir, "_fast_minimax.pyx")

# We need to be in the 'src' directory so inplace build puts the .so next to the .pyx
src_root = os.path.join(src_dir, "..", "..", "..", "..")
src_root = os.path.normpath(src_root)  # project root
os.chdir(os.path.join(src_root, "src"))

ext = Extension(
    "trickster.games.snapszer._fast_minimax",
    sources=[os.path.relpath(pyx_path)],
)

dist = Distribution({"ext_modules": cythonize([ext], language_level=3)})

cmd = dist.get_command_obj("build_ext")
cmd.inplace = True
cmd.ensure_finalized()
cmd.run()

print("Build complete!")
