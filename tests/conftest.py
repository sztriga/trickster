from __future__ import annotations

import sys
from pathlib import Path


# Allow `import trickster` when running tests without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

