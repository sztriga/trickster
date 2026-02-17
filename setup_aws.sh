#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Trickster — Setup"
echo "============================================"
echo ""

# ── Detect OS ────────────────────────────────────────
OS="$(uname -s)"

# ── System dependencies ──────────────────────────────
echo "[1/6] Installing system dependencies..."
if [ "$OS" = "Linux" ]; then
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-pip python3-venv python3-dev gcc git-lfs > /dev/null 2>&1
        echo "  Done (apt)."
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y -q python3.11 python3.11-pip python3.11-devel gcc git-lfs > /dev/null 2>&1
        # Make python3.11 the default python3 if system python is too old
        if python3 --version 2>&1 | grep -qE '3\.(8|9)'; then
            sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 2>/dev/null || true
        fi
        echo "  Done (dnf, Python 3.11)."
    elif command -v yum &> /dev/null; then
        sudo yum install -y -q python3.11 python3.11-pip python3.11-devel gcc git-lfs > /dev/null 2>&1
        if python3 --version 2>&1 | grep -qE '3\.(8|9)'; then
            sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 2>/dev/null || true
        fi
        echo "  Done (yum, Python 3.11)."
    else
        echo "  WARNING: No supported package manager found. Install python3-dev, gcc, and git-lfs manually."
    fi
    # Initialize Git LFS hooks
    git lfs install > /dev/null 2>&1 || true
elif [ "$OS" = "Darwin" ]; then
    # macOS: assume Xcode CLI tools are installed (provides gcc/clang)
    if ! command -v python3 &> /dev/null; then
        echo "  ERROR: python3 not found. Install via brew: brew install python"
        exit 1
    fi
    echo "  Done (macOS — using system toolchain)."
else
    echo "  WARNING: Unknown OS '$OS'. Skipping system deps."
fi

# ── Virtual environment ──────────────────────────────
echo "[2/6] Creating virtual environment..."
# Prefer python3.11 if available (Amazon Linux ships with 3.9)
PY=$(command -v python3.11 || command -v python3)
if [ ! -d ".venv" ]; then
    $PY -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
echo "  Done. Using $(python3 --version)"

# ── Python dependencies ──────────────────────────────
echo "[3/6] Installing Python packages..."
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy cython fastapi uvicorn onnxruntime
pip install -e . -q
echo "  Done."

# ── Cython extensions ─────────────────────────────────
echo "[4/6] Building Cython extensions (Ulti solver + Snapszer minimax)..."
python3 setup_cython.py build_ext --inplace 2>&1 | tail -5
echo ""

# ── Verify ───────────────────────────────────────────
echo "[5/6] Running verification..."
FAIL=0

python3 -c "import torch; print(f'  PyTorch {torch.__version__}: OK')" || FAIL=1
python3 -c "import numpy; print(f'  NumPy {numpy.__version__}: OK')" || FAIL=1
python3 -c "import onnxruntime as ort; print(f'  ONNX Runtime {ort.__version__}: OK')" || FAIL=1
python3 -c "from trickster._solver_core import solve_root; print('  Cython solver (Ulti): OK')" || FAIL=1
python3 -c "from trickster.games.snapszer._fast_minimax import c_alphabeta; print('  Cython solver (Snapszer): OK')" || FAIL=1
python3 -c "
from trickster.training.tiers import TIERS
from trickster.training.model_io import resolve_paths
from trickster.training.contract_loop import train_one_tier
from trickster.training.bidding_loop import train_with_bidding
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
print(f'  Training modules: OK ({len(TIERS)} tiers, solver={SOLVER_ENGINE})')
" || FAIL=1

# Git LFS
if command -v git-lfs &> /dev/null || git lfs version &> /dev/null; then
    echo "  Git LFS: OK ($(git lfs version 2>&1 | head -1))"
else
    echo "  Git LFS: NOT FOUND — model push will fail without it"
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "  ERROR: Some checks failed. See above."
    exit 1
fi

CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")

echo ""
echo "============================================"
echo "  Setup complete! ($CORES cores detected)"
echo "============================================"
echo ""
echo "  IMPORTANT: activate the venv first (setup can't do this for you):"
echo "    source .venv/bin/activate"
echo ""
echo "  Train:"
echo "    python3 scripts/train_e2e.py scout --workers $CORES"
echo "    python3 scripts/train_e2e.py knight --workers $CORES"
echo "    python3 scripts/train_e2e.py bishop --workers $CORES"
echo ""
echo "  Evaluate:"
echo "    python3 scripts/eval_bidding.py --seats bishop knight knight --games 10000 --workers $CORES"
echo ""
echo "  Push trained models (Git LFS handles the heavy files):"
echo "    git add models/"
echo "    git commit -m 'trained <tier> on AWS'"
echo "    git push"
echo ""
