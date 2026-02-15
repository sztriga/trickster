#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Trickster — Setup"
echo "============================================"
echo ""

# ── Detect OS ────────────────────────────────────────
OS="$(uname -s)"

# ── System dependencies ──────────────────────────────
echo "[1/5] Installing system dependencies..."
if [ "$OS" = "Linux" ]; then
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-pip python3-venv python3-dev gcc > /dev/null 2>&1
        echo "  Done (apt)."
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y -q python3-pip python3-devel gcc > /dev/null 2>&1
        echo "  Done (dnf)."
    elif command -v yum &> /dev/null; then
        sudo yum install -y -q python3-pip python3-devel gcc > /dev/null 2>&1
        echo "  Done (yum)."
    else
        echo "  WARNING: No supported package manager found. Install python3-dev and gcc manually."
    fi
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
echo "[2/5] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
echo "  Done. Using $(python3 --version)"

# ── Python dependencies ──────────────────────────────
echo "[3/5] Installing Python packages..."
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy cython fastapi uvicorn
pip install -e . -q
echo "  Done."

# ── Cython solver ────────────────────────────────────
echo "[4/5] Building Cython alpha-beta solver..."
python3 setup_cython.py build_ext --inplace 2>&1 | tail -3
echo ""

# ── Verify ───────────────────────────────────────────
echo "[5/5] Running verification..."
FAIL=0

python3 -c "import torch; print(f'  PyTorch {torch.__version__}: OK')" || FAIL=1
python3 -c "import numpy; print(f'  NumPy {numpy.__version__}: OK')" || FAIL=1
python3 -c "from trickster._solver_core import solve_root; print('  Cython solver: OK')" || FAIL=1
python3 -c "
from trickster.training.tiers import TIERS
from trickster.training.model_io import resolve_paths
from trickster.training.contract_loop import train_one_tier
from trickster.training.bidding_loop import train_with_bidding
from trickster.hybrid import HybridPlayer, SOLVER_ENGINE
print(f'  Training modules: OK ({len(TIERS)} tiers, solver={SOLVER_ENGINE})')
" || FAIL=1

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
echo "  Activate the environment:"
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
