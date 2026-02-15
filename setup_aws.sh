#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Trickster — AWS Setup"
echo "============================================"
echo ""

# ── System dependencies ──────────────────────────────
echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv python3-dev gcc > /dev/null 2>&1
echo "  Done."

# ── Virtual environment ──────────────────────────────
echo "[2/4] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
echo "  Done. Using $(python3 --version)"

# ── Python dependencies ──────────────────────────────
echo "[3/4] Installing Python packages..."
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy cython fastapi uvicorn
pip install -e . -q
echo "  Done."

# ── Cython solver ────────────────────────────────────
echo "[4/4] Building Cython alpha-beta solver..."
python3 setup_cython.py build_ext --inplace 2>&1 | tail -1
python3 -c "from trickster._solver_core import solve_root; print('  Cython solver: OK')"

# ── Verify ───────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Example commands:"
echo ""
echo "  # Train scout (fast, ~3 min)"
echo "  python3 scripts/train_e2e.py scout --workers \$(nproc)"
echo ""
echo "  # Train knight (~25 min)"
echo "  python3 scripts/train_e2e.py knight --workers \$(nproc)"
echo ""
echo "  # Train bishop (~1h)"
echo "  python3 scripts/train_e2e.py bishop --workers \$(nproc)"
echo ""
echo "  # Evaluate bishop vs knight"
echo "  python3 scripts/eval_bidding.py --seats bishop knight knight --games 10000 --workers \$(nproc)"
echo ""
echo "  # Evaluate with deeper search"
echo "  python3 scripts/eval_bidding.py --seats bishop knight knight --games 2000 --speed normal --workers \$(nproc)"
echo ""
