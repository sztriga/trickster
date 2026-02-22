#!/usr/bin/env bash
# train_grid.sh — Sequential grid-search training.
# Intended for AWS EC2 (multi-core CPU instance).
#
# Usage:
#   chmod +x train_grid.sh
#   ./train_grid.sh
set -euo pipefail

# ── Setup ──
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

WORKERS=$(python3 -c "import os; print(max(1, os.cpu_count() - 1))")
echo "=== Train grid: $WORKERS workers ==="

TIERS=(scout knight bishop bronze trinity)

# ── Train each tier sequentially ──
for tier in "${TIERS[@]}"; do
    echo ""
    echo "=== Training: $tier ==="
    python3 scripts/train_e2e.py "$tier" --workers "$WORKERS"
done

# ── Commit & push ──
echo ""
echo "=== Committing models ==="
git add models/
git commit -m "Grid search: ${TIERS[*]} ($(date +%Y-%m-%d))"
git push

echo "=== Done ==="
