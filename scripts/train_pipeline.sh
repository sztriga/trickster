#!/usr/bin/env bash
#
# Full training pipeline: scout → knight → bishop → bronze → trinity
# then round-robin tournament of all tiers.
#
# Each tier warm-starts from the previous one where architecturally
# compatible (same network size).  Larger nets start from scratch.
# Pure self-play — no opponent pooling.
#
# Usage:
#   ./scripts/train_pipeline.sh              # default 6 workers
#   ./scripts/train_pipeline.sh 4            # custom worker count
#   GAMES=300 ./scripts/train_pipeline.sh    # custom tournament games

set -euo pipefail

WORKERS="${1:-6}"
TOURNEY_GAMES="${GAMES:-500}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ULTI TRAINING PIPELINE                                     ║"
echo "║  scout → knight → bishop → bronze → trinity + tournament    ║"
echo "║  Workers: $WORKERS                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Scout: from scratch, pure self-play ──
echo "━━━ Stage 1/5: scout (from scratch) ━━━"
python3 scripts/train_e2e.py scout --workers "$WORKERS"

# ── 2. Knight: warm-start from scout ──
echo "━━━ Stage 2/5: knight (from scout) ━━━"
python3 scripts/train_e2e.py knight \
    --from scout \
    --workers "$WORKERS"

# ── 3. Bishop: larger net (384×4), starts fresh ──
echo "━━━ Stage 3/5: bishop (fresh 384×4 net) ━━━"
python3 scripts/train_e2e.py bishop \
    --workers "$WORKERS"

# ── 4. Bronze: same arch as knight (256×4), warm-start, 2× data ──
echo "━━━ Stage 4/5: bronze (from knight) ━━━"
python3 scripts/train_e2e.py bronze \
    --from knight \
    --workers "$WORKERS"

# ── 5. Trinity: bishop arch (384×4), warm-start from bishop ──
echo "━━━ Stage 5/5: trinity (from bishop) ━━━"
python3 scripts/train_e2e.py trinity \
    --from bishop \
    --workers "$WORKERS"

# ── Tournament: all 5 tiers + random baseline ──
echo ""
echo "━━━ TOURNAMENT ━━━"
python3 scripts/tournament.py scout knight bishop bronze trinity random \
    --games "$TOURNEY_GAMES" \
    --workers "$WORKERS"

echo ""
echo "Pipeline complete."
