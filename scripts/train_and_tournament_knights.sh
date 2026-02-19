#!/usr/bin/env bash
# Train all four knight variants (knight_light, knight_balanced, knight_heavy, knight_pure)
# sequentially, then run a round-robin tournament between them.
#
# Usage:
#   ./scripts/train_and_tournament_knights.sh
#   ./scripts/train_and_tournament_knights.sh --workers 6
#   ./scripts/train_and_tournament_knights.sh --workers 6 --games 300
#
# Options passed to both training and tournament:
#   --workers N    (default: 6 for training, 6 for tournament)
#   --games N      tournament deals per matchup (default: 1000)
#   --verbose      pass -v to training
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

WORKERS=6
GAMES=1000
VERBOSE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --games)
      GAMES="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE="-v"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--workers N] [--games N] [--verbose]"
      echo ""
      echo "  Train knight_light, knight_balanced, knight_heavy, knight_pure,"
      echo "  then run a round-robin tournament between them."
      echo ""
      echo "  --workers N   Workers for training and tournament (default: 6)"
      echo "  --games N     Tournament deals per matchup (default: 300)"
      echo "  --verbose     Verbose training output"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run $0 --help for usage."
      exit 1
      ;;
  esac
done

KNIGHTS="knight_light knight_balanced knight_heavy knight_pure"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TRAIN ALL KNIGHTS + TOURNAMENT                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Tiers: $KNIGHTS"
echo "  Workers: $WORKERS"
echo "  Tournament games per matchup: $GAMES"
echo ""

# Phase 1: Train all knight tiers
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1 — TRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/train_e2e.py $KNIGHTS --workers "$WORKERS" $VERBOSE

# Phase 2: Tournament
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2 — TOURNAMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/tournament.py $KNIGHTS --games "$GAMES" --workers "$WORKERS"

echo ""
echo "  Done."
