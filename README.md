# Trickster

AI framework for Hungarian trick-taking card games. Train AlphaZero-style agents using MCTS + neural networks, then play against them in a React web UI.

## Games

**Ulti** (3-player, 32-card Tell deck) — the primary focus. Features asymmetric 1v2 play with a complex bidding system, contract types (Parti, Betli, Durchmars, Ulti), kontra/rekontra, and silent bonuses. See [ULTI_AI.md](ULTI_AI.md) for the training framework.

**Snapszer** (2-player, 20-card) — the original game. Fully playable with trained models up to T6-Captain tier.

## Quick Start

```bash
# Install Python package
pip install -e .

# Install frontend
cd apps/web && npm install

# Run the API server
PYTHONPATH=src uvicorn apps.api.main:app --reload --port 8000

# Run the frontend dev server
cd apps/web && npm run dev
```

Open http://localhost:5173 to play.

## Ulti AI

The Ulti AI uses PyTorch with a shared-backbone AlphaZero architecture (UltiNet, 259-dim input, 308k params). It learns through MCTS self-play with determinization for imperfect information.

```bash
# Train (mixed Simple + Betli curriculum)
python scripts/train_baseline.py --mode mixed --steps 600 --sims 20 --def-sims 8

# Evaluate two checkpoints head-to-head
python scripts/eval_head2head.py \
    --model-a models/checkpoints/mixed/step_00200.pt \
    --model-b models/checkpoints/mixed/step_00600.pt \
    --games 200

# Play against the trained model in the React UI
# (model auto-loads from models/ulti_mixed.pt)
```

See [ULTI_AI.md](ULTI_AI.md) for full documentation of the training pipeline, evaluation framework, and roadmap.

## Snapszer AI

```bash
# Train the strength ladder (T0 through T6)
python scripts/train_ladder.py

# Play in the web UI — press ESC to pick an opponent
```

## Project Structure

```
src/trickster/
  games/
    ulti/                # Ulti game engine
      game.py            # Core game logic (deal, tricks, scoring)
      cards.py           # Card definitions, Tell deck
      adapter.py         # UltiGame — MCTS interface + PIMC determinization
      encoder.py         # 259-dim "detective" state encoder
      auction_encoder.py # 116-dim auction phase encoder
    snapszer/            # Snapszer game engine
  model.py               # UltiNet (PyTorch) — policy + value + auction heads
  evaluator.py           # Oracle hand evaluator (shallow MCTS)
  train_utils.py         # Reward engine, replay buffer
  mcts.py                # MCTS with PIMC determinization

scripts/
  train_baseline.py      # Ulti curriculum training (simple/betli/mixed/auto)
  eval_head2head.py      # Head-to-head model comparison
  train_ladder.py        # Snapszer strength ladder

apps/
  api/
    main.py              # FastAPI backend (Snapszer)
    ulti.py              # FastAPI backend (Ulti) — serves trained AI
  web/                   # React + TypeScript frontend

models/                  # Trained model checkpoints
```
