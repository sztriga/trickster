# Trickster

AI framework for **Snapszer** (Hungarian Schnapsen) — a two-player imperfect-information trick-taking card game.

Train AlphaZero-style agents using MCTS + neural networks, then play against them in a React web UI with live analysis mode.

## Quick Start

```bash
# Install
pip install -e .
cd apps/web && npm install

# Run
uvicorn apps.api.main:app --reload   # backend
cd apps/web && npm run dev            # frontend (dev)
```

Open http://localhost:5173, press ESC to pick an opponent, and play.

## What's Inside

- **AlphaZero training** with MCTS, determinization for imperfect info, hybrid bootstrap
- **Pure NumPy** neural networks (no PyTorch/TF) — dual-head SharedAlphaNet (value + policy)
- **React web UI** with live play, speech bubbles, analysis mode (progressive MCTS evaluation)
- **Strength ladder** — automated multi-tier training pipeline (T0-Direct through T8-Marshal)
- **25-action space**: 20 cards + close talon + 4 marriage declarations

## Trained Models

| Model | Architecture | Training Games |
|-------|-------------|----------------|
| T4-Bishop | 128×2/h64 | 15k |
| T5-Rook | 128×2/h64 | 20k |
| T6-Captain | 128×3/h64 | 48k |

## Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical details — game rules, neural network architecture, MCTS implementation, training pipeline, API endpoints, and design decisions.

## Training

```bash
# Train the full strength ladder
python scripts/train_ladder.py

# Train specific tiers
python scripts/train_ladder.py --tiers 4
```
