# Trickster — Ulti AI

An end-to-end AI for **Ulti**, the Hungarian 3-player trick-taking card game. Neural networks learn to bid contracts, discard cards, play tricks, and decide kontra/rekontra — all through self-play. The AI plays against humans in a React web UI.

The project also includes a fully playable **Snapszer** (2-player, 20-card) engine with trained models.

## Quick Start

```bash
pip install -e .

# Build the Cython solver (requires cython)
pip install cython
python setup_cython.py build_ext --inplace

# Train end-to-end (base models + bidding in one pipeline)
python scripts/train_e2e.py knight_light --workers 6

# Evaluate
python scripts/eval_bidding.py --seats knight_light knight_light scout --games 2000 --workers 6

# Play in the browser
PYTHONPATH=src uvicorn apps.api.ulti:app --reload --port 8000
cd apps/web && npm install && npm run dev
```

Open http://localhost:5173 to play.

---

## The Game of Ulti

Ulti is played with a 32-card Hungarian Tell deck by 3 players. One player (the **soloist**) bids a contract and plays alone against the other two (**defenders**) who form a silent coalition.

### Contracts

Playable contracts in ascending bid strength:

| Contract | Rules | Win / Loss (per defender) |
|---|---|---|
| **Piros Parti** | Hearts trump, soloist scores more card points than defenders | +2 / −2 |
| **40-100** | Soloist has trump K+Q, must reach 100+ card points | +4 / −4 |
| **Ulti** | Parti + soloist must win last trick with trump 7 | +5 / −9 (combined) |
| **Betli** | No trump, soloist must take zero tricks | +5 / −5 |
| **Piros 40-100** | 40-100 with Hearts trump (2x stakes) | +8 / −8 |
| **Piros Ulti** | Ulti with Hearts trump (2x stakes) | +10 / −18 |
| **Piros Betli** | Betli with 2x stakes (Rebetli) | +10 / −10 |

Plain Parti (non-red) cannot be played. If the soloist cannot bid anything better, they **pass** and pay 2 points to each defender.

### Compound Contracts

Ulti is a compound contract: Parti and Ulti are evaluated independently. The soloist can win Parti but lose Ulti (or vice versa). Each component has its own scoring.

### Silent Bonuses

Even when not announced, certain outcomes trigger half-value bonuses:
- **Silent 40-100 / 20-100**: Replaces Parti when soloist (or defenders) reach 100+ with a marriage
- **Silent Ulti**: Trump 7 played on the last trick (±2 win / ±4 loss)
- **Silent Durchmars**: One side takes all 10 tricks (±3)

### Kontra & Rekontra

After the first trick is played:
1. Each **defender** sees the soloist's card and can **kontra** (2x stakes)
2. The **soloist** can respond with **rekontra** (4x stakes)

For trump (adu) games, defenders share a single kontra decision. For Betli, each defender kontras independently.

**Bukott (Fallen) Ulti** — special rule: when Ulti is kontrad and lost, the penalty is `win_value × (2^k + 1)` instead of the standard `loss_value × 2^k`. This means kontra on Ulti costs 12 (not 16) and rekontra costs 20 (not 32).

### Piros (Red)

Hearts trump doubles all stakes. The strategy doesn't change (same trump mechanics), but the risk/reward is doubled. The AI learns piros dynamics directly through the value head.

---

## Training

Training is a unified two-phase pipeline via `train_e2e.py`.

### Usage

```bash
# Full pipeline (Phase 1 + Phase 2)
python scripts/train_e2e.py knight_light --workers 6

# Skip Phase 1, use existing base models (--base-from <tier>)
python scripts/train_e2e.py knight_light --base-from knight_light --workers 6
python scripts/train_e2e.py silver --base-from bronze --workers 6  # use bronze base

# Verbose output (per-step details)
python scripts/train_e2e.py knight_light --workers 6 -v

# Override bidding steps
python scripts/train_e2e.py knight_light --workers 6 --bidding-steps 4000
```

### Phase 1: Base Contract Training

Each contract type (parti, ulti, betli, 40-100) is trained independently via hybrid self-play:
1. Deal cards, pick up talon (soloist sees 12 cards)
2. Neural discard: value head evaluates discard pairs, picks the best
3. Play 10 tricks via `HybridPlayer` (MCTS for opening, alpha-beta solver for endgame)
4. Value-head-driven kontra/rekontra after trick 1
5. Record `(state, policy_target, reward)` for every decision
6. Train on mini-batches: `loss = Huber(value, outcome) + CrossEntropy(policy, target)`

Models are saved to `models/<contract>/<prefix><tier>/model.pt`.

### Phase 2: End-to-End Bidding Training

All contracts train together. The value heads decide which contract to bid each deal:
1. Deal 12 cards to the soloist
2. Each contract model evaluates the hand (value head predicts expected game-point stakes)
3. Softmax-temperature sampling picks a contract (annealed from exploratory to greedy)
4. The chosen contract's model trains on the game result
6. All models improve on hands they'd actually be bid on — realistic training distribution

Models are saved to `models/e2e/<tier>/`.

### Tiers

Each tier defines the full training budget — network size, MCTS search depth, game count. All tiers are defined in `src/trickster/training/tiers.py`.

| Tier | Net | Base games | E2E games | MCTS (sol/def) |
|---|---|---|---|---|
| Scout | 256×4 | 4k | 8k | 40 / 20 |
| Knight Light | 256×4 | 16k | 16k | 60 / 24 |
| Bishop | 384×4 | 24k | 24k | 80 / 30 |
| Rook | 512×6 | 24k | 24k | 80 / 30 |

Additional tiers (bronze, silver, gold, hawk, eagle, falcon, trinity, morpheus) explore volume, search depth, and hybrid dimensions.

### Model Naming

| Key | Description | Path |
|---|---|---|
| `knight_light_base` | Pre-trained Knight Light models (Phase 1) | `models/<contract>/X2-Knight-Light/` |
| `knight_light` | Bidding-trained Knight Light models (Phase 2) | `models/e2e/knight_light/` |

---

## Evaluation

Full end-to-end evaluation with 3 seats, bidding, kontra, and mixed model tiers:

```bash
# E2E models head-to-head
python scripts/eval_bidding.py --seats knight_light knight_light scout --games 2000 --workers 6

# Different search speeds
python scripts/eval_bidding.py --seats knight_light:deep knight_light:fast scout:normal --games 2000 --workers 6

# Global speed override
python scripts/eval_bidding.py --seats knight_light knight_light scout --speed normal --games 2000 --workers 6
```

Speed presets (estimated for 2000 deals @ 6 workers):

| Speed | MCTS sims | Dets | PIMC | Endgame | Time |
|---|---|---|---|---|---|
| fast | 40 | 2 | 20 | 6 tricks | ~1 min |
| normal | 80 | 3 | 25 | 6 tricks | ~5 min |
| deep | 200 | 6 | 50 | 7 tricks | ~15 min |

---

## Architecture

### UltiNet (~333k parameters)

```
Input (state vector)
    |
    v
Shared Backbone (4 x 256 Linear + LayerNorm + ReLU)
    |
    +-- Policy Head (sol)  -> Linear(256, 32) -> LogSoftmax
    +-- Policy Head (def)  -> Linear(256, 32) -> LogSoftmax
    +-- Value Head (sol)   -> Linear(256, 1)
    +-- Value Head (def)   -> Linear(256, 1)
```

- **Role-specific heads**: Soloist and defender have separate policy and value heads, routed by `forward_dual()`. The shared backbone learns features common to both roles.
- **Unbounded value head**: No tanh — the value head is a linear regressor that predicts expected game-point stakes directly (including piros multiplier and kontra dynamics). Gradient clipping provides stability.
- **Huber loss**: Robust to the large outlier rewards of kontrad piros games (e.g. a rekontra'd piros ulti loss can be −5.6 normalised).

### State Encoder

| Feature | Dims | Purpose |
|---|---|---|
| My hand | 32 | Binary bitmap of cards held |
| Per-player captured cards | 96 | Card counting — who took what |
| Current trick cards | 64 | What's on the table |
| Trump suit | 4 | One-hot (zeros for Betli) |
| Scalars | 6 | Betli flag, is_soloist, trick#, scores |
| Opponent void flags | 8 | Inferred from play |
| Marriage bitmask | 4 | K+Q pairs held |
| Seven-in-hand | 1 | Trump 7 (Ulti card) |
| Contract DNA | 8 | Component flags (parti, ulti, betli, 40, 100, etc.) |
| Trick momentum | 20 | Leader + winner per trick |
| Auction context | 10 | Bid rank, seat, kontras |
| Marriage memory | 6 | Declared marriages |
| Is-red flag | 1 | Piros game indicator |

All positions are encoded **relative to the current player** (me / left / right) so one network plays all 3 seats.

### HybridPlayer (MCTS + Alpha-Beta Solver)

| Tricks remaining | Engine | Signal quality |
|---|---|---|
| 5-10 (opening) | Neural MCTS with policy priors + value head | Noisy (visit distribution) |
| 1-4 (endgame) | PIMC + Cython alpha-beta solver | Exact |

The Cython solver provides exact game-theoretic values for ~60% of training positions.

### Cython Alpha-Beta Solver

The endgame solver (`_solver_core.pyx`) supports pluggable contract evaluators:

| Tricks remaining | Cython time | Python fallback | Speedup |
|---|---|---|---|
| 7 | 6 ms | 879 ms | 143x |
| 6 | 0.66 ms | 93 ms | 141x |
| 5 | 0.031 ms | 3.8 ms | 120x |

Build: `python setup_cython.py build_ext --inplace`

### PIMC Determinization

For imperfect information, Perfect Information Monte Carlo:
1. Sample plausible opponent hands (respecting void + auction constraints)
2. Solve each determinization exactly (endgame) or search via MCTS (opening)
3. Average values across determinizations

### Bidding System

No separate bidding neural network. The play-phase value heads evaluate contract feasibility:
1. For each contract x trump suit x discard pair, encode the resulting game state
2. Batch-evaluate via the contract model's value head
3. Pick the (contract, trump, discard) with the highest expected stakes
4. Bid if expected value beats the pass penalty (-2/defender)

### Replay Buffer

- **Reservoir sampling** (Vitter 1985): Equal retention probability across training history
- **Outcome-balanced batching**: 15%+ soloist-win samples per batch
- **Soloist-weighted sampling**: 3x weight (correcting for the 2:1 defender ratio in self-play)

---

## Scoring Engine

`simple_outcome()` in `train_utils.py` computes normalised rewards:

- Per-component evaluation (Parti and Ulti scored independently)
- Kontra multipliers from `state.component_kontras`
- Piros multiplier (x2) applied to all components
- Silent bonuses at half value (40-100, 20-100, Ulti, Durchmars)
- Bukott Ulti special formula: `win_value x (2^k + 1)`
- Normalised by `_GAME_PTS_MAX = 10`

The same scoring engine is used for training, evaluation, and the UI.

---

## Scripts

| Script | Purpose | Example |
|---|---|---|
| `train_e2e.py` | Full training pipeline (base + bidding) | `python scripts/train_e2e.py knight_light --workers 6` |
| `eval_bidding.py` | 3-seat end-to-end evaluation | `python scripts/eval_bidding.py --seats knight_light knight_light scout --games 2000 --workers 6` |

---

## Project Structure

```
src/trickster/
  model.py                 # UltiNet (PyTorch) — dual-head architecture
  train_utils.py           # Scoring engine, replay buffer
  mcts.py                  # MCTS with PIMC determinization
  hybrid.py                # HybridPlayer (MCTS + solver)
  solver.py                # Alpha-beta solver (Python fallback)
  _solver_core.pyx         # Cython alpha-beta solver (143x faster)

  training/
    tiers.py               # Tier and contract definitions (hyperparameters)
    contract_loop.py       # Single-contract training loop (Phase 1)
    bidding_loop.py        # Multi-contract bidding training loop (Phase 2)
    model_io.py            # Model loading, path conventions, display labels
    progress.py            # Training progress callbacks (bar + verbose)
    ulti_hybrid.py         # Core hybrid self-play engine

  bidding/
    registry.py            # Contract definitions (ContractDef, bid ranks)
    evaluator.py           # Hand evaluation using value heads

  games/
    ulti/
      game.py              # Core game logic (deal, tricks, scoring)
      cards.py             # 32-card Hungarian Tell deck
      adapter.py           # UltiGame — MCTS interface + PIMC
      encoder.py           # State encoder (relative positioning)
      rules.py             # Follow-suit and trick-winning rules
      auction.py           # Auction system (bidding rules)
    snapszer/              # Snapszer game engine (2-player)
    interface.py           # GameInterface protocol

scripts/
  train_e2e.py             # Full training pipeline (base + bidding)
  eval_bidding.py          # 3-seat evaluation

apps/
  api/ulti.py              # FastAPI backend (Ulti)
  api/main.py              # FastAPI backend (Snapszer)
  web/                     # React + TypeScript frontend

models/
  e2e/
    knight_light/         # E2E trained models
  snapszer/                # Snapszer models
```
