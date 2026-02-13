# Trickster — Ulti AI

An AlphaZero-style AI framework for **Ulti**, the Hungarian 3-player trick-taking card game. A neural network learns to play through hybrid self-play (MCTS + exact alpha-beta solver), then plays against humans in a React web UI.

The project also includes a fully playable **Snapszer** (2-player, 20-card) engine with trained models, though the primary focus is Ulti.

## Quick Start

```bash
# Install Python package
pip install -e .

# Build the Cython solver (one-time, requires cython)
pip install cython
python setup_cython.py build_ext --inplace

# Run the API server
PYTHONPATH=src uvicorn apps.api.ulti:app --reload --port 8000

# Run the frontend
cd apps/web && npm install && npm run dev
```

Open http://localhost:5173 to play.

---

## The Game of Ulti

Ulti is played with a 32-card Hungarian Tell deck by 3 players. One player (the **soloist**) bids a contract and plays alone against the other two (**defenders**) who form a silent coalition.

| Contract | Description |
|---|---|
| **Parti (Simple)** | Soloist must score more card points than the defenders combined |
| **Betli** | Soloist must take zero tricks (no trump suit) |
| **Durchmars** | Soloist must win all 10 tricks |
| **Ulti** | Soloist must win the last trick with the 7 of trumps |
| **40-100** | Soloist declares trump marriage (K+Q) and scores 100+ points |
| **20-100** | Soloist declares off-suit marriage and scores 100+ points |

Contracts can be compounded (e.g. "Parti + Ulti"), doubled with kontra/rekontra, played in Hearts (doubled stakes) or open (soloist's hand visible). A full game involves an auction phase where players bid contracts, followed by 10 tricks of play.

---

## Training Approach

Training happens in two separate stages. Only Stage 1 is currently implemented.

### Stage 1: Play Phase (current)

Teach the network to play tricks well for Parti contracts (all 4 trump suits including Hearts/red). The network learns from **hybrid self-play**: MCTS with neural guidance for the opening, exact alpha-beta solving for the endgame.

The game setup for training:
1. Deal 10+10+10 cards + 2-card talon
2. Soloist picks up talon (sees 12 cards), discards 2 via greedy heuristic
3. Trump chosen randomly from soloist's suits
4. Contract is always Parti (forced — no auction)

The discard heuristic is not being trained — it just produces reasonable starting positions. The talon's card points count for the defenders (matching real Ulti rules), and only the soloist knows which cards were discarded.

**Self-play with HybridPlayer:**

Each decision point uses one of two engines depending on game phase:

| Tricks remaining | Engine | Labels |
|---|---|---|
| 5-10 (opening) | Neural MCTS (20 sims soloist, 8 defenders) | Noisy (visit counts) |
| 1-4 (endgame) | PIMC + Cython alpha-beta (20 determinizations) | Exact |

The solver produces **exact game-theoretic values** for ~60% of training positions (tricks 5-10). The network gets clean, perfect targets for the endgame, while MCTS handles the imperfect-information opening where exact solving is too expensive.

A single `UltiNet` plays all 3 seats. The state encoder uses relative positioning (me / left / right), so the same weights work from any seat.

**Training loop:**
1. Self-play N games with HybridPlayer (all 3 seats)
2. Record `(state, legal_mask, policy_target, reward)` for every non-forced decision
3. Push samples into a replay buffer (reservoir sampling, outcome-balanced batching)
4. Train on mini-batches: `loss = MSE(value, outcome) + CrossEntropy(policy, target)`
5. Evaluate periodically against random opponents

```bash
# Quick training run
python scripts/train_baseline.py --steps 200

# Full training with checkpoints
python scripts/train_baseline.py \
    --steps 5000 \
    --games-per-step 4 \
    --sims 20 --def-sims 8 \
    --endgame-tricks 6 --pimc-dets 20 \
    --lr 1e-3 \
    --checkpoint-interval 500 \
    --eval-interval 100 \
    --eval-games 30

# Continue from checkpoint
python scripts/train_baseline.py --steps 5000 --load models/checkpoints/simple/step_02000.pt

# Parallel self-play
python scripts/train_baseline.py --workers 4 --games-per-step 16
```

### Stage 2: Auction + Discard (future)

Train the network to:
- **Evaluate hands**: Given the soloist's 12 cards + trump suit, predict win probability
- **Choose discards**: Pick the best 2 cards to discard from the 12-card hand
- **Bid contracts**: Decide which contract to play based on hand strength

This is a separate training process because the play network doesn't need to learn these skills — it only sees the post-discard 10-card game state.

**Key insight for discard training**: The training data must include bad discards (e.g. discarding Tens/Aces) so the network learns that giving away points is costly. The solver plays out games from different discard choices and the outcome teaches which discards are good or bad. The greedy heuristic used in Stage 1 always discards optimally, so it never generates these "don't do this" examples.

The infrastructure for this exists in `scripts/train_solver_nets.py` (hand evaluator + policy network trained from solver self-play) but is not yet connected to the main training pipeline.

---

## Architecture

### UltiNet (~325k parameters)

```
Input (259-dim state vector)
    |
    v
Shared Backbone (4 x 256 Linear + LayerNorm + ReLU)
    |
    +-- Policy Head  -> Linear(256, 32) -> LogSoftmax
    +-- Value Head   -> Linear(256, 1)  -> Tanh
    +-- Auction Head (for future use)
```

The network has role-specific heads for soloist and defender, routed via `forward_dual()`. The shared backbone learns features common to both roles (card counting, suit tracking), while separate heads specialize in their distinct objectives.

### Cython Alpha-Beta Solver

The endgame solver (`_solver_core.pyx`) provides exact play for the last N tricks:

| Tricks remaining | Cython solve time | Python fallback | Speedup |
|---|---|---|---|
| 7 | 6 ms | 879 ms | 143x |
| 6 | 0.66 ms | 93 ms | 141x |
| 5 | 0.031 ms | 3.8 ms | 120x |
| 4 | 0.009 ms | 0.76 ms | 89x |

Supports pluggable contract evaluators: parti, betli, durchmars, parti_ulti.

Build with: `python setup_cython.py build_ext --inplace`

### State Encoder (259 dimensions)

| Feature | Dims | Purpose |
|---|---|---|
| My hand | 32 | Binary bitmap of cards held |
| Per-player captured cards | 96 | Card counting — who took what |
| Current trick cards | 64 | What's on the table |
| Trump suit | 4 | One-hot (zeros for Betli) |
| Scalars | 6 | Betli flag, soloist, trick#, scores |
| Opponent void flags | 8 | Inferred from play |
| Marriage bitmask | 4 | K+Q pairs held |
| Seven-in-hand | 1 | Trump 7 (Ulti card) |
| Contract DNA | 8 | Component flags |
| Trick momentum | 20 | Leader + winner per trick |
| Auction context | 10 | Bid rank, seat, kontras |
| Marriage memory | 6 | Declared marriages |

Positions are encoded **relative to the current player** so the network learns position-invariant play.

### PIMC Determinization

For imperfect information, we use Perfect Information Monte Carlo:
1. Sample plausible opponent hands (respecting void + auction constraints)
2. Solve each determinization exactly (endgame) or search via MCTS (opening)
3. Average values across determinizations

---

## Replay Buffer

- **Reservoir sampling** (Vitter 1985): Equal retention probability across training history
- **Outcome-balanced batching**: 15%+ soloist-win samples per batch
- **Soloist-weighted sampling**: 3x weight (correcting for 2:1 defender ratio)

Optional **PFSP league training** (`--pool-interval`): defenders drawn from a checkpoint pool, weighted toward ~50% win rate opponents.

---

## Project Structure

```
src/trickster/
  games/
    ulti/                  # Ulti game engine
      game.py              # Core logic (deal, tricks, scoring)
      cards.py             # Card definitions, Tell deck
      adapter.py           # UltiGame — MCTS interface + PIMC
      encoder.py           # 259-dim state encoder
      auction.py           # Auction system (bidding rules)
      rules.py             # Follow-suit and trick-winning rules
    snapszer/              # Snapszer game engine (2-player)
    interface.py           # GameInterface protocol
  model.py                 # UltiNet (PyTorch)
  evaluator.py             # Oracle hand evaluator
  train_utils.py           # Rewards, replay buffer, PFSP pool
  mcts.py                  # MCTS with PIMC determinization
  solver.py                # Alpha-beta solver (Python fallback)
  hybrid.py                # HybridPlayer (MCTS + solver)
  _solver_core.pyx         # Cython alpha-beta solver

scripts/
  train_baseline.py        # Hybrid training (Parti)
  train_solver_nets.py     # Hand evaluator + policy from solver data
  test_cython_solver.py    # Solver verification + benchmark
  eval_head2head.py        # Head-to-head model comparison

apps/
  api/ulti.py              # FastAPI backend (Ulti)
  api/main.py              # FastAPI backend (Snapszer)
  web/                     # React + TypeScript frontend

models/                    # Trained model checkpoints
```
