# Trickster — Architecture & Training Guide

## Overview

Trickster is a framework for training AI agents to play **Snapszer** (Hungarian Schnapsen), a two-player imperfect-information trick-taking card game. It combines AlphaZero-style self-play (MCTS + neural network) with determinization for imperfect information handling. The project includes a React web UI with live analysis mode and an automated training pipeline.

**Tech stack**: Pure NumPy neural networks (no PyTorch/TF), FastAPI backend, React + TypeScript frontend.

---

## The Game: Snapszer

- **Deck**: 20 cards — 4 Hungarian suits (Hearts/Piros, Bells/Tök, Leaves/Zöld, Acorns/Makk) × 5 ranks (Jack=2, Queen=3, King=4, Ten=10, Ace=11)
- **Deal**: Each player gets 5 cards; remaining 10 form the talon (draw pile) with one trump upcard
- **Play**: 10 tricks; after each trick both players draw back to 5 (winner first)
- **Scoring**: Card points equal their rank values; first to 66 wins immediately; game points (1–3) awarded based on margin
- **Two phases**:
  - **Open phase**: Talon is available, no obligation to follow suit
  - **Closed phase**: After talon is exhausted or closed — must follow suit, must beat if possible, must trump if can't follow
- **Special actions**:
  - **Exchange trump jack**: Swap the trump Jack (2 pts) for the higher-value upcard. Always beneficial — executed automatically in MCTS rollouts and by the AI during play.
  - **Close talon (betakarás)**: Leader declares no more drawing; strict follow/beat rules apply immediately. Closer must reach 66 or loses 2–3 game points to the opponent. This is a strategic gamble.
  - **Declare marriage (20/40)**: Announce King+Queen of same suit for 20 bonus points (40 if trump suit). Must then lead K or Q of that suit. This is a **learned decision** — the AI chooses whether and when to declare via MCTS.
- **Imperfect information**: Players can't see the opponent's hand or the draw pile order
- **Winning the match**: Game points accumulate across deals; first to 7 match points wins

### Action Space

The AlphaZero adapter (`adapter.py`) defines 25 discrete actions:

| Index | Action |
|-------|--------|
| 0–19 | Play one of 20 cards |
| 20 | Close talon (betakarás) |
| 21–24 | Declare marriage (Hearts, Bells, Leaves, Acorns) |

Trump exchange is NOT in the action space — it is always executed automatically when legal (it's always beneficial).

---

## Training Methods

### 1. Direct Self-Play (Legacy)

**Location**: `src/trickster/training/self_play.py`

A simpler approach where two copies of the same policy play against each other. Each decision is encoded as a feature vector, and the model learns to score actions by their quality.

- **Architecture**: Separate lead and follow MLP models, stored in a `TrainedPolicy` dataclass
- **Training signal**: Win/loss outcome from each game
- **Action selection**: Epsilon-greedy over model-scored actions
- **Limitations**: No search — the agent plays greedily from its learned policy. Cannot learn strategic actions like marriage declarations or talon closing (these are auto-decided).

### 2. AlphaZero (MCTS + Neural Network)

**Location**: `src/trickster/training/alpha_zero.py`

The full AlphaZero loop adapted for imperfect information via determinization (PIMC).

#### Training Loop

Each iteration consists of:

1. **Self-play with MCTS**: The current network guides Monte Carlo Tree Search. For each decision point, MCTS produces a visit-count distribution over actions (the "expert policy" π). The chosen action is sampled from this distribution.

2. **Data collection**: Each decision yields a training sample `(state_features, legal_mask, π, z)` where `z` is the **actual game outcome** (not the NN prediction).

3. **SGD training**: Mini-batches are sampled from a replay buffer and the network is updated with the combined loss.

**Important**: The value target `z` always comes from the real game outcome (`game.outcome()`), NOT the value head's prediction. This means even if the value head gives wrong estimates for certain positions (e.g., post-betakarás), the training signal is always correct.

#### Handling Imperfect Information

Since the agent can't see the opponent's hand or draw pile:

- **Determinization (PIMC)**: Before each MCTS run, hidden information is sampled randomly (consistent with observations). Multiple determinizations are aggregated. Each determinization produces a separate MCTS tree; visit counts are summed across all trees.
- **Bayesian void tracking**: When a player doesn't follow suit in the closed phase, we infer they lack that suit. This constrains future determinizations for more accurate sampling.

#### Hybrid Bootstrap

The **cold-start problem**: An untrained network provides useless value estimates, so MCTS search degrades.

**Solution**: For the first N games (`bootstrap_games`), MCTS uses **random rollouts** instead of the value head for leaf evaluation. The policy head still provides priors, and both heads are trained on the resulting data. After the bootstrap phase, the network switches to full AlphaZero mode (value head evaluation).

Training logs show this as `[BOOT]` vs `[AZ]` phases.

#### Parallel Self-Play

Self-play games within each iteration are distributed across `num_workers` processes using `ProcessPoolExecutor`. Each worker receives the latest network weights. This provides near-linear speedup for the game-generation phase without affecting learning dynamics (the network is only updated after all games in an iteration are collected).

---

## Neural Network Architecture

**Location**: `src/trickster/models/alpha_net.py`

All networks are pure NumPy — no PyTorch/TensorFlow dependency.

### SharedAlphaNet (Dual-Head Architecture)

The primary architecture for AlphaZero training:

```
state_features (126-dim)
        |
   [Shared Body MLP]          ← body_layers × body_units (e.g. 3×128)
        |
    body_output
     /       \
[Value Head]  [Policy Head]
    |              |
  tanh           masked softmax
    |              |
 V(s) in        π(a|s) over
 [-1, +1]      25 actions
```

**Shared Body**: `body_layers` fully-connected layers with `body_units` neurons each (ReLU activation, capped at 20.0). This is the feature extraction backbone shared by both heads.

**Value Head**: One hidden layer (`head_units` neurons, ReLU) → scalar output with tanh activation. Predicts the expected normalized game outcome V(s) ∈ [-1, +1].

**Policy Head**: One hidden layer (`head_units` neurons, ReLU) → 25 logits. Illegal actions are masked to -∞ before softmax.

**Combined Loss**:

```
L = (z − v)² − π·log(p) + c·‖θ‖²
```

- `(z − v)²`: Value MSE — how accurately the value head predicts actual game outcomes
- `−π·log(p)`: Policy cross-entropy — how well the policy head matches MCTS visit distributions
- `c·‖θ‖²`: L2 regularization (weight decay)

**Stability features**:
- Xavier weight initialization
- Gradient clipping (max norm = 5.0)
- Weight clipping ([-5, +5])
- Activation clipping (ReLU capped at 20.0)
- Logit clipping ([-15, +15])

### State Features (126 dimensions)

Encoded by `AlphaEncoder` in `src/trickster/games/snapszer/features.py`:

- **Hand** (~29 features): Binary presence for each of 20 cards, per-suit count, per-suit max rank, hand size
- **Public information** (~52 features): Draw pile fraction, phase (open/closed), trump color one-hot, trump upcard info, captured cards for both players (binary per card), unseen card tracking
- **Opponent inference** (~20 features): Known opponent cards in closed phase
- **Phase + lead card** (~7 features): Is-following flag, lead card color/rank/is-trump

### Training Metrics

- **VMSE** (Value Mean Squared Error): `mean((z − v)²)` — lower is better. Typically starts ~0.28 (random guessing) and drops to ~0.20–0.22 during bootstrap, then may rise slightly during AZ phase transition before settling.
- **PCE** (Policy Cross-Entropy): `−mean(Σ(π·log(p)))` — lower is better. Starts ~1.48 (near-uniform over ~5 legal actions) and declines as the policy learns to match MCTS search results.

---

## MCTS Implementation

**Location**: `src/trickster/mcts.py`

### Core Algorithm

1. **Selection**: PUCT formula — walk down the tree picking the child with highest `Q(a) + c_puct · P(a) · √(N_parent) / (1 + N_child)` where Q is the mean value and P is the prior from the policy head
2. **Expansion**: Lazy — child states are only computed when first visited (saves ~80% of `game.apply` calls)
3. **Evaluation**: Either value head forward pass (AlphaZero mode) or random rollout to terminal (bootstrap mode)
4. **Backpropagation**: Value propagated up the tree, flipping sign for opponent nodes

### Determinization

For each move decision:
1. Sample N determinized worlds (shuffle unknown cards consistently with observations)
2. Run a full MCTS tree on each determinized world
3. Aggregate visit counts across all determinizations
4. Choose the action with the most total visits

### Fast Rollout

`adapter.py` provides a specialized `fast_rollout()` method that:
- Uses a single mutable clone (no per-move cloning)
- Inlines trick resolution and draw logic
- Auto-declares marriages and auto-exchanges trump
- Skips `close_talon` (random agents never close)

This eliminates ~50% of MCTS time compared to the generic rollout loop.

### Key Functions

- `_run_mcts(state, game, net, perspective, config)` → `(visit_counts, root_value)` — single-determinization MCTS
- `alpha_mcts_choose(state, game, net, player, config, rng)` → action — aggregated determinized MCTS for play
- `alpha_mcts_policy(state, game, net, player, config, rng)` → `(π, action)` — for training (returns full visit distribution)

---

## Model Tiers (Strength Ladder)

Defined in `scripts/train_ladder.py`. Each tier increases network capacity and training budget:

| Tier | Name | Type | Network | Games | MCTS (sims × dets) | Est. Time |
|------|------|------|---------|-------|---------------------|-----------|
| T0 | Direct | Self-play MLP | 128×2 (separate) | 20k episodes | None | ~5s |
| T1 | Pawn | AlphaZero | 32×1/h16 | 2k | 30 × 3 | ~30s |
| T2 | Scout | AlphaZero | 64×2/h32 | 4.5k | 40 × 3 | ~1–2 min |
| T3 | Knight | AlphaZero | 64×2/h32 | 8k | 50 × 4 | ~3–5 min |
| T4 | Bishop | AlphaZero | 128×2/h64 | 15k | 60 × 5 | ~10–15 min |
| T5 | Rook | AlphaZero | 128×2/h64 | 20k | 60 × 5 | ~15–20 min |
| T6 | Captain | AlphaZero | 128×3/h64 | 48k | 80 × 6 | ~1–1.5 hr |
| T7 | General | AlphaZero | 256×4/h128 | 100k | 150 × 8 | ~5–8 hr |
| T8 | Marshal | AlphaZero | 256×4/h128 | 200k | 200 × 12 | ~12–20 hr |

**Network notation**: `body_units × body_layers / h(head_units)` — e.g., `128×3/h64` means 3 body layers of 128 units each, head layers of 64 units.

**Bootstrap**: Each AZ tier uses hybrid bootstrap for the first ~40% of training games to avoid the cold-start problem.

### Running the Ladder

```bash
# Train all tiers sequentially
python scripts/train_ladder.py

# Train first 4 tiers only
python scripts/train_ladder.py --tiers 4

# Train from a specific tier upward
python scripts/train_ladder.py --from-tier Bishop

# Evaluate existing models only (no training)
python scripts/train_ladder.py --eval-only
```

The script automatically:
- Cleans up any existing model pickles
- Trains each tier and registers it (visible in the web UI)
- Runs a round-robin evaluation at the end to establish strength order

### Currently Trained Models

The `models/` directory contains:
- **T4-Bishop** (128×2/h64)
- **T5-Rook** (128×2/h64)
- **T6-Captain** (128×3/h64) — strongest available

---

## Evaluation

### Round-Robin (train_ladder.py)

After training, the ladder script runs every agent pair against each other over N deals (default 200). Results are reported as:
- **Points per deal**: Average game points scored (1–3 per deal)
- **Margin table**: Head-to-head difference showing which agent beats which

### MCTS Strength at Play Time

The same model can play at different strengths by adjusting MCTS parameters at play time. Higher simulations and determinizations = stronger but slower play. Verified by `eval_mcts_strength.py` which shows that doubling MCTS budget measurably increases win rate against the same model with lower budget.

### Expected Strength Order

```
Random << T0-Direct < T1-Pawn ≈ T2-Scout ≈ T3-Knight < T4-Bishop < T5-Rook < T6-Captain
```

---

## Playing Against the AI

### React Web UI

**Backend**: `apps/api/main.py` (FastAPI)
**Frontend**: `apps/web/src/ui/` (React + TypeScript)

Start the server:

```bash
uvicorn apps.api.main:app --reload
```

Start the frontend dev server:

```bash
cd apps/web && npm run dev
```

Or serve the pre-built frontend from `apps/web/dist/`.

#### UI Features

- **Menu system**: Press ESC (or click ☰) to open the menu with New Game, Settings, and Return options
- **Opponent selection**: Choose any trained model by name (e.g., T6-Captain)
- **Live play**: Click cards to lead/follow; illegal cards are visually dimmed
- **Special actions**: Exchange trump, close talon, declare marriage via buttons
- **Speech bubbles**: Both AI and player actions appear as floating speech bubbles in Hungarian ("Betakarok!", "Cserélek!", "Van 20-am!", "Van 40-em!")
- **Scoreboard**: Shows deal score, match score, trump suit indicator (always visible)
- **Visual talon**: Stacked card backs represent the draw pile; face-down upcard after betakarás
- **Captured cards**: Toggle visibility; shown as card backs when hidden

#### Analysis Mode

Toggle in Settings. When enabled:

- **Progressive MCTS analysis**: A background thread runs the same MCTS engine the AI uses, progressively searching through determinized worlds
- **Probability bars**: Thin colored bars under each card show MCTS visit probabilities. Colors are **relative** — best move is always bright green, worst is always red, regardless of raw probability spread. Percentage shown in the bar.
- **Position evaluation**: Badge in the status bar shows the MCTS-backed value (not raw NN — MCTS root value accounts for actual game outcomes through tree search, including betakarás failure penalties)
- **Live updates**: Bars and evaluation update in real time as the search deepens (~800ms polling). Progress indicator shows `(3/12)` while searching.
- **Action analysis**: Betakarás and marriage buttons also get probability bars

The analysis runs in a separate thread with a cloned game state, so it never interferes with the main game or AI decisions.

#### MCTS Play Parameters

Adjustable in Settings:
- **Simulations** (10–500): How many moves the AI explores per determinized world
- **Determinizations** (1–30): How many possible hidden card distributions are sampled
- Total search budget = sims × dets per move. More = stronger but slower.

These persist across deals within a match and are restored when starting a new game.

### How AI Plays at Game Time

- **AlphaZero models**: Full MCTS search with determinization. MCTS decides all strategic actions including close talon and marriage declarations. Trump exchange is automatic (always beneficial).
- **Direct models**: Greedy action selection from learned policy. Strategic actions are auto-played.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models` | List available model labels |
| POST | `/api/new` | Start a new game |
| POST | `/api/new_deal` | Deal next round (same match) |
| POST | `/api/continue` | Continue after trick pause |
| POST | `/api/action` | Play a card / close talon / exchange trump / declare marriage |
| POST | `/api/settings` | Update MCTS parameters (sims, dets) |
| GET | `/api/analyze/{id}` | Get progressive MCTS analysis for current position |

---

## Project Structure

```
src/trickster/
  games/snapszer/
    cards.py          # Card, Color, deck definitions
    game.py           # Core game logic (deal, play_trick, legal_actions, ...)
    rules.py          # Follow/beat rules for closed phase
    agent.py          # LearnedAgent (for direct-trained models)
    adapter.py        # SnapszerGame — GameInterface for MCTS/AlphaZero
    features.py       # Feature encoders (FastEncoder, AlphaEncoder)
  models/
    alpha_net.py      # SharedAlphaNet, ValueHead, PolicyHead
    mlp.py            # MLP model for direct training
    linear.py         # Linear model for direct training
  training/
    alpha_zero.py     # AlphaZero training loop (self-play + SGD)
    self_play.py      # Direct self-play training loop
    eval.py           # Round-robin evaluation
    policy.py         # TrainedPolicy dataclass
    model_spec.py     # Model metadata (spec.json), label generation
    model_store.py    # Save/load model checkpoints
  mcts.py             # MCTS implementation (PUCT, determinization, rollouts)
  cli.py              # CLI training entry point
  card_graphics/      # Card images (Hungarian deck) + card back

apps/
  api/main.py         # FastAPI backend for React UI
  web/src/ui/
    App.tsx            # Main React component
    api.ts             # API client + types
    cards.ts           # Card image URL helpers
    styles.css         # Full UI styling

scripts/
  train_ladder.py     # Automated multi-tier training + evaluation
  train_alpha_zero.py # CLI for single AlphaZero training run
  train.py            # CLI for direct training
  eval.py             # Standalone evaluation script

models/               # Trained model checkpoints (net.pkl + spec.json)
eval_mcts_strength.py # Script to evaluate MCTS parameter impact on play strength
```

---

## Key Design Decisions

1. **Pure NumPy neural networks**: No framework dependency. Enables easy pickling, simple deployment, and full control over the training loop. Trade-off: slower than GPU-accelerated frameworks, but sufficient for the game's complexity.

2. **Fixed action space with masking**: The policy head outputs 25 logits (one per possible action). Illegal actions are masked to -∞ before softmax. This is simpler and faster than variable-length action encoding.

3. **Marriage as a learned decision**: Marriage declarations are part of the 25-action space (indices 21–24). The AI must learn *when* to declare — the 20/40 points are free, but forcing a K/Q lead may not always be optimal. Random rollouts in MCTS always declare marriages (free points), while the full MCTS tree explores both options.

4. **Trump exchange is automatic**: Unlike marriages, exchanging the trump jack is always beneficial (trade 2-point Jack for a higher-value card). It's not in the action space — it's executed automatically after trick resolution when legal, both in real play and in MCTS simulations.

5. **Determinization for imperfect info (PIMC)**: Rather than implementing a full POMDP solver, we use Perfect Information Monte Carlo — sample possible hidden states, run standard MCTS on each, and aggregate visit counts. Bayesian void tracking improves sampling quality.

6. **Hybrid bootstrap**: Solves the AlphaZero cold-start problem without curriculum learning or expert demonstrations. The bootstrap phase uses random rollouts (which work even with an untrained network), then transitions to value-head evaluation once the network has learned basic position assessment.

7. **MCTS root value for analysis**: The analysis mode displays the MCTS-backed root value (`root.value_sum / root.visits`), not the raw neural network prediction. This is important because the NN may not accurately evaluate edge cases (e.g., post-betakarás positions), while MCTS search reaches terminal states and computes correct outcomes through actual game simulation.

8. **Background analysis thread**: Analysis runs in a daemon thread with a cloned game state, its own `SnapszerGame` instance, and its own RNG. The shared neural network is safe to use from multiple threads (NumPy forward passes are read-only). Analysis is cancelled when the game state changes.

---

## Setup & Installation

```bash
# Install Python package (editable)
pip install -e .

# Install frontend dependencies
cd apps/web && npm install

# Start backend
uvicorn apps.api.main:app --reload

# Start frontend dev server
cd apps/web && npm run dev
```

Requirements: Python 3.11+, NumPy, FastAPI, uvicorn. See `requirements.txt`.
