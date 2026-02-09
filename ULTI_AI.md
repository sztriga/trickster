# Ulti AI — Training & Evaluation Framework

## Overview

The Ulti AI learns to play the Hungarian 3-player trick-taking card game through AlphaZero-style self-play. A single neural network (UltiNet) plays all three seats, using Monte Carlo Tree Search (MCTS) with determinization to handle hidden information.

The system currently trains on **Simple (Parti)** and **Betli** contracts. After a proper training session, the AI can play these games in the React web UI with configurable strength.

---

## Architecture

### UltiNet (PyTorch, ~308k parameters)

```
Input (259-dim state vector)
    │
    ▼
Shared Backbone (4 × 256 Linear + LayerNorm + ReLU)
    │
    ├── Policy Head  → Linear(256, 32) → LogSoftmax
    │                   (probability over 32 cards)
    │
    ├── Value Head   → Linear(256, 1)  → Tanh
    │                   (predicted outcome ∈ [-1, +1])
    │
    └── Auction Head (multi-component)
         ├── Trump sub-head  → Linear(128, 5) → Softmax
         │                      (Hearts, Bells, Leaves, Acorns, NoTrump)
         └── Flags sub-head  → Linear(128, 4) → Sigmoid
                                (Is_100, Is_Ulti, Is_Betli, Is_Durchmars)
```

The **shared backbone** means the same card-counting features feed into all heads. The network learns a unified understanding of card value that adapts based on the contract type.

### State Encoder (259 dimensions)

The "detective model" encoder gives the AI everything a human expert would track:

| Feature | Dims | Purpose |
|---|---|---|
| My hand | 32 | Binary bitmap of cards held |
| Player 0/1/2 captured cards | 96 | Card counting — who took what |
| Current trick cards (slots 0, 1) | 64 | What's on the table right now |
| Trump suit | 4 | One-hot (all zeros for Betli) |
| Scalars (betli, soloist, trick#, scores, trick_count) | 6 | Game context |
| Opponent void flags | 8 | 2 opponents × 4 suits — inferred from play |
| Marriage bitmask | 4 | Do I hold King+Queen pairs |
| Seven-in-hand | 1 | Do I hold 7 of trumps (Ulti card) |
| Contract DNA | 8 | Component flags: parti, ulti, betli, durchmars, 40-100, 20-100, red, open |
| Trick momentum (leaders + winners) | 20 | Who led and won each of the 10 tricks |
| Auction context | 10 | Bid rank, talon flag, seat, kontra levels, has_ulti |
| Marriage memory | 6 | Per-player declared marriage totals |

The **Contract DNA** bits act as a context switch — when the "betli" bit is on, the AI's internal neurons learn to reverse card rankings (the 10 becomes weak instead of strong).

### MCTS with PIMC Determinization

Since players can't see each other's hands, we use **Perfect Information Monte Carlo** (PIMC):

1. **Determinize**: Randomly assign unknown cards to opponents (respecting void constraints and auction signals)
2. **Search**: Run standard MCTS on the determinized game
3. **Aggregate**: Sum visit counts across multiple determinizations
4. **Pick**: Choose the most-visited action

Void tracking ensures the sampler won't give Hearts to a player who already showed they're void in Hearts. Auction constraints (e.g. a player who bid 40-100 likely holds the trump marriage) further narrow the sampling space.

---

## Training Pipeline

### Script: `scripts/train_baseline.py`

The training loop repeats:

1. **Self-play**: Generate 4 games per step. All 3 seats use the same UltiNet via MCTS.
   - Soloist: 20 MCTS simulations (more exploration)
   - Defenders: 8 MCTS simulations (lighter budget for speed)
2. **Collect**: Store (state, MCTS_policy, final_reward, is_soloist) in a replay buffer
3. **Train**: Sample mini-batches (64) from the buffer, weighted 3x toward soloist experiences
4. **Evaluate**: Periodically play the trained model (with MCTS) vs random opponents

### Curriculum Modes

| Mode | Contract | What it teaches |
|---|---|---|
| `simple` | Parti only (random trump) | Trick-taking, trump management, point collection |
| `betli` | Betli only (no trump) | Trick avoidance, dumping high cards, inverse strategy |
| `mixed` | 50/50 Simple + Betli | Context switching based on Contract DNA bits |
| `auto` | AI picks contract via Auction Head | Contract selection, neural discard (future) |

### Reward Engine

Per-component asymmetric rewards, normalized to [-1, +1]:

| Component | Win | Loss | Notes |
|---|---|---|---|
| Parti | +1 | -2 | Basic "take more points than defenders" |
| Ulti | +4 | -8 | Win last trick with 7 of trumps |
| Betli | +5 | -10 | Take zero tricks |
| Durchmars | +6 | -12 | Win all 10 tricks |
| 40-100 | +4 | -8 | Declare trump marriage + reach 100 points |
| 20-100 | +8 | -16 | Declare non-trump marriage + reach 100 points |

**Compound contracts** are scored component by component. In a "Parti + Ulti" contract, the soloist might win Parti (+1) but lose Ulti (-8), netting -7 raw reward.

The **double penalty on loss** teaches the AI that being soloist is risky — it learns to play defensively as defender and aggressively (but carefully) as soloist.

### Talon Discard

After the soloist picks up the 2-card talon (giving them 12 cards), they must discard 2 to get back to 10. Currently uses a **greedy heuristic**:

- **Simple**: Discard the two weakest non-trump cards
- **Betli**: Discard the two highest-strength cards (Aces, Kings)

A **neural discard** option exists (`--neural-discard`) that evaluates all 66 possible discards through the value head and picks the one with the highest predicted outcome.

### Replay Buffer

- Capacity: 50,000 samples (FIFO)
- **Weighted sampling**: Soloist experiences are sampled 3x more frequently than defender experiences, correcting for the ~67% of positions where the AI plays defender (which are typically easier to learn)

### Training Command

```bash
# Recommended: mixed curriculum, 600 steps, self-play defenders
python scripts/train_baseline.py \
    --mode mixed \
    --steps 600 \
    --games-per-step 4 \
    --sims 20 \
    --def-sims 8 \
    --eval-interval 200 \
    --eval-games 30 \
    --checkpoint-interval 200 \
    --seed 42
```

This runs ~2400 games in about 1 minute on CPU. For meaningful learning, aim for **5000+ steps** (20k+ games) which takes ~15-20 minutes.

Key parameters:
- `--sims`: MCTS simulations per move for soloist (more = stronger play, slower training)
- `--def-sims`: MCTS sims for defenders (lower keeps training fast)
- `--lr`: Learning rate (default 1e-3, consider 3e-4 for long runs)
- `--body-units` / `--body-layers`: Network size (default 256×4)
- `--load`: Continue training from a checkpoint
- `--neural-discard`: Use value head for talon discard instead of heuristic

### Checkpoints

Saved to `models/checkpoints/{mode}/step_XXXXX.pt` containing:
- Model weights
- Architecture config (body_units, body_layers)
- Training progress (step, games, samples, best win rate)

Final model saved to `models/ulti_{mode}.pt`.

---

## Evaluation Framework

### Script: `scripts/eval_head2head.py`

Pits two models (or one model vs random) against each other with full diagnostic output.

```bash
# Model vs random baseline
python scripts/eval_head2head.py \
    --model-a models/ulti_mixed.pt \
    --model-b random \
    --games 200 --sims 15 --dets 3

# Checkpoint vs checkpoint
python scripts/eval_head2head.py \
    --model-a models/checkpoints/mixed/step_00200.pt \
    --model-b models/checkpoints/mixed/step_00600.pt \
    --games 200
```

Each evaluation plays `N games × 2 modes × 2 roles = 4N total games` (each model plays as soloist and defender in both Simple and Betli).

### Diagnostic Output

The evaluator reports:

1. **Overall results**: Total wins for each model
2. **Per-contract breakdown**: Win rates as soloist vs defender, average points scored
3. **Simple by trump suit**: Per-suit soloist win rates (reveals trump-specific weaknesses)
4. **Notable events**: Ulti outcomes, Durchmars, 0-trick games
5. **Points distribution**: Average/min/max/median soloist points, Betli 0-trick rate
6. **Game highlights**: Biggest wins, closest games, Betli failures (with seeds for replay)

### Built-in Eval (during training)

The training script evaluates periodically:

- **vs Random**: Model + MCTS vs random opponents (standard benchmark)
- **Elite Benchmark** (`--elite-interval`): Neural net (policy only, 0 MCTS) vs Oracle (MCTS with 50 sims) — measures how much the network has internalized search

### What the Metrics Mean

| Metric | Random baseline | Decent model | Strong model |
|---|---|---|---|
| Overall WR vs Random | 50% | 55-60% | 65%+ |
| Simple soloist WR | ~33% | 35-45% | 50%+ |
| Betli soloist WR | ~10% | 15-25% | 30%+ |
| Defender WR | ~67% | 70-80% | 80%+ |
| Value loss | ~0.8 | ~0.05 | <0.02 |

Note: Soloist is inherently disadvantaged (1v2), so 33% baseline is expected. A strong model wins more as soloist through better card play and as defender through coordinated card counting.

---

## What the AI Currently Learns

### Contracts Covered

| Contract | Status | Notes |
|---|---|---|
| **Parti (Simple)** | Trained | All 4 trump suits. AI learns trick-taking, trump management, point counting |
| **Betli** | Trained | No trump, avoid all tricks. AI learns to dump high cards, inverse strategy |
| **Durchmars** | Not yet | Win all 10 tricks — needs curriculum level 4 |
| **Ulti** | Partially | As a component in Simple+Ulti combos. Dedicated training pending |
| **40-100 / 20-100** | Not yet | Marriage-based contracts, needs auction training |
| **Piros (Hearts)** | Not yet | Doubled values. Structurally ready (Contract DNA has is_red flag) |
| **Teritett (Open)** | Not yet | Soloist's hand visible. PIMC handles it but not trained |

### Skills Acquired

After a proper training session (5000+ steps), the AI demonstrates:

1. **Card counting**: Tracks which cards each player has captured, deduces remaining holdings
2. **Trump management**: Learns when to lead trump vs when to hold it
3. **Context switching**: Recognizes Betli vs Simple via Contract DNA bits and adjusts strategy
4. **Void exploitation**: Infers when opponents are void in a suit and adjusts accordingly
5. **Point awareness**: Prioritizes high-value cards (Aces, 10s) in Simple, avoids them in Betli
6. **Defensive coordination**: As defenders, both AI seats work to prevent soloist from winning

### Known Limitations

- **Betli discard is heuristic**: The greedy discard (dump highest cards) is reasonable but not optimal
- **No auction learning yet**: Contract selection is forced during training, not learned
- **No kontra decisions**: The AI doesn't learn when to double
- **Short training runs overfit**: The value loss drops quickly but policy doesn't fully converge
- **CPU-bound**: Training on CPU only; GPU support would enable longer, more productive runs

---

## Playing Against the AI (React UI)

The trained model loads automatically in the FastAPI backend from `models/ulti_mixed.pt`.

### AI Play Modes

| Mode | How it works | Speed | Strength |
|---|---|---|---|
| **Neural** | Policy head only (argmax) | Instant | Weakest — raw intuition, no search |
| **MCTS** | Policy + value heads guide tree search | 0.5-3s/move | Strongest — proper lookahead |
| **Random** | Random legal card | Instant | Baseline |

### MCTS Strength Presets

| Preset | Sims | Speed | Level |
|---|---|---|---|
| Fast | 10 sims × 2 dets | ~0.3s | Casual |
| Medium | 30 sims × 3 dets | ~1s | Intermediate |
| Strong | 80 sims × 5 dets | ~3s | Challenging |

These are selectable in the UI top bar. The AI mode and strength can be changed mid-game.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/ulti/model-info` | Check if a model is loaded |
| POST | `/ulti/{id}/ai-settings` | Change AI mode/strength |
| POST | `/ulti/new` | Start a new Ulti game |

---

## Testing a Properly Trained Model

### Step 1: Train for Real

A meaningful training run needs 5000+ steps. Recommended:

```bash
# ~20 minutes on a modern CPU
python scripts/train_baseline.py \
    --mode mixed \
    --steps 5000 \
    --games-per-step 4 \
    --sims 20 \
    --def-sims 8 \
    --lr 3e-4 \
    --eval-interval 500 \
    --eval-games 50 \
    --checkpoint-interval 1000 \
    --seed 42
```

For an overnight run:

```bash
# ~2-4 hours
python scripts/train_baseline.py \
    --mode mixed \
    --steps 30000 \
    --games-per-step 4 \
    --sims 30 \
    --def-sims 10 \
    --lr 1e-4 \
    --eval-interval 2000 \
    --eval-games 100 \
    --checkpoint-interval 5000 \
    --elite-interval 10000 \
    --elite-games 20 \
    --seed 42
```

### Step 2: Evaluate Progress

Compare checkpoints against random and each other:

```bash
# Each checkpoint vs random (200 games each = 800 per eval)
for step in 01000 02000 03000 04000 05000; do
    echo "=== Step $step vs Random ==="
    python scripts/eval_head2head.py \
        --model-a models/checkpoints/mixed/step_$step.pt \
        --model-b random \
        --games 200 --sims 15 --dets 3
done

# Latest vs earliest (skill gap measurement)
python scripts/eval_head2head.py \
    --model-a models/checkpoints/mixed/step_01000.pt \
    --model-b models/checkpoints/mixed/step_05000.pt \
    --games 300 --sims 20 --dets 3 --highlights 10
```

### Step 3: Look For

**Signs of learning:**
- Overall WR vs Random climbing above 55%
- Simple soloist WR above 40% (beating the 33% baseline)
- Betli 0-trick rate (as soloist) above 20%
- Value loss stable below 0.05
- Later checkpoints consistently beating earlier ones head-to-head

**Signs of trouble:**
- WR plateaus or drops after initial improvement → lower the learning rate
- Betli soloist WR near 0% → the Betli discard heuristic might be failing
- Value loss rises → possible overfitting, increase buffer size or reduce LR
- Step N beats Step N+1000 → training is unstable, reduce LR or increase MCTS sims

### Step 4: Play Against It

```bash
# Start the API (loads models/ulti_mixed.pt automatically)
PYTHONPATH=src uvicorn apps.api.main:app --reload --port 8000

# Start the frontend
cd apps/web && npm run dev
```

In the UI, set AI Mode to "MCTS" and Strength to "Strong" for the best opponent. The model-info endpoint (`GET /ulti/model-info`) confirms which model file is loaded.

---

## The Rare Contract Problem

Not all contracts are equally likely from random deals:

| Contract | Roughly viable | Why |
|---|---|---|
| Parti (Simple) | ~90%+ of deals | Almost any hand can try |
| Betli | ~10-15% | Need mostly low cards, no dangerous holdings |
| Ulti | ~30% | Need the 7 of trumps (1 specific card out of 32) |
| 40-100 | ~10-15% | Need K+Q of trump + enough strength for 100 pts |
| Durchmars | ~1-3% | Need overwhelming dominance — top trumps, side aces |
| 20-100 | ~5% | Need off-suit marriage + 100 points |

If the AI only trained on naturally-dealt tournament games, it might see a viable Durchmars hand once every 50-100 games. That's not enough signal to learn from.

### How the curriculum solves this

The curriculum **forces the contract** on a fully random deal. The cards are always natural — only the contract assignment is overridden. So in `--mode betli`, the AI gets a random hand but is told "you're playing Betli with this."

Most of the time the hand will be terrible for that contract, and the AI loses. But this is valuable — it learns both sides:

- "Hands like this lose Betli" → useful for future contract selection (auction head)
- The rare times the random hand is actually strong → "this is how you win Betli"

No positions are hand-crafted. The enrichment is purely about ensuring rare game types get enough training reps.

### End-to-end training strategy

When tournament-style training is ready (roadmap item 13), the training script will mix natural and forced games:

- ~60% free choice (tournament-style, natural contract distribution)
- ~15% forced Betli
- ~10% forced Ulti
- ~10% forced 40-100 / 20-100
- ~5% forced Durchmars

This ensures the AI gets enough reps on rare contracts while still learning realistic contract selection from the free-choice games. The curriculum enrichment never fully goes away — it becomes a background supplement mixed into natural play.

This is a standard RL technique for **sparse reward on rare events** — the same approach used in AlphaGo for unusual board positions.

---

## Roadmap

The end goal is **three AIs playing full Ulti autonomously** — deal, auction, contract selection, kontra, play, scoring, rotating dealer. Each roadmap item builds toward that, in dependency order.

### Near-term (next training upgrades)

1. **Durchmars curriculum**: Add `--mode durchmars` for "win all 10 tricks" training
2. **Ulti-specific curriculum**: Dedicated training for the "win last trick with 7 of trumps" component
3. **Lower learning rate schedule**: Implement cosine or step-decay LR for longer runs
4. **GPU training**: Move to CUDA for 5-10x speedup on self-play generation

### Medium-term (new capabilities)

5. **Auction Head training** (`--mode auto`): Let the AI choose its own contracts using the Oracle evaluator as teacher
6. **Neural discard** by default: Replace greedy heuristic with value-head evaluation
7. **Kontra decisions**: Train a small head for kontra/rekontra timing
8. **40-100 / 20-100 contracts**: Marriage-based contracts with point threshold win conditions

### Long-term (full game)

9. **Complete bidding system**: Full auction with pickup/pass decisions, overbidding strategy
10. **Piros (Hearts) contracts**: Doubled stakes with all contract types
11. **Teritett (Open) games**: Open-hand play where opponents see soloist's cards
12. **ELO rating system**: Track model strength progression over time
13. **Tournament mode**: Multiple consecutive games with rotating dealer — the full end-to-end experience where 3 AIs sit down and play real Ulti

Once item 13 is complete, training becomes a single command that handles the curriculum internally: forced-contract enrichment for rare game types mixed with free-choice tournament games, progressively shifting toward more natural play as the AI improves.
