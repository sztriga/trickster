# Trickster — Ulti AI

An AlphaZero-style AI framework for **Ulti**, the Hungarian 3-player trick-taking card game. A neural network learns to play through MCTS self-play with determinization for imperfect information, then plays against humans in a React web UI.

The project also includes a fully playable **Snapszer** (2-player, 20-card) engine with trained models up to the T6-Captain tier, though the primary focus is Ulti.

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

---

## The Game of Ulti

Ulti is played with a 32-card Hungarian Tell deck by 3 players. One player (the **soloist**) bids a contract and plays alone against the other two (**defenders**) who form a silent coalition. The game has a rich contract system:

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

## Project Goal

The end goal is **three AIs that play full Ulti autonomously** — deal, auction, contract selection, kontra decisions, trick play, scoring, rotating dealer. The system should handle every contract type, every modifier, and play at a level that challenges experienced human players.

This is a hard problem. Ulti sits at an unusual intersection of challenges for game AI:

- **Imperfect information**: You can't see the other players' cards. Unlike poker (where hidden info is mostly about opponent holdings), Ulti's hidden state directly affects the mechanics — you must follow suit with cards you can't see, and the optimal play depends on where specific cards sit across three hidden hands.
- **Asymmetric objectives**: The soloist plays offense (1 player), the defenders play defense (2 players, silently cooperating). A single network must learn both roles — and they require opposite strategies.
- **Compound contracts**: A hand might involve Parti + Ulti simultaneously, where winning one component and losing the other produces a net result. The AI needs to evaluate multi-objective tradeoffs.
- **Small deck, deep strategy**: Only 32 cards and 10 tricks, but the game tree is rich. Expert play involves counting cards, inferring holdings from void signals, managing trump sequences, and planning multi-trick endgames.

We are pursuing this incrementally. The current focus is **Phase 1: learn to play tricks well** — teach the AI solid trick-taking (Parti) and trick-avoidance (Betli) through curriculum self-play. Once the play phase is strong, Phase 2 layers on auction and contract selection, and Phase 3 assembles the complete game with all contract types.

---

## Architecture

### UltiNet (PyTorch, ~308k parameters)

A shared-backbone AlphaZero network with role-specific heads:

```
Input (259-dim state vector)
    |
    v
Shared Backbone (4 x 256 Linear + LayerNorm + ReLU)
    |
    +-- [Soloist] Policy Head  -> Linear(256, 32) -> LogSoftmax
    |              Value Head   -> Linear(256, 1)  -> Tanh
    |
    +-- [Defender] Policy Head -> Linear(256, 32) -> LogSoftmax
    |              Value Head   -> Linear(256, 1)  -> Tanh
    |
    +-- Shared Policy Head (backbone training signal)
    |   Shared Value Head
    |
    +-- Auction Head (multi-component)
         +-- Trump sub-head  -> Linear(128, 5) -> Softmax
         |                      (Hearts, Bells, Leaves, Acorns, NoTrump)
         +-- Flags sub-head  -> Linear(128, 4) -> Sigmoid
                                (Is_100, Is_Ulti, Is_Betli, Is_Durchmars)
```

The **dual-head architecture** is a key design decision. During training, each sample is routed through its role-specific head (soloist or defender) so that the two roles never push conflicting gradients through the same output layer. The shared backbone learns perceptual features common to both roles (card counting, suit tracking), while the role heads specialize in their distinct objectives (soloist: "how do I win this contract?", defender: "how do I stop the soloist?").

During inference (MCTS), the `is_soloist` flag in the state features auto-selects the correct head.

### State Encoder (259 dimensions)

The "detective model" encoder gives the AI everything a human expert would track:

| Feature | Dims | Purpose |
|---|---|---|
| My hand | 32 | Binary bitmap of cards held |
| Player 0/1/2 captured cards | 96 | Card counting — who took what |
| Current trick cards (slots 0, 1) | 64 | What's on the table right now |
| Trump suit | 4 | One-hot (all zeros for Betli) |
| Scalars (betli, soloist, trick#, scores, trick_count) | 6 | Game context |
| Opponent void flags | 8 | 2 opponents x 4 suits — inferred from play |
| Marriage bitmask | 4 | Do I hold King+Queen pairs |
| Seven-in-hand | 1 | Do I hold 7 of trumps (Ulti card) |
| Contract DNA | 8 | Component flags: parti, ulti, betli, durchmars, 40-100, 20-100, red, open |
| Trick momentum (leaders + winners) | 20 | Who led and won each of the 10 tricks |
| Auction context | 10 | Bid rank, talon flag, seat, kontra levels, has_ulti |
| Marriage memory | 6 | Per-player declared marriage totals |

The **Contract DNA** bits act as a context switch — when the "betli" bit is on, the AI's internal neurons learn to reverse card rankings (the 10 becomes weak instead of strong). This lets a single network handle fundamentally different contract strategies.

Positions are encoded **relative to the current player** (me / left opponent / right opponent) so the network learns position-invariant play.

### MCTS with PIMC Determinization

Since players can't see each other's hands, we use **Perfect Information Monte Carlo** (PIMC):

1. **Determinize**: Randomly assign unknown cards to opponents, respecting void constraints (a player who failed to follow suit is void in that suit) and auction constraints (a player who bid 40-100 must hold the trump King+Queen).
2. **Search**: Run standard AlphaZero MCTS (PUCT selection + neural policy priors + value head evaluation) on the determinized game.
3. **Aggregate**: Sum visit counts across multiple determinizations.
4. **Pick**: Choose the most-visited action.

The MCTS implementation uses **lazy expansion** (child states computed on first visit, not when the parent is expanded) which avoids ~80% of `game.apply` calls and makes the tree search significantly faster.

**Coalition-aware backpropagation**: In the 3-player game, defenders are allies. During MCTS backprop, both defenders receive the same value sign — this teaches the AI to coordinate defensively even though the defenders never explicitly communicate.

---

## Training Pipeline

### How It Works

The training loop (in `scripts/train_baseline.py`) repeats:

1. **Self-play**: Generate games where all 3 seats use the same UltiNet via MCTS. The soloist gets a higher MCTS budget (default 20 sims) for more thorough exploration, while defenders use a lighter budget (default 8 sims) to keep training fast.
2. **Collect**: Store `(state, legal_mask, MCTS_policy, final_reward, is_soloist)` tuples in a replay buffer.
3. **Train**: Sample mini-batches from the buffer, route through the appropriate role head via `forward_dual()`, and update weights with the AlphaZero loss: `L = MSE(value, reward) + CrossEntropy(policy, MCTS_visits)`.
4. **Evaluate**: Periodically play the trained model (with MCTS) against random opponents.

### Curriculum Modes

| Mode | Contract | What it teaches |
|---|---|---|
| `simple` | Parti only (random trump) | Trick-taking, trump management, point collection |
| `betli` | Betli only (no trump) | Trick avoidance, dumping high cards, inverse strategy |
| `mixed` | 50/50 Simple + Betli | Context switching based on Contract DNA bits |
| `auto` | AI picks contract via Auction Head | Contract selection + neural discard |

The curriculum **forces the contract** on a fully random deal. The cards are always natural — only the contract assignment is overridden. So in `--mode betli`, the AI gets a random hand but is told "you're playing Betli." Most of the time the hand will be terrible for that contract and the AI loses, but this is valuable — it learns both what losing hands look like (useful for future contract selection) and what winning looks like when the random hand happens to be strong.

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

**Compound contracts** are scored component by component. In a "Parti + Ulti" contract, the soloist might win Parti (+1) but lose Ulti (-8), netting -7 raw reward. The **double penalty on loss** teaches the AI that being soloist is risky.

During the current curriculum phase, training uses a simpler +1/-1 reward that matches the MCTS value scale. The compound reward engine is ready for when multi-component contracts are trained.

### Replay Buffer

- **Reservoir sampling** (Algorithm R, Vitter 1985) instead of FIFO eviction — every sample ever pushed has an equal probability of being retained, preserving diversity across training history.
- **Outcome-balanced batching**: Each mini-batch guarantees at least 15% of its samples come from winning soloist outcomes, preventing self-play collapse where the AI only sees defender wins.
- **Weighted sampling**: Soloist experiences are sampled 3x more frequently than defender experiences, correcting for the ~67% of positions where the AI plays defender.

### Talon Discard

After the soloist picks up the 2-card talon (giving them 12 cards), they must discard 2. Two strategies are available:

- **Greedy heuristic** (default): Simple heuristic — discard weakest non-trump cards for Parti, discard highest-strength cards for Betli.
- **Neural discard** (`--neural-discard`): Evaluates all C(12,2)=66 possible discards through the value head in a single batch and picks the one with the highest predicted outcome.

### League Training (PFSP)

When `--pool-interval` is set, the system implements **Prioritized Fictitious Self-Play** (from DeepMind's StarCraft II paper). Periodic snapshots of the network are stored in a checkpoint pool. During self-play, the soloist uses the current network while defenders are sampled from the pool with weights favoring opponents near a 50% win rate — neither too easy nor too hard. This prevents the network from overfitting to its own play style.

### Oracle Hand Evaluator

The Oracle evaluates a 12-card hand across all 5 contract types by:
1. Finding the best discard for each contract type via the value head.
2. Playing shallow MCTS rollouts to estimate soloist win probability.
3. Returning a "Contract Heatmap" of win rates.

The Oracle serves as a **teacher** for the Auction Head — the training target is the Oracle's best contract choice, trained via cross-entropy loss. This avoids the chicken-and-egg problem: the play AI learns to play well first, and the Oracle uses that play ability to evaluate which contracts are viable.

### Training Command

```bash
# Recommended: mixed curriculum, 5000 steps
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

# With league training and parallel workers
python scripts/train_baseline.py \
    --mode mixed \
    --steps 10000 \
    --games-per-step 16 \
    --workers 4 \
    --pool-interval 500 \
    --pool-size 10 \
    --sims 20 \
    --def-sims 8 \
    --lr 1e-4

# Auto mode with Oracle-guided auction training
python scripts/train_baseline.py \
    --mode auto \
    --steps 5000 \
    --neural-discard \
    --oracle-rollouts 3
```

Key parameters:
- `--sims` / `--def-sims`: MCTS simulations for soloist / defenders
- `--lr`: Learning rate (1e-3 for short runs, 3e-4 or 1e-4 for longer runs)
- `--body-units` / `--body-layers`: Network size (default 256x4)
- `--workers`: Parallel self-play processes
- `--pool-interval` / `--pool-size`: League training with PFSP
- `--load`: Continue training from a checkpoint
- `--neural-discard`: Use value head for talon discard

---

## Evaluation

### Head-to-Head Comparison

```bash
python scripts/eval_head2head.py \
    --model-a models/checkpoints/mixed/step_01000.pt \
    --model-b models/checkpoints/mixed/step_05000.pt \
    --games 200 --sims 15 --dets 3
```

Each evaluation plays `N games x 2 modes x 2 roles = 4N total games`. Reports per-contract breakdown, per-suit soloist win rates, points distributions, and game highlights with seeds for replay.

### Built-in Eval (During Training)

- **vs Random**: Model + MCTS vs random opponents (standard benchmark)
- **Elite Benchmark** (`--elite-interval`): Neural net (policy only, 0 MCTS sims) vs Oracle (MCTS with 50 sims) — measures how much the network has internalized search

### What the Metrics Mean

| Metric | Random baseline | Decent model | Strong model |
|---|---|---|---|
| Overall WR vs Random | 50% | 55-60% | 65%+ |
| Simple soloist WR | ~33% | 35-45% | 50%+ |
| Betli soloist WR | ~10% | 15-25% | 30%+ |
| Defender WR | ~67% | 70-80% | 80%+ |
| Value loss | ~0.8 | ~0.05 | <0.02 |

Note: Soloist is inherently disadvantaged (1v2), so 33% baseline is expected.

---

## Training Campaign Log

### The Soloist Plateau Problem

Across four training runs with different configurations, we've identified a consistent pattern: the AI learns defense quickly but struggles to improve as soloist. Every run follows the same arc:

1. **Fast early gains** (steps 1-300): vloss drops, overall WR jumps from 50% to 60%+.
2. **Defender WR climbs to 75-85%**: the 2v1 structural advantage makes defense easy to learn.
3. **Soloist WR stalls at 25-45%**: barely above the 33% random baseline for a 1v2 game.
4. **Buffer composition locks**: `buf_swr` (fraction of soloist-win samples) stops changing, learning signal dries up.
5. **Internal losses keep improving** (vloss, ploss decrease) but play strength flatlines.

### Run History

**Run 1 — Pure Simple, conservative LR** (`--mode simple`, LR=1e-4, 1 det, 2000 steps)
- Peaked at 64% overall WR. Soloist WR oscillated 26-48%, never stabilized. Value loss kept bouncing (0.33-0.43). buf_swr stuck at 33%.
- Lesson: LR too low, single-contract mode gave no representation diversity.

**Run 2 — Mixed mode, aggressive LR** (`--mode mixed`, LR=1e-3, 2 dets, 3000 steps)
- Fast early gains: vloss dropped from 0.90 to 0.24 in 500 steps, ploss from 1.19 to 0.87. Peaked at 63% WR by step 300. Betli soloist briefly spiked to 41% then collapsed. Plateaued by step 1000.
- Lesson: High LR good for initial exploration, but oscillated once near convergence. The Betli spike proved the network *can* learn soloist strategies — it just couldn't hold onto them.

**Run 3 — Two-stage LR** (loaded Run 2's step 1000, LR=3e-4)
- Hit **66% WR** — the all-time best. Simple soloist reached 45%. ploss improved to 0.77. Then plateaued again. buf_swr locked at 27% for the entire run.
- Lesson: Two-stage LR (1e-3 then 3e-4) produced the best result. But internal improvements stopped translating to better play.

**Run 4 — PFSP league training** (loaded Run 2's step 1000, LR=3e-4, pool every 300 steps)
- Same 60-64% band. Pool win rates at 27-28% — even against weaker past selves the soloist couldn't win consistently. No improvement over Run 3.
- Lesson: Opponent diversity didn't help because the bottleneck isn't who the soloist plays against — it's the quality of the MCTS search guiding the soloist.

### Diagnosis

The MCTS search at 20 sims with 1-2 determinizations produces **noisy policy targets**. The network learns to match those targets (ploss drops steadily), but the targets themselves aren't good enough to teach strong soloist play. Winning as soloist requires multi-trick planning — stripping opponents' trumps, sequencing Aces, setting up endgames. A shallow, noisy search finds these lines by accident sometimes but can't teach them systematically.

Defense doesn't have this problem because it's reactive and local — "follow suit, play high to block, trump when you can" is learnable from noisy signal. Offense requires a plan.

### What Worked

- **Mixed mode over pure Simple**: Betli adds diversity, forces Contract DNA to matter, gives the backbone richer representations.
- **Two-stage LR (1e-3 then 3e-4)**: Fast exploration followed by refinement produced the best result (66%).
- **Dual-head architecture**: Eliminated gradient conflict between soloist and defender roles. Both roles improve monotonically within each run.
- **Outcome-balanced batching**: Prevented total self-play collapse. The soloist never dropped to 0% even when buf_swr was stuck.
- **The network learns real skills**: ploss consistently decreases across all runs. Card counting, void tracking, basic trump management, and Betli/Simple context switching all emerge.

### Next Steps

Higher search quality: `--sims 40 --dets 4` with fewer games per step. If the ceiling is MCTS target quality rather than data quantity, cleaner policy targets should break through. If not, the bottleneck may be deeper — the encoder, the greedy discard, or the network capacity.

---

## Challenges and Design Decisions

### 1. The Soloist vs Defender Conflict

The fundamental tension: soloist and defender have opposite objectives, but they share a backbone. Early architectures used a single policy/value head, which led to **gradient interference** — defender updates would undo soloist learning and vice versa.

**Solution**: Dual role-specific heads. The backbone learns shared perceptual features (card counting, suit tracking), while separate policy and value heads specialize for each role. During training, `forward_dual()` routes each sample through the correct head. This eliminated the tug-of-war and both roles now improve monotonically.

### 2. Self-Play Collapse

When the AI only plays against itself with a single network, it can enter a collapse mode where the soloist never wins. Since the defenders are 2-against-1, a mediocre defense easily beats a mediocre offense. The buffer fills with defender-win samples, the value head learns "soloist always loses", and learning stalls.

**Solutions** (layered):
- **Outcome-balanced batching**: Each mini-batch guarantees a minimum fraction (15%) of winning-soloist samples, even when they're rare in the buffer.
- **Weighted sampling**: Soloist experiences get 3x sampling weight.
- **Asymmetric MCTS budget**: The soloist gets more search (20 sims vs 8 for defenders), giving it a better chance of finding winning lines during self-play.
- **Reservoir sampling**: Preserves sample diversity across training history instead of FIFO (which biases toward recent, potentially degenerate play).
- **PFSP league training**: Defenders are drawn from a pool of past checkpoints, preventing co-adaptation.

These mitigations prevent collapse but haven't yet solved the deeper problem: the soloist plateaus at ~35-45% WR because the MCTS policy targets are too noisy to teach long-range planning.

### 3. The Rare Contract Problem

Not all contracts are equally likely from random deals:

| Contract | Roughly viable | Why |
|---|---|---|
| Parti | ~90%+ of deals | Almost any hand can try |
| Betli | ~10-15% | Need mostly low cards |
| Ulti | ~30% | Need the 7 of trumps specifically |
| 40-100 | ~10-15% | Need K+Q of trump + enough for 100 pts |
| Durchmars | ~1-3% | Need overwhelming dominance |

If the AI only trained on naturally-dealt games, it might see a viable Durchmars hand once every 50-100 games — not enough signal to learn from.

**Solution**: The curriculum forces the contract on random deals. In `--mode betli`, the AI gets a random hand but must play Betli regardless. Most of the time the hand is terrible and the AI loses, but this is by design — it learns both "what losing looks like" (useful for future contract selection) and "what winning looks like" on the rare strong hands. When tournament-style training arrives, forced-contract enrichment will be mixed with free-choice games (~60% natural + ~40% forced across the rare types).

### 4. Imperfect Information

Unlike chess or Go, Ulti has hidden information — you can't see opponents' hands. Standard MCTS assumes perfect information.

**Solution**: PIMC determinization. Before each MCTS search, we sample a plausible assignment of hidden cards to opponents, constrained by:
- **Void tracking**: If a player failed to follow suit, they can't hold that suit.
- **Auction constraints**: If the soloist bid 40-100, they must hold the trump King+Queen. If they bid Ulti, they likely hold the trump 7.
- **Marriage declarations**: A player who declared a marriage held K+Q of that suit.

Multiple determinizations (typically 2-6) are aggregated to smooth out the randomness. This is approximate — it doesn't achieve the theoretical guarantees of more sophisticated approaches like Information Set MCTS — but it's fast and works well in practice for trick-taking games. The current evidence suggests that too few determinizations (1-2) produce noisy targets that limit learning; 4+ may be needed for clean signal.

### 5. Betli vs Simple: Opposite Strategies

Betli (take zero tricks) requires the opposite of Parti (take the most points). Card rankings reverse, high cards become dangerous instead of valuable, and the AI must learn a completely different playstyle.

**Solution**: The Contract DNA encoding explicitly tells the network what contract is being played. The "betli" bit flips the AI's internal strategy. Mixed training (50/50 Simple + Betli) forces the network to learn context-dependent play, which it does surprisingly well — the same network can play both aggressive trick-taking and cautious trick-avoidance. Training logs confirm that the Betli soloist can spike to 41% WR when the network "discovers" the dump-high-cards strategy, though maintaining this under mixed training has been inconsistent.

### 6. Training Speed vs Search Quality

MCTS self-play is CPU-intensive. Each game involves hundreds of MCTS calls, each requiring game state cloning, neural network inference, and tree traversal. There is a fundamental tension between:
- **More games with shallow search** (fast, noisy targets) — good for data quantity
- **Fewer games with deep search** (slow, clean targets) — good for data quality

Our training campaign suggests we've been too far on the "quantity" side. At 20 sims with 1-2 determinizations, the policy targets are noisy enough that the network learns to predict them (ploss drops) but the targets themselves don't encode strong enough play to teach genuine strategic depth.

**Speed mitigations** (to make deeper search affordable):
- **Lazy expansion**: MCTS child states are only computed on first visit (~80% fewer `game.apply` calls).
- **Inlined PUCT**: The hot selection loop avoids per-child function calls.
- **Asymmetric budgets**: Defenders get fewer sims (8-15 vs 20-40 for soloist).
- **Parallel self-play**: `--workers N` distributes games across processes.
- **Combined forward pass**: `predict_both()` computes policy and value in a single backbone pass.

GPU training is not yet implemented but would allow higher search budgets within the same wall-clock time.

---

## Playing Against the AI (Web UI)

The trained model loads automatically in the FastAPI backend from `models/ulti_mixed.pt`.

| Mode | How it works | Speed | Strength |
|---|---|---|---|
| **Neural** | Policy head only (argmax) | Instant | Weakest — raw intuition, no search |
| **MCTS** | Policy + value heads guide tree search | 0.5-3s/move | Strongest — proper lookahead |
| **Random** | Random legal card | Instant | Baseline |

MCTS strength presets (selectable in the UI):

| Preset | Sims | Speed | Level |
|---|---|---|---|
| Fast | 10 sims x 2 dets | ~0.3s | Casual |
| Medium | 30 sims x 3 dets | ~1s | Intermediate |
| Strong | 80 sims x 5 dets | ~3s | Challenging |

---

## Current Status

### Performance (Best Results)

| Metric | Best achieved | Run | Notes |
|---|---|---|---|
| Overall WR vs Random | **66%** | Run 3 (step 700) | Two-stage LR, mixed mode |
| Simple soloist WR | **45%** | Run 3 (step 700) | Above 33% baseline = real learning |
| Betli soloist WR | **41%** | Run 2 (step 400) | Brief spike, not sustained |
| Defender WR | **84%** | Run 3 (step 1500) | Consistently the strongest side |
| Best ploss | **0.77** | Run 3 (step 1400) | Gut instinct approaching MCTS quality |
| Best vloss | **0.17** | Run 3 (step 1500) | Accurate mid-game win prediction |

The AI beats a random player ~62-66% of the time. It has learned basic card play, defensive coordination, and context-dependent strategy (Betli vs Parti). It has not yet achieved strong soloist play — the 1v2 attacker role remains the primary unsolved challenge.

### Contracts Covered

| Contract | Status | Notes |
|---|---|---|
| **Parti (Simple)** | Trained | All 4 trump suits. Basic trick-taking, trump management, point counting |
| **Betli** | Trained | No trump, avoid all tricks. Can dump high cards, but inconsistent |
| **Durchmars** | Not yet | Win all 10 tricks — needs dedicated curriculum |
| **Ulti** | Partially | As a component in Parti + Ulti combos. Dedicated training pending |
| **40-100 / 20-100** | Not yet | Marriage-based contracts, needs auction training |
| **Piros (Hearts)** | Not yet | Doubled values. Structurally ready (Contract DNA has is_red flag) |
| **Teritett (Open)** | Not yet | Soloist's hand visible. PIMC handles it but not trained |

### Skills Demonstrated

The AI reliably demonstrates:

1. **Card counting**: Tracks which cards each player has captured, deduces remaining holdings
2. **Trump management**: Learns when to lead trump vs when to hold it
3. **Context switching**: Recognizes Betli vs Simple via Contract DNA and adjusts strategy
4. **Void exploitation**: Infers when opponents are void and adjusts accordingly
5. **Point awareness**: Prioritizes Aces and 10s in Simple, avoids them in Betli
6. **Defensive coordination**: Both AI defenders work together through coalition-aware MCTS

Skills not yet reliably demonstrated:
- Multi-trick planning as soloist (stripping trumps, setting up endgames)
- Consistent Betli soloist strategy (learned once, lost it)
- Strong offensive card play against non-random opponents

### Known Limitations

- **Soloist plateau at ~35-45%**: The core unsolved problem. Likely caused by noisy MCTS policy targets at current search depth.
- **No auction learning yet**: Contract selection is forced during training, not learned
- **No kontra decisions**: The AI doesn't learn when to double
- **CPU-bound**: GPU support would enable deeper search within the same wall-clock time
- **Greedy discard**: The talon discard heuristic may limit soloist potential (neural discard available but slower)
- **PIMC is approximate**: Doesn't handle strategic signaling between defenders
- **Eval noise**: 200-game evaluations produce high variance on soloist WR (only ~70 soloist games per eval), making it hard to distinguish real improvement from noise

---

## Project Structure

```
src/trickster/
  games/
    ulti/                  # Ulti game engine
      game.py              # Core game logic (deal, tricks, scoring)
      cards.py             # Card definitions, Tell deck
      adapter.py           # UltiGame — MCTS interface + PIMC determinization
      encoder.py           # 259-dim "detective" state encoder
      auction_encoder.py   # 116-dim auction phase encoder
      auction.py           # Auction system (bidding rules)
      rules.py             # Follow-suit and trick-winning rules
    snapszer/              # Snapszer game engine (2-player)
    interface.py           # GameInterface protocol
  model.py                 # UltiNet (PyTorch) — dual-head policy + value + auction
  evaluator.py             # Oracle hand evaluator (neural discard + shallow MCTS)
  train_utils.py           # Reward engine, replay buffer, checkpoint pool (PFSP)
  mcts.py                  # MCTS with PIMC determinization
  solver.py                # Alpha-beta solver (pure-Python fallback)
  hybrid.py                # HybridPlayer — MCTS + alpha-beta endgame solver
  _solver_core.pyx         # Cython alpha-beta solver (pluggable contract evaluators)
  models/
    alpha_net.py           # SharedAlphaNet (for Snapszer)
  training/
    alpha_zero.py          # Generic AlphaZero training loop

scripts/
  train_baseline.py        # Ulti curriculum training (simple/betli/mixed/auto)
  eval_head2head.py        # Head-to-head model comparison
  test_cython_solver.py    # Cython solver verification + benchmark
  train_ladder.py          # Snapszer strength ladder
  train_alpha_zero.py      # Snapszer AlphaZero training

apps/
  api/
    main.py                # FastAPI backend (Snapszer)
    ulti.py                # FastAPI backend (Ulti) — serves trained AI
  web/                     # React + TypeScript frontend

models/                    # Trained model checkpoints and specs
```

---

## Roadmap

### Immediate (breaking the soloist plateau)
1. **Higher search quality**: `--sims 40 --dets 4` with fewer games per step — test whether cleaner MCTS targets break the 66% ceiling
2. **GPU training**: CUDA support would allow deep search (80+ sims, 6+ dets) without prohibitive wall-clock time
3. **Learning rate schedule**: Cosine or step-decay to automate the two-stage LR approach that produced the best results
4. **Larger eval sets**: 500+ games per eval to reduce variance on soloist WR measurements

### Near-term (more contracts)
5. Durchmars curriculum (`--mode durchmars`)
6. Ulti-specific curriculum (win last trick with 7 of trumps)
7. Neural discard as default (replace greedy heuristic)

### Medium-term (auction and strategy)
8. Auction Head training (`--mode auto`) with Oracle teacher
9. Kontra/rekontra decision head
10. 40-100 / 20-100 marriage-based contracts

### Long-term (full game)
11. Complete bidding system with overbidding strategy
12. Piros (Hearts) contracts — doubled stakes
13. Teritett (Open) games — open-hand play
14. ELO rating system
15. Tournament mode — 3 AIs play full Ulti with rotating dealer, end-to-end
