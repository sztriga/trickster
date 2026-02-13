# Training Notes & Future Plans

---

## Current: Stage 1 — Play Phase Training

### Ulti — Tiered Parti Training

**Primary script**: `scripts/train_ulti.py` (tiered, current focus)
**Single-run script**: `scripts/train_baseline.py` (original, still usable)

Trains UltiNet to play Parti (Simple) contracts using hybrid
self-play: neural MCTS for the opening + exact Cython alpha-beta
solver for the endgame.

- Contract: Parti only, all 4 trump suits (including Hearts/red)
- No auction — contract is forced on random deals
- Greedy discard heuristic — not learning to discard
- Single network plays all 3 seats (relative encoding)
- Solver provides exact labels for ~60% of decisions (tricks 5-10)

#### Ulti training tiers

Training uses a tiered curriculum. Each tier uses the same 256×4
UltiNet architecture but with increasing self-play budgets:

| Tier | Name | Steps | Games | MCTS (sol/def) | SGD/step | LR schedule | Status |
|---|---|---|---|---|---|---|---|
| U1 | Scout | 500 | 4,000 | 20s / 8s | 50 | 1e-3 → 2e-4 | **Trained** |
| U2 | Knight | 2,000 | 16,000 | 30s / 12s | 80 | 1e-3 → 1e-4 | **Trained** |
| U3 | Bishop | 8,000 | 64,000 | 30s / 12s | 100 | 1e-3 → 5e-5 | Next target |

All tiers use endgame-tricks=6 (Cython solver, 20 PIMC
determinizations, solver-temp=0.5).

```bash
# Train all 3 tiers + round-robin eval
python scripts/train_ulti.py

# Train a single tier
python scripts/train_ulti.py --tiers knight

# Evaluate existing models only
python scripts/train_ulti.py --eval-only
```

Models are saved to `models/ulti/<tier-name>/` (e.g.
`models/ulti/U2-Knight/model.pt`).

#### Key improvements over train_baseline.py

The tiered script (`train_ulti.py`) adds:

- **Cosine LR annealing** — learning rate decays smoothly from
  `lr_start` to `lr_end` over training
- **Decoupled SGD steps** — a fixed number of gradient steps per
  iteration regardless of self-play batch size
- **Per-role value loss** — tracks soloist vs defender value head
  accuracy separately (catches role-specific collapse early)
- **Policy accuracy metric** — top-1 match between predicted and
  target policy (more intuitive than cross-entropy alone)
- **Round-robin eval** — every trained tier plays every other tier
  + a random baseline, producing a ranking table

The original `train_baseline.py` is still valid for single-run
experiments and supports deal enrichment, parallel workers, and
opponent pool training not yet ported to the tiered script.

### Snapszer — AlphaZero + Hybrid Training

Snapszer (2-player, 20-card) has two parallel training pipelines
with fully trained models.

#### Pure AlphaZero ladder (`scripts/train_ladder.py`)

Standard self-play: MCTS with neural value+policy heads, no solver.
7 tiers from tiny to full-strength:

| Tier | Name | Net | Games | MCTS | Status |
|---|---|---|---|---|---|
| T1 | Pawn | 32×1/h16 | 2,000 | 30s×3d | Trained |
| T2 | Scout | 64×2/h32 | 4,500 | 40s×3d | Trained |
| T3 | Knight | 64×2/h32 | 8,000 | 50s×4d | Trained |
| T4 | Bishop | 128×2/h64 | 15,000 | 60s×5d | Trained |
| T5 | Rook | 128×2/h64 | 20,000 | 60s×5d | Trained |
| T6 | Captain | 128×3/h64 | 48,000 | 80s×6d | Trained |
| T7 | General | 256×4/h128 | 100,000 | 150s×8d | Defined (AWS) |

```bash
python scripts/train_ladder.py                    # all tiers
python scripts/train_ladder.py --range 5:6        # Rook + Captain
python scripts/train_ladder.py --eval-only        # round-robin eval
```

#### Hybrid MCTS+Minimax (`scripts/train_hybrid.py`)

Uses PIMC + exact Minimax for endgame positions during self-play
(analogous to the Ulti Cython solver approach). The hybrid method
produces cleaner training data because endgame positions are solved
exactly.

| Tier | Name | Net | Games | PIMC | Status |
|---|---|---|---|---|---|
| H1 | Scout | 64×2/h32 | 4,500 | 30 samples | Defined |
| H2 | Knight | 64×2/h32 | 8,000 | 30 samples | **Trained** (v1-v3) |
| H3 | Bishop | 128×2/h64 | 15,000 | 30 samples | Defined |
| H4 | Rook | 128×2/h64 | 20,000 | 30 samples | Defined |
| H5 | Captain | 128×3/h64 | 48,000 | 30 samples | Defined |

```bash
python scripts/train_hybrid.py --tier knight      # train one tier
python scripts/train_hybrid.py --eval-only        # hybrid vs old AZ models
```

### Solver engine choice

The `--endgame-tricks` parameter controls when the solver takes over.
With the Cython solver (Ulti):

| endgame-tricks | MCTS tricks | Solver tricks | Exact label % | Per-game overhead |
|---|---|---|---|---|
| 4 | 1-6 | 7-10 | ~40% | minimal |
| 6 (default) | 1-4 | 5-10 | ~60% | ~12ms at trick 5 |
| 7 | 1-3 | 4-10 | ~70% | ~100ms at trick 4 |
| 8 | 1-2 | 3-10 | ~80% | ~1s at trick 3 |

Higher values = cleaner training signal but slower self-play. 6 is
the practical default; 7-8 require the Cython solver to be feasible.

---

## Future: Stage 2 — Auction + Discard Training

Trains the network to evaluate hands, choose discards, and bid
contracts. This is **separate** from play training.

### Why separate?

The play network only sees the post-discard 10-card hand. It doesn't
need to learn which cards to discard or which contract to bid. Those
are pre-play decisions that can be trained independently:

1. **Hand evaluator**: Given 12 cards + trump suit, predict P(win).
   Trained from solver self-play with binary cross-entropy.

2. **Discard network**: Given 12 cards, evaluate all C(12,2)=66
   possible discards via the hand evaluator or value head. The network
   learns which 10-card hands lead to wins.

3. **Auction/bidding**: Given hand + position, choose contract type
   and trump suit. Trained via an Oracle teacher that evaluates each
   contract option using the play network.

### Why the discard training needs bad examples

The greedy heuristic in Stage 1 always discards the weakest non-trump
cards.  This means the play training data never includes games where
the soloist gave away Aces/Tens (20 points) to the defenders via the
talon.

For discard training, the network needs to see **both good and bad
discards** and learn from the outcomes. The solver plays games with
different discard choices; games where the soloist discarded a Ten
will tend to lose, teaching "don't put Tens in the talon."

### Talon rules (for reference)

- Soloist picks up 2-card talon (sees 12 cards)
- Soloist discards 2 cards face-down
- Discarded cards' **points count for the defenders** (not the soloist)
- Only the soloist knows which cards were discarded
- Defenders must infer from the unknown pool during PIMC determinization

### Infrastructure

`scripts/train_solver_nets.py` has the foundation:
- Phase 1: Generate data (solver vs solver self-play)
- Phase 2: Train hand evaluator (36-dim input, binary CE)
- Phase 3: Train card play policy (259-dim input, CE over 32 actions)

Not yet connected to the main training pipeline.

---

## Future: Stage 3 — Additional Contracts

Once Parti play is strong, extend to other contract types:

| Contract | Training approach | Notes |
|---|---|---|
| Betli | Forced curriculum, separate value targets | Opposite strategy — avoid tricks |
| Durchmars | Forced curriculum, rare natural occurrence | Need overwhelming hand strength |
| Ulti | As compound with Parti, 7-of-trumps tracking | Trump 7 management |
| 40-100 | Marriage-conditional, needs auction | K+Q of trump required |
| 20-100 | Marriage-conditional, needs auction | K+Q of off-suit required |
| Piros (red) | Hearts trump, doubled stakes | Structurally ready (Contract DNA has is_red flag) |
| Teritett (open) | Soloist hand visible | PIMC already handles it |

The Contract DNA encoding (8 bits in the state vector) acts as a
context switch, telling the network which contract is active. A single
network can learn multiple contract strategies.

---

## Design Decisions

### Relative encoding (single network for all seats)

The state encoder uses relative player positions (me / left / right)
rather than absolute indices. This means the same network plays all
3 seats. Samples from all seats go into the same replay buffer,
tripling the effective training data.

### Solver as anchor

The hybrid approach provides **exact** training labels for endgame
positions. The value head learns that certain mid-game positions
reliably lead to wins/losses (because the solver proved it), which
improves MCTS quality at earlier tricks. This creates a virtuous
cycle: better value estimates -> better MCTS -> better early-game
labels -> better overall play.

### Outcome-balanced batching

The soloist is 1-vs-2, so defenders win more often in self-play.
Without correction, the buffer fills with defender-win samples and
the value head learns "soloist always loses." The replay buffer
guarantees 15%+ soloist-win samples per batch and weights soloist
positions 3x to prevent this collapse.

### Deal enrichment (curriculum learning via value head)

In a random deal, the soloist is statistically disadvantaged — roughly
50-70% of random deals are unwinnable. This creates a class imbalance
that drowns the policy signal.

Deal enrichment uses the **network's own value head** to filter deals.
A deal is re-rolled (up to 20 attempts) until the value head rates the
soloist's position above a threshold. The threshold anneals over training:

| Phase | Steps | Threshold | Effect |
|---|---|---|---|
| warmup | first N steps | disabled | Value head is untrained, skip filtering |
| 1 (easy) | 0%-25% | v ≥ 0.00 | Only "even or better" deals |
| 2 (medium) | 25%-60% | v ≥ −0.25 | Slightly losing deals mixed in |
| 3 (full) | 60%+ | disabled | All deals accepted |

The warmup period (default 20 steps, `--enrich-warmup`) is critical:
at the start of training the value head outputs near-zero for
everything, so filtering by it would be meaningless. Once the value
head has learned the basics from unfiltered games, enrichment kicks
in and provides a better balance of soloist-winnable games.

30% of games at every phase are **always fully random** to prevent
distribution shift. This ensures the network never forgets how to
handle genuinely bad positions.

Disable with `--no-enrichment`. Adjust the random fraction with
`--enrich-random-frac` (default 0.3).

### Solver temperature

`--solver-temp` controls how sharp the solver's policy labels are.
Lower temperature (default 0.5) makes the solver prefer its best
move more strongly, giving the network cleaner gradient signals
during the endgame. At 1.0, the solver distributes probability
more evenly across "good enough" moves.
