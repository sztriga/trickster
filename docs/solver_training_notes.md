# Training Notes & Future Plans

---

## Current: Stage 1 — Play Phase Training

**Script**: `scripts/train_baseline.py`

Trains the UltiNet to play Parti (Simple) contracts using hybrid
self-play: neural MCTS for the opening + exact Cython alpha-beta
solver for the endgame.

- Contract: Parti only, all 4 trump suits (including Hearts/red)
- No auction — contract is forced on random deals
- Greedy discard heuristic — not learning to discard
- Single network plays all 3 seats (relative encoding)
- Solver provides exact labels for ~60% of decisions (tricks 5-10)

### Solver engine choice

The `--endgame-tricks` parameter controls when the solver takes over.
With the Cython solver:

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
