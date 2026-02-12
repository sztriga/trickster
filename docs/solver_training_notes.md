# Solver-to-NN Training Pipeline — Design Notes & Concerns

This document captures design decisions and open concerns for the
`scripts/train_solver_nets.py` pipeline that trains neural networks
from solver-generated Parti data.

---

## 1. Policy data is collected from ALL players, not just the soloist

The solver makes decisions for all three seats (1 soloist + 2 defenders).
Every non-forced move by any player is recorded as a policy training
example.  This is intentional and correct:

- The policy network sees a **relative** state encoding (position 0 =
  "me"), so a single network can learn to play any seat.
- Defender moves are just as informative as soloist moves.
- With ~30 moves per game and many being forced (single legal card),
  collecting only soloist moves would yield very few training examples
  per game (~3-5 non-forced moves).

Collecting from all three players roughly triples the policy dataset.
For 2 000 games this should produce approximately 20 000–30 000 useful
policy samples.

---

## 2. Hand evaluator sample size may be insufficient

The hand evaluator gets exactly **one sample per game** (the soloist's
hand + trump + win/loss outcome).  With the default 2 000 games, that
is only 2 000 training examples for a 36-dimensional input.

**Risk:** The network may overfit or show poor calibration in the
extreme buckets (0–20 % and 80–100 %).

**Mitigation options (in order of preference):**

1. Increase `--num-games` to 5 000–10 000 if wall-clock time allows.
2. Data augment by permuting the non-trump suits in the hand encoding
   (there are up to 6 permutations of 3 off-suits, each producing a
   valid training example with the same label).
3. Use stronger regularisation (dropout, weight decay) during training.

The script already prints calibration buckets so this issue will be
immediately visible after training.

---

## 3. Solver quality ceiling

The solver uses a **depth-limited PIMC** approach:

- The last `--solver-depth` tricks (default 6) are solved exactly via
  alpha-beta.
- Earlier tricks rely on PIMC averaging over determinisations.

This means the training data has an inherent quality ceiling: the
solver is not perfect, especially in the opening tricks where heuristic
playout (or shallow evaluation) is used.  The neural networks can only
be as good as the data they learn from.

This is acceptable for the first iteration.  Later, the trained policy
network can itself be used as the MCTS prior in the hybrid player,
generating stronger data in a self-improvement loop.

---

## 4. Greedy discard heuristic limits hand diversity

The data generation uses a simple heuristic for talon discards ("discard
the two weakest non-trump cards").  This means:

- The distribution of starting hands seen during training is biased
  towards a particular discard style.
- In real play (with a learned or human-like discard strategy), the
  soloist's 10-card hand may look different.

For the hand evaluator this is a moderate concern — the network may
calibrate well for heuristic discards but poorly for creative ones.

For the policy network this is a minor concern — the play-phase state
space is large enough that individual discard choices only slightly
shift the distribution.

**Future fix:** Replace the greedy discard with a neural discard model
(already exists in `train_baseline.py` as `make_neural_discard_fn`)
once the first iteration is trained.

---

## 5. Forced moves are excluded from policy training

When a player has only one legal card, that move is recorded for game
progression but **not** added to the policy dataset (there is no
decision to learn).  This is correct and reduces noise, but it means
the policy dataset is smaller than `num_games × 30`.

The script logs the count of skipped forced moves for transparency.

---

## 6. No marriage-conditional hand evaluation (yet)

The hand evaluator input (36-dim) does not encode potential marriages.
Two hands with identical cards but different marriage potential have the
same feature vector.

For Parti this is a minor issue because marriages contribute relatively
few points (20 or 40) compared to the 90-point total.  For contracts
where marriages are critical (e.g. 100-as Parti), the hand evaluator
input would need to be extended.

---

## 7. Calibration validation is essential before deployment

The script prints a 5-bucket calibration table after training the hand
evaluator.  **This must be checked before using the network in
production.**  Good calibration means:

| Bucket   | Expected actual WR |
|----------|--------------------|
| 0–20 %   | ~10 %              |
| 20–40 %  | ~30 %              |
| 40–60 %  | ~50 %              |
| 60–80 %  | ~70 %              |
| 80–100 % | ~90 %              |

If calibration is poor (e.g. all predictions cluster around 50 %),
the most likely cause is insufficient data — increase `--num-games`.
