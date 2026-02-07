# Trickster — Training & Evaluation

## Method: Expert Iteration

The AI learns Snapszer by imitating a self-play agent.

### How it works

1. **An agent plays games against itself.** Either a pure MCTS expert or
   random self-play generates thousands of games.

2. **Record decisions and outcomes.** For each decision point we store
   the (state + chosen action) feature vector and which player made it.
   After the game ends, each sample is labeled with the game outcome
   scaled by points (1.0 = 3pt win, 0.667 = 1pt win, 0.333 = 1pt loss,
   0.0 = 3pt loss).

3. **Train an MLP to predict outcomes.** A small neural network
   (`MLPBinaryModel`) learns P(win | state, action) from these labeled
   samples using binary cross-entropy (log-loss). Separate models are
   trained for lead and follow decisions.

4. **At game time, no search.** For each legal action the trained model
   scores it instantly. The AI picks the highest-scoring action.

### Training

```bash
# Expert iteration with parallel self-play
python3 scripts/train.py \
  --method expert \
  --kind mlp --hidden 128 --layers 2 --activation relu \
  --iters 500 --games-per-iter 200 --train-steps 400 \
  --sims 50 --dets 4 --lr 0.03 \
  --buf-cap 500000 --workers 4 --seed 42 \
  --name MyModel

# Direct self-play (simpler, no MCTS)
python3 scripts/train.py \
  --method direct \
  --kind mlp --hidden 128 --layers 2 --activation relu \
  --episodes 100000 --lr 0.03 \
  --name MyModel
```

### Evaluation

```bash
python3 scripts/eval.py vs-random models/Wolverine --deals 2000 --workers 4
python3 scripts/eval.py compare models/Wolverine models/Batman --deals 2000 --workers 4
python3 scripts/eval.py list
```

## Models

| Model | Architecture | LR | Games | vs Random |
|---|---|---|---|---|
| **Wolverine** | h=128 L=2 relu | 0.03 | 100k | **1.54/deal** |
| **Batman** | h=128 L=2 relu | 0.01 | 100k | 1.53/deal |
| **Spiderman** | h=64 L=3 relu | 0.01 | 100k | 1.53/deal |
| Mewtwo | h=128 L=2 relu | 0.03 | 20k (working MCTS) | 1.00/deal |

## Key finding: MCTS bug and the exploration-exploitation tradeoff

### The bug

The original MCTS implementation had a critical bug: root children's states
were never stored in the node-state lookup table. Every simulation hit
`if leaf_state is None: continue` and skipped. The MCTS was doing zero
actual search — it just picked the first legal action (effectively random).

This means **Wolverine, Batman, and Spiderman were trained on random
self-play data**, not MCTS expert data. They reached 1.54/deal despite this
because outcome-based learning works well with diverse/random data.

### The fix

The MCTS now uses lazy expansion: child states are computed on-demand when
first selected, not eagerly during parent expansion. This correctly handles
root children and all deeper nodes. The MCTS genuinely searches now
(50 sims x 4 determinizations with random rollouts).

### The paradox

With the bug fixed, MCTS self-play produces **weaker** training data for
outcome-based learning:

- **Random play** → diverse outcomes (big wins, big losses) → wide spread
  of targets → strong learning gradients
- **Strong MCTS play** → near-optimal play on both sides → games are close
  → outcomes cluster around 0.5 → weak learning signal

This is a known issue in expert iteration: when the expert is too strong
and plays optimally against itself, the training signal degrades because
there's less variance in outcomes.

**Mewtwo** (20k games with working MCTS, 7 min) scored 1.00/deal vs random,
significantly below Wolverine's 1.54/deal from 100k games of (accidentally)
random self-play.

### Performance optimizations done

- **Inlined PUCT selection**: eliminated 1.35M function calls per 100 games,
  reduced selection from 55% to 3.4% of MCTS runtime
- **Lazy tree expansion**: child states computed on-demand, ~80% fewer
  `game.apply` calls in the tree
- **Cached `make_deck()`**: module-level frozenset instead of recreating per
  determinization
- **Multiprocessing**: `--workers N` flag for parallel self-play, ~3x speedup
  with 4 workers (14.3 → 44.2 games/sec)

### Current bottleneck

With the MCTS working, **random rollouts are 82%** of runtime. Each MCTS
simulation plays random moves to terminal to estimate position value. This
is the AlphaGo approach. The AlphaZero approach (value network instead of
rollouts) would eliminate this entirely.

## Next steps (pick up here)

The core question: how to get the benefits of MCTS search quality while
maintaining the data diversity needed for outcome-based learning.

### Option A: Noisy MCTS
Use fewer simulations (`--sims 5 --dets 1`) for weaker but still strategic
play. The MCTS adds some tactical quality while maintaining diverse outcomes.

### Option B: Epsilon-greedy expert
Add an `--eps` flag to choose a random action X% of the time during MCTS
self-play. This injects diversity while keeping most decisions high-quality.

### Option C: MCTS visit distribution targets
Instead of training on game outcomes, train on "which action did MCTS
prefer" (soft targets from visit counts). This directly uses the search
quality. The challenge: gradient cancellation when multiple actions from the
same state share state features (all positive/negative gradients cancel on
the state weights). This was tried before and didn't work with the current
architecture. May need a different model design (e.g., state-action
separation, attention, or a value-only approach).

### Option D: Value network (full AlphaZero loop)
Train a value network V(state) from MCTS games. Use it to replace rollouts
in MCTS (`--use-value-head`). Then iterate: better value network → better
MCTS → better training data → better value network. This eliminates the
rollout bottleneck (82% of runtime) AND improves data quality over time.
The infrastructure for this exists (`AlphaNet`, `use_value_head` flag) but
the policy head component failed previously.

### Option E: Hybrid buffer
Mix random self-play games (high diversity) with MCTS games (high quality)
in the training buffer. Tune the ratio to balance signal strength vs data
quality.

### Other improvements
- **Scale to Ulti.** The `GameInterface` / adapter / generic MCTS is
  game-agnostic. Adding Ulti requires only a new adapter and feature encoder.
- **AWS scaling.** Self-play is embarrassingly parallel. With `--workers 16`
  on an EC2 instance, 100k MCTS games would take ~15 min.
- **Native rollouts.** Moving the random rollout loop to C/Cython would
  cut the 82% rollout bottleneck significantly without changing the algorithm.
