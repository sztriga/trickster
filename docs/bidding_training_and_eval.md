# Bidding Training and Evaluation — Detailed Description

This document describes how the Ulti bidding system works during training and evaluation, including the heuristics used and where they fall short.

---

## 1. Overview

The bidding system uses the **play-phase value heads** of the contract models — there is no separate bidding neural network. Each contract model (parti, ulti, 40-100, betli) has a value head that predicts expected game-point outcome for a given hand/contract/trump/discard combination. Bidding is driven by these value-head predictions.

**Key parameters:**
- `min_bid_pts` (default: -2.0): Minimum expected pts/defender to place a bid. Matches the pass penalty so the AI bids whenever it expects to do better than passing.
- `pass_penalty` (default: 2.0): Pts per defender when everyone passes (soloist pays 2 to each).
- `max_discards` (default: 15): Max discard pairs evaluated per (contract, trump) combo.

---

## 2. Training Flow (`bidding_loop.py`)

Training uses a **simplified, non-competitive** auction:

### 2.1 Per-game flow

1. **Deal**: 12 cards to soloist, 10 to each defender. Soloist is always `next_player(dealer)` — i.e. the first bidder. There is no talon-passing; the soloist is given 12 cards directly.

2. **Evaluate**: `evaluate_all_contracts()` runs on the soloist's 12-card hand. For each feasible (contract, trump, discard) combo, the value head predicts expected stakes. Returns a sorted list of `ContractEval` by `stakes_pts` descending.

3. **Pick bid**: Softmax-sample over `stakes_pts` with an annealed temperature (`bid_temp_start` → `bid_temp_end` via cosine schedule). High temperature early in training explores diverse contracts; low temperature late in training converges to greedy selection. If no feasible contract exists → treat as pass (no game played).

4. **Setup**: Apply the chosen bid (contract, trump, discards) to the game state.

5. **Play**: HybridPlayer (MCTS + solver) plays the game. Soloist and defenders use the same or pool models.

6. **Collect samples**: Each (state, mask, policy, reward) is pushed to the contract's replay buffer. Off-policy samples (defenders in pool games) have policy loss masked.

### 2.2 What training does *not* do

- **No competitive auction**: Only one player ever bids. No talon passing, no overbidding, no "should I pick up?" decisions.
- **No pickup decision**: The soloist always has 12 cards. The model never learns "pick up vs pass" in training.
- **No overbid pressure**: The soloist never faces an existing bid they must beat.

---

## 3. Evaluation Flow (`eval_bidding.py`, `tournament.py`)

Evaluation uses a **full competitive 3-player auction**:

### 3.1 Auction flow (`_run_auction`)

1. **First bidder**: Gets the talon (12 cards). Must bid at least Passz.

2. **Each player in turn**:
   - If `awaiting_bid` (has 12 cards): Evaluate contracts via `evaluate_all_contracts()`, then `_best_legal_nn_bid()`. Only bids that are legal overbids and have `game_pts >= min_bid_pts` are considered. If a profitable legal bid exists → submit it. Otherwise:
     - If `current_bid is None` (first bidder): **Heuristic fallback** → `ai_initial_bid(hand)` (Passz + 2 weakest cards).
     - Else (picked up but can't overbid): **Heuristic fallback** → bid `legal[0]` with `discards = hands[:2]` (first two cards).
   - If not `awaiting_bid` (has 10 cards): Decide whether to pick up. Simulate 12-card hand, run `_best_legal_nn_bid()`. If profitable legal bid exists → pick up. Else → pass.

3. **All-pass**: If the winning bid is Passz, no game is played; first bidder pays 2 to each defender.

### 3.2 Play phase

Same as training: HybridPlayer with MCTS + solver. Kontra/rekontra decided after trick 1 using the value head.

---

## 4. Contract Evaluation (`bidding/evaluator.py`)

### 4.1 `evaluate_contract()`

For a single (contract, trump, is_piros):

1. **Feasibility**: Betli needs no trump; 40-100 needs K+Q in trump; ulti needs trump 7.

2. **Candidate discards**: All C(12,2) = 66 pairs. If `len > max_discards` (15) and not betli:
   - **Heuristic pruning** (see §5.1): Score each pair; keep the "best" 15 by a hand-crafted score.
   - Betli: no pruning (fewer combos matter).

3. **Batch value-head inference**: Encode each (hand, contract, trump, discard) as state; run `wrapper.batch_value()`. Pick the discard with highest value.

4. **Un-normalise**: `game_pts = value * _GAME_PTS_MAX / 2`.

### 4.2 `evaluate_all_contracts()`

Iterates over all contracts × trumps (or piros variants). Returns list sorted by `stakes_pts` descending.

---

## 5. The "Embarrassing" Heuristics

### 5.1 Discard pruning heuristic (evaluator.py, ~line 186)

When there are more than `max_discards` (15) discard pairs, we **prune** before running the value head:

```python
def _discard_score(pair):
    score = 0.0
    for c in pair:
        if c.suit == trump:        score += 100   # penalise discarding trump
        score -= c.points()                        # prefer discarding low-value
        if 40-100: K/Q in trump → score += 1000   # never discard these
        if ulti: trump 7 → score += 1000          # never discard these
    return score  # lower = better to discard
all_discards.sort(key=_discard_score)
all_discards = all_discards[:max_discards]
```

**Problem**: The value head is never shown the pruned pairs. The heuristic can drop the true best discard. The NN could learn that e.g. discarding a specific trump is good in some positions, but we never evaluate those discards.

**Mitigation**: `max_discards=15` keeps 15/66 ≈ 23% of pairs. For betli, no pruning (all pairs evaluated).

### 5.2 First-bidder forced Passz (`ai_initial_bid`, auction.py)

When the first bidder has 12 cards and **no** eval meets `min_bid_pts`, they still must bid at least Passz. Fallback:

```python
def ai_initial_bid(hand):
    candidates = sorted(hand, key=lambda c: (c.points(), c.strength()))
    discards = candidates[:2]  # 2 weakest cards
    return BID_PASSZ, discards
```

**Problem**: No NN involved. Discards are "2 weakest" by points+strength — a crude heuristic. The value head said "nothing is profitable" but we force a bid anyway.

### 5.3 Picked-up-but-can't-overbid fallback (eval_bidding.py, ~line 296)

When a player picked up the talon but finds no profitable legal overbid:

```python
bid_obj = legal[0]           # minimum legal bid
discards = gs.hands[player][:2]  # first two cards in hand order!
```

**Problem**: `hands[:2]` is arbitrary — hand order is not sorted by strength. We bid the minimum legal with a random-looking discard.

### 5.4 No-model heuristic (`ai_should_pickup`, auction.py)

When a seat has no loaded model (e.g. "random" or typo) and must decide whether to pick up:

```python
def ai_should_pickup(hand, auction):
    aces = sum(1 for c in hand if c.rank == Rank.ACE)
    tens = sum(1 for c in hand if c.rank == Rank.TEN)
    strong = aces + tens
    return strong >= 5 and can_pickup(auction)
```

**Problem**: "5+ aces/tens" is a crude rule. Used only when no NN is available (e.g. random player).

---

## 6. Train vs Eval Mismatch

| Aspect | Training | Eval |
|--------|----------|------|
| Auction | Single player, always has 12 cards | Full 3-player, talon passing |
| Pickup decision | Never (always 12 cards) | Yes — simulate 12-card, check profitability |
| Overbidding | Never | Yes — must beat current bid |
| First bidder | Can "pass" (no game) if nothing profitable | Must bid Passz; heuristic if nothing profitable |
| Fallbacks | None | ai_initial_bid, legal[0]+hands[:2] |

The model learns bidding only in the "I have 12 cards, what should I bid?" setting. It never sees competitive auctions or pickup decisions during training. The eval environment is richer and triggers heuristic fallbacks when the NN says "pass" in situations where a bid is required.

---

## 7. Value Head Interpretation

The value head outputs a normalised value in `[-1, 1]` (roughly). Un-normalised:

- `game_pts = value * _GAME_PTS_MAX / 2` → expected per-defender stakes from the soloist's perspective.
- Positive = soloist expects to win (defenders pay).
- Negative = soloist expects to lose (soloist pays).
- `min_bid_pts = -2.0` matches the pass penalty: bid if you expect to do better than paying 2 to each defender.

---

## 8. Summary

- **Training**: Simplified single-player bidding; value head drives contract/discard choice; no auction dynamics.
- **Eval**: Full competitive auction; same value head; heuristic fallbacks when NN says "no bid" but a bid is required.
- **Heuristics**: Discard pruning (limits NN's discard choices), forced Passz with weak-card discards, minimum bid + arbitrary discards when stuck.
