# Ulti AI — Observation & Input Specification

This document describes **what the AI can observe** at each decision point
in an Ulti game, what is already encoded, and what still needs to be built.

---

## Decision Points

The AI must act at **four** distinct phases. Each has its own observation
space and action space.

| Phase | Actor | Action space | Encoder status |
|---|---|---|---|
| **Bidding** (discard + bid) | Current bidder | 2 discards + 1 bid from legal bids | ✅ `auction_encoder.py` (116-dim) |
| **Kontra / Rekontra** | Defender / Soloist | Subset of kontrable components | Pending |
| **Trump selection** | Soloist | 3 non-red suits | Pending |
| **Play** (trick-taking) | Current player | Legal cards from hand | ✅ `encoder.py` (259-dim, v2) |

---

## Phase 1 — Bidding / Auction

### What the player knows

| Observation | Available? | Notes |
|---|---|---|
| Own hand (10 or 12 cards) | Yes (hands) | 12 after talon pickup, 10 after discard |
| Talon cards (if picked up) | Yes | Only visible to the player who picked up |
| Who is the first bidder / dealer | Yes (AuctionState) | |
| Current highest bid | Yes (AuctionState.current_bid) | |
| Who holds the current bid | Yes (AuctionState.holder) | |
| Full auction history | Yes (AuctionState.history) | List of (player, action, bid?) tuples |
| Number of consecutive passes | Yes (AuctionState.consecutive_passes) | |
| Whether I already picked up the talon | Yes (AuctionState.awaiting_bid) | |
| Legal bids I can make | Yes (legal_bids()) | |
| Can I pick up the talon? | Yes (can_pickup()) | |
| Other players' hands | **No** | Hidden information |

### Action

- If `awaiting_bid`: choose 2 cards to discard + 1 bid from `legal_bids()`
- If auction phase: choose `pickup` or `pass` (or `stand` if holder)

### Encoding — BUILT (`auction_encoder.py`, 116 dims)

| Feature | Dim | Notes |
|---|---|---|
| Raw hand | 32 | Binary bitmap of 10–12 cards |
| Talon bitmap | 32 | Cards seen in talon (if picked up) |
| Talon seen flag | 1 | Has the player seen the talon? |
| Current bid rank | 1 | Normalised 0–1 |
| Am I the holder | 1 | Boolean |
| Consecutive passes | 1 | Normalised |
| Seat relative to dealer | 3 | One-hot |
| Auction history (last 3) | 21 | Per entry: player_rel + action_type(5) + bid_rank |
| Suit strength | 16 | Per suit: count, high-card count, has_ace, has_ten |
| Marriage potential | 4 | K+Q pair per suit |
| High-card counts | 4 | A+10+K count per suit |
| **Total** | **116** | |

---

## Phase 2 — Kontra / Rekontra

### What the player knows

| Observation | Available? | Notes |
|---|---|---|
| The winning bid (full details) | Yes | Rank, name, components, values |
| Kontrable components | Yes (kontrable_units()) | e.g. ["parti", "ulti"] |
| Current kontra levels per component | Yes (component_kontras) | 0/1/2 per unit |
| Trump suit (if chosen) | Yes | |
| Own hand | Yes | |
| All auction history | Yes | Who bid what, who passed |
| Whether I am soloist or defender | Yes | |
| Who the soloist is | Yes | |
| Marriages I hold | Derivable | K+Q pairs in hand |
| Other players' hands | **No** | |

### Action

- **Defender**: select a subset of kontrable components to kontra (or pass)
- **Soloist**: select a subset of kontra'd components to rekontra (or pass)

### Encoding — NOT YET BUILT

Suggested features:
- Hand bitmap (32)
- Winning bid rank (scalar, normalised)
- Component presence flags (parti, ulti, durchmars, betli, 40-100, 20-100) — 6 bits
- Is soloist / is defender (1 bit)
- Is red / is open (2 bits)
- Current kontra levels per component (up to 6 scalars)
- Marriage bitmask (4 bits — do I hold K+Q pairs?)
- Trump suit (4 one-hot)

---

## Phase 3 — Trump Selection

### What the player knows

Same as kontra phase, minus kontra info. The soloist chooses a suit.

### Action

- One of 3 non-red suits (Tök, Zöld, Makk)

### Encoding — NOT YET BUILT

Likely a small network or even heuristic. Features:
- Hand bitmap (32)
- Bid rank (scalar)
- Suit strength features (count per suit, high-card presence)

---

## Phase 4 — Play (Trick-Taking)

### What the player knows

This is the most complex observation space, encoded in
`encoder.py` (259-dimensional vector, v2 "detective model").

| Observation | Encoded? | Feature | Dim | Notes |
|---|---|---|---|---|
| **My hand** | ✅ | Binary bitmap | 32 | 1 = card in hand |
| **P0 captured cards** | ✅ | Binary bitmap | 32 | Card counting: who took what |
| **P1 captured cards** | ✅ | Binary bitmap | 32 | Card counting: who took what |
| **P2 captured cards** | ✅ | Binary bitmap | 32 | Card counting: who took what |
| **Current trick card 0** | ✅ | One-hot | 32 | First card played this trick |
| **Current trick card 1** | ✅ | One-hot | 32 | Second card played this trick |
| **Trump suit** | ✅ | One-hot | 4 | All zeros for Betli |
| **Is Betli** | ✅ | Scalar | 1 | |
| **Am I the soloist** | ✅ | Scalar | 1 | |
| **Trick number** | ✅ | Scalar | 1 | Normalised 0–1 (trick_no / 10) |
| **My score** | ✅ | Scalar | 1 | Normalised by total points (90) |
| **Enemy score** | ✅ | Scalar | 1 | Defenders see soloist; soloist sees sum of defenders |
| **Cards in trick so far** | ✅ | Scalar | 1 | 0, 0.5, or 1.0 |
| **Opponent void flags** | ✅ | Binary | 8 | 2 opponents × 4 suits |
| **Marriage bitmask** | ✅ | Binary | 4 | Do I hold K+Q of each suit |
| **Seven-in-hand** | ✅ | Binary | 1 | Do I hold 7 of trumps |
| **Contract DNA** | ✅ | Binary | 8 | Component flags: parti, ulti, betli, durchmars, 40, 20, is_red, is_open |
| **Trick leaders** | ✅ | Relative | 10 | Who led each trick (0.33=me, 0.67=left, 1.0=right) |
| **Trick winners** | ✅ | Relative | 10 | Who won each trick |
| **Auction context** | ✅ | Mixed | 10 | Bid rank, talon flag, seat position, kontra levels, has_ulti |
| **Marriage memory** | ✅ | Scalar | 6 | Per-player marriage totals (normalised) + has_marriage flags |
| | | | **259** | |

### What is STILL MISSING from the play encoder

| Observation | Available in state? | Priority | Notes |
|---|---|---|---|
| **Talon cards (if soloist)** | ✅ (session-level) | Medium | What the soloist discarded — only the soloist knows. Affects card counting |
| **Auction history** | ✅ `AuctionState.history` | **Medium** | What opponents bid/passed reveals information about their hands. E.g. if someone picked up and bid 40-100, they likely hold trump K+Q |
| **Terített — soloist's hand visible** | ✅ `is_open` | Medium | In terített games, opponents can see the soloist's hand — handled by PIMC determinize() but not directly encoded |
| **Defender tricks count** | Derivable | Low | How many tricks each side has won (relevant for Durchmars) — derivable from trick winner history |

### Action

- Play one card from `legal_actions(state)` (respects suit-following, trump obligation, 7esre tartás)
- Action space: 32 (one index per card in the deck), masked by legality

---

## Memory / History — "Remembering What Happened"

A key challenge for imperfect-information games: the AI should **remember**
past events to make informed decisions. Currently the encoder is
**memoryless** — it only sees the current snapshot.

### What should be remembered

| Memory | Source | Why it matters |
|---|---|---|
| **Card counting** (which cards remain) | `captured` + `hand` | Deducing opponent holdings. The "played-out" bitmap helps, but per-player captured cards are more informative |
| **Who played what card on which trick** | Trick log | Reveals opponent strategy, void suits, strength signals |
| **Bidding history** | `AuctionState.history` | A player who bid 40-100 likely holds trump K+Q. A player who passed early is weak. Pickup decisions reveal hand quality |
| **Talon contents (soloist only)** | Session data | The soloist saw the talon and chose what to discard. This is private info that affects card counting |
| **Void inference** | `known_voids` | Already tracked in `UltiNode`. When a player fails to follow suit, they're void. When they don't trump, they're also void in trump |
| **Marriage declarations** | `marriages` | Knowing that opponent declared 40 (trump marriage) means they held K+Q of trumps at game start |
| **Kontra decisions** | `component_kontras` | A defender who kontra'd ulti believes the soloist's ulti is weak |

### Approaches to encode history

1. **Fixed-size summary** (current approach, extended):
   - Per-trick winner history (10 bits)
   - Per-player captured card bitmaps (3 × 32 = 96 bits)
   - Bidding summary (bid rank, holder, number of passes)
   - Total state dim ~300–400

2. **Sequence model** (future):
   - Feed the full history as a sequence to a transformer or LSTM
   - Each timestep = one card play or auction action
   - More expressive but harder to train

3. **Belief state** (PIMC / determinization):
   - Already implemented in `determinize()`: sample hidden cards
     consistent with observations
   - MCTS runs many determinizations to approximate the belief
   - Memory is implicit: void constraints narrow the sampling

---

## Determinization (PIMC)

Already implemented in `adapter.py`:

- Gathers all cards visible to the current player (own hand + all captured + trick cards)
- Shuffles remaining unknown cards among opponents
- Respects void constraints (won't give a voided suit to a player who showed they're void)
- **Not yet considered**: auction-derived constraints (e.g. "this player bid 40-100 so they probably hold trump K+Q")

---

## Summary — Build Status

| Item | Status | Notes |
|---|---|---|
| Per-player captured card bitmaps (card counting) | ✅ Done | 3 × 32 = 96 bits in play encoder |
| Contract DNA (component flags) | ✅ Done | 8 bits in play encoder |
| Trick momentum (leader/winner history) | ✅ Done | 20 bits in play encoder |
| Auction context (bid, seat, kontras) | ✅ Done | 10 bits in play encoder |
| Marriage memory (per-player) | ✅ Done | 6 bits in play encoder |
| Kontra levels per component | ✅ Done | In auction context section |
| is_red, is_open flags | ✅ Done | In contract DNA |
| has_ulti flag | ✅ Done | In auction context |
| Auction encoder | ✅ Done | 116-dim, `auction_encoder.py` |
| Inference-enhanced PIMC | ✅ Done | Auction constraints in `determinize()` |
| Curriculum training_mode flag | ✅ Done | In `GameState` |
| Kontra phase encoder (separate network head) | Pending | |
| Trump selection encoder | Pending | |
| Talon memory for soloist | Pending | |
| Terített hand encoding for opponents | Pending | Handled by PIMC, not directly encoded |
| Sequence-based history model | Pending | Future upgrade |
