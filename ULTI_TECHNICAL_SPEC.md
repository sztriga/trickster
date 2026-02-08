# 🏛️ Ulti Technical Specification: The Source of Truth (v2.0)

## 1. Game Setup & Card Mechanics
* **Players:** 3 (Soloist vs. Defenders).
* **Deck:** 32 Hungarian cards (Tell deck).
* **Card Points:** * Ace (11), 10 (10), King (4), Over (3), Under (2).
    * IX, VIII, VII (0 points).
    * **Total Card Points:** 120 + 10 points for the last trick = 130 total points.
* **Rankings:**
    * **Trump/Simple:** A, 10, K, O, U, IX, VIII, VII.
    * **Betli/Durchmars:** A, K, O, U, **10**, IX, VIII, VII (10 moves down).
* **Deal (Counter-Clockwise):**
    * Dealer deals 5 to each, then 2 to Starting Bidder, then 5 to each.
    * Distribution: 12 cards for the first bidder, 10 cards for others.

---

## 2. The Auction Phase (Licitálás)
The auction determines the **Soloist**, the **Contract**, and the **Trump Suit**.

### 2.1 The "Mondás" vs. "Tartás" Logic
* **Mondás (Saying):** A player picks up the 2-card Talon, discards 2, and announces a *higher* bid.
* **Tartás (Holding):** A player who already made a bid may "Hold" the auction if a subsequent player tries to bid a contract of equal value. To "overbid" a holder, you must name a contract of strictly higher value.
* **Double Value:** Bids in **Hearts (Piros)** are always worth double the base value.

### 2.2 The Bidding Hierarchy (The Ladder)
Precedence rule: If two bids have the same value, the one with **fewer components** is higher.
* *Example:* **Betli** (5 pts, 1 component) > **Ulti-Passz** (4+1=5 pts, 2 components).

| Rank | Contract | Point Value (Base) | Heart Value |
| :--- | :--- | :--- | :--- |
| 1 | **Passz (Simple)** | 1 | 2 |
| 2 | **40-100** | 4 | 8 |
| 3 | **Ulti** | 4 (+1 penalty) | 8 (+2 penalty) |
| 4 | **Betli** | 5 | 10 |
| 5 | **Durchmars** | 6 | 12 |
| 6 | **40-100 Ulti** | 8 | 16 |
| 7 | **20-100** | 8 | 16 |
| 8 | **40-100 Durchmars** | 10 | 20 |
| 9 | **Ulti Durchmars** | 10 | 20 |
| 10 | **20-100 Ulti** | 12 | 24 |
| 11 | **Terített (Open) Betli** | 15 | 30 |
| 12 | **20-100 Durchmars** | 14 | 28 |
| 13 | **Open Durchmars** | 24 | 48 |

---

## 3. Play Rules (Mandatory)
* **Suiting (Színkötelezettség):** You must follow the lead suit.
* **Must-Beat (Felülütés):** You **must** play a higher card than the current winner if you have one of that suit.
* **Trumping (Adukötelezettség):** If you can't follow suit, you **must** play a trump. If multiple trumps are played, you must beat the highest trump currently on the table.
* **Betli Variation:** Standard tournament rules: No Trump, Must Follow Suit, **No Must-Beat**.

---

## 4. Scoring & Rewards (The AI Target)
Ulti is zero-sum. The Soloist wins or loses against **each** defender individually.

### 4.1 Announced vs. Silent Bonuses
* **Announced (Licitált):** Full contract value. If lost, Soloist pays $Value + Penalty$ to each defender.
* **Silent (Csendes):**
    * Silent Ulti (7 of trump wins trick 10): +2 points.
    * Silent 100 (Reaching 100 pts without bidding it): +2 points.
    * Silent Durchmars (Winning all tricks): +3 points (replaces Simple).

### 4.2 Doubling (Kontra)
* **Kontra:** Doubles the component value (e.g., Kontra Simple, Kontra Ulti).
* **Penalty Logic:** Doubling applies to the **Base**, but the **Loss Penalty** for Ulti is usually a flat addition *after* the doubling calculation to prevent infinite scaling of penalties.

---

## 5. Implementation Notes for Developer AI
* **The Talon:** Discarded cards count as "captured" by the Soloist at the start of the game for scoring points (Aces/10s).
* **MCTS State:** Feature vector must include `contract_type`, `trump_suit`, `bidding_history`, and `void_flags` for opponents.
* **The 2v1 Coalition:** During MCTS, if `current_player` is a Defender, the reward $z$ used for backpropagation must be $z_{def1} + z_{def2}$. They act as a single unit with two hands.