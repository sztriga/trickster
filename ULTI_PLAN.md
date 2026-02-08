# Trickster AI: Ulti Expansion Development Plan

This document outlines the step-by-step transition from Snapszer (2-player) to Ulti (3-player, asymmetric). The development follows a **Strict TDD (Test-Driven Development)** methodology to manage the increased complexity of bidding and contract-specific rules.

---

## Phase 1: The Core 3-Player Engine
**Objective:** Transition the game state from 1v1 to a 1v2 asymmetric environment.

### 1.1 32-Card Tell Deck & Dealing
* **Implementation:** Define the 32-card Hungarian (Tell) deck. Implement the dealer rotation and the specific 7-5-5, 5-5-5 distribution logic.
* **Test Case (`test_dealing.py`):** * Verify total card count is 32.
    * Verify Player 1 (Dealer's right) receives 12 cards, while others receive 10.
    * Verify the "Talon" is empty after the soloist picks up their extra 2 cards.

### 1.2 Asymmetric Player Roles
* **Implementation:** Logic to identify the "Soloist" (Declarer) and the "Defenders" (Opponents).
* **Test Case (`test_roles.py`):**
    * Given an index $i$, verify `is_soloist(i)` is true and `is_defender(j)` is true for all $j \neq i$.
    * Ensure the turn order cycles correctly: $0 \to 1 \to 2 \to 0$.

---

## Phase 2: The Bidding & Talon System
**Objective:** Handle the "Game-within-a-Game" where the contract is determined.

### 2.1 The Bidding Ladder (54 Bids)
* **Implementation:** Define the hierarchy of Ulti bids (Simple, 40-100, Ulti, Betli, Durchmars, etc.) and their point values.
* **Test Case (`test_bidding.py`):**
    * Assert `Bid("Ulti") > Bid("Simple")`.
    * Assert `Bid("Heart-Simple") > Bid("Leaf-Simple")`.
    * Validate the `get_legal_bids()` mask returns only bids higher than the current auction price.

### 2.2 Soloist Talon Exchange
* **Implementation:** The mechanic where the Soloist picks up 2 cards and discards 2 hidden cards to "set" their hand.
* **Test Case (`test_talon_exchange.py`):**
    * Verify the Soloist cannot discard 10s or Aces (depending on house rules selected).
    * Verify discarded cards are removed from play and added to the Soloist's "captured" pile for scoring.

---

## Phase 3: Contract-Specific Logic (The "Brain Switch")
**Objective:** Dynamically change rules based on the active contract.

### 3.1 Rule Variant Factory
* **Implementation:** A logic controller that switches between Trump-based play (Simple), No-Trump (Betli), and specific obligations (Ulti).
* **Test Case (`test_rule_variants.py`):**
    * **Betli:** Assert that "Must follow suit" is active but "Must beat" is inactive.
    * **100-Point:** Assert that the scoring logic tracks "Marriages" (20/40) during play.

### 3.2 The "Trump 7" (Ulti) Constraint
* **Implementation:** Logic to track if the Trump 7 is played in the 10th trick.
* **Test Case (`test_ulti_contract.py`):**
    * Simulate Soloist playing Trump 7 on Trick 10 $\to$ Success.
    * Simulate Soloist playing Trump 7 on Trick 9 $\to$ Failure.
    * Simulate Defender beating the Soloist's Trump 7 on Trick 10 $\to$ Failure.

---

## Phase 4: AI & Reward Engineering
**Objective:** Train the model to understand asymmetric values.

### 4.1 Zero-Sum Asymmetric Rewards
* **Implementation:** A reward function $z$ that maps contract outcomes to a 3-way vector.
* **Test Case (`test_rewards.py`):**
    * If Soloist wins a 4-point game: `rewards = [4, -2, -2]`.
    * If Soloist loses a 4-point game: `rewards = [-8, 4, 4]` (Double penalty for losing).

### 4.2 Hierarchical Action Space
* **Implementation:** Update `adapter.py` to handle 2 phases: Bidding Action Space (Index 0-53) and Play Action Space (Index 54-85).
* **Test Case (`test_action_masking.py`):**
    * In the Bidding phase, assert play-card actions are masked to $-\infty$.
    * In the Play phase, assert bidding actions are masked to $-\infty$.

---

## Phase 5: MCTS & Determinization
**Objective:** Handle hidden information across two opponents.

### 5.1 3-Player PIMC (Determinization)
* **Implementation:** Sample hidden cards by shuffling remaining cards and splitting them into two pools for the defenders.
* **Test Case (`test_pimc_sampling.py`):**
    * Ensure no card is assigned to both defenders simultaneously.
    * Incorporate "Void Tracking": If Defender A failed to follow Hearts previously, ensure the sampler gives them 0 Hearts.

---

## 📋 Summary of Development Steps

| Step | Component | TDD Verification |
| :--- | :--- | :--- |
| 1 | `UltiGame` State | 32-card deal & 3-player rotation tests. |
| 2 | `BidLadder` | Hierarchy validation & legal bid masking. |
| 3 | `TalonExchange` | 12-card hand $\to$ 10-card hand transition. |
| 4 | `ContractRules` | Success/Failure logic for Betli and Ulti. |
| 5 | `AsymmetricMCTS` | Joint defender rewards vs. Soloist rewards. |
| 6 | `SharedAlphaNet` | Updated input features for 3 players. |

---

### Next Immediate Task
1.  **Define `UltiBid` Enum:** List all 54 possible bids.
2.  **Write `test_bidding.py`:** Define the "overbidding" logic.
3.  **Implement `BidMasker`:** To be used by the AI's Policy Head.