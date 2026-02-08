#!/usr/bin/env python3
"""Evaluate whether stronger MCTS parameters improve play strength.

Loads the same neural network (T6-Captain) and plays it against itself
with different MCTS configurations.
"""

import pickle
import random
import time
from pathlib import Path

from trickster.games.snapszer.adapter import SnapszerGame, SnapszerNode
from trickster.games.snapszer.game import (
    deal, deal_awarded_game_points, is_terminal, legal_actions, play_trick,
    can_exchange_trump_jack, exchange_trump_jack,
)
from trickster.mcts import MCTSConfig, alpha_mcts_choose

MODEL_PATH = Path("models/T6-Captain/net.pkl")
GAME = SnapszerGame()


def load_net():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def make_config(sims: int, dets: int) -> MCTSConfig:
    return MCTSConfig(
        simulations=sims,
        determinizations=dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.0,
        visit_temp=0.1,
    )


def az_choose(net, gs, pending_lead, player, cfg, rng):
    node = SnapszerNode(gs=gs, pending_lead=pending_lead, known_voids=(frozenset(), frozenset()))
    actions = GAME.legal_actions(node)
    # Filter out string actions (close_talon, marry_*) for simplicity
    card_actions = [a for a in actions if not isinstance(a, str)]
    if len(card_actions) <= 1:
        return card_actions[0] if card_actions else actions[0]
    return alpha_mcts_choose(node, GAME, net, player, cfg, rng)


def play_deal(net, cfg_a: MCTSConfig, cfg_b: MCTSConfig, a_idx: int, seed: int):
    """Play one deal. Player a_idx uses cfg_a, the other uses cfg_b."""
    st = deal(seed=seed, starting_leader=0)
    rng_a = random.Random(seed ^ 0xA1FA)
    rng_b = random.Random(seed ^ 0xB2FB)

    while not is_terminal(st):
        leader = st.leader
        responder = 1 - leader

        # Auto-exchange for leader
        if can_exchange_trump_jack(st, leader):
            exchange_trump_jack(st, leader)

        if leader == a_idx:
            lead_card = az_choose(net, st, None, leader, cfg_a, rng_a)
        else:
            lead_card = az_choose(net, st, None, leader, cfg_b, rng_b)

        # Handle string actions (skip them, pick a card)
        if isinstance(lead_card, str):
            cards = [c for c in legal_actions(st, leader, None) if not isinstance(c, str)]
            lead_card = random.choice(cards)

        if responder == a_idx:
            resp_card = az_choose(net, st, lead_card, responder, cfg_a, rng_a)
        else:
            resp_card = az_choose(net, st, lead_card, responder, cfg_b, rng_b)

        if isinstance(resp_card, str):
            cards = [c for c in legal_actions(st, responder, lead_card) if not isinstance(c, str)]
            resp_card = random.choice(cards)

        st, _ = play_trick(st, lead_card, resp_card)

    winner, pts, reason = deal_awarded_game_points(st)
    return winner, pts


def run_match(net, cfg_a, cfg_b, label_a, label_b, n_deals=200):
    a_total = 0
    b_total = 0
    t0 = time.time()
    for g in range(n_deals):
        # Alternate sides
        if g % 2 == 0:
            a_idx = 0
        else:
            a_idx = 1
        seed = 10000 + (g // 2)
        winner, pts = play_deal(net, cfg_a, cfg_b, a_idx, seed)
        if winner == a_idx:
            a_total += pts
        else:
            b_total += pts
        if (g + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{g+1:3d}/{n_deals}]  {label_a}: {a_total}  {label_b}: {b_total}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    a_ppd = a_total / n_deals
    b_ppd = b_total / n_deals
    print(f"\n  Result ({n_deals} deals, {elapsed:.1f}s):")
    print(f"    {label_a:20s}  pts={a_total:5d}  ppd={a_ppd:.3f}")
    print(f"    {label_b:20s}  pts={b_total:5d}  ppd={b_ppd:.3f}")
    print(f"    Δ = {a_ppd - b_ppd:+.3f} ({label_a} {'stronger' if a_ppd > b_ppd else 'weaker'})")
    return a_ppd, b_ppd


def main():
    print("Loading T6-Captain net...")
    net = load_net()

    configs = [
        ("Base (50s×6d)",    make_config(50, 6)),
        ("Medium (100s×10d)", make_config(100, 10)),
        ("Strong (200s×12d)", make_config(200, 12)),
    ]

    print(f"\n{'='*60}")
    print("  MCTS Strength Test — same net, different search budgets")
    print(f"{'='*60}\n")

    n_deals = 200

    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            label_a, cfg_a = configs[i]
            label_b, cfg_b = configs[j]
            print(f"\n┌─ {label_a}  vs  {label_b} ─────────────────")
            run_match(net, cfg_a, cfg_b, label_a, label_b, n_deals)
            print()


if __name__ == "__main__":
    main()
