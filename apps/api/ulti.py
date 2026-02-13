"""FastAPI router for Ulti game sessions.

Human is always player 0 (bottom of screen).  AI uses trained UltiNet.
Game flow: BID → AUCTION → TRUMP_SELECT → KONTRA → PLAY → DONE.

AI play modes (configurable per session):
  - "neural"  : UltiNet policy head only — instant, no search.
  - "mcts"    : UltiNet + MCTS search — stronger but slower.
  - "solver"  : PIMC with exact alpha-beta solver — strongest, may be slow.
  - "random"  : Random legal card (legacy / fallback).

The model is loaded lazily from ``models/ulti_mixed.pt`` on first use.
"""

from __future__ import annotations

import logging
import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from trickster.games.ulti.adapter import (
    UltiGame,
    UltiNode,
    _CARD_IDX,
    _IDX_TO_CARD,
    build_auction_constraints,
)
from trickster.games.ulti.cards import Card, Rank, Suit
from trickster.mcts import MCTSConfig, alpha_mcts_policy
from trickster.model import UltiNet, UltiNetWrapper
from trickster.solver import SolverPIMC

log = logging.getLogger(__name__)
from trickster.games.ulti.auction import (
    ALL_BIDS,
    AuctionState,
    Bid,
    BID_PASSZ,
    COMP_100,
    COMP_20,
    COMP_40,
    COMP_BETLI,
    COMP_DURCHMARS,
    COMP_PARTI,
    COMP_ULTI,
    SUIT_NAMES,
    ai_bid_after_pickup,
    ai_initial_bid,
    ai_should_pickup,
    can_pickup,
    component_value_map,
    contract_loss_value,
    contract_value,
    create_auction,
    kontrable_units,
    legal_bids,
    marriage_restriction,
    submit_bid,
    submit_pass,
    submit_pickup,
)
from trickster.games.ulti.game import (
    GameState,
    current_player,
    deal,
    declare_all_marriages,
    defender_has_20,
    defender_has_40,
    defender_points,
    defender_won_durchmars,
    is_terminal,
    last_trick_ulti_check,
    legal_actions,
    marriage_points,
    next_player,
    pickup_talon,
    play_card,
    set_contract,
    soloist_has_20,
    soloist_has_40,
    soloist_lost_betli,
    soloist_points,
    soloist_tricks,
    soloist_won_durchmars,
    soloist_won_simple,
)
from trickster.games.ulti.rules import TrickResult

# ---------------------------------------------------------------------------
#  Neural AI — model loading (lazy singleton)
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_DEFAULT_MODEL = "ulti_parti.pt"  # legacy; also checks models/ulti/*/model.pt

# Singleton: loaded once, shared across all sessions.
_ulti_game: UltiGame | None = None
_ulti_wrapper: UltiNetWrapper | None = None
_model_path_loaded: str = ""


def _get_ulti_game() -> UltiGame:
    global _ulti_game
    if _ulti_game is None:
        _ulti_game = UltiGame()
    return _ulti_game


def _load_ulti_model(model_path: str | None = None) -> UltiNetWrapper | None:
    """Load a trained UltiNet from disk (cached singleton).

    Falls back through: explicit path → env var ``ULTI_MODEL`` →
    ``models/ulti_mixed.pt`` → any ``models/ulti_*.pt``.
    Returns ``None`` if no model file is found.
    """
    global _ulti_wrapper, _model_path_loaded

    if model_path is None:
        model_path = os.environ.get("ULTI_MODEL")
    if model_path is None:
        candidate = _MODEL_DIR / _DEFAULT_MODEL
        if candidate.exists():
            model_path = str(candidate)
        else:
            # Try tiered models (prefer strongest tier)
            ulti_dir = _MODEL_DIR / "ulti"
            if ulti_dir.exists():
                for tier_dir in sorted(ulti_dir.iterdir(), reverse=True):
                    mp = tier_dir / "model.pt"
                    if mp.exists():
                        model_path = str(mp)
                        break
        if model_path is None:
            # Fallback: any ulti_*.pt
            for f in sorted(_MODEL_DIR.glob("ulti_*.pt")):
                model_path = str(f)
                break

    if model_path is None:
        log.warning("No Ulti model found — AI will play randomly.")
        return None

    # Return cached if same path
    if _ulti_wrapper is not None and _model_path_loaded == model_path:
        return _ulti_wrapper

    try:
        checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
        body_units = checkpoint.get("body_units", 256)
        body_layers = checkpoint.get("body_layers", 4)
        input_dim = checkpoint.get("input_dim", 291)
        action_dim = checkpoint.get("action_dim", 32)

        net = UltiNet(
            input_dim=input_dim,
            body_units=body_units,
            body_layers=body_layers,
            action_dim=action_dim,
        )
        net.load_state_dict(checkpoint["model_state_dict"])

        _ulti_wrapper = UltiNetWrapper(net, device="cpu")
        _model_path_loaded = model_path
        log.info("Loaded Ulti model from %s (%d params)",
                 model_path, sum(p.numel() for p in net.parameters()))
        return _ulti_wrapper
    except Exception:
        log.exception("Failed to load Ulti model from %s", model_path)
        return None


def _session_to_node(sess: "UltiSession") -> UltiNode:
    """Build a UltiNode from a live session for neural inference.

    Populates known_voids, contract_components, kontras, and
    auction constraints so the encoder produces a full 259-dim vector.
    """
    st = sess.state
    bid = sess.winning_bid

    # Contract components
    comps: frozenset[str] | None = None
    is_red = False
    is_open = False
    bid_rank = 0
    if bid is not None:
        comps = bid.components
        is_red = bid.is_red
        is_open = bid.is_open
        bid_rank = bid.rank

    # Build auction constraints
    constraints = build_auction_constraints(st, comps)

    # Known voids: not tracked in the UI session, start empty
    # (could be inferred from trick history, but not critical)
    empty_voids = (frozenset[Suit](), frozenset[Suit](), frozenset[Suit]())

    return UltiNode(
        gs=st,
        known_voids=empty_voids,
        bid_rank=bid_rank,
        is_red=is_red,
        is_open=is_open,
        contract_components=comps,
        dealer=st.dealer,
        component_kontras=dict(sess.component_kontras),
        must_have=constraints,
    )


# Solver PIMC configs for different strength levels
_SOLVER_CONFIGS: dict[str, dict[str, int]] = {
    "fast": {"num_determinizations": 5, "max_exact_tricks": 5},
    "medium": {"num_determinizations": 15, "max_exact_tricks": 6},
    "strong": {"num_determinizations": 30, "max_exact_tricks": 6},
}

# Lazy singleton solver instances
_solver_instances: dict[str, SolverPIMC] = {}


def _get_solver(strength: str = "medium") -> SolverPIMC:
    """Get or create a SolverPIMC instance for the given strength."""
    if strength not in _solver_instances:
        cfg = _SOLVER_CONFIGS.get(strength, _SOLVER_CONFIGS["medium"])
        _solver_instances[strength] = SolverPIMC(**cfg)
    return _solver_instances[strength]


# MCTS configs for different AI strength levels
_MCTS_CONFIGS: dict[str, MCTSConfig] = {
    "fast": MCTSConfig(
        simulations=5, determinizations=1,
        c_puct=1.5, dirichlet_alpha=0.0, dirichlet_weight=0.0,
        use_value_head=True, use_policy_priors=True, visit_temp=0.1,
    ),
    "medium": MCTSConfig(
        simulations=20, determinizations=2,
        c_puct=1.5, dirichlet_alpha=0.0, dirichlet_weight=0.0,
        use_value_head=True, use_policy_priors=True, visit_temp=0.1,
    ),
    "strong": MCTSConfig(
        simulations=50, determinizations=3,
        c_puct=1.5, dirichlet_alpha=0.0, dirichlet_weight=0.0,
        use_value_head=True, use_policy_priors=True, visit_temp=0.1,
    ),
}


# ---------------------------------------------------------------------------
#  JSON helpers
# ---------------------------------------------------------------------------

_RANK_MAP: dict[str, Rank] = {r.name: r for r in Rank}
_SUIT_MAP: dict[str, Suit] = {s.value: s for s in Suit}


def _card_json(c: Card) -> dict[str, str]:
    return {"suit": c.suit.value, "rank": c.rank.name}


def _card_from_json(obj: dict[str, Any]) -> Card:
    try:
        return Card(_SUIT_MAP[obj["suit"]], _RANK_MAP[obj["rank"]])
    except Exception as e:
        raise HTTPException(400, f"Invalid card: {obj!r} ({e})")


def _trick_json(tr: TrickResult) -> dict[str, Any]:
    return {
        "cards": [_card_json(c) for c in tr.cards],
        "players": list(tr.players),
        "winner": tr.winner,
    }


def _bid_json(bid: Bid) -> dict[str, Any]:
    return {
        "rank": bid.rank,
        "name": bid.name,
        "winValue": bid.win_value,
        "lossValue": bid.loss_value,
        "displayWin": bid.display_win(),
        "displayLoss": bid.display_loss(),
        "trumpMode": bid.trump_mode,
        "isOpen": bid.is_open,
        "label": bid.label(),
    }


def _auction_history_json(
    history: list[tuple[int, str, Optional[Bid]]],
) -> list[dict[str, Any]]:
    result = []
    for player, action_type, bid in history:
        entry: dict[str, Any] = {"player": player, "action": action_type}
        if bid is not None:
            entry["bid"] = _bid_json(bid)
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
#  Session
# ---------------------------------------------------------------------------

PLAYER_LABELS = ["Te", "Gép 1", "Gép 2"]


@dataclass
class UltiSession:
    id: str
    state: GameState
    talon: list[Card]  # original deal talon (reference only)
    phase: str  # "bid"|"auction"|"trump_select"|"kontra"|"rekontra"|"play"|"done"
    seed: int
    auction: Optional[AuctionState] = None
    winning_bid: Optional[Bid] = None
    # Per-component kontra levels (adu games — shared across defenders)
    # component label → 0 (none) / 1 (kontra) / 2 (rekontra)
    component_kontras: dict[str, int] = field(default_factory=dict)
    # Colorless games: per-defender kontra level on the single component
    # defender player index → 0 / 1 / 2
    colorless_kontras: dict[int, int] = field(default_factory=dict)
    kontra_defender_idx: int = 0  # which defender is up (0=first, 1=second)
    needs_continue: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)
    # Speech-bubble events: [{player: int, text: str}, ...]
    bubbles: list[dict[str, Any]] = field(default_factory=list)
    # AI play mode: "neural" | "mcts" | "random"
    ai_mode: str = "neural"
    # MCTS strength level: "fast" | "medium" | "strong" (only for mcts mode)
    ai_strength: str = "medium"


_sessions: dict[str, UltiSession] = {}

HUMAN = 0  # human is always player 0


# ---------------------------------------------------------------------------
#  State serialisation
# ---------------------------------------------------------------------------


def _sort_hand(hand: list[Card]) -> list[Card]:
    return sorted(hand, key=lambda c: (c.suit.value, c.rank.value))


def _public_state(sess: UltiSession) -> dict[str, Any]:
    st = sess.state
    cp = current_player(st) if sess.phase == "play" and not sess.needs_continue else None

    # Legal cards (for play or discard selection)
    legal: list[dict[str, str]] = []
    if sess.phase == "play" and cp == HUMAN and not sess.needs_continue:
        legal = [_card_json(c) for c in legal_actions(st)]
    elif sess.phase == "bid" and sess.auction and sess.auction.turn == HUMAN:
        # During bid phase all 12 cards are selectable for discard
        legal = [_card_json(c) for c in st.hands[HUMAN]]

    # Auction data
    auction_data: dict[str, Any] | None = None
    if sess.auction is not None:
        legal_bid_list: list[dict[str, Any]] = []
        pickup_ok = False
        is_holder = False

        a = sess.auction
        if a.turn == HUMAN and not a.done:
            if a.awaiting_bid:
                # Human must bid — show legal bids
                legal_bid_list = [_bid_json(b) for b in legal_bids(a)]
            else:
                # Human in auction phase — can pickup or pass
                pickup_ok = can_pickup(a)
                is_holder = (a.turn == a.holder)

        auction_data = {
            "turn": a.turn,
            "currentBid": _bid_json(a.current_bid) if a.current_bid else None,
            "holder": a.holder,
            "firstBidder": a.first_bidder,
            "history": _auction_history_json(a.history),
            "done": a.done,
            "winner": a.winner,
            "legalBids": legal_bid_list,
            "canPickup": pickup_ok,
            "isHolderTurn": is_holder,
            "awaitingBid": a.awaiting_bid,
        }

    # Kontra data
    kontra_data: dict[str, Any] | None = None
    if sess.phase in ("kontra", "rekontra"):
        bid = sess.winning_bid
        is_colorless = bid is not None and bid.is_colorless
        units = kontrable_units(bid) if bid else []
        defenders = [i for i in range(3) if i != st.soloist]

        if sess.phase == "kontra":
            turn = defenders[sess.kontra_defender_idx] if sess.kontra_defender_idx < len(defenders) else None
        else:
            turn = st.soloist

        kontra_data = {
            "phase": sess.phase,  # "kontra" or "rekontra"
            "turn": turn,
            "isMyTurn": turn == HUMAN,
            "kontrable": units,
            "currentKontras": dict(sess.component_kontras),
            "isColorless": is_colorless,
        }

    # Contract info (after auction)
    contract_info: dict[str, Any] | None = None
    if sess.winning_bid is not None:
        bid = sess.winning_bid
        contract_info = {
            "bid": _bid_json(bid),
            "componentKontras": dict(sess.component_kontras),
            "displayWin": bid.display_win(),
            "displayLoss": bid.display_loss(),
        }

    # Trump selection options
    trump_options: list[str] | None = None
    if sess.phase == "trump_select":
        trump_options = [s.value for s in Suit if s != Suit.HEARTS]

    # Result message and settlement
    result_msg: str | None = None
    settlement_data: dict[str, Any] | None = None
    if sess.phase == "done" and sess.winning_bid is not None:
        bid = sess.winning_bid
        sol_label = PLAYER_LABELS[st.soloist]

        settlement = _compute_final_settlement(st, bid, sess.component_kontras)
        settlement_data = settlement

        if bid.rank == BID_PASSZ.rank and st.trick_no == 0:
            result_msg = f"Mindenki passzolt — {sol_label} fizet 2-2 pontot."
        else:
            won = settlement["soloistWon"]
            net = settlement["netPerDefender"]
            silent_parts = settlement["silentBonuses"]

            if won:
                base_str = f"+{settlement['contractResult']}"
            else:
                base_str = f"{settlement['contractResult']}"

            if silent_parts:
                silent_str = ", ".join(
                    f"{sb['label']} ({'+' if sb['points'] > 0 else ''}{sb['points']})"
                    for sb in silent_parts
                )
                result_msg = (
                    f"{sol_label} {'nyert' if won else 'vesztett'}! "
                    f"({bid.label()}, {base_str}/védő) "
                    f"+ {silent_str} = {'+' if net > 0 else ''}{net}/védő"
                )
            else:
                result_msg = (
                    f"{sol_label} {'nyert' if won else 'vesztett'}! "
                    f"({bid.label()}, {'+' if net > 0 else ''}{net}/védő)"
                )

    # Terített: the SOLOIST shows their cards to all players.
    is_teritett = sess.winning_bid is not None and sess.winning_bid.is_open
    soloist_hand: list[dict[str, str]] | None = None
    if is_teritett and sess.phase in ("play", "done") and st.soloist != HUMAN:
        # AI soloist: send their hand so the human can see it.
        soloist_hand = [_card_json(c) for c in _sort_hand(st.hands[st.soloist])]

    # Per-player declared marriage totals (visible to all — suit hidden).
    declared_marriages = [marriage_points(st, p) for p in range(3)]

    # My captured cards grouped as tricks (3 cards per group).
    my_captured = [_card_json(c) for c in st.captured[HUMAN]]
    captured_tricks: list[list[dict[str, str]]] = []
    for i in range(0, len(my_captured), 3):
        captured_tricks.append(my_captured[i : i + 3])

    # Consume bubbles (send once, then clear).
    bubbles = list(sess.bubbles)
    sess.bubbles.clear()

    return {
        "gameId": sess.id,
        "phase": sess.phase,
        "hand": [_card_json(c) for c in _sort_hand(st.hands[HUMAN])],
        "aiHandSizes": [len(st.hands[1]), len(st.hands[2])],
        "trickCards": [{"player": p, "card": _card_json(c)} for p, c in st.trick_cards],
        "scores": list(st.scores),
        "currentPlayer": cp,
        "leader": st.leader,
        "trickNo": st.trick_no,
        "soloist": st.soloist,
        "trump": st.trump.value if st.trump else None,
        "betli": st.betli,
        "legalCards": legal,
        "lastTrick": _trick_json(st.last_trick) if st.last_trick else None,
        "needsContinue": sess.needs_continue,
        "dealOver": sess.phase == "done",
        "log": sess.log,
        "seed": sess.seed,
        "dealer": st.dealer,
        "soloistPoints": soloist_points(st),
        # During play, show only trick-based defender points (no talon).
        # At game end, include talon points.
        "defenderPoints": defender_points(st) if sess.phase == "done" else (
            sum(s for i, s in enumerate(st.scores) if i != st.soloist)
        ),
        "auction": auction_data,
        "kontra": kontra_data,
        "contract": contract_info,
        "trumpOptions": trump_options,
        "resultMessage": result_msg,
        "declaredMarriages": declared_marriages,
        "isTeritett": is_teritett,
        "soloistHand": soloist_hand,
        "settlement": settlement_data,
        "capturedTricks": captured_tricks,
        "bubbles": bubbles,
        "aiMode": sess.ai_mode,
        "aiStrength": sess.ai_strength,
        # Talon cards: revealed only when the game is over.
        "talonCards": [_card_json(c) for c in st.talon_discards] if (
            sess.phase == "done" and st.talon_discards
        ) else None,
    }


# ---------------------------------------------------------------------------
#  Win condition check
# ---------------------------------------------------------------------------


def _soloist_won(st: GameState, bid: Bid) -> bool:
    """Evaluate whether the soloist won the contracted game.

    Each bid is decomposed into components (from its ``components``
    frozenset).  ALL components must be satisfied; if any one fails
    the entire contract is lost.

    Component rules (Hungarian Tournament)
    ---------------------------------------
    - **parti**: Soloist total > defenders total (foundation of all Adu games).
    - **40**:    Declared 40 (trump K+Q marriage).
    - **20**:    Declared 20 (non-trump K+Q marriage).
    - **100**:   Total points (tricks + marriages) >= 100.
    - **ulti**:  Won the last trick with 7 of trumps.
    - **durchmars**: Won all 10 tricks.
    - **betli**:  Won zero tricks (no trump, no Simple).
    """
    comps = bid.components

    # --- Betli: completely separate logic ---
    if COMP_BETLI in comps:
        return not soloist_lost_betli(st)

    # --- Build component checks for Adu games ---
    checks: list[bool] = []

    if COMP_PARTI in comps:
        checks.append(soloist_won_simple(st))

    if COMP_40 in comps:
        checks.append(soloist_has_40(st))

    if COMP_20 in comps:
        checks.append(soloist_has_20(st))

    if COMP_100 in comps:
        checks.append(soloist_points(st) >= 100)

    if COMP_ULTI in comps:
        if st.last_trick is None or st.trump is None:
            checks.append(False)
        else:
            seven = Card(st.trump, Rank.SEVEN)
            checks.append(
                st.last_trick.winner == st.soloist
                and seven in st.last_trick.cards
            )

    if COMP_DURCHMARS in comps:
        checks.append(soloist_won_durchmars(st))

    return all(checks)


# ---------------------------------------------------------------------------
#  Silent (Csendes) bonus evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SilentBonus:
    """One silent achievement."""
    label: str       # e.g. "Csendes 40-100", "Bukott csendes ulti"
    points: int      # per-defender value (positive = soloist earns, negative = pays)


def _compute_silent_bonuses(
    st: GameState,
    bid: Bid,
    kontras: dict[str, int],
) -> tuple[list[SilentBonus], set[str]]:
    """Compute all silent bonuses for a completed game.

    Silent bonus = half the component's win value.
    Applied per-defender (soloist earns/pays X to EACH defender).

    Rules:
      - Silent 100 (40-100 or 20-100): REPLACES parti.
        The parti component is removed from the base contract result,
        and the full silent value (×kontra of the parti slot) is charged.
      - Silent Durchmars: always STACKS (additive, no kontra inherited).
      - Silent Ulti: always STACKS (additive, no kontra inherited).
      - Fallen (bukott) silent ulti: penalty = component value.
      - Only in adu (trump) games — not Betli or colorless.
      - Both soloist and defenders can earn/lose silent bonuses.

    Returns ``(bonuses, replaced_units)`` where ``replaced_units`` is the
    set of kontrable unit labels (e.g. ``{"parti"}``) whose contribution
    should be excluded from the announced base contract result.
    """
    comps = bid.components
    bonuses: list[SilentBonus] = []
    replaced: set[str] = set()

    # No silent bonuses in Betli or colorless games (no trump → no marriages/ulti)
    if COMP_BETLI in comps or st.trump is None:
        return bonuses, replaced

    M = 2 if bid.is_red else 1

    # Silent component half-values (per defender, before kontra)
    SILENT_100_40 = 2 * M   # half of 40-100 component (4*M / 2)
    SILENT_100_20 = 4 * M   # half of 20-100 component (8*M / 2)
    SILENT_ULTI = 2 * M     # half of ulti component (4*M / 2)
    SILENT_DURI = 3 * M     # half of durchmars component (6*M / 2)
    BUKOTT_ULTI = 4 * M     # fallen ulti = component value (4*M*2 / 2)

    # Determine the "parti slot" — which kontra key guards the base game.
    # Silent 100 replaces parti, so it inherits this kontra multiplier.
    vm = component_value_map(bid)
    parti_slot: str | None = "parti" if "parti" in vm else None

    # Kontra multiplier for the parti slot
    parti_kontra_mult = 2 ** kontras.get(parti_slot, 0) if parti_slot else 1

    # Check if there's a separate parti component that can be replaced
    has_separate_parti = parti_slot is not None

    # ---------------------------------------------------------------
    #  Soloist silent achievements
    # ---------------------------------------------------------------

    sol_has_silent_100 = False
    sol_silent_100_val = 0
    sol_silent_100_label = ""

    # Silent 100 (only if 100 not already announced)
    if COMP_100 not in comps:
        sol_total = soloist_points(st)
        if soloist_has_40(st) and sol_total >= 100:
            sol_silent_100_val = max(sol_silent_100_val, SILENT_100_40)
            sol_silent_100_label = "Csendes 40-100"
            sol_has_silent_100 = True
        if soloist_has_20(st) and sol_total >= 100:
            sol_silent_100_val = max(sol_silent_100_val, SILENT_100_20)
            sol_silent_100_label = "Csendes 20-100"
            sol_has_silent_100 = True

    # Silent Durchmars (only if duri not already announced) — always stacks
    if COMP_DURCHMARS not in comps and soloist_won_durchmars(st):
        bonuses.append(SilentBonus("Csendes durchmars", SILENT_DURI))

    # Silent 100: replaces parti — full value charged with kontra
    if sol_has_silent_100:
        if has_separate_parti:
            replaced.add("parti")
            bonuses.append(SilentBonus(
                sol_silent_100_label,
                sol_silent_100_val * parti_kontra_mult,
            ))
        else:
            # No separate parti to replace — pure additive
            bonuses.append(SilentBonus(sol_silent_100_label, sol_silent_100_val))

    # Silent Ulti — always additive (only if ulti not already announced)
    if COMP_ULTI not in comps:
        side, won = last_trick_ulti_check(st)
        if side == "soloist" and won:
            bonuses.append(SilentBonus("Csendes ulti", SILENT_ULTI))
        elif side == "soloist" and not won:
            bonuses.append(SilentBonus("Bukott csendes ulti", -BUKOTT_ULTI))

    # ---------------------------------------------------------------
    #  Defender silent achievements (negative = soloist pays)
    # ---------------------------------------------------------------

    def_has_silent_100 = False
    def_silent_100_val = 0
    def_silent_100_label = ""

    if COMP_100 not in comps:
        def_total = defender_points(st)
        if defender_has_40(st) and def_total >= 100:
            def_silent_100_val = max(def_silent_100_val, SILENT_100_40)
            def_silent_100_label = "Védők: csendes 40-100"
            def_has_silent_100 = True
        if defender_has_20(st) and def_total >= 100:
            def_silent_100_val = max(def_silent_100_val, SILENT_100_20)
            def_silent_100_label = "Védők: csendes 20-100"
            def_has_silent_100 = True

    # Defender silent Durchmars — always stacks
    if COMP_DURCHMARS not in comps and defender_won_durchmars(st):
        bonuses.append(SilentBonus("Védők: csendes durchmars", -SILENT_DURI))

    # Defender silent 100: replaces parti — full value charged with kontra
    if def_has_silent_100:
        if has_separate_parti:
            replaced.add("parti")
            bonuses.append(SilentBonus(
                def_silent_100_label,
                -(def_silent_100_val * parti_kontra_mult),
            ))
        else:
            bonuses.append(SilentBonus(def_silent_100_label, -def_silent_100_val))

    # Defender silent/fallen Ulti
    if COMP_ULTI not in comps:
        side, won = last_trick_ulti_check(st)
        if side == "defender" and won:
            bonuses.append(SilentBonus("Védők: csendes ulti", -SILENT_ULTI))
        elif side == "defender" and not won:
            bonuses.append(SilentBonus("Védők: bukott csendes ulti", BUKOTT_ULTI))

    return bonuses, replaced


def _compute_final_settlement(
    st: GameState,
    bid: Bid,
    kontras: dict[str, int],
) -> dict[str, Any]:
    """Compute the complete per-defender settlement for a finished game.

    Uses per-component kontra levels.
    Each component's win/loss value is multiplied by ``2 ** kontra_level``.
    Components replaced by silent bonuses (e.g. parti → silent 100) are
    excluded from the base and charged at the full silent value instead.

    Returns a dict with:
      - ``contractResult``: per-defender points from the main contract
      - ``silentBonuses``: list of {label, points} dicts
      - ``netPerDefender``: total per-defender settlement
      - ``soloistTotal``: net points for soloist (×2 defenders)
      - ``soloistWon``: bool
    """
    comps = bid.components

    # --- All-pass rule (no play) ---
    # Only applies when all three players passed in the auction (no tricks played).
    if bid.rank == BID_PASSZ.rank and st.trick_no == 0:
        return {
            "contractResult": -2,
            "silentBonuses": [],
            "netPerDefender": -2,
            "soloistTotal": -4,
            "soloistWon": False,
        }

    # --- Determine win/loss ---
    won = _soloist_won(st, bid)

    # --- Silent bonuses (computed first to know which components are replaced) ---
    silent_list, replaced = _compute_silent_bonuses(st, bid, kontras)
    silent_total = sum(sb.points for sb in silent_list)

    # --- Main contract: per-component scoring (skip replaced units) ---
    vm = component_value_map(bid)
    base = 0
    for unit, (wv, lv) in vm.items():
        if unit in replaced:
            continue  # This component is replaced by a silent bonus
        mult = 2 ** kontras.get(unit, 0)
        if won:
            base += wv * mult
        else:
            base -= lv * mult

    net = base + silent_total

    return {
        "contractResult": base,
        "silentBonuses": [{"label": sb.label, "points": sb.points} for sb in silent_list],
        "netPerDefender": net,
        "soloistTotal": net * 2,
        "soloistWon": won,
    }


# ---------------------------------------------------------------------------
#  AI logic
# ---------------------------------------------------------------------------


def _advance_ai_auction(sess: UltiSession) -> None:
    """Auto-play AI turns in the auction until it's the human's turn or done."""
    a = sess.auction
    if a is None:
        return

    while not a.done and a.turn != HUMAN:
        player = a.turn
        hand = sess.state.hands[player]

        if a.awaiting_bid:
            # AI must discard + bid (first bid or after picking up)
            if a.current_bid is None:
                bid, discards = ai_initial_bid(hand)
            else:
                bid, discards = ai_bid_after_pickup(hand, a)
            # Remove discards from hand
            for c in discards:
                hand.remove(c)
            submit_bid(a, player, bid, discards)
        else:
            # AI decides: pick up or pass
            if ai_should_pickup(hand, a):
                # Pick up talon
                hand.extend(a.talon)
                submit_pickup(a, player)
                # Next iteration will handle the bid (awaiting_bid=True)
            else:
                submit_pass(a, player)

    if a.done:
        _resolve_auction(sess)
    elif a.turn == HUMAN:
        # It's human's turn
        if a.awaiting_bid:
            sess.phase = "bid"
        else:
            sess.phase = "auction"


def _resolve_auction(sess: UltiSession) -> None:
    """Transition from auction to the next phase.

    Handles the all-pass rule: if the winning bid is Passz (rank 1)
    and nobody challenged, the game is skipped and the first bidder
    pays 2 to each defender.
    """
    a = sess.auction
    assert a is not None and a.done
    assert a.winner is not None  # Auction always has a winner

    winner = a.winner
    bid = a.current_bid
    assert bid is not None

    sess.winning_bid = bid
    sess.state.soloist = winner
    sess.state.contract_type = bid.rank

    # --- All-pass rule ---
    # If everyone accepted the minimum bid (Passz), skip the game.
    if bid.rank == BID_PASSZ.rank:
        # First bidder pays 2 to each defender.
        for i in range(3):
            if i != winner:
                sess.state.scores[i] += 2
        sess.state.scores[winner] -= 4  # pays 2 × 2 defenders
        sess.phase = "done"
        return

    # Store the soloist's talon discards separately.
    # Their card-point value counts for the defenders (not the soloist).
    # Only the soloist knows which cards were discarded.
    sess.state.talon_discards = list(a.talon)

    # Determine trump and proceed.
    if bid.is_colorless:
        # No trump (Betli / Színtelen games).
        _setup_contract(sess, bid, trump=None)
        _start_kontra_phase(sess)
    elif bid.is_red:
        # Red = Hearts trump.
        _setup_contract(sess, bid, trump=Suit.HEARTS)
        _start_kontra_phase(sess)
    else:
        # Non-red adu: winner must choose trump suit.
        if winner == HUMAN:
            sess.phase = "trump_select"
        else:
            # AI chooses trump (most common non-Hearts suit).
            trump = _ai_choose_trump(sess, winner)
            _setup_contract(sess, bid, trump=trump)
            _start_kontra_phase(sess)


def _ai_choose_trump(sess: UltiSession, player: int) -> Suit:
    """AI picks trump = most common non-Hearts suit in hand."""
    hand = sess.state.hands[player]
    counts = Counter(c.suit for c in hand)
    non_hearts = [(cnt, suit) for suit, cnt in counts.items() if suit != Suit.HEARTS]
    non_hearts.sort(reverse=True)
    return non_hearts[0][1] if non_hearts else Suit.ACORNS


def _setup_contract(sess: UltiSession, bid: Bid, trump: Suit | None) -> None:
    """Configure game state for the contracted game.

    All colorless games (Betli, Színtelen, Redurchmars) use the
    ``betli`` flag for card ordering and no-trump rules.
    The soloist always leads the first trick.
    """
    set_contract(
        sess.state,
        soloist=sess.state.soloist,
        trump=trump,
        betli=bid.is_colorless,
    )
    # Soloist leads trick 1.
    sess.state.leader = sess.state.soloist
    # Set has_ulti flag for 7esre tartás enforcement.
    sess.state.has_ulti = COMP_ULTI in bid.components


def _start_kontra_phase(sess: UltiSession) -> None:
    """Start the Kontra phase (defenders choose per-component kontras)."""
    bid = sess.winning_bid
    assert bid is not None
    units = kontrable_units(bid)

    if not units:
        # Nothing to kontra (should not happen, but safety)
        _start_play_phase(sess)
        return

    sess.phase = "kontra"
    sess.component_kontras = {u: 0 for u in units}
    sess.colorless_kontras = {}
    sess.kontra_defender_idx = 0
    _advance_ai_kontra(sess)


def _defender_at(sess: UltiSession, idx: int) -> int | None:
    """Return the player index of the idx-th defender (0 or 1)."""
    defenders = [i for i in range(3) if i != sess.state.soloist]
    if idx < len(defenders):
        return defenders[idx]
    return None


def _advance_ai_kontra(sess: UltiSession) -> None:
    """Auto-play AI kontra/rekontra decisions."""
    if sess.phase == "kontra":
        while sess.kontra_defender_idx < 2:
            defender = _defender_at(sess, sess.kontra_defender_idx)
            if defender is None:
                break
            if defender == HUMAN:
                return  # Human's turn to decide
            # AI defender: for now, always passes (no kontra)
            sess.kontra_defender_idx += 1

        # Both defenders done — check if any kontras were made
        if any(v > 0 for v in sess.component_kontras.values()):
            sess.phase = "rekontra"
            if sess.state.soloist != HUMAN:
                # AI soloist: for now, never rekontra
                _start_play_phase(sess)
            # else: human soloist decides
        else:
            _start_play_phase(sess)

    elif sess.phase == "rekontra":
        # AI soloist: never rekontra
        if sess.state.soloist != HUMAN:
            _start_play_phase(sess)


def _start_play_phase(sess: UltiSession) -> None:
    """Transition to the play phase.

    Before the first trick, all players declare their marriages
    (K+Q pairs) — points are added immediately.

    Marriage restriction is applied based on the contract:
      - 40-100: soloist only declares trump K+Q (40).
      - 20-100: soloist only declares one non-trump K+Q (20).
    """
    sess.phase = "play"

    # Determine marriage restriction from the winning bid.
    restrict = marriage_restriction(sess.winning_bid) if sess.winning_bid else None

    # Declare marriages for all players before any tricks are played.
    declare_all_marriages(sess.state, soloist_marriage_restrict=restrict)

    # Generate speech bubbles for declared marriages.
    for player in range(3):
        pts = marriage_points(sess.state, player)
        if pts > 0:
            # Build text from individual marriages for this player.
            parts = []
            for p, _suit, mpts in sess.state.marriages:
                if p == player:
                    parts.append(f"Van {mpts}-{'em' if mpts == 40 else 'am'}!")
            sess.bubbles.append({"player": player, "text": " ".join(parts)})

    cp = current_player(sess.state)
    if cp != HUMAN:
        _advance_ai_one(sess)


def _ai_pick_card(sess: UltiSession) -> Card:
    """Select a card for the AI using the configured mode.

    Modes:
      - "neural":  Pure policy head (instant, no search).
      - "mcts":    MCTS search with configurable strength.
      - "solver":  PIMC with exact alpha-beta solver.
      - "random":  Random legal card (fallback).
    """
    st = sess.state
    actions = legal_actions(st)
    if len(actions) == 1:
        return actions[0]

    mode = sess.ai_mode

    # Solver mode: no neural network needed
    if mode == "solver":
        solver = _get_solver(sess.ai_strength)
        node = _session_to_node(sess)
        player = current_player(st)
        rng = random.Random(sess.seed ^ (st.trick_no * 13 + len(st.trick_cards) * 7))
        return solver.choose_action(node, player, rng)

    wrapper = _load_ulti_model()

    if wrapper is None or mode == "random":
        rng = random.Random(sess.seed ^ (st.trick_no * 13 + len(st.trick_cards) * 7))
        return rng.choice(actions)

    game = _get_ulti_game()
    node = _session_to_node(sess)
    player = current_player(st)

    if mode == "mcts":
        config = _MCTS_CONFIGS.get(sess.ai_strength, _MCTS_CONFIGS["medium"])
        rng = random.Random(sess.seed ^ (st.trick_no * 13 + len(st.trick_cards) * 7))
        _pi, action = alpha_mcts_policy(node, game, wrapper, player, config, rng)
        return action
    else:
        # "neural" — pure policy head, no search
        feats = game.encode_state(node, player)
        mask = game.legal_action_mask(node)
        probs = wrapper.predict_policy(feats, mask)
        best_idx = int(np.argmax(probs))
        return _IDX_TO_CARD[best_idx]


def _advance_ai_one(sess: UltiSession) -> None:
    """Play exactly ONE AI card."""
    st = sess.state

    if is_terminal(st):
        sess.phase = "done"
        return

    cp = current_player(st)
    if cp == HUMAN:
        return

    card = _ai_pick_card(sess)
    result = play_card(st, card)

    if result is not None:
        sess.log.append({"trick": st.trick_no, "result": _trick_json(result)})
        sess.needs_continue = True
        return

    if is_terminal(st):
        sess.phase = "done"


# ---------------------------------------------------------------------------
#  Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/ulti", tags=["ulti"])


@router.post("/new")
def new_game(body: dict[str, Any] = {}) -> dict[str, Any]:
    """Start a new Ulti game.

    The first bidder picks up the talon (12 cards) and must discard 2
    + bid.  If the first bidder is AI, they auto-bid and the talon
    passes around until it reaches the human.

    Optional body fields:
      - ``seed``:  int — deal seed
      - ``dealer``: int — dealer index (0–2)
      - ``aiMode``: "neural" | "mcts" | "random"
      - ``aiStrength``: "fast" | "medium" | "strong" (for mcts mode)
    """
    seed = body.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31)
    seed = int(seed)
    dealer_arg = body.get("dealer")
    if dealer_arg is not None:
        dealer = int(dealer_arg)
    else:
        dealer = random.randint(0, 2)

    # AI configuration
    ai_mode = body.get("aiMode", "neural")
    if ai_mode not in ("neural", "mcts", "solver", "random"):
        ai_mode = "neural"
    ai_strength = body.get("aiStrength", "medium")
    if ai_strength not in ("fast", "medium", "strong"):
        ai_strength = "medium"

    st, talon = deal(seed=seed, dealer=dealer)

    # First bidder picks up the talon (12 cards).
    first_bidder = next_player(dealer)
    pickup_talon(st, first_bidder, talon)

    # Create auction.
    auction = create_auction(first_bidder, talon)

    sess = UltiSession(
        id=str(uuid.uuid4()),
        state=st,
        talon=talon,
        phase="bid",  # will be set by _advance_ai_auction
        seed=seed,
        auction=auction,
        ai_mode=ai_mode,
        ai_strength=ai_strength,
    )
    _sessions[sess.id] = sess

    # Eagerly try to load the model on first game
    if ai_mode != "random":
        _load_ulti_model()

    if first_bidder == HUMAN:
        # Human has 12 cards — must discard 2 + bid.
        sess.phase = "bid"
    else:
        # AI first bidder: auto-bid, then advance.
        _advance_ai_auction(sess)

    return _public_state(sess)


@router.post("/{game_id}/bid")
def bid_action(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Human discards 2 cards and announces a bid.

    Body: { discards: [card, card], bidRank: 1 }
    """
    sess = _sessions.get(game_id)
    if not sess:
        raise HTTPException(404, "Game not found")
    if sess.phase != "bid":
        raise HTTPException(400, f"Not in bid phase (phase={sess.phase})")
    a = sess.auction
    if a is None or a.done:
        raise HTTPException(400, "Auction is finished")
    if a.turn != HUMAN or not a.awaiting_bid:
        raise HTTPException(400, "Not your turn to bid")

    # Parse discards.
    discards_raw = body.get("discards", [])
    if len(discards_raw) != 2:
        raise HTTPException(400, "Must discard exactly 2 cards")
    discards = [_card_from_json(d) for d in discards_raw]
    for c in discards:
        if c not in sess.state.hands[HUMAN]:
            raise HTTPException(400, f"Card {c} not in your hand")

    # Parse bid by rank.
    bid_rank = body.get("bidRank")
    if bid_rank is None:
        raise HTTPException(400, "Missing bidRank")
    from trickster.games.ulti.auction import BID_BY_RANK
    bid = BID_BY_RANK.get(int(bid_rank))
    if bid is None:
        raise HTTPException(400, f"Invalid bid rank: {bid_rank}")

    # Validate bid is legal.
    legal = legal_bids(a)
    if bid not in legal:
        raise HTTPException(400, f"Illegal bid: {bid.label()}")

    # Remove discards from hand.
    for c in discards:
        sess.state.hands[HUMAN].remove(c)

    # Submit bid.
    submit_bid(a, HUMAN, bid, discards)

    # Advance AI turns.
    sess.phase = "auction"
    _advance_ai_auction(sess)

    return _public_state(sess)


@router.post("/{game_id}/auction")
def auction_action(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Human picks up talon, passes, or accepts (stands).

    Body: { action: "pickup" | "pass" }
    """
    sess = _sessions.get(game_id)
    if not sess:
        raise HTTPException(404, "Game not found")
    if sess.phase != "auction":
        raise HTTPException(400, f"Not in auction phase (phase={sess.phase})")
    a = sess.auction
    if a is None or a.done:
        raise HTTPException(400, "Auction is finished")
    if a.turn != HUMAN or a.awaiting_bid:
        raise HTTPException(400, "Not your turn in the auction")

    action = body.get("action")

    if action == "pickup":
        if not can_pickup(a):
            raise HTTPException(400, "Cannot pick up — no higher bids available")
        # Add talon to human's hand.
        sess.state.hands[HUMAN].extend(a.talon)
        submit_pickup(a, HUMAN)
        # Human now has 12 cards — enter bid phase.
        sess.phase = "bid"

    elif action == "pass":
        submit_pass(a, HUMAN)
        if a.done:
            _resolve_auction(sess)
        else:
            _advance_ai_auction(sess)

    else:
        raise HTTPException(400, f"Invalid auction action: {action}")

    return _public_state(sess)


@router.post("/{game_id}/trump")
def choose_trump(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Human chooses trump suit for a non-red game.

    Body: { suit: "ACORNS" | "BELLS" | "LEAVES" }
    """
    sess = _sessions.get(game_id)
    if not sess:
        raise HTTPException(404, "Game not found")
    if sess.phase != "trump_select":
        raise HTTPException(400, f"Not in trump selection phase (phase={sess.phase})")

    suit_val = body.get("suit")
    if suit_val not in _SUIT_MAP:
        raise HTTPException(400, f"Invalid suit: {suit_val}")
    suit = _SUIT_MAP[suit_val]
    if suit == Suit.HEARTS:
        raise HTTPException(400, "Cannot choose Hearts — that would be a red game")

    assert sess.winning_bid is not None
    _setup_contract(sess, sess.winning_bid, trump=suit)
    _start_kontra_phase(sess)

    return _public_state(sess)


@router.post("/{game_id}/kontra")
def kontra_action(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Human responds to kontra/rekontra opportunity.

    Kontra phase (defender):
      body: { "action": "kontra", "components": ["parti", "ulti"] }
      body: { "action": "pass" }

    Rekontra phase (soloist):
      body: { "action": "rekontra", "components": ["parti"] }
      body: { "action": "pass" }
    """
    sess = _sessions.get(game_id)
    if not sess:
        raise HTTPException(404, "Game not found")
    if sess.phase not in ("kontra", "rekontra"):
        raise HTTPException(400, f"Not in kontra/rekontra phase (phase={sess.phase})")

    action = body.get("action")

    if sess.phase == "kontra":
        # Human is a defender
        defender = _defender_at(sess, sess.kontra_defender_idx)
        if defender != HUMAN:
            raise HTTPException(400, "Not your turn for kontra")

        if action == "kontra":
            components = body.get("components", [])
            for comp in components:
                if comp in sess.component_kontras:
                    sess.component_kontras[comp] = max(sess.component_kontras[comp], 1)
            if components:
                sess.bubbles.append({"player": HUMAN, "text": f"Kontra! ({', '.join(components)})"})
        elif action != "pass":
            raise HTTPException(400, f"Invalid kontra action: {action}")
            raise HTTPException(400, f"Invalid kontra action: {action}")

        sess.kontra_defender_idx += 1
        _advance_ai_kontra(sess)

    elif sess.phase == "rekontra":
        # Human is the soloist
        if sess.state.soloist != HUMAN:
            raise HTTPException(400, "Not your turn for rekontra")

        if action == "rekontra":
            components = body.get("components", [])
            for comp in components:
                if comp in sess.component_kontras and sess.component_kontras[comp] == 1:
                    sess.component_kontras[comp] = 2
            if components:
                sess.bubbles.append({"player": HUMAN, "text": f"Rekontra! ({', '.join(components)})"})
        elif action != "pass":
            raise HTTPException(400, f"Invalid rekontra action: {action}")
            raise HTTPException(400, f"Invalid rekontra action: {action}")

        _start_play_phase(sess)

    return _public_state(sess)


@router.post("/{game_id}/play")
def play(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Play a card from the human's hand."""
    sess = _sessions.get(game_id)
    if not sess:
        raise HTTPException(404, "Game not found")
    if sess.phase != "play":
        raise HTTPException(400, f"Not in play phase (phase={sess.phase})")
    if sess.needs_continue:
        raise HTTPException(400, "Call /continue first")

    cp = current_player(sess.state)
    if cp != HUMAN:
        raise HTTPException(400, f"Not your turn (current={cp})")

    card = _card_from_json(body.get("card", {}))
    legal = legal_actions(sess.state)
    if card not in legal:
        raise HTTPException(400, f"Illegal card: {card}")

    result = play_card(sess.state, card)

    if result is not None:
        sess.log.append({"trick": sess.state.trick_no, "result": _trick_json(result)})
        # Always pause to show the completed trick — even the last one.
        sess.needs_continue = True

    return _public_state(sess)


@router.post("/{game_id}/continue")
def continue_game(game_id: str) -> dict[str, Any]:
    """Advance past a completed trick."""
    sess = _sessions.get(game_id)
    if not sess:
        raise HTTPException(404, "Game not found")

    sess.needs_continue = False

    if is_terminal(sess.state):
        sess.phase = "done"
        return _public_state(sess)

    _advance_ai_one(sess)
    return _public_state(sess)


@router.post("/{game_id}/ai-settings")
def ai_settings(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Update AI play settings for this session.

    Body: { aiMode: "neural"|"mcts"|"random", aiStrength: "fast"|"medium"|"strong" }
    """
    sess = _sessions.get(game_id)
    if not sess:
        raise HTTPException(404, "Game not found")

    if "aiMode" in body:
        mode = body["aiMode"]
        if mode in ("neural", "mcts", "solver", "random"):
            sess.ai_mode = mode
    if "aiStrength" in body:
        strength = body["aiStrength"]
        if strength in ("fast", "medium", "strong"):
            sess.ai_strength = strength

    return {
        "aiMode": sess.ai_mode,
        "aiStrength": sess.ai_strength,
    }


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    """Return info about the loaded AI model."""
    wrapper = _load_ulti_model()
    if wrapper is None:
        return {
            "loaded": False,
            "path": None,
            "params": 0,
            "message": "No model found — AI plays randomly. "
                       "Train one with: python scripts/train_baseline.py --steps 500",
        }
    return {
        "loaded": True,
        "path": _model_path_loaded,
        "params": sum(p.numel() for p in wrapper.net.parameters()),
    }


# ---------------------------------------------------------------------------
#  Parti Practice — training-style deals (no auction, straight to play)
# ---------------------------------------------------------------------------


@router.post("/parti/new")
def parti_new_game(body: dict[str, Any] = {}) -> dict[str, Any]:
    """Start a Parti practice game (same deals as training).

    Deals 10-10-10 + 2 talon.  Talon goes to defender points.
    Hearts trump (Piros Passz).  No auction, no discard,
    straight to the play phase.

    Soloist = next_player(dealer), rotating each round.

    Optional body fields:
      - ``seed``: int — deal seed
      - ``dealer``: int — dealer index (0-2), rotates soloist
      - ``aiMode``: "neural" | "mcts" | "solver" | "random"
      - ``aiStrength``: "fast" | "medium" | "strong"
    """
    seed = body.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31)
    seed = int(seed)

    ai_mode = body.get("aiMode", "mcts")
    if ai_mode not in ("neural", "mcts", "solver", "random"):
        ai_mode = "mcts"
    ai_strength = body.get("aiStrength", "medium")
    if ai_strength not in ("fast", "medium", "strong"):
        ai_strength = "medium"

    dealer_arg = body.get("dealer")
    if dealer_arg is not None:
        dealer = int(dealer_arg) % 3
    else:
        dealer = 0  # default: soloist = player 1

    game = _get_ulti_game()
    wrapper = _load_ulti_model()

    # Deal with value-head enrichment: keep competitive deals only.
    # Reject hands where the soloist is hopeless OR dominant.
    VAL_LO = -0.20   # soloist not too weak
    VAL_HI = +0.35   # soloist not too strong (no free 40+20 marriages)
    MAX_ATTEMPTS = 50
    best_node = None
    best_val = -999.0
    best_dist = 999.0  # distance from 0 (most balanced = best)
    for attempt in range(MAX_ATTEMPTS):
        attempt_seed = seed + attempt * 100_000
        node = game.new_game(
            seed=attempt_seed,
            training_mode="simple",
            starting_leader=dealer,
        )
        # Override trump to Hearts before evaluation
        set_contract(node.gs, node.gs.soloist, trump=Suit.HEARTS)

        if wrapper is not None:
            sol = node.gs.soloist
            feats = game.encode_state(node, sol)
            val = wrapper.predict_value(feats)
            dist = abs(val)
            # Track the most balanced deal we've seen
            if dist < best_dist:
                best_node, best_val, best_dist = node, val, dist
            # Accept if within the competitive band
            if VAL_LO <= val <= VAL_HI:
                best_node, best_val = node, val
                break
        else:
            best_node, best_val = node, 0.0
            break

    node = best_node
    gs = node.gs

    from trickster.games.ulti.auction import BID_BY_RANK
    parti_bid = BID_BY_RANK[2]   # Piros passz (red Parti)

    sess = UltiSession(
        id=str(uuid.uuid4()),
        state=gs,
        talon=list(gs.talon_discards),
        phase="play",
        seed=seed,
        auction=None,
        winning_bid=parti_bid,
        component_kontras={"parti": 0},
        ai_mode=ai_mode,
        ai_strength=ai_strength,
    )
    _sessions[sess.id] = sess

    # Declare marriages before play
    declare_all_marriages(gs)

    # Generate marriage bubbles
    for player in range(3):
        pts = marriage_points(gs, player)
        if pts > 0:
            parts = []
            for p, _suit, mpts in gs.marriages:
                if p == player:
                    parts.append(f"Van {mpts}-{'em' if mpts == 40 else 'am'}!")
            sess.bubbles.append({"player": player, "text": " ".join(parts)})

    # If AI leads, play one card
    cp = current_player(gs)
    if cp != HUMAN:
        _advance_ai_one(sess)

    result = _public_state(sess)
    result["dealValue"] = round(best_val, 3)  # value-head assessment
    return result
