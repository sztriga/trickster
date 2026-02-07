from __future__ import annotations

import pickle
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import random

from trickster.games.snapszer.agent import LearnedAgent
from trickster.games.snapszer.cards import Card, Color
from trickster.games.snapszer.adapter import SnapszerGame, SnapszerNode
from trickster.games.snapszer.game import (
    can_close_talon,
    can_declare_marriage,
    can_exchange_trump_jack,
    close_talon,
    deal_awarded_game_points,
    deal,
    declare_marriage,
    exchange_trump_jack,
    is_terminal,
    legal_actions,
    play_trick,
    talon_size,
)
from trickster.mcts import MCTSConfig, alpha_mcts_choose
from trickster.models.alpha_net import SharedAlphaNet
from trickster.training.model_spec import list_model_dirs, model_label_from_dir
from trickster.training.model_store import load_slot


def _card_to_json(c: Card) -> dict[str, Any]:
    return {"color": c.color.value, "number": int(c.number)}


def _card_from_json(obj: dict[str, Any]) -> Card:
    try:
        return Card(Color(str(obj["color"])), int(obj["number"]))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid card: {obj!r} ({e})")


@dataclass(slots=True)
class Session:
    state: object
    pending_lead: Optional[Card]
    model_label: str
    created_at: float
    needs_continue: bool = False
    deal_over: bool = False
    match_points: list[int] = field(default_factory=lambda: [0, 0])
    last_award: Optional[dict[str, Any]] = None
    seed: int = 0
    deal_starting_leader: int = 0


_SESSIONS: dict[str, Session] = {}

# ---------------------------------------------------------------------------
#  AlphaZero MCTS agent
# ---------------------------------------------------------------------------

_ALPHA_NET_PATH = Path("models/AlphaZero/net.pkl")
_ALPHA_LABEL = "AlphaZero (MCTS)"
_ALPHA_EVAL_CONFIG = MCTSConfig(
    simulations=50,
    determinizations=6,
    use_value_head=True,
    use_policy_priors=True,
    dirichlet_alpha=0.0,   # no noise at play time
    visit_temp=0.1,        # near-greedy
)
_ALPHA_GAME = SnapszerGame()

# Cache the loaded net so we don't reload every move
_alpha_net_cache: dict[str, SharedAlphaNet] = {}


def _load_alpha_net() -> Optional[SharedAlphaNet]:
    """Load the SharedAlphaNet for MCTS play (cached)."""
    if not _ALPHA_NET_PATH.exists():
        return None
    cache_key = str(_ALPHA_NET_PATH)
    if cache_key not in _alpha_net_cache:
        with open(_ALPHA_NET_PATH, "rb") as f:
            _alpha_net_cache[cache_key] = pickle.load(f)  # noqa: S301
    return _alpha_net_cache[cache_key]


def _is_alpha_model(label: str) -> bool:
    return label == _ALPHA_LABEL


def _alpha_choose_action(
    gs: object,
    pending_lead: Optional[Card],
    ai_player: int,
    rng: random.Random,
) -> Any:
    """Use MCTS search to pick a move for the AlphaZero agent."""
    net = _load_alpha_net()
    if net is None:
        raise HTTPException(status_code=500, detail="AlphaZero net not found")
    node = SnapszerNode(
        gs=gs, pending_lead=pending_lead,
        known_voids=(frozenset(), frozenset()),
    )
    actions = _ALPHA_GAME.legal_actions(node)
    if len(actions) <= 1:
        return actions[0]
    return alpha_mcts_choose(
        node, _ALPHA_GAME, net, ai_player, _ALPHA_EVAL_CONFIG, rng,
    )


# ---------------------------------------------------------------------------
#  Legacy (direct-trained) model support
# ---------------------------------------------------------------------------


def _load_policy_by_label(label: str) -> Optional[Any]:
    """Load a trained policy by its GUI label."""
    if not label or _is_alpha_model(label):
        return None
    dirs = list_model_dirs(root="models")
    by_label = {model_label_from_dir(d): d for d in dirs}
    d = by_label.get(label)
    if d is None:
        return None
    if not (d / "latest.pkl").exists():
        return None
    return load_slot("latest", models_dir=d)


def _ai_agent(policy, *, seed: int) -> Optional[LearnedAgent]:
    if policy is None:
        return None
    return LearnedAgent(policy.lead_model, policy.follow_model, random.Random(seed), epsilon=0.0)


def _random_choice(legal):
    lst = list(legal)
    return random.choice(lst)


def _public_state(sess: Session) -> dict[str, Any]:
    st = sess.state
    pending = sess.pending_lead
    last = st.last_trick
    # Available marriages for the human (leader-only).
    marriages: list[dict[str, Any]] = []
    if sess.pending_lead is None and st.leader == 0 and not sess.deal_over:
        for suit in Color:
            if can_declare_marriage(st, 0, suit):
                pts = 40 if suit == st.trump_color else 20
                marriages.append({"suit": suit.value, "points": pts})
    return {
        "gameId": None,
        "needsContinue": bool(sess.needs_continue),
        "dealOver": bool(sess.deal_over),
        "matchPoints": list(sess.match_points),
        "lastAward": sess.last_award,
        "seed": sess.seed,
        "pendingLead": None if pending is None else _card_to_json(pending),
        "lastTrick": None
        if last is None
        else {
            "leaderCard": _card_to_json(last.leader_card),
            "responderCard": _card_to_json(last.responder_card),
            "winner": int(last.winner),
        },
        "scores": list(st.scores),
        "leader": int(st.leader),
        "trickNo": int(st.trick_no),
        "talon": {
            "size": int(talon_size(st)),
            "drawPileSize": int(len(st.draw_pile)),
            "closed": bool(st.talon_closed or (len(st.draw_pile) == 0 and st.trump_card is None)),
            "isClosedByTakaras": bool(st.talon_closed),
            "trumpColor": st.trump_color.value,
            "trumpUpcard": None
            if (st.trump_card is None or not st.trump_upcard_visible)
            else _card_to_json(st.trump_card),
        },
        "announcements": {
            "marriages": [{"player": p, "suit": suit.value, "points": pts} for (p, suit, pts) in st.declared_marriages],
        },
        "available": {
            "canCloseTalon": bool(can_close_talon(st, 0) if (sess.pending_lead is None and not sess.deal_over) else False),
            "marriages": marriages,
        },
        "hands": {
            "human": [_card_to_json(c) for c in st.hands[0]],
            # do NOT leak AI hand
        },
        "captured": {
            "human": [_card_to_json(c) for c in st.captured[0]],
            "ai": [_card_to_json(c) for c in st.captured[1]],
        },
        "terminal": bool(is_terminal(st)),
        "canExchangeTrumpJack": bool(sess.pending_lead is None and st.leader == 0 and can_exchange_trump_jack(st, 0)),
    }


def _finalize_deal(sess: Session) -> str:
    if sess.deal_over:
        w = sess.last_award.get("winner") if sess.last_award else None
        return "Deal over." if w is None else ("Deal over: you win." if w == 0 else "Deal over: AI wins.")

    st = sess.state
    winner, pts, reason = deal_awarded_game_points(st)
    sess.match_points[winner] += int(pts)
    sess.deal_over = True
    sess.needs_continue = False
    sess.pending_lead = None
    sess.last_award = {
        "winner": int(winner),
        "points": int(pts),
        "reason": str(reason),
        "scores": list(st.scores),
        "matchPoints": list(sess.match_points),
    }
    return "Deal over: you win." if winner == 0 else "Deal over: AI wins."


def _advance_ai(sess: Session) -> str:
    """
    Run AI moves until it's human's turn/response, terminal,
    or until a trick completes (so the UI can show the two cards briefly).
    """
    st = sess.state
    use_alpha = _is_alpha_model(sess.model_label)

    # Deterministic per-session AI RNG
    ai_seed = int(sess.seed) ^ 0x51F15E
    ai_rng = random.Random(ai_seed + st.trick_no * 7)

    if not use_alpha:
        policy = _load_policy_by_label(sess.model_label)
        ai = _ai_agent(policy, seed=ai_seed)
    else:
        policy = None
        ai = None

    sess.needs_continue = False

    while True:
        if is_terminal(st):
            return _finalize_deal(sess)

        leader = st.leader
        responder = 1 - leader

        if sess.pending_lead is None:
            if leader == 1:
                if use_alpha:
                    action = _alpha_choose_action(st, None, 1, ai_rng)
                    if action == "close_talon":
                        close_talon(st, 1)
                        action = _alpha_choose_action(st, None, 1, ai_rng)
                    sess.pending_lead = action
                else:
                    # Legacy LearnedAgent path
                    if can_exchange_trump_jack(st, 1):
                        exchange_trump_jack(st, 1)
                    can_close = can_close_talon(st, 1)
                    lead_legal_pre = legal_actions(st, leader, None)
                    trump_up = st.trump_card if st.trump_upcard_visible else None
                    if ai is not None and can_close:
                        do_close, _best_card = ai.choose_lead_or_close_talon(
                            st.hands[leader], lead_legal_pre,
                            can_close_talon=can_close,
                            draw_pile_size=len(st.draw_pile),
                            captured_self=st.captured[leader],
                            captured_opp=st.captured[responder],
                            trump_color=st.trump_color, trump_upcard=trump_up,
                        )
                        if do_close:
                            close_talon(st, 1)
                    lead_legal = legal_actions(st, leader, None)
                    lead = (
                        ai.choose_lead(
                            st.hands[leader], lead_legal,
                            draw_pile_size=len(st.draw_pile),
                            captured_self=st.captured[leader],
                            captured_opp=st.captured[responder],
                            trump_color=st.trump_color,
                            trump_upcard=st.trump_card if st.trump_upcard_visible else None,
                        )
                        if ai is not None
                        else _random_choice(lead_legal)
                    )
                    sess.pending_lead = lead
                return "AI led. Pick your response."
            return "Your turn: lead a card."

        lead = sess.pending_lead
        if responder == 1:
            if use_alpha:
                resp = _alpha_choose_action(st, lead, 1, ai_rng)
            else:
                resp_legal = legal_actions(st, responder, lead)
                resp = (
                    ai.choose_follow(
                        st.hands[responder], lead, resp_legal,
                        draw_pile_size=len(st.draw_pile),
                        captured_self=st.captured[responder],
                        captured_opp=st.captured[leader],
                        trump_color=st.trump_color,
                        trump_upcard=st.trump_card if st.trump_upcard_visible else None,
                    )
                    if ai is not None
                    else _random_choice(resp_legal)
                )
            st, _ = play_trick(st, lead, resp)
            sess.pending_lead = None
            if is_terminal(st):
                return _finalize_deal(sess)
            sess.needs_continue = True
            return "Trick complete."

        return "Respond to the lead."


app = FastAPI(title="Trickster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CARDS_DIR = Path(__file__).resolve().parents[2] / "src" / "trickster" / "card_graphics"
app.mount("/cards", StaticFiles(directory=str(CARDS_DIR)), name="cards")


@app.get("/api/models")
def list_models() -> list[str]:
    labels = sorted([model_label_from_dir(d) for d in list_model_dirs(root="models")])
    if _ALPHA_NET_PATH.exists() and _ALPHA_LABEL not in labels:
        labels.append(_ALPHA_LABEL)
    return labels


@app.post("/api/new")
def new_game(payload: dict[str, Any]) -> dict[str, Any]:
    if "seed" in payload and payload.get("seed", None) is not None and str(payload.get("seed")).strip() != "":
        seed = int(payload["seed"])
    else:
        seed = random.randint(0, 2_147_483_647)
    model_label = str(payload.get("modelLabel", "") or "")
    gid = str(uuid.uuid4())
    starting = random.Random(int(seed)).randrange(2)
    st = deal(seed=seed, starting_leader=starting)
    sess = Session(state=st, pending_lead=None, model_label=model_label, created_at=time.time(), needs_continue=False)
    sess.match_points = [0, 0]
    sess.deal_over = False
    sess.last_award = None
    sess.seed = seed
    sess.deal_starting_leader = int(st.leader)
    _SESSIONS[gid] = sess
    prompt = _advance_ai(sess)
    out = _public_state(sess)
    out["gameId"] = gid
    out["prompt"] = prompt
    return out


@app.post("/api/new_deal")
def new_deal(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", "") or "")
    sess = _SESSIONS.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown gameId")
    if "seed" in payload and payload.get("seed", None) is not None and str(payload.get("seed")).strip() != "":
        seed = int(payload["seed"])
    else:
        seed = random.randint(0, 2_147_483_647)
    starting = 1 - int(getattr(sess, "deal_starting_leader", 0))
    st = deal(seed=seed, starting_leader=starting)
    sess.state = st
    sess.pending_lead = None
    sess.needs_continue = False
    sess.deal_over = False
    sess.last_award = None
    sess.seed = seed
    sess.deal_starting_leader = int(st.leader)
    prompt = _advance_ai(sess)
    out = _public_state(sess)
    out["gameId"] = game_id
    out["prompt"] = prompt
    return out


@app.get("/api/state/{game_id}")
def get_state(game_id: str) -> dict[str, Any]:
    sess = _SESSIONS.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown gameId")
    out = _public_state(sess)
    out["gameId"] = game_id
    if sess.deal_over:
        out["prompt"] = "Deal over."
        return out
    if sess.needs_continue:
        out["prompt"] = "Trick complete."
    elif sess.pending_lead is not None:
        out["prompt"] = "Respond to the lead."
    else:
        out["prompt"] = "Your turn."
    return out


@app.post("/api/continue")
def continue_game(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", "") or "")
    sess = _SESSIONS.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown gameId")
    if sess.deal_over:
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = "Deal over."
        return out
    if not sess.needs_continue:
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = "Your turn." if sess.pending_lead is None else "Respond to the lead."
        return out
    prompt = _advance_ai(sess)
    out = _public_state(sess)
    out["gameId"] = game_id
    out["prompt"] = prompt
    return out


@app.post("/api/action")
def action(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", "") or "")
    sess = _SESSIONS.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown gameId")
    if sess.deal_over:
        raise HTTPException(status_code=400, detail="Deal is over. Start a new deal.")

    st = sess.state
    typ = str(payload.get("type", "") or "")

    if typ == "close_talon":
        if not (sess.pending_lead is None and st.leader == 0 and can_close_talon(st, 0)):
            raise HTTPException(status_code=400, detail="Takaras not allowed now.")
        close_talon(st, 0)
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = "Talon closed."
        return out

    if typ == "declare_marriage":
        suit_s = str(payload.get("suit", "") or "")
        try:
            suit = Color(suit_s)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid suit for marriage.")
        if not (sess.pending_lead is None and st.leader == 0 and can_declare_marriage(st, 0, suit)):
            raise HTTPException(status_code=400, detail="Marriage not allowed now.")
        pts = declare_marriage(st, 0, suit)
        if is_terminal(st):
            prompt = _finalize_deal(sess)
        else:
            prompt = f"Declared {pts}."
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = prompt
        return out

    if typ == "exchange_trump_jack":
        if not (sess.pending_lead is None and st.leader == 0 and can_exchange_trump_jack(st, 0)):
            raise HTTPException(status_code=400, detail="Exchange not allowed now.")
        exchange_trump_jack(st, 0)
        try:
            prompt = _advance_ai(sess)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = prompt
        return out

    if typ != "play_card":
        raise HTTPException(status_code=400, detail="Unknown action type")

    card = _card_from_json(payload.get("card") or {})
    leader = st.leader
    responder = 1 - leader

    if sess.pending_lead is None:
        if leader != 0:
            raise HTTPException(status_code=400, detail="Not your turn to lead.")
        legal = set(legal_actions(st, 0, None))
        if card not in legal:
            raise HTTPException(status_code=400, detail="Illegal lead card.")
        if getattr(st, "pending_marriage", None) is not None:
            p, suit, _pts = st.pending_marriage
            if int(p) == 0:
                allowed = {Card(suit, 4), Card(suit, 3)}
                if card not in allowed:
                    raise HTTPException(
                        status_code=400,
                        detail="After declaring a marriage you must lead the King or Queen of that suit.",
                    )
        sess.pending_lead = card
        try:
            prompt = _advance_ai(sess)
        except ValueError as e:
            sess.pending_lead = None
            raise HTTPException(status_code=400, detail=str(e))
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = prompt
        return out
    else:
        if responder != 0:
            raise HTTPException(status_code=400, detail="Not your turn to respond.")
        lead = sess.pending_lead
        legal = set(legal_actions(st, 0, lead))
        if card not in legal:
            raise HTTPException(status_code=400, detail="Illegal response card.")
        st, _ = play_trick(st, lead, card)
        sess.pending_lead = None

    if is_terminal(st):
        prompt = _finalize_deal(sess)
    else:
        sess.needs_continue = True
        prompt = "Trick complete."
    out = _public_state(sess)
    out["gameId"] = game_id
    out["prompt"] = prompt
    return out
