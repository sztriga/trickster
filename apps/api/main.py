from __future__ import annotations

import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import random

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
from trickster.mcts import MCTSConfig, alpha_mcts_choose, _run_mcts
from trickster.models.alpha_net import SharedAlphaNet
from trickster.training.model_spec import list_model_dirs, model_label_from_dir


def _card_to_json(c: Card) -> dict[str, Any]:
    return {"color": c.color.value, "number": int(c.number)}


def _card_from_json(obj: dict[str, Any]) -> Card:
    try:
        return Card(Color(str(obj["color"])), int(obj["number"]))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid card: {obj!r} ({e})")


_DEFAULT_MCTS_SIMS = 50
_DEFAULT_MCTS_DETS = 6


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
    ai_bubble: Optional[str] = None
    mcts_sims: int = _DEFAULT_MCTS_SIMS
    mcts_dets: int = _DEFAULT_MCTS_DETS


_SESSIONS: dict[str, Session] = {}

# ---------------------------------------------------------------------------
#  AlphaZero MCTS agent
# ---------------------------------------------------------------------------

_ALPHA_GAME = SnapszerGame()

# Cache the loaded nets so we don't reload every move: label -> (net, path)
_alpha_net_cache: dict[str, SharedAlphaNet] = {}

# Map model label -> directory (populated lazily)
_label_to_dir: dict[str, Path] = {}


def _refresh_label_map() -> None:
    """Rebuild the label -> dir mapping from disk."""
    _label_to_dir.clear()
    for d in list_model_dirs(root="models"):
        label = model_label_from_dir(d)
        _label_to_dir[label] = d


def _load_alpha_net(label: str = "") -> Optional[SharedAlphaNet]:
    """Load the SharedAlphaNet for a given model label (cached)."""
    if not _label_to_dir:
        _refresh_label_map()
    d = _label_to_dir.get(label)
    if d is None:
        # Fallback: try any alphazero model
        for lbl, dd in _label_to_dir.items():
            if (dd / "net.pkl").exists():
                sp = dd / "spec.json"
                if sp.exists():
                    from trickster.training.model_spec import read_spec
                    try:
                        if read_spec(sp).kind == "alphazero":
                            d = dd
                            break
                    except Exception:
                        continue
    if d is None:
        return None
    net_path = d / "net.pkl"
    if not net_path.exists():
        return None
    cache_key = str(net_path)
    if cache_key not in _alpha_net_cache:
        with open(net_path, "rb") as f:
            _alpha_net_cache[cache_key] = pickle.load(f)  # noqa: S301
    return _alpha_net_cache[cache_key]


def _alpha_choose_action(
    gs: object,
    pending_lead: Optional[Card],
    ai_player: int,
    rng: random.Random,
    *,
    sims: int = _DEFAULT_MCTS_SIMS,
    dets: int = _DEFAULT_MCTS_DETS,
    model_label: str = "",
) -> Any:
    """Use MCTS search to pick a move for the AlphaZero agent."""
    net = _load_alpha_net(model_label)
    if net is None:
        raise HTTPException(status_code=500, detail="AlphaZero net not found")
    node = SnapszerNode(
        gs=gs, pending_lead=pending_lead,
        known_voids=(frozenset(), frozenset()),
    )
    actions = _ALPHA_GAME.legal_actions(node)
    if len(actions) <= 1:
        return actions[0]
    cfg = MCTSConfig(
        simulations=sims,
        determinizations=dets,
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.0,
        visit_temp=0.1,
    )
    return alpha_mcts_choose(
        node, _ALPHA_GAME, net, ai_player, cfg, rng,
    )


def _random_choice(legal):
    lst = list(legal)
    return random.choice(lst)


def _public_state(sess: Session, *, consume_bubble: bool = True) -> dict[str, Any]:
    st = sess.state
    pending = sess.pending_lead
    last = st.last_trick
    bubble = sess.ai_bubble
    if consume_bubble:
        sess.ai_bubble = None
    # Available marriages for the human (leader-only).
    marriages: list[dict[str, Any]] = []
    if sess.pending_lead is None and st.leader == 0 and not sess.deal_over:
        for suit in Color:
            if can_declare_marriage(st, 0, suit):
                pts = 40 if suit == st.trump_color else 20
                marriages.append({"suit": suit.value, "points": pts})
    # Legal cards for the human player (for UI highlighting).
    legal_cards: list[dict[str, Any]] = []
    if not sess.deal_over and not sess.needs_continue:
        if pending is not None and st.leader != 0:
            # Human is follower — legal response cards
            legal_cards = [_card_to_json(c) for c in legal_actions(st, 0, pending)]
        elif pending is None and st.leader == 0:
            # Human is leader — legal lead cards
            lead_legal = legal_actions(st, 0, None)
            # Respect pending marriage constraint
            if st.pending_marriage is not None:
                _p, m_suit, _pts = st.pending_marriage
                lead_legal = [c for c in lead_legal if c.color == m_suit and c.number in (3, 4)]
            legal_cards = [_card_to_json(c) for c in lead_legal]
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
        "legalCards": legal_cards,
        "captured": {
            "human": [_card_to_json(c) for c in st.captured[0]],
            "ai": [_card_to_json(c) for c in st.captured[1]],
        },
        "terminal": bool(is_terminal(st)),
        "canExchangeTrumpJack": bool(sess.pending_lead is None and st.leader == 0 and can_exchange_trump_jack(st, 0)),
        "aiBubble": bubble,
        "mctsSettings": {"sims": sess.mcts_sims, "dets": sess.mcts_dets},
    }


def _finalize_deal(sess: Session) -> str:
    if sess.deal_over:
        w = sess.last_award.get("winner") if sess.last_award else None
        return "Kör vége." if w is None else ("Kör vége — te nyertél!" if w == 0 else "Kör vége — a gép nyert.")

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
    return "Kör vége — te nyertél!" if winner == 0 else "Kör vége — a gép nyert."


def _advance_ai(sess: Session) -> str:
    """
    Run AI moves until it's human's turn/response, terminal,
    or until a trick completes (so the UI can show the two cards briefly).
    """
    st = sess.state

    # Deterministic per-session AI RNG
    ai_seed = int(sess.seed) ^ 0x51F15E
    ai_rng = random.Random(ai_seed + st.trick_no * 7)

    sess.needs_continue = False
    sess.ai_bubble = None  # clear previous bubble

    while True:
        if is_terminal(st):
            return _finalize_deal(sess)

        leader = st.leader
        responder = 1 - leader

        if sess.pending_lead is None:
            if leader == 1:
                bubbles: list[str] = []

                # Exchange trump jack (always beneficial)
                if can_exchange_trump_jack(st, 1):
                    exchange_trump_jack(st, 1)
                    bubbles.append("Cserélek!")

                # MCTS decides close_talon, marriages, and lead card.
                # Loop until we get a card action.
                while True:
                    action = _alpha_choose_action(st, None, 1, ai_rng, sims=sess.mcts_sims, dets=sess.mcts_dets, model_label=sess.model_label)
                    if action == "close_talon":
                        close_talon(st, 1)
                        bubbles.append("Betakarok!")
                        continue
                    if isinstance(action, str) and action.startswith("marry_"):
                        suit = Color(action[6:])
                        pts = declare_marriage(st, 1, suit)
                        bubbles.append("Van 40-em!" if pts == 40 else "Van 20-am!")
                        if is_terminal(st):
                            sess.ai_bubble = " ".join(bubbles) if bubbles else None
                            return _finalize_deal(sess)
                        continue
                    break  # action is a card
                sess.pending_lead = action

                sess.ai_bubble = " ".join(bubbles) if bubbles else None
                return "A gép kijátszott. Válaszolj!"
            return "Te jössz — játssz ki egy lapot."

        lead = sess.pending_lead
        if responder == 1:
            resp = _alpha_choose_action(st, lead, 1, ai_rng, sims=sess.mcts_sims, dets=sess.mcts_dets, model_label=sess.model_label)
            st, _ = play_trick(st, lead, resp)
            sess.pending_lead = None
            sess.needs_continue = True
            return "Ütés kész."

        return "Válaszolj a kijátszásra."


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
    _refresh_label_map()
    return sorted(_label_to_dir.keys())


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
        out["prompt"] = "Kör vége."
        return out
    if sess.needs_continue:
        out["prompt"] = "Ütés kész."
    elif sess.pending_lead is not None:
        out["prompt"] = "Válaszolj a kijátszásra."
    else:
        out["prompt"] = "Te jössz."
    return out


@app.post("/api/continue")
def continue_game(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", "") or "")
    _cancel_analysis(game_id)
    sess = _SESSIONS.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown gameId")
    if sess.deal_over:
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = "Kör vége."
        return out
    if not sess.needs_continue:
        out = _public_state(sess)
        out["gameId"] = game_id
        out["prompt"] = "Te jössz." if sess.pending_lead is None else "Válaszolj a kijátszásra."
        return out
    prompt = _advance_ai(sess)
    out = _public_state(sess)
    out["gameId"] = game_id
    out["prompt"] = prompt
    return out


# ---------------------------------------------------------------------------
#  Background analysis (progressive MCTS)
# ---------------------------------------------------------------------------


@dataclass
class _AnalysisState:
    """Accumulated results from background MCTS determinizations."""
    position_key: str  # identifies the position this analysis belongs to
    total_visits: dict[Any, float] = field(default_factory=dict)
    value_sum: float = 0.0
    value_count: int = 0
    dets_done: int = 0
    dets_target: int = 0
    running: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: Optional[threading.Thread] = None


_ANALYSIS: dict[str, _AnalysisState] = {}  # game_id -> analysis state


def _cancel_analysis(game_id: str) -> None:
    """Cancel any in-progress analysis for a game."""
    existing = _ANALYSIS.pop(game_id, None)
    if existing is not None:
        existing.cancel.set()


def _position_key(sess: Session) -> str:
    """Cheap key that changes whenever the board position changes."""
    st = sess.state
    return f"{id(st)}:{getattr(st, 'trick_number', 0)}:{hash(sess.pending_lead)}"


def _analysis_worker(
    game_id: str,
    analysis: _AnalysisState,
    node: SnapszerNode,
    net: SharedAlphaNet,
    sims: int,
    dets: int,
) -> None:
    """Run determinizations one at a time, accumulating visit counts."""
    game = SnapszerGame()
    rng = random.Random()
    cfg = MCTSConfig(
        simulations=sims,
        determinizations=1,  # we do one det per iteration for progressive updates
        use_value_head=True,
        use_policy_priors=True,
        dirichlet_alpha=0.0,
        visit_temp=0.5,
    )
    for i in range(dets):
        if analysis.cancel.is_set():
            break
        det = game.determinize(node, 0, rng)
        rollout_rng = random.Random(rng.randrange(1 << 30))
        visits, root_val = _run_mcts(det, game, net, 0, cfg, rollout_rng)
        with analysis.lock:
            for act, cnt in visits.items():
                analysis.total_visits[act] = analysis.total_visits.get(act, 0.0) + cnt
            analysis.value_sum += root_val  # MCTS-backed value, not raw NN
            analysis.value_count += 1
            analysis.dets_done = i + 1
    with analysis.lock:
        analysis.running = False


def _build_analysis_response(analysis: _AnalysisState) -> dict[str, Any]:
    """Convert accumulated visit counts into a response dict."""
    from trickster.games.snapszer.adapter import _IDX_TO_CARD, _ACTION_IDX
    from trickster.games.snapszer.cards import ALL_COLORS
    from trickster.games.snapszer.adapter import _MARRY_ACTIONS

    with analysis.lock:
        total = dict(analysis.total_visits)
        val = analysis.value_sum / analysis.value_count if analysis.value_count > 0 else 0.0
        dets_done = analysis.dets_done
        dets_target = analysis.dets_target
        running = analysis.running

    # Convert visit counts to probabilities
    visit_sum = sum(total.values())
    actions_out = []
    for act, cnt in total.items():
        prob = cnt / visit_sum if visit_sum > 0 else 0.0
        if prob < 0.005:
            continue
        if isinstance(act, str):
            if act == "close_talon":
                actions_out.append({"type": "close_talon", "prob": round(prob, 4)})
            elif act.startswith("marry_"):
                suit = act.replace("marry_", "")
                actions_out.append({
                    "type": "marriage",
                    "suit": suit,
                    "prob": round(prob, 4),
                })
        else:
            # It's a Card
            actions_out.append({
                "type": "card",
                "card": _card_to_json(act),
                "prob": round(prob, 4),
            })
    actions_out.sort(key=lambda x: x["prob"], reverse=True)
    return {
        "value": round(float(val), 4),
        "actions": actions_out,
        "progress": dets_done,
        "total": dets_target,
        "searching": running,
    }


@app.get("/api/analyze/{game_id}")
def analyze(game_id: str) -> dict[str, Any]:
    """Return progressive MCTS analysis for the current position."""
    sess = _SESSIONS.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown gameId")
    net = _load_alpha_net(sess.model_label)
    if net is None:
        return {"value": 0.0, "actions": [], "progress": 0, "total": 0, "searching": False}

    pos_key = _position_key(sess)

    # Check if we already have an analysis running for this position
    existing = _ANALYSIS.get(game_id)
    if existing is not None and existing.position_key == pos_key:
        # Return current accumulated results
        return _build_analysis_response(existing)

    # Cancel any old analysis
    if existing is not None:
        existing.cancel.set()
        if existing.thread is not None:
            existing.thread.join(timeout=2.0)

    # Start new analysis — clone state so background thread is fully isolated
    node = SnapszerNode(
        gs=sess.state.clone(), pending_lead=sess.pending_lead,
        known_voids=(frozenset(), frozenset()),
    )
    # Use more determinizations for analysis (progressive reveal)
    dets = max(sess.mcts_dets, 12)
    sims = sess.mcts_sims
    analysis = _AnalysisState(
        position_key=pos_key,
        dets_target=dets,
        running=True,
    )
    _ANALYSIS[game_id] = analysis
    t = threading.Thread(
        target=_analysis_worker,
        args=(game_id, analysis, node, net, sims, dets),
        daemon=True,
    )
    analysis.thread = t
    t.start()
    # Return empty initial state — frontend will poll
    return {"value": 0.0, "actions": [], "progress": 0, "total": dets, "searching": True}


@app.post("/api/settings")
def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", "") or "")
    sess = _SESSIONS.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown gameId")
    if "sims" in payload:
        sess.mcts_sims = max(1, min(500, int(payload["sims"])))
    if "dets" in payload:
        sess.mcts_dets = max(1, min(30, int(payload["dets"])))
    return {"sims": sess.mcts_sims, "dets": sess.mcts_dets}


@app.post("/api/action")
def action(payload: dict[str, Any]) -> dict[str, Any]:
    game_id = str(payload.get("gameId", "") or "")
    _cancel_analysis(game_id)
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
        out["prompt"] = "Betakarva."
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
            prompt = f"{pts} bemondva."
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

    sess.needs_continue = True
    prompt = "Ütés kész."
    # If terminal, _finalize_deal happens on the next /api/continue
    out = _public_state(sess)
    out["gameId"] = game_id
    out["prompt"] = prompt
    return out
