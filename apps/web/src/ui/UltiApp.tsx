import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ultiAnalyze,
  ultiAuction,
  ultiBid,
  ultiContinue,
  ultiKontra,
  ultiListModels,
  ultiNewGame,
  ultiPartiNew,
  ultiPlay,
  ultiTrump,
  type AnalysisResult,
  type UltiCard,
  type UltiState,
} from "./ulti-api";
import { ultiCardBackUrl, ultiCardLabel, ultiCardUrl } from "./ulti-cards";

const SUIT_SYMBOL: Record<string, string> = {
  HEARTS: "\u2665", BELLS: "\u2666", LEAVES: "\u2660", ACORNS: "\u2663",
};
const SUIT_NAME: Record<string, string> = {
  HEARTS: "Piros", BELLS: "Tök", LEAVES: "Zöld", ACORNS: "Makk",
};
const KONTRA_LEVEL_LABEL: Record<number, string> = { 1: "Kontra", 2: "Rekontra" };

/** Bid ranks for contracts we have trained models for.
 *  1=Passz, 2=P.Passz, 3=40-100, 4=Ulti, 5=Betli,
 *  8=P.40-100, 10=P.Ulti, 11=P.Betli
 */
const SUPPORTED_BID_RANKS = new Set([1, 2, 3, 4, 5, 8, 10, 11]);

/** One completed round (party) in a match. */
type RoundRecord = {
  round: number;
  bidLabel: string;
  soloist: number;
  soloistWon: boolean;
  /** Per-player score delta for this round (positive = gained). */
  deltas: [number, number, number];
};

function cardKey(c: UltiCard): string {
  return `${c.suit}:${c.rank}`;
}

/** Lightweight speech bubble hook for a specific player position. */
function useUltiBubble(): {
  show: (text: string) => void;
  clear: () => void;
  node: React.ReactNode;
} {
  const [text, setText] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const timer = useRef<number | null>(null);

  const show = useCallback((msg: string) => {
    if (timer.current) window.clearTimeout(timer.current);
    setText(msg);
    setVisible(true);
    timer.current = window.setTimeout(() => {
      setVisible(false);
      timer.current = window.setTimeout(() => {
        setText(null);
        timer.current = null;
      }, 400);
    }, 2200);
  }, []);

  const clear = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current);
    setText(null);
    setVisible(false);
    timer.current = null;
  }, []);

  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);

  const node = text ? (
    <div className={`ulti-speech-bubble ${visible ? "ulti-bubble-in" : "ulti-bubble-out"}`}>
      <span>{text}</span>
    </div>
  ) : null;

  return { show, clear, node };
}

export function UltiApp() {
  // Screen: "lobby" (opponent select) or "game" (in-game)
  const [screen, setScreen] = useState<"lobby" | "game">("lobby");

  // Lobby state — fetch available models from the server
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [opp1, setOpp1] = useState("");
  const [opp2, setOpp2] = useState("");

  useEffect(() => {
    ultiListModels()
      .then((models) => {
        // Filter to e2e models only (no _base suffix)
        const e2e = models.filter((m) => !m.endsWith("_base"));
        setAvailableModels(e2e);
        if (e2e.length > 0) {
          setOpp1((prev) => prev || e2e[0]);
          setOpp2((prev) => prev || e2e[0]);
          setAnalysisSource((prev) => prev || e2e[e2e.length - 1]);
        }
      })
      .catch(() => setAvailableModels([]));
  }, []);

  const [state, setState] = useState<UltiState | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  // Match-level score tracking
  const [roundHistory, setRoundHistory] = useState<RoundRecord[]>([]);
  const [matchScores, setMatchScores] = useState<[number, number, number]>([0, 0, 0]);
  const [showScorecard, setShowScorecard] = useState(false);
  const recordedGameId = useRef<string | null>(null);

  // Current opponents for the active match
  const [activeOpponents, setActiveOpponents] = useState<[string, string]>(["random", "random"]);

  // Dynamic player labels based on selected opponents
  const PLAYER_LABEL = useMemo(() => {
    const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
    const name1 = capitalize(activeOpponents[0]);
    const name2 = capitalize(activeOpponents[1]);
    if (activeOpponents[0] === activeOpponents[1]) {
      return ["Te", `${name1}`, `Dark ${name2}`];
    }
    return ["Te", name1, name2];
  }, [activeOpponents]);

  // Bid phase state (discard selection + bid selection)
  const [selectedDiscards, setSelectedDiscards] = useState<Set<string>>(new Set());
  const [selectedBidIdx, setSelectedBidIdx] = useState<number>(0);

  // Analysis / settings state
  const [showSettings, setShowSettings] = useState(false);
  const [analysisEnabled, setAnalysisEnabled] = useState(false);
  const [analysisSource, setAnalysisSource] = useState("");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const analysisKey = useRef("");  // tracks position for cache invalidation

  const legalSet = useMemo(() => {
    const s = new Set<string>();
    for (const c of state?.legalCards ?? []) s.add(cardKey(c));
    return s;
  }, [state]);

  // Speech bubbles for all 3 players
  const bubble0 = useUltiBubble(); // Human (player 0)
  const bubble1 = useUltiBubble(); // Gép 1 (player 1)
  const bubble2 = useUltiBubble(); // Gép 2 (player 2)
  const bubbleOf = [bubble0, bubble1, bubble2];

  // Show bubbles when state provides them
  const lastBubbleKey = useRef("");
  useEffect(() => {
    if (!state?.bubbles?.length) return;
    // Deduplicate: stringify bubbles to avoid re-triggering on same data
    const key = JSON.stringify(state.bubbles);
    if (key === lastBubbleKey.current) return;
    lastBubbleKey.current = key;

    state.bubbles.forEach((b, i) => {
      // Stagger multiple bubbles slightly
      window.setTimeout(() => {
        bubbleOf[b.player]?.show(b.text);
      }, i * 300);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.bubbles]);

  // Record round result when a deal finishes (exactly once per gameId)
  useEffect(() => {
    if (!state || state.phase !== "done") return;
    if (!state.settlement || !state.contract) return;
    if (recordedGameId.current === state.gameId) return; // already scored
    recordedGameId.current = state.gameId;

    const sol = state.soloist;
    const net = state.settlement.netPerDefender; // from soloist's perspective
    const deltas: [number, number, number] = [0, 0, 0];
    deltas[sol] = net * 2; // soloist gains/loses vs 2 defenders
    for (let i = 0; i < 3; i++) {
      if (i !== sol) deltas[i] = -net; // each defender gets opposite
    }

    const record: RoundRecord = {
      round: roundHistory.length + 1,
      bidLabel: state.contract.bid.label,
      soloist: sol,
      soloistWon: state.settlement.soloistWon,
      deltas,
    };

    setRoundHistory((prev) => [...prev, record]);
    setMatchScores((prev) => [
      prev[0] + deltas[0],
      prev[1] + deltas[1],
      prev[2] + deltas[2],
    ]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phase, state?.gameId]);

  // Auto-continue after trick pause
  useEffect(() => {
    if (!state || busy || !state.needsContinue) return;
    const t = window.setTimeout(() => {
      ultiContinue(state.gameId).then(setState).catch((e) => setErr(String(e)));
    }, 1800);
    return () => window.clearTimeout(t);
  }, [state?.gameId, state?.needsContinue, busy]);

  // Auto-advance when it's an AI's turn during play
  useEffect(() => {
    if (!state || busy || state.needsContinue || state.phase !== "play" || state.dealOver) return;
    if (state.currentPlayer === 0 || state.currentPlayer === null) return;
    const t = window.setTimeout(() => {
      ultiContinue(state.gameId).then(setState).catch((e) => setErr(String(e)));
    }, 1800);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.gameId, state?.currentPlayer, state?.needsContinue, state?.phase, state?.dealOver, state?.trickNo, state?.trickCards?.length, busy]);


  // Analysis polling: fetch AI evaluation when analysis mode is on
  useEffect(() => {
    if (!analysisEnabled || !state || !state.gameId || !analysisSource) return;
    // Only analyze when it's human's turn in relevant phases
    const phase = state.phase;
    const isHumanTurn =
      (phase === "play" && state.currentPlayer === 0 && !state.needsContinue) ||
      phase === "bid" ||
      phase === "auction" ||
      phase === "kontra" ||
      phase === "rekontra";
    if (!isHumanTurn) {
      setAnalysis(null);
      return;
    }

    // Position key to detect changes
    const posKey = `${state.gameId}:${phase}:${state.trickNo}:${state.trickCards?.length ?? 0}`;
    if (posKey !== analysisKey.current) {
      analysisKey.current = posKey;
      setAnalysis(null);  // clear stale analysis
    }

    let cancelled = false;
    const fetchAnalysis = async () => {
      try {
        const result = await ultiAnalyze(state.gameId, analysisSource, 40, 2);
        if (!cancelled) setAnalysis(result);
      } catch { /* ignore */ }
    };

    fetchAnalysis();
    // Re-run with more sims after a delay for progressive deepening
    const t = window.setTimeout(async () => {
      try {
        const result = await ultiAnalyze(state.gameId, analysisSource, 100, 3);
        if (!cancelled) setAnalysis(result);
      } catch { /* ignore */ }
    }, 2000);

    return () => { cancelled = true; window.clearTimeout(t); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisEnabled, analysisSource, state?.gameId, state?.phase, state?.currentPlayer, state?.trickNo, state?.trickCards?.length, state?.needsContinue]);

  // Build a map of card key -> analysis score for easy lookup
  const cardScores = useMemo(() => {
    const m = new Map<string, { score: number; visits: number; best: boolean }>();
    if (!analysis?.actions) return m;
    for (const a of analysis.actions) {
      m.set(cardKey(a.card), { score: a.score, visits: a.visits, best: a.best });
    }
    return m;
  }, [analysis]);

  // Recommended discard cards from analysis (bid phase)
  const recDiscardKeys = useMemo(() => {
    const s = new Set<string>();
    if (!analysis?.recDiscard) return s;
    for (const c of analysis.recDiscard) s.add(cardKey(c));
    return s;
  }, [analysis]);

  /** Start the next round within the current match. */
  async function nextRound() {
    setErr(""); setBusy(true);
    try {
      const nextDealer = state ? (state.dealer + 1) % 3 : undefined;
      const st = await ultiNewGame(null, nextDealer, activeOpponents);
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
      setShowCaptured(false);
      bubble0.clear(); bubble1.clear(); bubble2.clear();
      lastBubbleKey.current = "";
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  /** Start a brand new match from the lobby. */
  async function startMatch(opponents: [string, string]) {
    setActiveOpponents(opponents);
    setRoundHistory([]);
    setMatchScores([0, 0, 0]);
    recordedGameId.current = null;
    setShowScorecard(false);
    setIsPartiMode(false);
    setScreen("game");
    setErr(""); setBusy(true);
    try {
      const st = await ultiNewGame(null, undefined, opponents);
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
      setShowCaptured(false);
      bubble0.clear(); bubble1.clear(); bubble2.clear();
      lastBubbleKey.current = "";
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  /** Go back to lobby. */
  function backToLobby() {
    setState(null);
    setRoundHistory([]);
    setMatchScores([0, 0, 0]);
    recordedGameId.current = null;
    setShowScorecard(false);
    setIsPartiMode(false);
    setScreen("lobby");
  }

  /** Submit discard + bid (bid phase). */
  async function submitBid() {
    if (!state || !state.auction) return;
    const discards = state.hand.filter((c) => selectedDiscards.has(cardKey(c)));
    if (discards.length !== 2) { setErr("Válassz ki 2 lapot az eldobáshoz!"); return; }
    const bid = state.auction.legalBids[selectedBidIdx];
    if (!bid) { setErr("Válassz licitet!"); return; }
    setErr(""); setBusy(true);
    try {
      const st = await ultiBid(state.gameId, discards, bid.rank);
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  /** Pickup or pass during auction phase. */
  async function handleAuction(action: "pickup" | "pass") {
    if (!state) return;
    setErr(""); setBusy(true);
    try {
      const st = await ultiAuction(state.gameId, action);
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  /** Choose trump suit. */
  async function handleTrump(suit: string) {
    if (!state) return;
    setErr(""); setBusy(true);
    try { setState(await ultiTrump(state.gameId, suit)); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  // Parti practice mode
  const [isPartiMode, setIsPartiMode] = useState(false);

  /** Start a Parti practice game (training-style deal, no auction). */
  async function startParti() {
    setRoundHistory([]);
    setMatchScores([0, 0, 0]);
    recordedGameId.current = null;
    setShowScorecard(false);
    setIsPartiMode(true);
    setErr(""); setBusy(true);
    try {
      const st = await ultiPartiNew(null, undefined, activeOpponents);
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
      setShowCaptured(false);
      bubble0.clear(); bubble1.clear(); bubble2.clear();
      lastBubbleKey.current = "";
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  /** Next round in Parti practice — rotate dealer so soloist cycles. */
  async function nextPartiRound() {
    setErr(""); setBusy(true);
    try {
      const nextDealer = state ? (state.dealer + 1) % 3 : 0;
      const st = await ultiPartiNew(null, nextDealer, activeOpponents);
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
      setShowCaptured(false);
      bubble0.clear(); bubble1.clear(); bubble2.clear();
      lastBubbleKey.current = "";
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }


  // Captured cards reveal toggle
  const [showCaptured, setShowCaptured] = useState(false);

  // Per-component kontra selection (checkboxes)
  const [kontraSelection, setKontraSelection] = useState<Set<string>>(new Set());

  function toggleKontraComponent(comp: string) {
    setKontraSelection((prev) => {
      const next = new Set(prev);
      if (next.has(comp)) next.delete(comp);
      else next.add(comp);
      return next;
    });
  }

  async function handleKontra(action: "kontra" | "rekontra" | "pass") {
    if (!state) return;
    setErr(""); setBusy(true);
    try {
      const comps = action !== "pass" ? Array.from(kontraSelection) : undefined;
      setState(await ultiKontra(state.gameId, action, comps));
      setKontraSelection(new Set());
    }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  async function playCard(card: UltiCard) {
    if (!state) return;
    setErr(""); setBusy(true);
    try { setState(await ultiPlay(state.gameId, card)); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  function toggleDiscard(card: UltiCard) {
    const key = cardKey(card);
    setSelectedDiscards((prev) => {
      const next = new Set(prev);
      if (next.has(key)) { next.delete(key); }
      else if (next.size < 2) { next.add(key); }
      return next;
    });
  }

  // --- Render ---
  const phase = state?.phase;
  const hand = state?.hand ?? [];
  const isMyTurn = phase === "play" && state?.currentPlayer === 0 && !state?.needsContinue;
  const auction = state?.auction;
  const isMyAuctionTurn = auction && auction.turn === 0 && !auction.done;
  const isBidPhase = phase === "bid" && isMyAuctionTurn && auction?.awaitingBid;
  const isAuctionPhase = phase === "auction" && isMyAuctionTurn && !auction?.awaitingBid;

  // ===== LOBBY SCREEN =====
  if (screen === "lobby") {
    return (
      <div className="app">
        <div className="ulti-lobby">
          <div className="ulti-lobby-card">
            <h1 className="ulti-lobby-title">Trickster Ulti</h1>
            <p className="ulti-lobby-subtitle">Válaszd ki az ellenfeleidet</p>

            <div className="ulti-lobby-seats">
              <div className="ulti-lobby-seat">
                <label className="ulti-lobby-label">Gép 1</label>
                <select
                  className="ulti-lobby-select"
                  value={opp1}
                  onChange={(e) => setOpp1(e.target.value)}
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>
                  ))}
                </select>
              </div>
              <div className="ulti-lobby-seat">
                <label className="ulti-lobby-label">Gép 2</label>
                <select
                  className="ulti-lobby-select"
                  value={opp2}
                  onChange={(e) => setOpp2(e.target.value)}
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>
                  ))}
                </select>
              </div>
            </div>

            <button
              className="btn btn-primary ulti-lobby-start"
              onClick={() => startMatch([opp1, opp2])}
              disabled={busy || availableModels.length === 0}
            >
              {busy ? "Indítás..." : availableModels.length === 0 ? "Nincs elérhető modell" : "Játék indítása"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ===== GAME SCREEN =====
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="title">Trickster Ulti</div>
          <div className="subtitle">
            {isPartiMode
              ? `Piros Parti gyakorlás | Te: ${matchScores[0]} | ${PLAYER_LABEL[1]}: ${matchScores[1]} | ${PLAYER_LABEL[2]}: ${matchScores[2]}`
              : roundHistory.length > 0
                ? `${roundHistory.length}. kör | Te: ${matchScores[0]} | ${PLAYER_LABEL[1]}: ${matchScores[1]} | ${PLAYER_LABEL[2]}: ${matchScores[2]}`
                : `${PLAYER_LABEL[1]} vs Te vs ${PLAYER_LABEL[2]}`}
          </div>
        </div>
        <div className="controls">
          <button className="btn btn-primary" onClick={backToLobby} disabled={busy}>Új játék</button>
          {roundHistory.length > 0 && (
            <button className="btn" onClick={() => setShowScorecard(true)} disabled={busy}>
              Pontszámok
            </button>
          )}
          <button className="btn btn-gear" onClick={() => setShowSettings(true)}>⚙</button>
        </div>
      </header>

      <main className="main" style={{ gridTemplateColumns: "1fr", maxWidth: 1000, margin: "0 auto", width: "100%" }}>
        <section className="table">
          {/* Contract badge — show during play/done */}
          {state && phase !== "bid" && phase !== "auction" && phase !== "trump_select" && state.contract && (
            <div className="ulti-contract-badge">
              <span className={`ulti-badge ${state.betli ? "ulti-badge-betli" : "ulti-badge-normal"}`}>
                {isPartiMode
                  ? (state.contract.bid.trumpMode === "red" ? "Piros Parti" : "Parti")
                  : state.contract.bid.label}
                {!state.betli && state.trump && (
                  <>
                    {" — Adu: "}
                    <span className={`trump-symbol trump-${state.trump.toLowerCase()}`}>
                      {SUIT_SYMBOL[state.trump] ?? "?"}
                    </span> {SUIT_NAME[state.trump] ?? ""}
                  </>
                )}
                {Object.entries(state.contract.componentKontras).filter(([, v]) => v > 0).map(([comp, lvl]) => (
                  <span key={comp} className="ulti-kontra-tag"> [{comp}: {KONTRA_LEVEL_LABEL[lvl] ?? "?"}]</span>
                ))}
              </span>
              <span className="ulti-badge-value">
                {state.contract.displayWin}p / -{state.contract.displayLoss}p
                {isPartiMode && state.dealValue != null && (
                  <span className="ulti-deal-value" title="Játékos esélye (value head)">
                    {" "}| v={state.dealValue > 0 ? "+" : ""}{state.dealValue.toFixed(2)}
                  </span>
                )}
              </span>
            </div>
          )}

          {/* AI hands */}
          <div className="ulti-ai-row">
            <div className="ulti-ai-hand">
              <div className="ulti-ai-label">
                {PLAYER_LABEL[2]}
                {state?.soloist === 2 && <span className="ulti-role-tag ulti-role-soloist"> Játékos</span>}
                {state?.soloist !== 2 && <span className="ulti-role-tag ulti-role-defender"> Védő</span>}
                {state?.isTeritett && state?.soloist === 2 && " — Terített"}
              </div>
              <div className="ulti-card-row">
                {state?.isTeritett && state?.soloistHand && state.soloist === 2 ? (
                  state.soloistHand.map((c, i) => (
                    <img key={i} className="card-thumb" src={ultiCardUrl(c)} alt={ultiCardLabel(c)} />
                  ))
                ) : (
                  Array.from({ length: state?.aiHandSizes[1] ?? 0 }).map((_, i) => (
                    <img key={i} className="card-thumb" src={ultiCardBackUrl()} alt="back" />
                  ))
                )}
              </div>
              <div className="ulti-bubble-anchor">{bubble2.node}</div>
            </div>
            <div className="ulti-ai-hand">
              <div className="ulti-ai-label">
                {PLAYER_LABEL[1]}
                {state?.soloist === 1 && <span className="ulti-role-tag ulti-role-soloist"> Játékos</span>}
                {state?.soloist !== 1 && <span className="ulti-role-tag ulti-role-defender"> Védő</span>}
                {state?.isTeritett && state?.soloist === 1 && " — Terített"}
              </div>
              <div className="ulti-card-row">
                {state?.isTeritett && state?.soloistHand && state.soloist === 1 ? (
                  state.soloistHand.map((c, i) => (
                    <img key={i} className="card-thumb" src={ultiCardUrl(c)} alt={ultiCardLabel(c)} />
                  ))
                ) : (
                  Array.from({ length: state?.aiHandSizes[0] ?? 0 }).map((_, i) => (
                    <img key={i} className="card-thumb" src={ultiCardBackUrl()} alt="back" />
                  ))
                )}
              </div>
              <div className="ulti-bubble-anchor">{bubble1.node}</div>
            </div>
          </div>

          {/* Scoreboard — visible during play, kontra, done */}
          {phase !== "bid" && phase !== "auction" && phase !== "trump_select" && state && (
            <div className="ulti-scores">
              <span>Ütés: {state.trickNo ?? 0}/10</span>
              <span className="ulti-score-sep">|</span>
              {(() => {
                const sol = state.soloist;
                const solPts = state.soloistPoints ?? 0;
                const defPts = state.defenderPoints ?? 0;
                const solLabel = PLAYER_LABEL[sol];
                const iAmSoloist = sol === 0;
                return (
                  <>
                    <span className={iAmSoloist ? "ulti-score-me" : ""}>
                      Játékos ({solLabel}): {solPts}
                    </span>
                    <span className="ulti-score-sep">|</span>
                    <span className={!iAmSoloist ? "ulti-score-me" : ""}>
                      Védők: {defPts}
                    </span>
                  </>
                );
              })()}
            </div>
          )}

          {/* =============== BID PHASE (discard + bid) =============== */}
          {isBidPhase && auction && (
            <div className="ulti-auction-overlay">
              <div className="ulti-auction-title">Licitálás</div>

              {/* Current bid info */}
              <div className="ulti-auction-current">
                {auction.currentBid ? (
                  <>
                    <span className="ulti-auction-label">Aktuális licit:</span>{" "}
                    <span className="ulti-auction-bid-text">{auction.currentBid.label}</span>
                    <span className="ulti-auction-holder"> — {PLAYER_LABEL[auction.holder ?? 0]}</span>
                  </>
                ) : (
                  <span className="ulti-auction-label">Első licit — válassz 2 lapot és licitet</span>
                )}
              </div>

              {/* History — only bids and passes, skip pickup */}
              {auction.history.filter((e: any) => e.action !== "pickup").length > 0 && (
                <div className="ulti-auction-history">
                  {auction.history.filter((e: any) => e.action !== "pickup").slice(-3).map((entry: any, i: number) => (
                    <div key={i} className="ulti-auction-history-entry">
                      <span className="ulti-auction-player">{PLAYER_LABEL[entry.player]}</span>
                      {entry.action === "pass" && <span className="ulti-auction-action pass">Passz</span>}
                      {entry.action === "stand" && <span className="ulti-auction-action hold">Elfogadva</span>}
                      {entry.action === "bid" && entry.bid && (
                        <span className="ulti-auction-action bid">{entry.bid.label} ({entry.bid.displayWin}p)</span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Bid selector + discard count */}
              <div className="ulti-bid-phase-info">
                Dobj el 2 lapot ({selectedDiscards.size}/2) és válassz licitet:
              </div>

              {(() => {
                const supported = auction.legalBids.filter((b) => SUPPORTED_BID_RANKS.has(b.rank));
                if (supported.length === 0) return null;
                return (
                  <div className="ulti-bid-select-row">
                    <select
                      className="ulti-bid-select"
                      value={selectedBidIdx}
                      onChange={(e) => setSelectedBidIdx(Number(e.target.value))}
                    >
                      {supported.map((b) => {
                        const origIdx = auction.legalBids.indexOf(b);
                        return (
                          <option key={b.rank} value={origIdx}>
                            {b.label} — {b.displayWin}p
                          </option>
                        );
                      })}
                    </select>
                    <button
                      className="btn ulti-bid-confirm"
                      onClick={submitBid}
                      disabled={busy || selectedDiscards.size !== 2}
                    >
                      Licitálok
                    </button>
                  </div>
                );
              })()}
            </div>
          )}

          {/* =============== AUCTION PHASE (pickup/pass) =============== */}
          {phase === "auction" && auction && (
            <div className="ulti-auction-overlay">
              <div className="ulti-auction-title">Licitálás</div>

              <div className="ulti-auction-current">
                {auction.currentBid && (
                  <>
                    <span className="ulti-auction-label">Aktuális licit:</span>{" "}
                    <span className="ulti-auction-bid-text">{auction.currentBid.label}</span>
                    <span className="ulti-auction-holder"> — {PLAYER_LABEL[auction.holder ?? 0]}</span>
                  </>
                )}
              </div>

              {/* History — only bids and passes, skip pickup */}
              {auction.history.filter((e: any) => e.action !== "pickup").length > 0 && (
                <div className="ulti-auction-history">
                  {auction.history.filter((e: any) => e.action !== "pickup").slice(-3).map((entry: any, i: number) => (
                    <div key={i} className="ulti-auction-history-entry">
                      <span className="ulti-auction-player">{PLAYER_LABEL[entry.player]}</span>
                      {entry.action === "pass" && <span className="ulti-auction-action pass">Passz</span>}
                      {entry.action === "stand" && <span className="ulti-auction-action hold">Elfogadva</span>}
                      {entry.action === "bid" && entry.bid && (
                        <span className="ulti-auction-action bid">{entry.bid.label} ({entry.bid.displayWin}p)</span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Pickup/Pass buttons */}
              {isAuctionPhase && (
                <div className="ulti-auction-actions">
                  <div className="ulti-bid-bottom-row">
                    {auction.canPickup && (
                      <button className="btn ulti-bid-hold" onClick={() => handleAuction("pickup")} disabled={busy}>
                        Felveszem
                      </button>
                    )}
                    <button className="btn ulti-bid-pass" onClick={() => handleAuction("pass")} disabled={busy}>
                      {auction.isHolderTurn ? "Elfogadom" : "Passz"}
                    </button>
                  </div>
                </div>
              )}

              {/* AI thinking */}
              {phase === "auction" && auction && !auction.done && auction.turn !== 0 && (
                <div className="ulti-auction-waiting">
                  {PLAYER_LABEL[auction.turn]} gondolkodik...
                </div>
              )}
            </div>
          )}

          {/* =============== TRUMP SELECTION =============== */}
          {phase === "trump_select" && state?.trumpOptions && (
            <div className="ulti-auction-overlay">
              <div className="ulti-auction-title">Adu választás</div>
              <div className="ulti-auction-current">
                <span className="ulti-auction-label">
                  Kontraktus: <strong>{state.contract?.bid.label}</strong> — válassz adu színt:
                </span>
              </div>
              <div className="ulti-trump-buttons">
                {state.trumpOptions.map((suit) => (
                  <button
                    key={suit}
                    className={`btn ulti-trump-btn trump-${suit.toLowerCase()}`}
                    onClick={() => handleTrump(suit)}
                    disabled={busy}
                  >
                    <span className="ulti-trump-symbol">{SUIT_SYMBOL[suit]}</span>
                    {" "}{SUIT_NAME[suit]}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* =============== KONTRA / REKONTRA OVERLAY =============== */}
          {(phase === "kontra" || phase === "rekontra") && state?.kontra && (
            <div className="ulti-kontra-overlay">
              {/* Big title */}
              <div className="ulti-kontra-title">
                {state.kontra.phase === "rekontra" ? "Rekontra!" : "Kontra!"}
              </div>

              {/* Explanation */}
              <div className="ulti-kontra-desc">
                {state.kontra.phase === "kontra"
                  ? `${state.contract?.bid.label ?? "Kontraktus"} — az 1. ütés után kontrázhatod a tétet.`
                  : "Kontrázták a tétet — visszakontrázhatod."}
                {Object.entries(state.kontra.currentKontras).filter(([, v]) => v > 0).length > 0 && (
                  <span className="ulti-kontra-active-tags">
                    {Object.entries(state.kontra.currentKontras).filter(([, v]) => v > 0).map(([comp, lvl]) => (
                      <span key={comp} className="ulti-kontra-active-tag">{comp} {KONTRA_LEVEL_LABEL[lvl]}</span>
                    ))}
                  </span>
                )}
              </div>

              {state.kontra.isMyTurn ? (
                <div className="ulti-kontra-actions">
                  {/* Contract label */}
                  <div className="ulti-kontra-contract">
                    {state.contract?.bid.label}
                  </div>

                  {/* Component toggle buttons */}
                  <div className="ulti-kontra-toggles">
                    {state.kontra.kontrable
                      .filter((comp) =>
                        state.kontra!.phase === "kontra"
                          ? state.kontra!.currentKontras[comp] === 0
                          : state.kontra!.currentKontras[comp] === 1
                      )
                      .map((comp) => {
                        const kontraVal = analysisEnabled && analysis?.phase && ["kontra", "rekontra"].includes(analysis.phase) ? analysis.value : undefined;
                        return (
                          <div key={comp} className="ulti-kontra-toggle-row">
                            <button
                              className={`ulti-kontra-toggle ${kontraSelection.has(comp) ? "ulti-kontra-toggle-on" : ""}`}
                              onClick={() => toggleKontraComponent(comp)}
                              disabled={busy}
                            >
                              {comp}
                            </button>
                            {kontraVal != null && (
                              <div className={`ulti-kontra-bar ${kontraVal > 0 ? "ulti-kontra-bar-good" : "ulti-kontra-bar-bad"}`}>
                                <div
                                  className="ulti-kontra-bar-fill"
                                  style={{ width: `${Math.min(Math.abs(kontraVal) * 100, 100)}%` }}
                                />
                                <span className="ulti-kontra-bar-label">
                                  {kontraVal > 0 ? "+" : ""}{kontraVal.toFixed(2)}
                                </span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                  </div>

                  {/* Action row */}
                  <div className="ulti-kontra-btn-row">
                    <button
                      className="btn ulti-kontra-confirm"
                      onClick={() => handleKontra(state.kontra!.phase === "rekontra" ? "rekontra" : "kontra")}
                      disabled={busy || kontraSelection.size === 0}
                    >
                      {state.kontra.phase === "rekontra" ? "Rekontra!" : "Kontra!"}
                    </button>
                    <button className="btn ulti-kontra-pass" onClick={() => handleKontra("pass")} disabled={busy}>
                      Passz
                    </button>
                  </div>
                </div>
              ) : (
                <div className="ulti-kontra-waiting">
                  {PLAYER_LABEL[state.kontra.turn ?? 0]} gondolkodik...
                </div>
              )}
            </div>
          )}

          {/* Declared marriages (shown during play/kontra/done if any exist) */}
          {state && (phase === "play" || phase === "done" || phase === "kontra" || phase === "rekontra") && state.declaredMarriages.some(v => v > 0) && (
            <div className="ulti-declared-marriages">
              {state.declaredMarriages.map((pts, i) =>
                pts > 0 ? (
                  <span key={i} className="ulti-marriage-badge">
                    {PLAYER_LABEL[i]}: +{pts}
                  </span>
                ) : null
              )}
            </div>
          )}

          {/* Trick area — also visible during kontra/rekontra (trick 1 result shown) */}
          {(phase === "play" || phase === "done" || phase === "kontra" || phase === "rekontra") && (
            <div className="ulti-trick-area">
              {(() => {
                const slots: Record<number, { card: UltiCard; winner?: boolean }> = {};
                if ((state?.needsContinue || phase === "done" || phase === "kontra" || phase === "rekontra") && state?.lastTrick) {
                  state.lastTrick.players.forEach((p, i) => {
                    slots[p] = { card: state.lastTrick!.cards[i], winner: p === state.lastTrick!.winner };
                  });
                } else if (state?.trickCards.length) {
                  state.trickCards.forEach((tc) => {
                    slots[tc.player] = { card: tc.card };
                  });
                }
                const renderSlot = (p: number) => (
                  <div key={p} className="ulti-trick-slot">
                    <div className="ulti-trick-player">
                      {PLAYER_LABEL[p]}
                      {slots[p]?.winner ? " \u2605" : ""}
                    </div>
                    {slots[p] ? (
                      <img className="card-large" src={ultiCardUrl(slots[p].card)} alt={ultiCardLabel(slots[p].card)} />
                    ) : (
                      <div className="card-placeholder-empty" />
                    )}
                  </div>
                );
                return (
                  <>
                    {renderSlot(2)}
                    {renderSlot(0)}
                    {renderSlot(1)}
                  </>
                );
              })()}

              {/* Talon — tucked into the bottom-left of the trick area */}
              {phase === "done" && state?.talonCards && state.talonCards.length > 0 && (
                <div className="ulti-talon-inline">
                  {state.talonCards.map((c, i) => (
                    <img key={i} className="ulti-talon-card" src={ultiCardUrl(c)} alt={ultiCardLabel(c)} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Status */}
          <div className="status-bar">
            {phase === "bid" && `Válassz 2 lapot és licitet (${selectedDiscards.size}/2 kiválasztva)`}
            {phase === "auction" && auction?.done && `Licitálás vége — ${PLAYER_LABEL[auction.winner ?? 0]} nyerte`}
            {phase === "trump_select" && "Válassz adu színt!"}
            {phase === "play" && !state?.needsContinue && isMyTurn && "Te jössz — válassz egy lapot!"}
            {phase === "play" && !state?.needsContinue && !isMyTurn && state?.currentPlayer !== null && "AI gondolkodik..."}
            {phase === "play" && state?.needsContinue && state.lastTrick && `Ütés ${state.trickNo}: ${PLAYER_LABEL[state.lastTrick.winner]} nyerte`}
            {phase === "done" && state?.resultMessage}
            {phase === "done" && !state?.resultMessage && `Vége! Játékos: ${state?.soloistPoints ?? 0} pont | Védők: ${state?.defenderPoints ?? 0} pont`}
            {phase === "done" && state?.settlement && state.settlement.silentBonuses.length > 0 && (
              <div style={{ marginTop: 4, fontSize: "0.85em", opacity: 0.9 }}>
                {state.settlement.silentBonuses.map((sb, i) => (
                  <span key={i} style={{ marginRight: 8, color: sb.points > 0 ? "#4caf50" : "#ef5350" }}>
                    {sb.label}: {sb.points > 0 ? "+" : ""}{sb.points}
                  </span>
                ))}
              </div>
            )}
          </div>


          {/* Next round + scorecard buttons when done */}
          {phase === "done" && (
            <div className="ulti-done-actions">
              <button className="btn btn-primary" onClick={isPartiMode ? nextPartiRound : nextRound} disabled={busy}>
                Következő kör
              </button>
              {roundHistory.length > 0 && (
                <button className="btn" onClick={() => setShowScorecard(true)}>
                  Pontszámok
                </button>
              )}
            </div>
          )}

          {/* Player bubble */}
          <div className="ulti-bubble-anchor ulti-bubble-player-anchor">{bubble0.node}</div>

          {/* Player role label */}
          {state && (phase === "play" || phase === "done" || phase === "kontra" || phase === "rekontra") && (
            <div className="ulti-player-role">
              Te
              {state.soloist === 0
                ? <span className="ulti-role-tag ulti-role-soloist"> Játékos</span>
                : <span className="ulti-role-tag ulti-role-defender"> Védő</span>}
            </div>
          )}

          {/* Hand */}
          <div className="hand ulti-hand">
            {hand.map((c) => {
              const key = cardKey(c);
              const isLegal = legalSet.size === 0 || legalSet.has(key);
              const isSelected = selectedDiscards.has(key);
              const canClick =
                phase === "bid" ||
                (phase === "play" && isMyTurn && isLegal);
              const scoreInfo = analysisEnabled ? cardScores.get(key) : undefined;
              const isRecDiscard = analysisEnabled && phase === "bid" && recDiscardKeys.has(key);
              return (
                <div key={key} className="hand-slot">
                  <button
                    className={`hand-card${!isLegal && phase === "play" ? " hand-card-illegal" : ""}${isSelected ? " hand-card-selected" : ""}${scoreInfo?.best ? " hand-card-best" : ""}${isRecDiscard ? " hand-card-discard" : ""}`}
                    onClick={() => {
                      if (phase === "bid") toggleDiscard(c);
                      else if (phase === "play" && isMyTurn && isLegal) playCard(c);
                    }}
                    disabled={!canClick}
                    style={isSelected ? { transform: "translateY(-20px)", zIndex: 10 } : undefined}
                  >
                    <img className="card-hand" src={ultiCardUrl(c)} alt={ultiCardLabel(c)} />
                    {scoreInfo != null && (
                      <span
                        className={`card-score ${scoreInfo.score >= 0.6 ? "card-score-good" : scoreInfo.score <= 0.3 ? "card-score-bad" : "card-score-neutral"}`}
                      >
                        {Math.round(scoreInfo.score * 100)}
                      </span>
                    )}
                    {isRecDiscard && (
                      <span className="card-discard-tag">Talon</span>
                    )}
                  </button>
                </div>
              );
            })}
          </div>

          {err ? <div className="error">{err}</div> : null}

          {/* Captured cards */}
          {state && state.capturedTricks && state.capturedTricks.length > 0 && (
            <div className="ulti-captured">
              <div className="ulti-captured-header">
                <span className="ulti-captured-title">
                  Ütéseim ({state.capturedTricks.length})
                </span>
                <button
                  className="btn ulti-captured-toggle"
                  onClick={() => setShowCaptured((v) => !v)}
                >
                  {showCaptured ? "Elrejt" : "Mutat"}
                </button>
              </div>
              {showCaptured ? (
                <div className="ulti-captured-grid">
                  {state.capturedTricks.map((trick, i) => (
                    <div key={i} className="ulti-captured-trick">
                      {trick.map((c, j) => (
                        <img key={j} className="card-micro" src={ultiCardUrl(c)} alt={ultiCardLabel(c)} />
                      ))}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="ulti-captured-stacks">
                  {state.capturedTricks.map((_, i) => (
                    <div key={i} className="ulti-captured-stack">
                      <img className="card-micro" src={ultiCardBackUrl()} alt="captured" />
                      <img className="card-micro ulti-stack-2" src={ultiCardBackUrl()} alt="captured" />
                      <img className="card-micro ulti-stack-3" src={ultiCardBackUrl()} alt="captured" />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      {/* Scorecard modal */}
      {showScorecard && (
        <div className="modal-overlay" onClick={() => setShowScorecard(false)}>
          <div className="modal ulti-scorecard" onClick={(e) => e.stopPropagation()}>
            <div className="modal-title">Pontszámok</div>
            <table className="ulti-score-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Játék</th>
                  <th>Játékos</th>
                  <th>{PLAYER_LABEL[0]}</th>
                  <th>{PLAYER_LABEL[1]}</th>
                  <th>{PLAYER_LABEL[2]}</th>
                </tr>
              </thead>
              <tbody>
                {roundHistory.map((r) => (
                  <tr key={r.round} className={r.soloistWon && r.soloist === 0 ? "ulti-row-win" : r.soloist === 0 && !r.soloistWon ? "ulti-row-loss" : ""}>
                    <td>{r.round}</td>
                    <td>{r.bidLabel}</td>
                    <td>{PLAYER_LABEL[r.soloist]}</td>
                    <td className={r.deltas[0] > 0 ? "ulti-sc-pos" : r.deltas[0] < 0 ? "ulti-sc-neg" : ""}>
                      {r.deltas[0] > 0 ? "+" : ""}{r.deltas[0]}
                    </td>
                    <td className={r.deltas[1] > 0 ? "ulti-sc-pos" : r.deltas[1] < 0 ? "ulti-sc-neg" : ""}>
                      {r.deltas[1] > 0 ? "+" : ""}{r.deltas[1]}
                    </td>
                    <td className={r.deltas[2] > 0 ? "ulti-sc-pos" : r.deltas[2] < 0 ? "ulti-sc-neg" : ""}>
                      {r.deltas[2] > 0 ? "+" : ""}{r.deltas[2]}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={3}>Összesen</td>
                  <td className={matchScores[0] > 0 ? "ulti-sc-pos" : matchScores[0] < 0 ? "ulti-sc-neg" : ""}>{matchScores[0]}</td>
                  <td className={matchScores[1] > 0 ? "ulti-sc-pos" : matchScores[1] < 0 ? "ulti-sc-neg" : ""}>{matchScores[1]}</td>
                  <td className={matchScores[2] > 0 ? "ulti-sc-pos" : matchScores[2] < 0 ? "ulti-sc-neg" : ""}>{matchScores[2]}</td>
                </tr>
              </tfoot>
            </table>
            <div className="modal-actions">
              <button className="btn" onClick={() => setShowScorecard(false)}>Bezár</button>
            </div>
          </div>
        </div>
      )}

      {/* Settings modal */}
      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="modal ulti-settings" onClick={(e) => e.stopPropagation()}>
            <div className="modal-title">Beállítások</div>

            <div className="ulti-setting-row">
              <label className="ulti-setting-label">AI elemzés</label>
              <button
                className={`ulti-toggle ${analysisEnabled ? "ulti-toggle-on" : ""}`}
                onClick={() => {
                  setAnalysisEnabled((v) => !v);
                  if (analysisEnabled) setAnalysis(null);
                }}
              >
                {analysisEnabled ? "Be" : "Ki"}
              </button>
            </div>

            {analysisEnabled && (
              <div className="ulti-setting-row">
                <label className="ulti-setting-label">Elemző modell</label>
                <select
                  className="ulti-lobby-select"
                  value={analysisSource}
                  onChange={(e) => { setAnalysisSource(e.target.value); setAnalysis(null); }}
                >
                  <option value="">— válassz —</option>
                  {availableModels.map((m) => (
                    <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>
                  ))}
                </select>
              </div>
            )}

            <div className="modal-actions">
              <button className="btn" onClick={() => setShowSettings(false)}>Bezár</button>
            </div>
          </div>
        </div>
      )}

      {/* Analysis panel — bid/auction phase */}
      {analysisEnabled && (analysis?.phase === "bid" || analysis?.phase === "auction") && analysis?.contracts && analysis.contracts.length > 0 && (
        <div className="ulti-analysis-panel">
          <div className="ulti-analysis-title">AI elemzés ({analysis.source})</div>
          <div className="ulti-analysis-contracts">
            {analysis.contracts.map((c, i) => (
              <div key={i} className={`ulti-analysis-row ${c.gamePts >= 0 ? "ulti-analysis-good" : "ulti-analysis-bad"}`}>
                <span className="ulti-analysis-label">{c.bidLabel}</span>
                <span className={`ulti-analysis-pts ${c.gamePts >= 0 ? "card-score-good" : "card-score-bad"}`}>
                  {c.gamePts >= 0 ? "+" : ""}{c.gamePts.toFixed(1)}
                </span>
              </div>
            ))}
          </div>
          {analysis.recommendation && (
            <div className="ulti-analysis-rec">
              Javaslat: <strong>{analysis.recommendation}</strong>
              {analysis.recDiscard && (
                <div className="ulti-analysis-discard">
                  Talon: {ultiCardLabel(analysis.recDiscard[0])}, {ultiCardLabel(analysis.recDiscard[1])}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Kontra analysis is shown inline — see kontra toggle section */}

      {/* Analysis info bar — play phase */}
      {analysisEnabled && analysis?.phase === "play" && analysis.value != null && (
        <div className="ulti-analysis-bar">
          <span className="ulti-analysis-source">{analysis.source}</span>
          <span className={`ulti-analysis-value ${analysis.value > 0 ? "card-score-good" : analysis.value < 0 ? "card-score-bad" : ""}`}>
            Pozíció: {analysis.value > 0 ? "+" : ""}{analysis.value.toFixed(2)}
          </span>
          {analysis.sims != null && <span className="ulti-analysis-sims">{analysis.sims} sim</span>}
        </div>
      )}
    </div>
  );
}
