import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ultiAiSettings,
  ultiAuction,
  ultiBid,
  ultiContinue,
  ultiKontra,
  ultiModelInfo,
  ultiNewGame,
  ultiPlay,
  ultiTrump,
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
const PLAYER_LABEL = ["Te", "Gép 1", "Gép 2"];
const KONTRA_LEVEL_LABEL: Record<number, string> = { 1: "Kontra", 2: "Rekontra" };

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
  const [state, setState] = useState<UltiState | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  // Match-level score tracking
  const [roundHistory, setRoundHistory] = useState<RoundRecord[]>([]);
  const [matchScores, setMatchScores] = useState<[number, number, number]>([0, 0, 0]);
  const [showScorecard, setShowScorecard] = useState(false);
  const [roundRecorded, setRoundRecorded] = useState(false);

  // Bid phase state (discard selection + bid selection)
  const [selectedDiscards, setSelectedDiscards] = useState<Set<string>>(new Set());
  const [selectedBidIdx, setSelectedBidIdx] = useState<number>(0);

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

  // Record round result when a deal finishes
  useEffect(() => {
    if (!state || state.phase !== "done" || roundRecorded) return;
    if (!state.settlement || !state.contract) return;

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
    setRoundRecorded(true);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phase, state?.gameId, roundRecorded]);

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


  /** Start the next round within the current match. */
  async function nextRound() {
    setErr(""); setBusy(true);
    try {
      // Rotate dealer from previous game (or random if first game).
      const nextDealer = state ? (state.dealer + 1) % 3 : undefined;
      const st = await ultiNewGame(null, nextDealer);
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
      setShowCaptured(false);
      setRoundRecorded(false);
      bubble0.clear(); bubble1.clear(); bubble2.clear();
      lastBubbleKey.current = "";
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  /** Start a brand new match (reset all scores). */
  async function newMatch() {
    setRoundHistory([]);
    setMatchScores([0, 0, 0]);
    setRoundRecorded(false);
    setShowScorecard(false);
    setErr(""); setBusy(true);
    try {
      const st = await ultiNewGame();
      setState(st);
      setSelectedDiscards(new Set());
      setSelectedBidIdx(0);
      setShowCaptured(false);
      bubble0.clear(); bubble1.clear(); bubble2.clear();
      lastBubbleKey.current = "";
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
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

  // AI mode
  const [aiMode, setAiMode] = useState<string>("neural");
  const [aiStrength, setAiStrength] = useState<string>("medium");
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  // Check model on mount
  useEffect(() => {
    ultiModelInfo().then((info) => {
      setModelLoaded(info.loaded);
      if (!info.loaded) setAiMode("random");
    }).catch(() => setModelLoaded(false));
  }, []);

  // Sync AI settings when changed
  useEffect(() => {
    if (!state?.gameId) return;
    ultiAiSettings(state.gameId, aiMode, aiStrength).catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiMode, aiStrength, state?.gameId]);

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

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="title">Trickster Ulti</div>
          <div className="subtitle">
            {roundHistory.length > 0
              ? `${roundHistory.length}. kör | Te: ${matchScores[0]} | G1: ${matchScores[1]} | G2: ${matchScores[2]}`
              : "3 játékos, 1 vs 2"}
          </div>
        </div>
        <div className="controls">
          <select
            className="btn ulti-ai-select"
            value={aiMode}
            onChange={(e) => setAiMode(e.target.value)}
            title="AI mód"
          >
            <option value="neural">AI: Neural</option>
            <option value="mcts">AI: MCTS</option>
            <option value="random">AI: Random</option>
          </select>
          {aiMode === "mcts" && (
            <select
              className="btn ulti-ai-select"
              value={aiStrength}
              onChange={(e) => setAiStrength(e.target.value)}
              title="AI erősség"
            >
              <option value="fast">Gyors (5)</option>
              <option value="medium">Közepes (20)</option>
              <option value="strong">Erős (50)</option>
            </select>
          )}
          {modelLoaded === false && (
            <span className="ulti-no-model" title="Nincs modell — tanítsd be: python scripts/train_baseline.py --mode mixed --steps 200">
              ⚠ Nincs modell
            </span>
          )}
          {roundHistory.length > 0 && (
            <button className="btn" onClick={() => setShowScorecard(true)} disabled={busy}>
              Pontszámok
            </button>
          )}
          <button className="btn btn-primary" onClick={newMatch} disabled={busy}>Új mérkőzés</button>
        </div>
      </header>

      <main className="main" style={{ gridTemplateColumns: "1fr", maxWidth: 1000, margin: "0 auto", width: "100%" }}>
        <section className="table">
          {/* Contract badge — show during play/done */}
          {state && phase !== "bid" && phase !== "auction" && phase !== "trump_select" && state.contract && (
            <div className="ulti-contract-badge">
              <span className={`ulti-badge ${state.betli ? "ulti-badge-betli" : "ulti-badge-normal"}`}>
                {state.contract.bid.label}
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
              <span className="ulti-badge-value">{state.contract.displayWin}p / -{state.contract.displayLoss}p</span>
            </div>
          )}

          {/* AI hands */}
          <div className="ulti-ai-row">
            <div className="ulti-ai-hand">
              <div className="ulti-ai-label">
                Gép 2 ({state?.scores[2] ?? 0} pont)
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
                Gép 1 ({state?.scores[1] ?? 0} pont)
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

          {/* Scoreboard — hide during bid/auction */}
          {phase !== "bid" && phase !== "auction" && phase !== "trump_select" && (
            <div className="ulti-scores">
              <span>Ütés: {state?.trickNo ?? 0}/10</span>
              <span className="ulti-score-sep">|</span>
              <span>Te: {state?.scores[0] ?? 0}</span>
              <span className="ulti-score-sep">|</span>
              <span>Védők: {(state?.scores[1] ?? 0) + (state?.scores[2] ?? 0)}</span>
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

              {/* History (last 3) */}
              {auction.history.length > 0 && (
                <div className="ulti-auction-history">
                  {auction.history.slice(-3).map((entry, i) => (
                    <div key={i} className="ulti-auction-history-entry">
                      <span className="ulti-auction-player">{PLAYER_LABEL[entry.player]}</span>
                      {entry.action === "pass" && <span className="ulti-auction-action pass">Passz</span>}
                      {entry.action === "stand" && <span className="ulti-auction-action hold">Elfogadva</span>}
                      {entry.action === "pickup" && <span className="ulti-auction-action pickup">Felvette</span>}
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

              {auction.legalBids.length > 0 && (
                <div className="ulti-bid-select-row">
                  <select
                    className="ulti-bid-select"
                    value={selectedBidIdx}
                    onChange={(e) => setSelectedBidIdx(Number(e.target.value))}
                  >
                    {auction.legalBids.map((b, i) => (
                      <option key={b.rank} value={i}>
                        {b.label} — {b.displayWin}p
                      </option>
                    ))}
                  </select>
                  <button
                    className="btn ulti-bid-confirm"
                    onClick={submitBid}
                    disabled={busy || selectedDiscards.size !== 2}
                  >
                    Licitálok
                  </button>
                </div>
              )}
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

              {/* History (last 3) */}
              {auction.history.length > 0 && (
                <div className="ulti-auction-history">
                  {auction.history.slice(-3).map((entry, i) => (
                    <div key={i} className="ulti-auction-history-entry">
                      <span className="ulti-auction-player">{PLAYER_LABEL[entry.player]}</span>
                      {entry.action === "pass" && <span className="ulti-auction-action pass">Passz</span>}
                      {entry.action === "stand" && <span className="ulti-auction-action hold">Elfogadva</span>}
                      {entry.action === "pickup" && <span className="ulti-auction-action pickup">Felvette</span>}
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
              <div className="ulti-auction-title">
                {state.kontra.phase === "rekontra" ? "Rekontra" : "Kontra"}
              </div>
              <div className="ulti-kontra-info">
                Kontraktus: <strong>{state.contract?.bid.label}</strong>
                {Object.entries(state.kontra.currentKontras).filter(([, v]) => v > 0).map(([comp, lvl]) => (
                  <span key={comp} className="ulti-kontra-tag"> [{comp}: {KONTRA_LEVEL_LABEL[lvl] ?? "?"}]</span>
                ))}
              </div>
              {state.kontra.isMyTurn ? (
                <div className="ulti-kontra-actions">
                  <div className="ulti-kontra-components">
                    {state.kontra.kontrable
                      .filter((comp) =>
                        state.kontra!.phase === "kontra"
                          ? state.kontra!.currentKontras[comp] === 0
                          : state.kontra!.currentKontras[comp] === 1
                      )
                      .map((comp) => (
                        <label key={comp} className="ulti-kontra-checkbox">
                          <input
                            type="checkbox"
                            checked={kontraSelection.has(comp)}
                            onChange={() => toggleKontraComponent(comp)}
                          />
                          {comp}
                        </label>
                      ))}
                  </div>
                  <div className="ulti-kontra-buttons">
                    <button
                      className="btn ulti-bid-hold"
                      onClick={() => handleKontra(state.kontra!.phase === "rekontra" ? "rekontra" : "kontra")}
                      disabled={busy || kontraSelection.size === 0}
                    >
                      {state.kontra.phase === "rekontra" ? "Rekontra!" : "Kontra!"}
                    </button>
                    <button className="btn ulti-bid-pass" onClick={() => handleKontra("pass")} disabled={busy}>
                      Passz
                    </button>
                  </div>
                </div>
              ) : (
                <div className="ulti-auction-waiting">
                  {PLAYER_LABEL[state.kontra.turn ?? 0]} gondolkodik...
                </div>
              )}
            </div>
          )}

          {/* Declared marriages (shown during play/done if any exist) */}
          {state && (phase === "play" || phase === "done") && state.declaredMarriages.some(v => v > 0) && (
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

          {/* Trick area */}
          {(phase === "play" || phase === "done") && (
            <div className="ulti-trick-area">
              {(() => {
                const slots: Record<number, { card: UltiCard; winner?: boolean }> = {};
                if ((state?.needsContinue || phase === "done") && state?.lastTrick) {
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
            {phase === "done" && !state?.resultMessage && `Vége! Te: ${state?.soloistPoints ?? 0} pont | Védők: ${state?.defenderPoints ?? 0} pont`}
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
              <button className="btn btn-primary" onClick={nextRound} disabled={busy}>
                Következő kör
              </button>
              {roundHistory.length > 0 && (
                <button className="btn" onClick={() => setShowScorecard(true)}>
                  Pontszámok
                </button>
              )}
            </div>
          )}

          {/* Next round button (shown when deal is done) */}
          {phase === "done" && (
            <div className="ulti-next-round">
              <button className="btn btn-primary" onClick={nextRound} disabled={busy}>
                Következő kör
              </button>
            </div>
          )}

          {/* Player bubble */}
          <div className="ulti-bubble-anchor ulti-bubble-player-anchor">{bubble0.node}</div>

          {/* Hand */}
          <div className="hand ulti-hand">
            {hand.map((c) => {
              const key = cardKey(c);
              const isLegal = legalSet.size === 0 || legalSet.has(key);
              const isSelected = selectedDiscards.has(key);
              const canClick =
                phase === "bid" ||
                (phase === "play" && isMyTurn && isLegal);
              return (
                <div key={key} className="hand-slot">
                  <button
                    className={`hand-card${!isLegal && phase === "play" ? " hand-card-illegal" : ""}${isSelected ? " hand-card-selected" : ""}`}
                    onClick={() => {
                      if (phase === "bid") toggleDiscard(c);
                      else if (phase === "play" && isMyTurn && isLegal) playCard(c);
                    }}
                    disabled={!canClick}
                    style={isSelected ? { transform: "translateY(-20px)", zIndex: 10 } : undefined}
                  >
                    <img className="card-hand" src={ultiCardUrl(c)} alt={ultiCardLabel(c)} />
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
                  <th>Te</th>
                  <th>Gép 1</th>
                  <th>Gép 2</th>
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
    </div>
  );
}
