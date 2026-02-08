import React, { useEffect, useMemo, useState } from "react";
import {
  apiActionCloseTalon,
  apiActionDeclareMarriage,
  apiActionExchangeTrumpJack,
  apiActionPlay,
  apiAnalyze,
  apiContinue,
  apiListModels,
  apiNewDeal,
  apiNewGame,
  apiUpdateSettings,
  type Analysis,
  type Card,
  type Color,
  type GameState,
} from "./api";
import { cardBackUrl, cardImageUrl, cardLabel } from "./cards";
import { AnalysisBar } from "./AnalysisBar";
import { useSpeechBubble } from "./SpeechBubble";
import { Scoreboard } from "./Scoreboard";
import { MenuModal, modelDisplayName } from "./MenuModal";

function sortHand(hand: Card[]): Card[] {
  return [...hand].sort((a, b) => (a.color < b.color ? -1 : a.color > b.color ? 1 : a.number - b.number));
}

export function App() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState<string>("");
  const [menuOpen, setMenuOpen] = useState<boolean>(false);
  const [menuTab, setMenuTab] = useState<"main" | "newgame" | "settings">("main");
  const [seedText, setSeedText] = useState<string>("");
  const [mctsSims, setMctsSims] = useState<number>(50);
  const [mctsDets, setMctsDets] = useState<number>(6);
  const [analysisOn, setAnalysisOn] = useState<boolean>(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [err, setErr] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [showCaptured, setShowCaptured] = useState<boolean>(false);

  const [aiBubble, aiBubbleNode] = useSpeechBubble("ai");
  const [playerBubble, playerBubbleNode] = useSpeechBubble("player");

  // Show bubble when AI does something notable
  useEffect(() => {
    if (state?.aiBubble) aiBubble.show(state.aiBubble);
  }, [state?.aiBubble, aiBubble]);

  // ESC key toggles the menu
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenuOpen((v) => {
          if (v) return false;
          setMenuTab("main");
          return true;
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Load model list on mount
  useEffect(() => {
    apiListModels()
      .then((xs) => {
        setModels(xs);
        if (xs.length && !model) setModel(xs[0]);
      })
      .catch(() => setModels([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const hand = useMemo(() => (state ? sortHand(state.hands.human) : []), [state]);

  const legalSet = useMemo(() => {
    const s = new Set<string>();
    for (const c of state?.legalCards ?? []) s.add(`${c.color}:${c.number}`);
    return s;
  }, [state]);

  const isLegal = (c: Card) => legalSet.size === 0 || legalSet.has(`${c.color}:${c.number}`);

  // Progressive MCTS analysis: poll backend while search is running
  useEffect(() => {
    if (!analysisOn || !state || state.needsContinue || state.dealOver) {
      setAnalysis(null);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = () => {
      if (cancelled) return;
      apiAnalyze(state.gameId)
        .then((a) => {
          if (cancelled) return;
          setAnalysis(a);
          if (a.searching) timer = setTimeout(poll, 800);
        })
        .catch(() => { if (!cancelled) setAnalysis(null); });
    };
    poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisOn, state?.gameId, state?.trickNo, state?.scores?.[0], state?.scores?.[1], state?.needsContinue, state?.dealOver, state?.prompt]);

  // Analysis helpers
  const cardProb = (c: Card): number => {
    if (!analysis) return 0;
    const a = analysis.actions.find((x) => x.type === "card" && x.card?.color === c.color && x.card?.number === c.number);
    return a ? a.prob : 0;
  };
  const actionProb = (type: string, suit?: string): number => {
    if (!analysis) return 0;
    const a = analysis.actions.find((x) => x.type === type && (suit === undefined || x.suit === suit));
    return a ? a.prob : 0;
  };

  // Normalize probabilities across all legal actions for relative coloring
  const allProbs: number[] = [];
  if (analysisOn && analysis) {
    for (const c of hand) allProbs.push(cardProb(c));
    const ct = actionProb("close_talon");
    if (ct > 0) allProbs.push(ct);
    for (const m of state?.available?.marriages ?? []) {
      const mp = actionProb("marriage", m.suit);
      if (mp > 0) allProbs.push(mp);
    }
  }
  const probMin = allProbs.length > 0 ? Math.min(...allProbs) : 0;
  const probMax = allProbs.length > 0 ? Math.max(...allProbs) : 1;
  const probRange = probMax - probMin || 1;
  const normProb = (p: number) => Math.max(0, Math.min(1, (p - probMin) / probRange));
  const hasAnalysisData = analysisOn && !!analysis && analysis.progress > 0;

  // Auto-continue after trick display
  useEffect(() => {
    if (!state || busy || !state.needsContinue) return;
    const t = window.setTimeout(() => {
      apiContinue(state.gameId).then((st) => setState(st)).catch((e) => setErr(String(e)));
    }, 2000);
    return () => window.clearTimeout(t);
  }, [state?.gameId, state?.needsContinue, busy]);

  // --- Actions ---

  async function newGame() {
    setErr(""); setBusy(true);
    try {
      const trimmed = seedText.trim();
      const seed = trimmed === "" ? null : Number(trimmed);
      const st = await apiNewGame(model, Number.isFinite(seed as number) ? (seed as number) : null);
      setState(st);
      setShowCaptured(false);
      aiBubble.clear(); playerBubble.clear();
      if (st.gameId) await apiUpdateSettings(st.gameId, mctsSims, mctsDets);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  async function newDeal() {
    if (!state) return;
    setErr(""); setBusy(true);
    try {
      const trimmed = seedText.trim();
      const seed = trimmed === "" ? null : Number(trimmed);
      const st = await apiNewDeal(state.gameId, Number.isFinite(seed as number) ? (seed as number) : null);
      setState(st);
      setShowCaptured(false);
      aiBubble.clear(); playerBubble.clear();
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  async function play(card: Card) {
    if (!state) return;
    setErr(""); setBusy(true);
    try { setState(await apiActionPlay(state.gameId, card)); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  async function exchangeTrumpJack() {
    if (!state) return;
    setErr(""); setBusy(true);
    try { playerBubble.show("Cserélek!"); setState(await apiActionExchangeTrumpJack(state.gameId)); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  async function closeTalon() {
    if (!state) return;
    setErr(""); setBusy(true);
    try { playerBubble.show("Betakarok!"); setState(await apiActionCloseTalon(state.gameId)); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  async function declareMarriage(suit: Color) {
    if (!state) return;
    setErr(""); setBusy(true);
    try {
      const isTrump = state.talon.trumpColor === suit;
      playerBubble.show(isTrump ? "Van 40-em!" : "Van 20-am!");
      setState(await apiActionDeclareMarriage(state.gameId, suit));
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }

  const actionsDisabled = busy || !!state?.needsContinue || !!state?.dealOver;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="title">Trickster</div>
          <div className="subtitle">
            {state && model ? `vs ${modelDisplayName(model)}` : "Játék a gép ellen"}
          </div>
        </div>
        <div className="controls">
          {state?.dealOver && (
            <button className="btn btn-primary" onClick={newDeal} disabled={busy}>Következő kör</button>
          )}
          <button className="menu-btn" onClick={() => { setMenuTab("main"); setMenuOpen(true); }}>
            <span className="menu-btn-icon">{"\u2630"}</span>
            <span className="menu-btn-esc">ESC</span>
          </button>
        </div>
      </header>

      <main className="main">
        <section className="panel">
          <div className="panel-head">
            <div className="panel-title">Fogott lapok</div>
            <button className="btn btn-small" onClick={() => setShowCaptured((v) => !v)} disabled={!state || busy} aria-pressed={showCaptured}>
              {showCaptured ? "Elrejt" : "Mutat"}
            </button>
          </div>
          <div className="panel-sub">{state ? `${state.captured.human.length} lap` : "\u2014"}</div>
          {showCaptured ? (
            <div className="captured-grid">
              {(state?.captured.human ?? []).map((c, i) => (
                <img key={i} className="card-thumb" src={cardImageUrl(c)} alt={cardLabel(c)} />
              ))}
            </div>
          ) : (
            <div className="captured-pile">
              {(state?.captured.human ?? []).length > 0 && (
                <div className="card-stack" style={{ height: Math.min(200, 60 + (state!.captured.human.length - 1) * 2) }}>
                  {state!.captured.human.map((_, i) => (
                    <img key={i} className="card-stack-item" src={cardBackUrl()} alt="captured card" style={{ top: i * 2, left: i * 1 }} />
                  ))}
                </div>
              )}
            </div>
          )}
        </section>

        <section className="table">
          <Scoreboard state={state} />

          <div className="center">
            <div className="bubble-column">
              {aiBubbleNode}
              <div className="bubble-spacer" />
              {playerBubbleNode}
            </div>

            <div className="trump">
              <div className="label">Pakli / Adu</div>
              <div className="talon-visual">
                {state?.talon.isClosedByTakaras ? (
                  <img className="card-up talon-cover" src={cardBackUrl()} alt="betakarva" />
                ) : state?.talon.trumpUpcard ? (
                  <img className="card-up" src={cardImageUrl(state.talon.trumpUpcard)} alt="adu" />
                ) : !state ? (
                  <div className="card-placeholder card-placeholder-up">{"\u2014"}</div>
                ) : null}
                {(state?.talon.drawPileSize ?? 0) > 0 && (
                  <div className="talon-pile">
                    {Array.from({ length: state!.talon.drawPileSize }).map((_, i) => (
                      <img key={i} className="talon-pile-card" src={cardBackUrl()} alt="talon lap" style={{ top: i * -2, left: i * 1 }} />
                    ))}
                  </div>
                )}
              </div>
              <div className="trump-actions">
                <div className="action-slot">
                  <button className="btn btn-small" onClick={exchangeTrumpJack} disabled={actionsDisabled || !state?.canExchangeTrumpJack} title="Az adu alsót elcseréled az adu lapra.">
                    Csere
                  </button>
                </div>
                <div className="action-slot">
                  <button className="btn btn-small" onClick={closeTalon} disabled={actionsDisabled || !state?.available?.canCloseTalon} title="Betakarás: nem húztok többet, kötelező színt és felülütést játszani.">
                    Betakarás
                  </button>
                  {analysisOn && <AnalysisBar prob={actionProb("close_talon")} norm={hasAnalysisData ? normProb(actionProb("close_talon")) : 0} hasData={hasAnalysisData} />}
                </div>
              </div>
              {state?.available?.marriages?.length ? (
                <div className="trump-actions" style={{ marginTop: 8, flexWrap: "wrap" }}>
                  {state.available.marriages.map((m) => {
                    const mp = actionProb("marriage", m.suit);
                    return (
                      <div key={`${m.suit}-${m.points}`} className="action-slot">
                        <button className="btn btn-small" onClick={() => declareMarriage(m.suit)} disabled={actionsDisabled} title="Bemondás: királyt vagy dámát kell kijátszanod ebből a színből.">
                          {m.points} bemondás
                        </button>
                        {analysisOn && <AnalysisBar prob={mp} norm={hasAnalysisData ? normProb(mp) : 0} hasData={hasAnalysisData} />}
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </div>

            <div className="trick">
              <div className="label">Kijátszás</div>
              {state?.pendingLead ? (
                <img className="card-large" src={cardImageUrl(state.pendingLead)} alt="lead card" />
              ) : state?.needsContinue && state?.lastTrick ? (
                <img className="card-large" src={cardImageUrl(state.lastTrick.leaderCard)} alt="last lead card" />
              ) : (
                <div className="card-placeholder-empty" />
              )}
            </div>

            <div className="trick">
              <div className="label">Válasz</div>
              {state?.needsContinue && state?.lastTrick ? (
                <img className="card-large" src={cardImageUrl(state.lastTrick.responderCard)} alt="response card" />
              ) : (
                <div className="card-placeholder-empty" />
              )}
            </div>
          </div>

          <div className="status-bar">
            {state?.prompt ?? "Kattints az Új játék gombra."}
            {analysisOn && state && !state.dealOver && (
              <span
                className={`eval-badge${hasAnalysisData ? (analysis!.value > 0.1 ? " eval-good" : analysis!.value < -0.1 ? " eval-bad" : "") : ""}`}
                title="Pozíció értékelés (neked)"
              >
                {hasAnalysisData ? `${analysis!.value > 0 ? "+" : ""}${analysis!.value.toFixed(2)}` : "..."}
                {analysis?.searching && <span className="eval-progress"> ({analysis.progress}/{analysis.total})</span>}
              </span>
            )}
          </div>

          <div className="hand">
            {hand.map((c, i) => {
              const disabled = !state || busy || !!state.needsContinue || !!state.dealOver;
              const legal = isLegal(c);
              const prob = cardProb(c);
              return (
                <div key={i} className="hand-slot">
                  <button className={`hand-card${!disabled && !legal ? " hand-card-illegal" : ""}`} onClick={() => play(c)} disabled={disabled}>
                    <img className="card-hand" src={cardImageUrl(c)} alt={cardLabel(c)} />
                  </button>
                  {analysisOn && <AnalysisBar prob={prob} norm={hasAnalysisData ? normProb(prob) : 0} hasData={hasAnalysisData} />}
                </div>
              );
            })}
          </div>

          {err ? <div className="error">{err}</div> : null}
        </section>
      </main>

      {menuOpen && (
        <MenuModal
          tab={menuTab}
          setTab={setMenuTab}
          onClose={() => setMenuOpen(false)}
          models={models}
          selectedModel={model}
          onSelectModel={setModel}
          onStartGame={async () => { setMenuOpen(false); await newGame(); }}
          busy={busy}
          seedText={seedText}
          onSeedChange={setSeedText}
          mctsSims={mctsSims}
          onSimsChange={setMctsSims}
          mctsDets={mctsDets}
          onDetsChange={setMctsDets}
          analysisOn={analysisOn}
          onAnalysisChange={setAnalysisOn}
          onSaveSettings={async () => {
            if (state) {
              try { await apiUpdateSettings(state.gameId, mctsSims, mctsDets); }
              catch (e) { setErr(String(e)); }
            }
            setMenuTab("main");
          }}
          state={state}
        />
      )}
    </div>
  );
}
