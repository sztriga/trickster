import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

function sortHand(hand: Card[]): Card[] {
  return [...hand].sort((a, b) => (a.color < b.color ? -1 : a.color > b.color ? 1 : a.number - b.number));
}

/** Strip params from model label: "T6-Captain  (alphazero 128x3 head=64)" → "T6-Captain" */
function modelDisplayName(label: string): string {
  const idx = label.indexOf("(");
  return idx > 0 ? label.substring(0, idx).trim() : label;
}

export function App() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState<string>("");
  const [menuOpen, setMenuOpen] = useState<boolean>(false);
  const [menuTab, setMenuTab] = useState<"main" | "newgame" | "settings">("main");
  const [seedText, setSeedText] = useState<string>(""); // empty => random
  const [mctsSims, setMctsSims] = useState<number>(50);
  const [mctsDets, setMctsDets] = useState<number>(6);
  const [analysisOn, setAnalysisOn] = useState<boolean>(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [err, setErr] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [showCaptured, setShowCaptured] = useState<boolean>(false);
  const [aiBubbleText, setAiBubbleText] = useState<string | null>(null);
  const [aiBubbleVisible, setAiBubbleVisible] = useState<boolean>(false);
  const aiBubbleTimer = useRef<number | null>(null);

  const [playerBubbleText, setPlayerBubbleText] = useState<string | null>(null);
  const [playerBubbleVisible, setPlayerBubbleVisible] = useState<boolean>(false);
  const playerBubbleTimer = useRef<number | null>(null);

  const showAiBubble = useCallback((text: string) => {
    if (aiBubbleTimer.current) window.clearTimeout(aiBubbleTimer.current);
    setAiBubbleText(text);
    setAiBubbleVisible(true);
    aiBubbleTimer.current = window.setTimeout(() => {
      setAiBubbleVisible(false);
      aiBubbleTimer.current = window.setTimeout(() => {
        setAiBubbleText(null);
        aiBubbleTimer.current = null;
      }, 400);
    }, 2000);
  }, []);

  const showPlayerBubble = useCallback((text: string) => {
    if (playerBubbleTimer.current) window.clearTimeout(playerBubbleTimer.current);
    setPlayerBubbleText(text);
    setPlayerBubbleVisible(true);
    playerBubbleTimer.current = window.setTimeout(() => {
      setPlayerBubbleVisible(false);
      playerBubbleTimer.current = window.setTimeout(() => {
        setPlayerBubbleText(null);
        playerBubbleTimer.current = null;
      }, 400);
    }, 2000);
  }, []);

  // Show bubble when AI does something notable
  useEffect(() => {
    if (state?.aiBubble) {
      showAiBubble(state.aiBubble);
    }
  }, [state?.aiBubble, showAiBubble]);

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

  useEffect(() => {
    apiListModels()
      .then((xs) => {
        setModels(xs);
        if (xs.length && !model) setModel(xs[0]);
      })
      .catch(() => {
        // ok: user may not have models yet
        setModels([]);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const hand = useMemo(() => (state ? sortHand(state.hands.human) : []), [state]);

  const legalSet = useMemo(() => {
    const s = new Set<string>();
    for (const c of state?.legalCards ?? []) {
      s.add(`${c.color}:${c.number}`);
    }
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
          // Keep polling while search is still running
          if (a.searching) {
            timer = setTimeout(poll, 800);
          }
        })
        .catch(() => { if (!cancelled) setAnalysis(null); });
    };
    poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisOn, state?.gameId, state?.trickNo, state?.scores?.[0], state?.scores?.[1], state?.needsContinue, state?.dealOver, state?.prompt]);

  // Helper to get probability for a card
  const cardProb = (c: Card): number | null => {
    if (!analysis) return null;
    const a = analysis.actions.find(
      (x) => x.type === "card" && x.card?.color === c.color && x.card?.number === c.number,
    );
    return a ? a.prob : 0;
  };

  const actionProb = (type: string, suit?: string): number | null => {
    if (!analysis) return null;
    const a = analysis.actions.find(
      (x) => x.type === type && (suit === undefined || x.suit === suit),
    );
    return a ? a.prob : 0;
  };

  // Compute all probs (cards + actions) to normalize min/max
  const allProbs: number[] = [];
  if (analysisOn && analysis) {
    for (const c of hand) {
      allProbs.push(cardProb(c) ?? 0);
    }
    const ct = actionProb("close_talon");
    if (ct !== null && ct > 0) allProbs.push(ct);
    for (const m of state?.available?.marriages ?? []) {
      const mp = actionProb("marriage", m.suit);
      if (mp !== null && mp > 0) allProbs.push(mp);
    }
  }
  const probMin = allProbs.length > 0 ? Math.min(...allProbs) : 0;
  const probMax = allProbs.length > 0 ? Math.max(...allProbs) : 1;
  const probRange = probMax - probMin || 1; // avoid div-by-zero

  // Normalize a probability to 0..1 within the range (0 = worst, 1 = best)
  const normProb = (p: number) => Math.max(0, Math.min(1, (p - probMin) / probRange));

  useEffect(() => {
    if (!state || busy) return;
    if (!state.needsContinue) return;
    const t = window.setTimeout(() => {
      apiContinue(state.gameId)
        .then((st) => setState(st))
        .catch((e) => setErr(String(e)));
    }, 2000);
    return () => window.clearTimeout(t);
  }, [state?.gameId, state?.needsContinue, busy]);

  async function newGame() {
    setErr("");
    setBusy(true);
    try {
      const trimmed = seedText.trim();
      const seed = trimmed === "" ? null : Number(trimmed);
      const st = await apiNewGame(model, Number.isFinite(seed as number) ? (seed as number) : null);
      setState(st);
      setShowCaptured(false);
      setAiBubbleText(null); setAiBubbleVisible(false);
      setPlayerBubbleText(null); setPlayerBubbleVisible(false);
      // Restore MCTS settings on the new session (new game creates a fresh session with defaults)
      if (st.gameId) {
        await apiUpdateSettings(st.gameId, mctsSims, mctsDets);
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function newDeal() {
    if (!state) return;
    setErr("");
    setBusy(true);
    try {
      const trimmed = seedText.trim();
      const seed = trimmed === "" ? null : Number(trimmed);
      const st = await apiNewDeal(state.gameId, Number.isFinite(seed as number) ? (seed as number) : null);
      setState(st);
      setShowCaptured(false);
      setAiBubbleText(null); setAiBubbleVisible(false);
      setPlayerBubbleText(null); setPlayerBubbleVisible(false);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function play(card: Card) {
    if (!state) return;
    setErr("");
    setBusy(true);
    try {
      const st = await apiActionPlay(state.gameId, card);
      setState(st);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function exchangeTrumpJack() {
    if (!state) return;
    setErr("");
    setBusy(true);
    try {
      showPlayerBubble("Cserélek!");
      const st = await apiActionExchangeTrumpJack(state.gameId);
      setState(st);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function closeTalon() {
    if (!state) return;
    setErr("");
    setBusy(true);
    try {
      showPlayerBubble("Betakarok!");
      const st = await apiActionCloseTalon(state.gameId);
      setState(st);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function declareMarriage(suit: Color) {
    if (!state) return;
    setErr("");
    setBusy(true);
    try {
      const isTrump = state.talon.trumpColor === suit;
      showPlayerBubble(isTrump ? "Van 40-em!" : "Van 20-am!");
      const st = await apiActionDeclareMarriage(state.gameId, suit);
      setState(st);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

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
            <button className="btn btn-primary" onClick={newDeal} disabled={busy}>
              Következő kör
            </button>
          )}
          <button className="menu-btn" onClick={() => { setMenuTab("main"); setMenuOpen(true); }}>
            <span className="menu-btn-icon">☰</span>
            <span className="menu-btn-esc">ESC</span>
          </button>
        </div>
      </header>

      <main className="main">
        <section className="panel">
          <div className="panel-head">
            <div className="panel-title">Fogott lapok</div>
            <button
              className="btn btn-small"
              onClick={() => setShowCaptured((v) => !v)}
              disabled={!state || busy}
              aria-pressed={showCaptured}
            >
              {showCaptured ? "Elrejt" : "Mutat"}
            </button>
          </div>
          <div className="panel-sub">
            {state ? `${state.captured.human.length} lap` : "—"}
          </div>
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
                    <img
                      key={i}
                      className="card-stack-item"
                      src={cardBackUrl()}
                      alt="captured card"
                      style={{ top: i * 2, left: i * 1 }}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </section>

        <section className="table">
          <div className="scoreboard">
            <div className="scoreboard-section">
              <div className="scoreboard-title">Kör</div>
              <div className="scoreboard-row">
                <div className="score-side score-you">
                  <div className="score-label">Te</div>
                  <div className="score-num">{state?.scores[0] ?? 0}</div>
                </div>
                <div className="score-divider">:</div>
                <div className="score-side score-ai">
                  <div className="score-num">{state?.scores[1] ?? 0}</div>
                  <div className="score-label">Gép</div>
                </div>
              </div>
            </div>
            <div className="scoreboard-sep" />
            <div className="scoreboard-section">
              <div className="scoreboard-title">Meccs</div>
              <div className="scoreboard-row">
                <div className="score-side score-you">
                  <div className="score-num score-num-match">{state?.matchPoints?.[0] ?? 0}</div>
                </div>
                <div className="score-divider score-divider-match">:</div>
                <div className="score-side score-ai">
                  <div className="score-num score-num-match">{state?.matchPoints?.[1] ?? 0}</div>
                </div>
              </div>
            </div>
            {state?.lastAward ? (
              <>
                <div className="scoreboard-sep" />
                <div className="scoreboard-award">
                  {state.lastAward.winner === 0 ? "Te" : "Gép"} +{state.lastAward.points}
                </div>
              </>
            ) : null}
            {state?.talon?.trumpColor && (
              <>
                <div className="scoreboard-sep" />
                <div className="scoreboard-trump">
                  Adu: <span className={`trump-symbol trump-${state.talon.trumpColor.toLowerCase()}`}>
                    {{ HEARTS: "♥", BELLS: "♦", LEAVES: "♠", ACORNS: "♣" }[state.talon.trumpColor] ?? "?"}
                  </span>
                  <span className="trump-name">
                    {{ HEARTS: "Piros", BELLS: "Tök", LEAVES: "Zöld", ACORNS: "Makk" }[state.talon.trumpColor] ?? ""}
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="center">
            <div className="bubble-column">
              {aiBubbleText && (
                <div className={`speech-bubble bubble-ai ${aiBubbleVisible ? "bubble-in-up" : "bubble-out-up"}`}>
                  <span>{aiBubbleText}</span>
                </div>
              )}
              <div className="bubble-spacer" />
              {playerBubbleText && (
                <div className={`speech-bubble bubble-player ${playerBubbleVisible ? "bubble-in-down" : "bubble-out-down"}`}>
                  <span>{playerBubbleText}</span>
                </div>
              )}
            </div>
            <div className="trump">
              <div className="label">Pakli / Adu</div>
              <div className="talon-visual">
                {state?.talon.isClosedByTakaras ? (
                  <img className="card-up talon-cover" src={cardBackUrl()} alt="betakarva" />
                ) : state?.talon.trumpUpcard ? (
                  <img className="card-up" src={cardImageUrl(state.talon.trumpUpcard)} alt="adu" />
                ) : !state ? (
                  <div className="card-placeholder card-placeholder-up">—</div>
                ) : null}
                {(state?.talon.drawPileSize ?? 0) > 0 && (
                  <div className="talon-pile">
                    {Array.from({ length: state!.talon.drawPileSize }).map((_, i) => (
                      <img
                        key={i}
                        className="talon-pile-card"
                        src={cardBackUrl()}
                        alt="talon lap"
                        style={{ top: i * -2, left: i * 1 }}
                      />
                    ))}
                  </div>
                )}
              </div>
              <div className="trump-actions">
                <div className="action-slot">
                  <button
                    className="btn btn-small"
                    onClick={exchangeTrumpJack}
                    disabled={busy || !!state?.needsContinue || !!state?.dealOver || !state?.canExchangeTrumpJack}
                    title="Az adu alsót elcseréled az adu lapra."
                  >
                    Csere
                  </button>
                </div>
                <div className="action-slot">
                  <button
                    className="btn btn-small"
                    onClick={closeTalon}
                    disabled={busy || !!state?.needsContinue || !!state?.dealOver || !state?.available?.canCloseTalon}
                    title="Betakarás: nem húztok többet, kötelező színt és felülütést játszani."
                  >
                    Betakarás
                  </button>
                  {analysisOn && (() => {
                    const cp = actionProb("close_talon") ?? 0;
                    const hasD = analysis && analysis.progress > 0;
                    const cn = hasD ? normProb(cp) : 0;
                    return (
                      <div
                        className="analysis-bar"
                        style={{ background: hasD ? `hsl(${cn * 130}, ${65 + cn * 15}%, ${45 + cn * 15}%)` : "hsl(0, 0%, 55%)" }}
                        title={hasD ? `${(cp * 100).toFixed(1)}%` : "..."}
                      >
                        <span className="analysis-label">{hasD ? `${(cp * 100).toFixed(0)}%` : "..."}</span>
                      </div>
                    );
                  })()}
                </div>
              </div>
              {state?.available?.marriages?.length ? (
                <div className="trump-actions" style={{ marginTop: 8, flexWrap: "wrap" }}>
                  {state.available.marriages.map((m) => {
                    const mp = analysisOn ? actionProb("marriage", m.suit) : null;
                    return (
                      <div key={`${m.suit}-${m.points}`} className="action-slot">
                        <button
                          className="btn btn-small"
                          onClick={() => declareMarriage(m.suit)}
                          disabled={busy || !!state.needsContinue || !!state.dealOver}
                          title="Bemondás: királyt vagy dámát kell kijátszanod ebből a színből."
                        >
                          {m.points} bemondás
                        </button>
                        {analysisOn && (() => {
                          const mpp = mp ?? 0;
                          const hasD = analysis && analysis.progress > 0;
                          const mn = hasD ? normProb(mpp) : 0;
                          return (
                            <div
                              className="analysis-bar"
                              style={{ background: hasD ? `hsl(${mn * 130}, ${65 + mn * 15}%, ${45 + mn * 15}%)` : "hsl(0, 0%, 55%)" }}
                              title={hasD ? `${(mpp * 100).toFixed(1)}%` : "..."}
                            >
                              <span className="analysis-label">{hasD ? `${(mpp * 100).toFixed(0)}%` : "..."}</span>
                            </div>
                          );
                        })()}
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
                className={`eval-badge${analysis && analysis.progress > 0 ? (analysis.value > 0.1 ? " eval-good" : analysis.value < -0.1 ? " eval-bad" : "") : ""}`}
                title="Pozíció értékelés (neked)"
              >
                {analysis && analysis.progress > 0
                  ? `${analysis.value > 0 ? "+" : ""}${analysis.value.toFixed(2)}`
                  : "..."}
                {analysis?.searching && (
                  <span className="eval-progress"> ({analysis.progress}/{analysis.total})</span>
                )}
              </span>
            )}
          </div>

          <div className="hand">
            {hand.map((c, i) => {
              const canPlay = !state || busy || !!state.needsContinue || !!state.dealOver;
              const legal = isLegal(c);
              const prob = analysisOn ? (cardProb(c) ?? 0) : null;
              const hasData = analysisOn && analysis && analysis.progress > 0;
              const norm = hasData && prob !== null ? normProb(prob) : 0;
              const hue = hasData ? norm * 130 : 0;
              const sat = hasData ? 65 + norm * 15 : 0;
              const lit = hasData ? 45 + norm * 15 : 55;
              return (
                <div key={i} className="hand-slot">
                  <button
                    className={`hand-card${!canPlay && !legal ? " hand-card-illegal" : ""}`}
                    onClick={() => play(c)}
                    disabled={canPlay}
                  >
                    <img className="card-hand" src={cardImageUrl(c)} alt={cardLabel(c)} />
                  </button>
                  {analysisOn && (
                    <div
                      className="analysis-bar"
                      style={{ background: hasData ? `hsl(${hue}, ${sat}%, ${lit}%)` : "hsl(0, 0%, 55%)" }}
                      title={hasData ? `${((prob ?? 0) * 100).toFixed(1)}%` : "..."}
                    >
                      <span className="analysis-label">{hasData ? `${((prob ?? 0) * 100).toFixed(0)}%` : "..."}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {err ? <div className="error">{err}</div> : null}
        </section>

      </main>

      {menuOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={(e) => {
            if (e.currentTarget === e.target) setMenuOpen(false);
          }}
        >
          <div className="modal menu-modal">
            {menuTab === "main" && (
              <>
                <div className="menu-title">Trickster</div>
                <div className="menu-grid">
                  <button className="menu-card" onClick={() => setMenuTab("newgame")}>
                    <span className="menu-card-icon">♠</span>
                    <span className="menu-card-label">Új játék</span>
                  </button>
                  <button className="menu-card" onClick={() => setMenuTab("settings")}>
                    <span className="menu-card-icon">⚙</span>
                    <span className="menu-card-label">Beállítások</span>
                  </button>
                  <button className="menu-card" onClick={() => setMenuOpen(false)}>
                    <span className="menu-card-icon">▶</span>
                    <span className="menu-card-label">Vissza</span>
                  </button>
                </div>
              </>
            )}

            {menuTab === "newgame" && (
              <>
                <div className="menu-title">Válassz ellenfelet</div>
                <div className="menu-opponents">
                  {models.map((m) => (
                    <button
                      key={m}
                      className={`menu-opponent${model === m ? " menu-opponent-active" : ""}`}
                      onClick={() => setModel(m)}
                    >
                      <span className="menu-opponent-name">{modelDisplayName(m)}</span>
                    </button>
                  ))}
                  <button
                    className={`menu-opponent${model === "" ? " menu-opponent-active" : ""}`}
                    onClick={() => setModel("")}
                  >
                    <span className="menu-opponent-name">Véletlen</span>
                  </button>
                </div>
                <div className="modal-actions">
                  <button
                    className="btn btn-primary btn-large"
                    disabled={busy}
                    onClick={async () => {
                      setMenuOpen(false);
                      await newGame();
                    }}
                  >
                    Indítás
                  </button>
                  <button className="btn" onClick={() => setMenuTab("main")}>
                    Vissza
                  </button>
                </div>
              </>
            )}

            {menuTab === "settings" && (
              <>
                <div className="menu-title">Beállítások</div>

                <div className="modal-row">
                  <label className="field" style={{ width: "100%" }}>
                    <span>Seed (opcionális)</span>
                    <input
                      type="text"
                      inputMode="numeric"
                      placeholder="véletlenszerű"
                      value={seedText}
                      onChange={(e) => setSeedText(e.target.value)}
                    />
                  </label>
                </div>
                <div className="modal-help">
                  Hagyd üresen véletlenszerű osztáshoz. Azonos seed azonos osztást ad.
                  {state?.seed !== undefined ? <span> Jelenlegi seed: {state.seed}</span> : null}
                </div>

                <div className="modal-divider" />

                <div className="modal-subtitle">AI gondolkodás (MCTS)</div>
                <div className="modal-sliders">
                  <label className="field field-short">
                    <span>Szimulációk: {mctsSims}</span>
                    <input
                      type="range"
                      min={10}
                      max={500}
                      step={10}
                      value={mctsSims}
                      onChange={(e) => setMctsSims(Number(e.target.value))}
                    />
                  </label>
                  <label className="field field-short">
                    <span>Determinizációk: {mctsDets}</span>
                    <input
                      type="range"
                      min={1}
                      max={30}
                      step={1}
                      value={mctsDets}
                      onChange={(e) => setMctsDets(Number(e.target.value))}
                    />
                  </label>
                </div>
                <div className="modal-help">
                  <strong>Szimulációk:</strong> hány lépést gondol végig az AI egy-egy világban.
                  <br />
                  <strong>Determinizációk:</strong> hány lehetséges kártyaosztást képzel el az ellenfélnél.
                  <br />
                  Összesen {mctsSims * mctsDets} keresés lépésenként. Több = erősebb, de lassabb.
                </div>

                <div className="modal-divider" />

                <div className="modal-subtitle">Elemzés mód</div>
                <label className="field toggle-field">
                  <input
                    type="checkbox"
                    checked={analysisOn}
                    onChange={(e) => setAnalysisOn(e.target.checked)}
                  />
                  <span>Pozíció értékelése és lépés-javaslatok megjelenítése</span>
                </label>
                <div className="modal-help">
                  Bekapcsolva a kártyák alatt egy sáv mutatja az AI által számított valószínűségeket.
                  <br />
                  Zöld = javasolt, piros = kerülendő. Az értékelés a státuszsávban jelenik meg.
                </div>

                <div className="modal-actions">
                  <button
                    className="btn btn-primary"
                    onClick={async () => {
                      if (state) {
                        try {
                          await apiUpdateSettings(state.gameId, mctsSims, mctsDets);
                        } catch (e) {
                          setErr(String(e));
                        }
                      }
                      setMenuTab("main");
                    }}
                  >
                    Mentés
                  </button>
                  <button className="btn" onClick={() => setMenuTab("main")}>
                    Vissza
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

