import React, { useEffect, useMemo, useState } from "react";
import {
  apiActionCloseTalon,
  apiActionDeclareMarriage,
  apiActionExchangeTrumpJack,
  apiActionPlay,
  apiContinue,
  apiListModels,
  apiNewDeal,
  apiNewGame,
  type Card,
  type Color,
  type GameState,
} from "./api";
import { cardImageUrl, cardLabel } from "./cards";

function sortHand(hand: Card[]): Card[] {
  return [...hand].sort((a, b) => (a.color < b.color ? -1 : a.color > b.color ? 1 : a.number - b.number));
}

export function App() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState<string>("");
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [seedText, setSeedText] = useState<string>(""); // empty => random
  const [state, setState] = useState<GameState | null>(null);
  const [err, setErr] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [showCaptured, setShowCaptured] = useState<boolean>(false);

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

  useEffect(() => {
    if (!state || busy) return;
    if (!state.needsContinue || state.terminal) return;
    const t = window.setTimeout(() => {
      apiContinue(state.gameId)
        .then((st) => setState(st))
        .catch((e) => setErr(String(e)));
    }, 2000);
    return () => window.clearTimeout(t);
  }, [state?.gameId, state?.needsContinue, state?.terminal, busy]);

  async function newGame() {
    setErr("");
    setBusy(true);
    try {
      const trimmed = seedText.trim();
      const seed = trimmed === "" ? null : Number(trimmed);
      const st = await apiNewGame(model, Number.isFinite(seed as number) ? (seed as number) : null);
      setState(st);
      setShowCaptured(false);
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
          <div className="subtitle">Play vs AI</div>
        </div>

        <div className="controls">
          <label className="field">
            <span>Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy}>
              <option value="">(random AI)</option>
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>

          <button className="btn btn-primary" onClick={newGame} disabled={busy}>
            New game
          </button>

          <button className="btn" onClick={newDeal} disabled={busy || !state || !state.dealOver}>
            Next deal
          </button>

          <button className="btn" onClick={() => setSettingsOpen(true)} disabled={busy}>
            Settings
          </button>
        </div>
      </header>

      <main className="main">
        <section className="panel">
          <div className="panel-head">
            <div className="panel-title">Your captured cards</div>
            <button
              className="btn btn-small"
              onClick={() => setShowCaptured((v) => !v)}
              disabled={!state || busy}
              aria-pressed={showCaptured}
            >
              {showCaptured ? "Hide" : "Show"}
            </button>
          </div>
          <div className="panel-sub">
            {state ? `Count: ${state.captured.human.length}` : "Count: —"}
          </div>
          {showCaptured ? (
            <div className="captured-grid">
              {(state?.captured.human ?? []).map((c, i) => (
                <img key={i} className="card-thumb" src={cardImageUrl(c)} alt={cardLabel(c)} />
              ))}
            </div>
          ) : (
            <div className="captured-hidden" />
          )}
        </section>

        <section className="table">
          <div className="hud">
            <div className="hud-item">
              <div className="hud-label">Scores</div>
              <div className="hud-value">{state ? `You ${state.scores[0]} — ${state.scores[1]} AI` : "—"}</div>
            </div>
            <div className="hud-item">
              <div className="hud-label">Match</div>
              <div className="hud-value">
                {state?.matchPoints ? `You ${state.matchPoints[0]} — ${state.matchPoints[1]} AI (to 7)` : "—"}
              </div>
              {state?.lastAward ? (
                <div className="hud-sub">
                  Last deal: {state.lastAward.winner === 0 ? "you" : "AI"} +{state.lastAward.points} (
                  {state.lastAward.reason})
                </div>
              ) : null}
            </div>
            <div className="hud-item">
              <div className="hud-label">Talon (draw pile)</div>
              <div className="hud-value">
                {state
                  ? `${state.talon.size} remaining — ${
                      state.talon.closed ? (state.talon.isClosedByTakaras ? "closed (takarás)" : "exhausted") : "open"
                    }`
                  : "—"}
              </div>
              {state ? (
                <div className="hud-sub">
                  {state.talon.drawPileSize} face-down + {state.talon.trumpUpcard ? 1 : 0} upcard
                </div>
              ) : null}
            </div>
            <div className="hud-item">
              <div className="hud-label">Trump</div>
              <div className="hud-value">{state ? state.talon.trumpColor : "—"}</div>
            </div>
          </div>

          <div className="center">
            <div className="trump">
              <div className="label">Upcard</div>
              {state?.talon.trumpUpcard ? (
                <img className="card-up" src={cardImageUrl(state.talon.trumpUpcard)} alt="trump upcard" />
              ) : (
                <div className="card-placeholder">No upcard</div>
              )}
              <div className="trump-actions">
                <button
                  className="btn btn-small"
                  onClick={exchangeTrumpJack}
                  disabled={busy || !!state?.needsContinue || !!state?.dealOver || !state?.canExchangeTrumpJack}
                  title="If you have the trump Jack and the talon has at least 2 cards (including the upcard), you can swap it with the upcard. Not allowed after the upcard has been picked up."
                >
                  Exchange trump
                </button>
                <button
                  className="btn btn-small"
                  onClick={closeTalon}
                  disabled={busy || !!state?.needsContinue || !!state?.dealOver || !state?.available?.canCloseTalon}
                  title="Close the talon (takarás): stop drawing; strict follow/beat rules apply."
                >
                  Close talon (takarás)
                </button>
              </div>
              {state?.available?.marriages?.length ? (
                <div className="trump-actions" style={{ marginTop: 8, flexWrap: "wrap" }}>
                  {state.available.marriages.map((m) => (
                    <button
                      key={`${m.suit}-${m.points}`}
                      className="btn btn-small"
                      onClick={() => declareMarriage(m.suit)}
                      disabled={busy || !!state.needsContinue || !!state.dealOver}
                      title="Declare 20/40. You must lead the King or Queen of that suit this trick."
                    >
                      Declare {m.points} ({m.suit})
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            <div className="trick">
              <div className="label">Lead</div>
              {state?.pendingLead ? (
                <img className="card-large" src={cardImageUrl(state.pendingLead)} alt="lead card" />
              ) : state?.needsContinue && state?.lastTrick ? (
                <img className="card-large" src={cardImageUrl(state.lastTrick.leaderCard)} alt="last lead card" />
              ) : (
                <div className="card-placeholder">—</div>
              )}
            </div>

            <div className="trick">
              <div className="label">Response</div>
              {state?.needsContinue && state?.lastTrick ? (
                <img className="card-large" src={cardImageUrl(state.lastTrick.responderCard)} alt="response card" />
              ) : (
                <div className="card-placeholder">—</div>
              )}
            </div>
          </div>

          {state?.announcements?.marriages?.length ? (
            <div className="prompt" style={{ marginTop: 10 }}>
              Announcements:{" "}
              {state.announcements.marriages
                .map((m) => `${m.player === 0 ? "You" : "AI"} ${m.points} (${m.suit})`)
                .join(" • ")}
            </div>
          ) : null}

          <div className="prompt">{state?.prompt ?? "Click New game to start."}</div>

          <div className="hand">
            {hand.map((c, i) => (
              <button
                key={i}
                className="hand-card"
                onClick={() => play(c)}
                disabled={!state || busy || !!state.needsContinue || !!state.dealOver}
              >
                <img className="card-hand" src={cardImageUrl(c)} alt={cardLabel(c)} />
              </button>
            ))}
          </div>

          {err ? <div className="error">{err}</div> : null}
        </section>

      </main>

      {settingsOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={(e) => {
            if (e.currentTarget === e.target) setSettingsOpen(false);
          }}
        >
          <div className="modal">
            <div className="modal-title">Settings</div>
            <div className="modal-row">
              <label className="field" style={{ width: "100%" }}>
                <span>Seed (optional)</span>
                <input
                  type="text"
                  inputMode="numeric"
                  placeholder="random"
                  value={seedText}
                  onChange={(e) => setSeedText(e.target.value)}
                />
              </label>
            </div>
            <div className="modal-help">
              Leave empty for a random seed. If set, the same seed produces the same deal.
              {state?.seed !== undefined ? <span> Current game seed: {state.seed}</span> : null}
            </div>
            <div className="modal-actions">
              <button className="btn" onClick={() => setSettingsOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

