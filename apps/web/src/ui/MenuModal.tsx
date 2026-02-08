import React from "react";
import type { GameState } from "./api";

/** Strip params from model label: "T6-Captain  (alphazero 128x3 head=64)" → "T6-Captain" */
export function modelDisplayName(label: string): string {
  const idx = label.indexOf("(");
  return idx > 0 ? label.substring(0, idx).trim() : label;
}

export type MenuModalProps = {
  tab: "main" | "newgame" | "settings";
  setTab: (t: "main" | "newgame" | "settings") => void;
  onClose: () => void;
  // New game
  models: string[];
  selectedModel: string;
  onSelectModel: (m: string) => void;
  onStartGame: () => void;
  busy: boolean;
  // Settings
  seedText: string;
  onSeedChange: (v: string) => void;
  mctsSims: number;
  onSimsChange: (v: number) => void;
  mctsDets: number;
  onDetsChange: (v: number) => void;
  analysisOn: boolean;
  onAnalysisChange: (v: boolean) => void;
  onSaveSettings: () => void;
  state: GameState | null;
};

export function MenuModal(props: MenuModalProps) {
  const {
    tab, setTab, onClose,
    models, selectedModel, onSelectModel, onStartGame, busy,
    seedText, onSeedChange, mctsSims, onSimsChange, mctsDets, onDetsChange,
    analysisOn, onAnalysisChange, onSaveSettings, state,
  } = props;

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      onMouseDown={(e) => { if (e.currentTarget === e.target) onClose(); }}
    >
      <div className="modal menu-modal">
        {tab === "main" && (
          <>
            <div className="menu-title">Trickster</div>
            <div className="menu-grid">
              <button className="menu-card" onClick={() => setTab("newgame")}>
                <span className="menu-card-icon">{"\u2660"}</span>
                <span className="menu-card-label">Új játék</span>
              </button>
              <button className="menu-card" onClick={() => setTab("settings")}>
                <span className="menu-card-icon">{"\u2699"}</span>
                <span className="menu-card-label">Beállítások</span>
              </button>
              <button className="menu-card" onClick={onClose}>
                <span className="menu-card-icon">{"\u25B6"}</span>
                <span className="menu-card-label">Vissza</span>
              </button>
            </div>
          </>
        )}

        {tab === "newgame" && (
          <>
            <div className="menu-title">Válassz ellenfelet</div>
            <div className="menu-opponents">
              {models.map((m) => (
                <button
                  key={m}
                  className={`menu-opponent${selectedModel === m ? " menu-opponent-active" : ""}`}
                  onClick={() => onSelectModel(m)}
                >
                  <span className="menu-opponent-name">{modelDisplayName(m)}</span>
                </button>
              ))}
              <button
                className={`menu-opponent${selectedModel === "" ? " menu-opponent-active" : ""}`}
                onClick={() => onSelectModel("")}
              >
                <span className="menu-opponent-name">Véletlen</span>
              </button>
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary btn-large" disabled={busy} onClick={onStartGame}>
                Indítás
              </button>
              <button className="btn" onClick={() => setTab("main")}>Vissza</button>
            </div>
          </>
        )}

        {tab === "settings" && (
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
                  onChange={(e) => onSeedChange(e.target.value)}
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
                <input type="range" min={10} max={500} step={10} value={mctsSims} onChange={(e) => onSimsChange(Number(e.target.value))} />
              </label>
              <label className="field field-short">
                <span>Determinizációk: {mctsDets}</span>
                <input type="range" min={1} max={30} step={1} value={mctsDets} onChange={(e) => onDetsChange(Number(e.target.value))} />
              </label>
            </div>
            <div className="modal-help">
              <strong>Szimulációk:</strong> hány lépést gondol végig az AI egy-egy világban.<br />
              <strong>Determinizációk:</strong> hány lehetséges kártyaosztást képzel el az ellenfélnél.<br />
              Összesen {mctsSims * mctsDets} keresés lépésenként. Több = erősebb, de lassabb.
            </div>

            <div className="modal-divider" />

            <div className="modal-subtitle">Elemzés mód</div>
            <label className="field toggle-field">
              <input type="checkbox" checked={analysisOn} onChange={(e) => onAnalysisChange(e.target.checked)} />
              <span>Pozíció értékelése és lépés-javaslatok megjelenítése</span>
            </label>
            <div className="modal-help">
              Bekapcsolva a kártyák alatt egy sáv mutatja az AI által számított valószínűségeket.<br />
              Zöld = javasolt, piros = kerülendő. Az értékelés a státuszsávban jelenik meg.
            </div>

            <div className="modal-actions">
              <button className="btn btn-primary" onClick={onSaveSettings}>Mentés</button>
              <button className="btn" onClick={() => setTab("main")}>Vissza</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
