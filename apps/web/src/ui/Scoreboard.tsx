import React from "react";
import type { GameState } from "./api";

const TRUMP_SYMBOLS: Record<string, string> = {
  HEARTS: "\u2665", BELLS: "\u2666", LEAVES: "\u2660", ACORNS: "\u2663",
};
const TRUMP_NAMES: Record<string, string> = {
  HEARTS: "Piros", BELLS: "Tök", LEAVES: "Zöld", ACORNS: "Makk",
};

export type ScoreboardProps = {
  state: GameState | null;
};

export function Scoreboard({ state }: ScoreboardProps) {
  return (
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
              {TRUMP_SYMBOLS[state.talon.trumpColor] ?? "?"}
            </span>
            <span className="trump-name">
              {TRUMP_NAMES[state.talon.trumpColor] ?? ""}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
