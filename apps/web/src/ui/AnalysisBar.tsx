import React from "react";

/** Compute HSL background string for analysis bar given a normalized 0..1 value. */
function barStyle(norm: number, hasData: boolean): React.CSSProperties {
  if (!hasData) return { background: "hsl(0, 0%, 55%)" };
  const hue = norm * 130;
  const sat = 65 + norm * 15;
  const lit = 45 + norm * 15;
  return { background: `hsl(${hue}, ${sat}%, ${lit}%)` };
}

export type AnalysisBarProps = {
  /** Raw probability (0..1) */
  prob: number;
  /** Normalized probability relative to all legal actions (0 = worst, 1 = best) */
  norm: number;
  /** Whether MCTS data has arrived */
  hasData: boolean;
};

export function AnalysisBar({ prob, norm, hasData }: AnalysisBarProps) {
  return (
    <div
      className="analysis-bar"
      style={barStyle(norm, hasData)}
      title={hasData ? `${(prob * 100).toFixed(1)}%` : "..."}
    >
      <span className="analysis-label">
        {hasData ? `${(prob * 100).toFixed(0)}%` : "..."}
      </span>
    </div>
  );
}
