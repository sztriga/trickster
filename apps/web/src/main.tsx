import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import { App } from "./ui/App";
import { UltiApp } from "./ui/UltiApp";
import "./ui/styles.css";

type Game = "menu" | "snapszer" | "ulti";

function Root() {
  const [game, setGame] = useState<Game>("menu");

  if (game === "snapszer") return <App />;
  if (game === "ulti") return <UltiApp />;

  return (
    <div className="game-selector">
      <div className="game-selector-title">Trickster</div>
      <div className="game-selector-grid">
        <button className="game-selector-card" onClick={() => setGame("snapszer")}>
          <span className="game-icon">{"\u2660"}</span>
          <span className="game-name">Snapszer</span>
          <span className="game-desc">2 játékos</span>
        </button>
        <button className="game-selector-card" onClick={() => setGame("ulti")}>
          <span className="game-icon">{"\u2663"}</span>
          <span className="game-name">Ulti</span>
          <span className="game-desc">3 játékos</span>
        </button>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
