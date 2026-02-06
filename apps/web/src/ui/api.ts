export type Color = "HEARTS" | "BELLS" | "LEAVES" | "ACORNS";

export type Card = {
  color: Color;
  number: number;
};

export type GameState = {
  gameId: string;
  prompt: string;
  seed?: number;
  needsContinue?: boolean;
  dealOver?: boolean;
  matchPoints?: [number, number] | number[];
  lastAward?: { winner: number; points: number; reason: string; scores: number[]; matchPoints: number[] } | null;
  pendingLead: Card | null;
  lastTrick?: { leaderCard: Card; responderCard: Card; winner: number } | null;
  scores: [number, number];
  leader: number;
  trickNo: number;
  terminal: boolean;
  canExchangeTrumpJack: boolean;
  talon: {
    size: number;
    drawPileSize: number;
    closed: boolean;
    isClosedByTakaras?: boolean;
    trumpColor: Color;
    trumpUpcard: Card | null;
  };
  announcements?: { marriages: { player: number; suit: Color; points: number }[] };
  available?: { canCloseTalon: boolean; marriages: { suit: Color; points: number }[] };
  hands: {
    human: Card[];
  };
  captured: {
    human: Card[];
    ai: Card[];
  };
};

export async function apiNewGame(modelLabel: string, seed?: number | null): Promise<GameState> {
  const body: Record<string, unknown> = { modelLabel };
  if (seed !== undefined && seed !== null) body.seed = seed;
  const res = await fetch("/api/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiNewDeal(gameId: string, seed?: number | null): Promise<GameState> {
  const body: Record<string, unknown> = { gameId };
  if (seed !== undefined && seed !== null) body.seed = seed;
  const res = await fetch("/api/new_deal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiGetState(gameId: string): Promise<GameState> {
  const res = await fetch(`/api/state/${encodeURIComponent(gameId)}`);
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiContinue(gameId: string): Promise<GameState> {
  const res = await fetch("/api/continue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gameId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiActionPlay(gameId: string, card: Card): Promise<GameState> {
  const res = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gameId, type: "play_card", card }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiActionExchangeTrumpJack(gameId: string): Promise<GameState> {
  const res = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gameId, type: "exchange_trump_jack" }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiActionCloseTalon(gameId: string): Promise<GameState> {
  const res = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gameId, type: "close_talon" }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiActionDeclareMarriage(gameId: string, suit: Color): Promise<GameState> {
  const res = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gameId, type: "declare_marriage", suit }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GameState;
}

export async function apiListModels(): Promise<string[]> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as string[];
}

