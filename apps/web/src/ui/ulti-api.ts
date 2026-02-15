/** Ulti API types and client functions. */

export type UltiCard = { suit: string; rank: string };

export type TrickEntry = {
  player: number;
  card: UltiCard;
};

export type TrickLog = {
  trick: number;
  result: {
    cards: UltiCard[];
    players: number[];
    winner: number;
  };
};

export type BidInfo = {
  rank: number;
  name: string;
  winValue: number;
  lossValue: number;
  displayWin: string;
  displayLoss: string;
  trumpMode: "choose" | "red" | "none";
  isOpen: boolean;
  label: string;
};

export type AuctionHistoryEntry = {
  player: number;
  action: "bid" | "pickup" | "pass" | "stand";
  bid?: BidInfo;
};

export type AuctionData = {
  turn: number;
  currentBid: BidInfo | null;
  holder: number | null;
  firstBidder: number;
  history: AuctionHistoryEntry[];
  done: boolean;
  winner: number | null;
  legalBids: BidInfo[];
  canPickup: boolean;
  isHolderTurn: boolean;
  awaitingBid: boolean;
};

export type KontraData = {
  phase: "kontra" | "rekontra";
  turn: number | null;
  isMyTurn: boolean;
  kontrable: string[];
  currentKontras: Record<string, number>;
  isColorless: boolean;
};

export type ContractInfo = {
  bid: BidInfo;
  componentKontras: Record<string, number>;
  displayWin: string;
  displayLoss: string;
};

export type SilentBonusEntry = {
  label: string;
  points: number;
};

export type Settlement = {
  contractResult: number;
  silentBonuses: SilentBonusEntry[];
  netPerDefender: number;
  soloistTotal: number;
  soloistWon: boolean;
};

export type UltiState = {
  gameId: string;
  phase: "bid" | "auction" | "trump_select" | "kontra" | "rekontra" | "play" | "done";
  hand: UltiCard[];
  aiHandSizes: [number, number];
  trickCards: TrickEntry[];
  scores: [number, number, number];
  currentPlayer: number | null;
  leader: number;
  trickNo: number;
  soloist: number;
  trump: string | null;
  betli: boolean;
  legalCards: UltiCard[];
  lastTrick: { cards: UltiCard[]; players: number[]; winner: number } | null;
  needsContinue: boolean;
  dealOver: boolean;
  log: TrickLog[];
  seed: number;
  dealer: number;
  soloistPoints: number;
  defenderPoints: number;
  auction: AuctionData | null;
  kontra: KontraData | null;
  contract: ContractInfo | null;
  trumpOptions: string[] | null;
  resultMessage: string | null;
  declaredMarriages: [number, number, number];
  isTeritett: boolean;
  soloistHand: UltiCard[] | null;
  settlement: Settlement | null;
  capturedTricks: UltiCard[][];
  bubbles: { player: number; text: string }[];
  opponents: [string, string];
  dealValue?: number;
  talonCards: UltiCard[] | null;
};

const BASE = "/api/ulti";

async function post(path: string, body: unknown = {}): Promise<UltiState> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as UltiState;
}

export function ultiNewGame(
  seed?: number | null,
  dealer?: number,
  opponents?: [string, string],
): Promise<UltiState> {
  const body: Record<string, unknown> = {};
  if (seed != null) body.seed = seed;
  if (dealer != null) body.dealer = dealer;
  if (opponents) body.opponents = opponents;
  return post("/new", body);
}

/** Submit discard + bid (during "bid" phase). */
export function ultiBid(
  gameId: string,
  discards: UltiCard[],
  bidRank: number,
): Promise<UltiState> {
  return post(`/${gameId}/bid`, { discards, bidRank });
}

/** Pick up or pass during "auction" phase. */
export function ultiAuction(
  gameId: string,
  action: "pickup" | "pass",
): Promise<UltiState> {
  return post(`/${gameId}/auction`, { action });
}

/** Choose trump suit (during "trump_select" phase). */
export function ultiTrump(gameId: string, suit: string): Promise<UltiState> {
  return post(`/${gameId}/trump`, { suit });
}

export function ultiKontra(
  gameId: string,
  action: "kontra" | "rekontra" | "pass",
  components?: string[],
): Promise<UltiState> {
  const body: Record<string, unknown> = { action };
  if (components) body.components = components;
  return post(`/${gameId}/kontra`, body);
}

export function ultiPlay(gameId: string, card: UltiCard): Promise<UltiState> {
  return post(`/${gameId}/play`, { card });
}

export function ultiContinue(gameId: string): Promise<UltiState> {
  return post(`/${gameId}/continue`);
}

/** Start a Parti practice game (training-style deal, no auction). */
export function ultiPartiNew(
  seed?: number | null,
  dealer?: number,
  opponents?: [string, string],
): Promise<UltiState> {
  const body: Record<string, unknown> = {};
  if (seed != null) body.seed = seed;
  if (dealer != null) body.dealer = dealer;
  if (opponents) body.opponents = opponents;
  return post("/parti/new", body);
}

/** List available model sources from the server. */
export async function ultiListModels(): Promise<string[]> {
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.models as string[];
}
