/** Card image helpers for Ulti (rank names → file codes). */

import type { UltiCard } from "./ulti-api";

const RANK_CODE: Record<string, string> = {
  SEVEN: "7", EIGHT: "8", NINE: "9",
  JACK: "J", QUEEN: "Q", KING: "K",
  TEN: "10", ACE: "A",
};

const RANK_LABEL: Record<string, string> = {
  SEVEN: "7", EIGHT: "8", NINE: "9",
  JACK: "J", QUEEN: "Q", KING: "K",
  TEN: "10", ACE: "A",
};

export function ultiCardUrl(card: UltiCard): string {
  const s = card.suit[0]; // HEARTS→H, BELLS→B, LEAVES→L, ACORNS→A
  const r = RANK_CODE[card.rank] ?? card.rank;
  return `/cards/card_piece_${s}${r}.jpg`;
}

export function ultiCardBackUrl(): string {
  return "/cards/card_back.png";
}

export function ultiCardLabel(card: UltiCard): string {
  return `${card.suit[0]}-${RANK_LABEL[card.rank] ?? card.rank}`;
}
