import type { Card, Color } from "./api";

function suitCode(c: Color): string {
  switch (c) {
    case "HEARTS":
      return "H";
    case "BELLS":
      return "B";
    case "LEAVES":
      return "L";
    case "ACORNS":
      return "A";
  }
}

function rankCode(n: number): string {
  // Matches backend + asset naming:
  // 11 Ace(A), 10 Ten(10), 4 King(K), 3 Queen(Q), 2 Jack(J), else numeric.
  if (n === 11) return "A";
  if (n === 10) return "10";
  if (n === 4) return "K";
  if (n === 3) return "Q";
  if (n === 2) return "J";
  return String(n);
}

export function cardImageUrl(card: Card): string {
  return `/cards/card_piece_${suitCode(card.color)}${rankCode(card.number)}.jpg`;
}

export function cardLabel(card: Card): string {
  // Debug label (for alt text).
  return `${card.color[0]}-${card.number}`;
}

