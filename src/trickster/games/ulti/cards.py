"""Card definitions for Ulti (Hungarian 3-player trick-taking game).

32-card Tell deck with two ranking orders:
  - Normal (trump/colored): A > 10 > K > Q > J > 9 > 8 > 7
  - Betli  (non-colored):   A > K > Q > J > 10 > 9 > 8 > 7

Scoring: Ace = 10, Ten = 10, all others = 0.
Total card points = 80.  Last trick bonus = 10.  Grand total = 90.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List


# ---------------------------------------------------------------------------
#  Suits (Hungarian Tell deck)
# ---------------------------------------------------------------------------


class Suit(str, Enum):
    HEARTS = "HEARTS"   # Piros
    BELLS = "BELLS"     # Tök
    LEAVES = "LEAVES"   # Zöld
    ACORNS = "ACORNS"   # Makk


ALL_SUITS: tuple[Suit, ...] = tuple(Suit)


# ---------------------------------------------------------------------------
#  Ranks — IntEnum values encode *normal* (trump) trick strength
# ---------------------------------------------------------------------------


class Rank(IntEnum):
    SEVEN = 0
    EIGHT = 1
    NINE = 2
    JACK = 3      # Alsó / Under
    QUEEN = 4     # Felső / Over
    KING = 5      # Király
    TEN = 6       # note: stronger than K/Q/J in normal play
    ACE = 7       # Ász


ALL_RANKS: tuple[Rank, ...] = tuple(Rank)


# In Betli the Ten drops below the Jack.
BETLI_STRENGTH: dict[Rank, int] = {
    Rank.SEVEN: 0,
    Rank.EIGHT: 1,
    Rank.NINE: 2,
    Rank.TEN: 3,     # ← drops here
    Rank.JACK: 4,
    Rank.QUEEN: 5,
    Rank.KING: 6,
    Rank.ACE: 7,
}


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

NUM_PLAYERS: int = 3
HAND_SIZE: int = 10          # cards per player during the play phase
TALON_SIZE: int = 2          # cards set aside for the soloist
TRICKS_PER_GAME: int = 10
TOTAL_CARD_POINTS: int = 80  # 4 Aces × 10 + 4 Tens × 10
LAST_TRICK_BONUS: int = 10
TOTAL_POINTS: int = TOTAL_CARD_POINTS + LAST_TRICK_BONUS  # 90


# ---------------------------------------------------------------------------
#  Card
# ---------------------------------------------------------------------------

_RANK_SHORT: dict[Rank, str] = {
    Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
    Rank.TEN: "10", Rank.ACE: "A",
}

_SUIT_SHORT: dict[Suit, str] = {
    Suit.HEARTS: "H", Suit.BELLS: "B",
    Suit.LEAVES: "L", Suit.ACORNS: "A",
}


@dataclass(frozen=True, slots=True)
class Card:
    suit: Suit
    rank: Rank

    def points(self) -> int:
        """Card point value: A = 10, 10 = 10, everything else = 0."""
        return 10 if self.rank in (Rank.ACE, Rank.TEN) else 0

    def strength(self, betli: bool = False) -> int:
        """Trick strength (higher = stronger)."""
        return BETLI_STRENGTH[self.rank] if betli else int(self.rank)

    def short(self) -> str:
        """Human-readable short label, e.g. 'H10', 'LA'."""
        return f"{_SUIT_SHORT[self.suit]}{_RANK_SHORT[self.rank]}"

    def __repr__(self) -> str:
        return f"Card({self.short()})"


# ---------------------------------------------------------------------------
#  Deck
# ---------------------------------------------------------------------------


def make_deck() -> List[Card]:
    """Create the full 32-card Tell deck."""
    return [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
