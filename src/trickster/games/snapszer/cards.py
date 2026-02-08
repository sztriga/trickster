from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Color(str, Enum):
    # Hungarian playing-card suits
    HEARTS = "HEARTS"   # Piros
    BELLS = "BELLS"     # Tök
    LEAVES = "LEAVES"   # Zöld
    ACORNS = "ACORNS"   # Makk


ALL_COLORS: tuple[Color, ...] = (Color.HEARTS, Color.BELLS, Color.LEAVES, Color.ACORNS)

# Deck configuration (Schnapsen / Hungarian "snapszer")
#
# Ranks are represented by their *point values* (which also preserve trick order):
# - 11 (Ace), 10 (Ten), 4 (King), 3 (Ober/Queen), 2 (Unter/Jack)
RANK_VALUES: tuple[int, ...] = (2, 3, 4, 10, 11)
MAX_RANK: int = 11
HAND_SIZE: int = 5
CARDS_PER_COLOR: int = len(RANK_VALUES)
TRICKS_PER_GAME: int = (len(ALL_COLORS) * CARDS_PER_COLOR) // 2  # 20 cards => 10 tricks


@dataclass(frozen=True, slots=True)
class Card:
    color: Color
    number: int  # one of RANK_VALUES

    def short(self) -> str:
        return f"{self.color.value[0]}{self.number}"

    def points(self) -> int:
        # In this simplified variant, points are encoded directly in `number`.
        return int(self.number)


def make_deck() -> List[Card]:
    return [Card(c, n) for c in ALL_COLORS for n in RANK_VALUES]



