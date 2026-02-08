"""Ulti auction (licitálás) — talon-passing bidding system.

Full contract table from the official rules:
https://hu.wikipedia.org/wiki/Rablóulti

Flow
----
1. First bidder picks up the 2-card talon → 12 cards in hand.
2. They discard 2 cards and announce a bid (at least Passz).
   Their discards become the new "talon" offered to the next player.
3. Each subsequent player can:
   - **Pick up** the talon → 12 cards → discard 2 → bid higher.
   - **Pass** → talon passes to next player.
4. When the turn returns to the holder (last bidder), they may
   pick up the talon again (must bid higher) or accept ("stand"),
   ending the auction.
5. The auction always produces a winner (the first bidder must bid
   at least Passz).

All-Pass Rule
-------------
If the winning bid is Passz and nobody challenged, the game is
skipped.  The first bidder pays 2 points to each defender.

Durchmars Rules
---------------
Standalone Durchmars is always colorless (Betli card ordering,
no trump).  Combined Durchmars (with 40-100, 20-100, or Ulti)
has a trump suit — the combined game's trump applies.
There is no "Piros Durchmars" (standalone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from trickster.games.ulti.cards import Card, Suit
from trickster.games.ulti.game import next_player, NUM_PLAYERS


# ---------------------------------------------------------------------------
#  Win-condition component identifiers
# ---------------------------------------------------------------------------

COMP_PARTI = "parti"          # Soloist total > defenders total
COMP_40 = "40"                # Declared 40 (trump K+Q)
COMP_20 = "20"                # Declared 20 (non-trump K+Q)
COMP_100 = "100"              # Total points >= 100
COMP_ULTI = "ulti"            # Won last trick with 7 of trumps
COMP_BETLI = "betli"          # Won 0 tricks
COMP_DURCHMARS = "durchmars"  # Won all 10 tricks


# ---------------------------------------------------------------------------
#  Bid definition (one row of the Wikipedia table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bid:
    """A valid Ulti bid — one row from the official bid table.

    ``rank`` determines the bidding hierarchy: a higher rank always
    beats a lower rank.  ``components`` lists the win-condition
    checks required for the soloist to win.
    """

    rank: int                      # 1 = weakest (Passz), 38 = strongest
    name: str                      # Hungarian display name
    win_value: int                 # Points soloist earns when winning
    loss_value: int                # Points soloist pays (to each defender) when losing
    trump_mode: str                # "choose" (3-féle) | "red" (Piros) | "none"
    components: frozenset[str]     # Win condition checks
    is_open: bool = False          # Terített — soloist shows cards

    def beats(self, other: Bid) -> bool:
        """Does this bid strictly beat *other* in the hierarchy?"""
        return self.rank > other.rank

    def label(self) -> str:
        return self.name

    @property
    def value(self) -> int:
        """Win value (for backward compat / display)."""
        return self.win_value

    @property
    def is_betli(self) -> bool:
        return COMP_BETLI in self.components

    @property
    def is_red(self) -> bool:
        return self.trump_mode == "red"

    @property
    def is_colorless(self) -> bool:
        """No-trump game (Betli/standalone Durchmars/Redurchmars)."""
        return self.trump_mode == "none"

    @property
    def has_ulti(self) -> bool:
        return COMP_ULTI in self.components

    @property
    def has_durchmars(self) -> bool:
        return COMP_DURCHMARS in self.components

    # ---- Display helpers (compound point breakdown) ----

    def _component_parts(self, loss: bool = False) -> list[int]:
        """Compute the display-value breakdown for compound bids.

        Component values:
          - 40-100:  win=4, loss=8  (× piros)
          - 20-100:  win=8, loss=16 (× piros)
          - Ulti:    win=4, loss=8  (× piros) — bukott ulti doubles
          - Durchmars: win=6, loss=12 (× piros × terített)
          - Parti:   win=1, loss=1  (× piros) — only when no 100-game
                     and no durchmars absorb it
          - Betli:   standalone (no breakdown)

        Order: base-game, ulti, durchmars, parti.
        """
        comps = self.components

        # Betli: always a single value, no breakdown
        if COMP_BETLI in comps:
            return [self.loss_value if loss else self.win_value]

        M = 2 if self.is_red else 1
        # Terített doubles the durchmars component (exception from the 4× rule)
        T = 2 if (self.is_open and COMP_DURCHMARS in comps) else 1
        parts: list[int] = []

        # 1. Base 100-game (includes/replaces parti)
        if COMP_100 in comps and COMP_20 in comps:
            parts.append((16 if loss else 8) * M)
        elif COMP_100 in comps and COMP_40 in comps:
            parts.append((8 if loss else 4) * M)

        # 2. Ulti addition (bukott = 2× on loss)
        if COMP_ULTI in comps:
            parts.append((8 if loss else 4) * M)

        # 3. Durchmars (standalone or combined addition)
        if COMP_DURCHMARS in comps:
            parts.append((12 if loss else 6) * M * T)

        # 4. Parti (only when no 100-game and no durchmars)
        if (COMP_PARTI in comps
                and COMP_100 not in comps
                and COMP_DURCHMARS not in comps):
            if COMP_ULTI in comps:
                # In compound (e.g. Ulti): parti = 1 on both win/loss
                parts.append(1 * M)
            else:
                # Standalone parti (Passz): loss = 2× win
                parts.append((2 if loss else 1) * M)

        return parts

    def display_win(self) -> str:
        """Formatted win value, e.g. ``'4+1'`` for Ulti."""
        parts = self._component_parts(loss=False)
        if len(parts) <= 1:
            return str(self.win_value)
        return "+".join(str(p) for p in parts)

    def display_loss(self) -> str:
        """Formatted loss value, e.g. ``'8+1'`` for Ulti."""
        parts = self._component_parts(loss=True)
        if len(parts) <= 1:
            return str(self.loss_value)
        return "+".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
#  Helper to define frozen component sets concisely
# ---------------------------------------------------------------------------

_P = frozenset  # shorthand


# ---------------------------------------------------------------------------
#  Complete bid table (38 entries)
#  Source: https://hu.wikipedia.org/wiki/Rablóulti
#
#  Standalone Durchmars is always colorless (no trump).
#  Combined Durchmars (with 40-100/20-100/Ulti) has trump.
#  No "Piros Durchmars" standalone.
#
#  Columns: rank, name, win, loss, trump_mode, components, is_open
# ---------------------------------------------------------------------------

ALL_BIDS: tuple[Bid, ...] = (
    #  1   Passz                         1 / 2     choose
    Bid(1, "Passz", 1, 2, "choose",
        _P({COMP_PARTI})),
    #  2   Piros passz                   2 / 4     red
    Bid(2, "Piros passz", 2, 4, "red",
        _P({COMP_PARTI})),
    #  3   40-100                        4 / 8     choose
    Bid(3, "40-100", 4, 8, "choose",
        _P({COMP_PARTI, COMP_40, COMP_100})),
    #  4   Ulti                          5 / 9     choose   (4+1 / 8+1)
    Bid(4, "Ulti", 5, 9, "choose",
        _P({COMP_PARTI, COMP_ULTI})),
    #  5   Betli                         5 / 5     none
    Bid(5, "Betli", 5, 5, "none",
        _P({COMP_BETLI})),
    #  6   Durchmars                     6 / 12    none  (always colorless)
    Bid(6, "Durchmars", 6, 12, "none",
        _P({COMP_DURCHMARS})),
    #  7   40-100 ulti                   8 / 16    choose
    Bid(7, "40-100 ulti", 8, 16, "choose",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_ULTI})),
    #  8   Piros 40-100                  8 / 16    red
    Bid(8, "Piros 40-100", 8, 16, "red",
        _P({COMP_PARTI, COMP_40, COMP_100})),
    #  9   20-100                        8 / 16    choose
    Bid(9, "20-100", 8, 16, "choose",
        _P({COMP_PARTI, COMP_20, COMP_100})),
    # 10   Piros ulti                    10 / 18   red      (8+2 / 16+2)
    Bid(10, "Piros ulti", 10, 18, "red",
        _P({COMP_PARTI, COMP_ULTI})),
    # 11   Piros betli / Rebetli         10 / 10   none (still colorless)
    Bid(11, "Piros betli", 10, 10, "none",
        _P({COMP_BETLI})),
    # 12   40-100 durchmars              10 / 20   choose (combined → has trump)
    Bid(12, "40-100 durchmars", 10, 20, "choose",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_DURCHMARS})),
    # 13   Ulti durchmars                10 / 20   choose (combined → has trump)
    Bid(13, "Ulti durchmars", 10, 20, "choose",
        _P({COMP_PARTI, COMP_ULTI, COMP_DURCHMARS})),
    # 14   20-100 ulti                   12 / 24   choose
    Bid(14, "20-100 ulti", 12, 24, "choose",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_ULTI})),
    # 15   Redurchmars                   12 / 24   none (colorless, like Rebetli)
    Bid(15, "Redurchmars", 12, 24, "none",
        _P({COMP_DURCHMARS})),
    # 16   Terített durchmars            12 / 24   none  (colorless, OPEN)
    Bid(16, "Terített durchmars", 12, 24, "none",
        _P({COMP_DURCHMARS}), is_open=True),
    # 17   40-100 ulti durchmars         14 / 28   choose
    Bid(17, "40-100 ulti durchmars", 14, 28, "choose",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_ULTI, COMP_DURCHMARS})),
    # 18   20-100 durchmars              14 / 28   choose
    Bid(18, "20-100 durchmars", 14, 28, "choose",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_DURCHMARS})),
    # 19   Piros 40-100 ulti             16 / 32   red
    Bid(19, "Piros 40-100 ulti", 16, 32, "red",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_ULTI})),
    # 20   Piros 20-100                  16 / 32   red
    Bid(20, "Piros 20-100", 16, 32, "red",
        _P({COMP_PARTI, COMP_20, COMP_100})),
    # 21   40-100 terített durchmars     16 / 32   choose   OPEN (combined → has trump)
    Bid(21, "40-100 terített durchmars", 16, 32, "choose",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_DURCHMARS}), is_open=True),
    # 22   Ulti terített durchmars       16 / 32   choose   OPEN (combined → has trump)
    Bid(22, "Ulti terített durchmars", 16, 32, "choose",
        _P({COMP_PARTI, COMP_ULTI, COMP_DURCHMARS}), is_open=True),
    # 23   20-100 ulti durchmars         18 / 36   choose
    Bid(23, "20-100 ulti durchmars", 18, 36, "choose",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_ULTI, COMP_DURCHMARS})),
    # 24   40-100 ulti terített durchmars 20 / 40  choose   OPEN
    Bid(24, "40-100 ulti terített durchmars", 20, 40, "choose",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_ULTI, COMP_DURCHMARS}), is_open=True),
    # 25   Piros 40-100 durchmars        20 / 40   red (combined → has trump)
    Bid(25, "Piros 40-100 durchmars", 20, 40, "red",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_DURCHMARS})),
    # 26   Piros ulti durchmars          20 / 40   red (combined → has trump)
    Bid(26, "Piros ulti durchmars", 20, 40, "red",
        _P({COMP_PARTI, COMP_ULTI, COMP_DURCHMARS})),
    # 27   20-100 terített durchmars     20 / 40   choose   OPEN
    Bid(27, "20-100 terített durchmars", 20, 40, "choose",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_DURCHMARS}), is_open=True),
    # 28   Terített betli                20 / 20   none     OPEN
    Bid(28, "Terített betli", 20, 20, "none",
        _P({COMP_BETLI}), is_open=True),
    # 29   20-100 ulti terített durchmars 24 / 48  choose   OPEN
    Bid(29, "20-100 ulti terített durchmars", 24, 48, "choose",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_ULTI, COMP_DURCHMARS}), is_open=True),
    # 30   Piros 20-100 ulti             24 / 48   red
    Bid(30, "Piros 20-100 ulti", 24, 48, "red",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_ULTI})),
    # 31   Piros 40-100 ulti durchmars   28 / 56   red
    Bid(31, "Piros 40-100 ulti durchmars", 28, 56, "red",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_ULTI, COMP_DURCHMARS})),
    # 32   Piros 20-100 durchmars        28 / 56   red
    Bid(32, "Piros 20-100 durchmars", 28, 56, "red",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_DURCHMARS})),
    # 33   Piros 40-100 terített durchmars  32/64  red      OPEN
    Bid(33, "Piros 40-100 terített durchmars", 32, 64, "red",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_DURCHMARS}), is_open=True),
    # 34   Piros ulti terített durchmars 32 / 64   red      OPEN
    Bid(34, "Piros ulti terített durchmars", 32, 64, "red",
        _P({COMP_PARTI, COMP_ULTI, COMP_DURCHMARS}), is_open=True),
    # 35   Piros 20-100 ulti durchmars   36 / 72   red
    Bid(35, "Piros 20-100 ulti durchmars", 36, 72, "red",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_ULTI, COMP_DURCHMARS})),
    # 36   Piros 40-100 ulti terített durchmars  40/80 red  OPEN
    Bid(36, "Piros 40-100 ulti terített durchmars", 40, 80, "red",
        _P({COMP_PARTI, COMP_40, COMP_100, COMP_ULTI, COMP_DURCHMARS}), is_open=True),
    # 37   Piros 20-100 terített durchmars  40/80  red      OPEN
    Bid(37, "Piros 20-100 terített durchmars", 40, 80, "red",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_DURCHMARS}), is_open=True),
    # 38   Piros 20-100 ulti terített durchmars  48/96 red  OPEN
    Bid(38, "Piros 20-100 ulti terített durchmars", 48, 96, "red",
        _P({COMP_PARTI, COMP_20, COMP_100, COMP_ULTI, COMP_DURCHMARS}), is_open=True),
)

# Quick lookups
BID_BY_RANK: dict[int, Bid] = {b.rank: b for b in ALL_BIDS}
BID_BY_NAME: dict[str, Bid] = {b.name: b for b in ALL_BIDS}

# The minimum bid (first bidder must bid at least this).
BID_PASSZ: Bid = ALL_BIDS[0]

# Suit display names.
SUIT_NAMES: dict[Suit, str] = {
    Suit.HEARTS: "Piros",
    Suit.BELLS: "Tök",
    Suit.LEAVES: "Zöld",
    Suit.ACORNS: "Makk",
}


# ---------------------------------------------------------------------------
#  Auction state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuctionState:
    """Mutable state of the talon-passing auction.

    ``awaiting_bid``
        When True the current-turn player has 12 cards and must
        discard 2 + announce a bid before the auction continues.
    ``talon``
        The 2 cards currently on the table, available for pickup.
    """

    num_players: int
    first_bidder: int
    talon: list[Card]               # 2 cards currently on offer

    current_bid: Bid | None = None
    holder: int | None = None       # who made the last bid
    turn: int = 0
    consecutive_passes: int = 0
    awaiting_bid: bool = True       # True → must discard + bid

    history: list[tuple[int, str, Optional[Bid]]] = field(default_factory=list)
    done: bool = False
    winner: int | None = None       # always set when done (no "fold")


def create_auction(first_bidder: int, talon: list[Card]) -> AuctionState:
    """Create a fresh auction.  The first bidder already has 12 cards."""
    return AuctionState(
        num_players=NUM_PLAYERS,
        first_bidder=first_bidder,
        talon=list(talon),
        turn=first_bidder,
        awaiting_bid=True,
    )


# ---------------------------------------------------------------------------
#  Legal actions
# ---------------------------------------------------------------------------


def legal_bids(auction: AuctionState) -> list[Bid]:
    """Bids that beat the current bid (or all bids if no bid yet)."""
    if auction.current_bid is None:
        return list(ALL_BIDS)
    return [b for b in ALL_BIDS if b.beats(auction.current_bid)]


def can_pickup(auction: AuctionState) -> bool:
    """Can the current-turn player pick up the talon?"""
    if auction.awaiting_bid or auction.done:
        return False
    return len(legal_bids(auction)) > 0


# ---------------------------------------------------------------------------
#  Placing actions
# ---------------------------------------------------------------------------


def submit_bid(
    auction: AuctionState,
    player: int,
    bid: Bid,
    discards: list[Card],
) -> None:
    """Player discards 2 cards and announces a bid.

    The discards become the new talon.  Must be called while
    ``awaiting_bid`` is True and it is *player*'s turn.
    """
    assert auction.awaiting_bid and auction.turn == player
    assert len(discards) == 2
    if auction.current_bid is not None:
        if not bid.beats(auction.current_bid):
            raise ValueError(
                f"Bid {bid.label()} does not beat {auction.current_bid.label()}"
            )

    auction.talon = list(discards)
    auction.current_bid = bid
    auction.holder = player
    auction.consecutive_passes = 0
    auction.awaiting_bid = False
    auction.history.append((player, "bid", bid))
    auction.turn = next_player(player)


def submit_pass(auction: AuctionState, player: int) -> None:
    """Player passes (does not pick up the talon).

    If the player IS the holder, this is interpreted as "accept" /
    "stand" — the auction ends and the holder wins.
    """
    assert not auction.awaiting_bid and auction.turn == player

    if player == auction.holder:
        # Holder accepts their bid — auction ends.
        auction.done = True
        auction.winner = auction.holder
        auction.history.append((player, "stand", None))
        return

    auction.consecutive_passes += 1
    auction.history.append((player, "pass", None))
    auction.turn = next_player(player)


def submit_pickup(auction: AuctionState, player: int) -> None:
    """Player picks up the talon.

    After this call ``awaiting_bid`` is True — the same player must
    call :func:`submit_bid` next.  The API layer is responsible for
    adding the talon cards to the player's hand.
    """
    assert not auction.awaiting_bid and auction.turn == player
    if not can_pickup(auction):
        raise ValueError("No higher bids available — cannot pick up")

    auction.awaiting_bid = True
    auction.history.append((player, "pickup", None))
    # Turn stays at the same player — they must bid next.


# ---------------------------------------------------------------------------
#  Contract scoring
# ---------------------------------------------------------------------------


def contract_value(bid: Bid, kontra_level: int = 0) -> int:
    """Win-side contract value after Kontra doubling."""
    return bid.win_value * (2 ** kontra_level)


def contract_loss_value(bid: Bid, kontra_level: int = 0) -> int:
    """Loss-side contract value after Kontra doubling."""
    return bid.loss_value * (2 ** kontra_level)


# ---------------------------------------------------------------------------
#  Marriage restriction helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Kontra helpers — per-component kontrable units and values
# ---------------------------------------------------------------------------


def kontrable_units(bid: Bid) -> list[str]:
    """Kontrable component labels for a bid.

    These correspond to the ``display_win`` breakdown parts.
    Defenders can kontra any of these individually.

    Example: Ulti (4+1) → ["ulti", "parti"]
             40-100 ulti (4+4) → ["40-100", "ulti"]
             Betli (5) → ["betli"]
             Durchmars standalone (6) → ["durchmars"]
    """
    comps = bid.components
    if COMP_BETLI in comps:
        return ["betli"]

    units: list[str] = []

    # Base game — 100-games absorb parti
    if COMP_100 in comps and COMP_20 in comps:
        units.append("20-100")
    elif COMP_100 in comps and COMP_40 in comps:
        units.append("40-100")
    elif COMP_PARTI in comps and COMP_DURCHMARS not in comps:
        units.append("parti")
    # standalone colorless durchmars: no parti

    if COMP_ULTI in comps:
        units.append("ulti")

    if COMP_DURCHMARS in comps:
        units.append("durchmars")

    return units


def component_value_map(bid: Bid) -> dict[str, tuple[int, int]]:
    """Map each kontrable unit to its ``(win_value, loss_value)``.

    These values match the breakdown from ``_component_parts``.
    """
    comps = bid.components
    M = 2 if bid.is_red else 1
    T = 2 if (bid.is_open and COMP_DURCHMARS in comps) else 1
    result: dict[str, tuple[int, int]] = {}

    if COMP_BETLI in comps:
        result["betli"] = (bid.win_value, bid.loss_value)
        return result

    # Base game
    if COMP_100 in comps and COMP_20 in comps:
        result["20-100"] = (8 * M, 16 * M)
    elif COMP_100 in comps and COMP_40 in comps:
        result["40-100"] = (4 * M, 8 * M)
    elif COMP_PARTI in comps and COMP_DURCHMARS not in comps:
        if COMP_ULTI in comps:
            # Compound: parti win=loss=1 (loss penalty comes from the other components)
            result["parti"] = (1 * M, 1 * M)
        else:
            # Standalone Passz: different loss value
            result["parti"] = (1 * M, 2 * M)

    if COMP_ULTI in comps:
        result["ulti"] = (4 * M, 8 * M)

    if COMP_DURCHMARS in comps:
        result["durchmars"] = (6 * M * T, 12 * M * T)

    return result


def marriage_restriction(bid: Bid) -> str | None:
    """Determine the soloist marriage restriction for a bid.

    Hungarian tournament rules:
      - 40-100 (and compounds): only the 40 marriage counts for the
        soloist; additional 20 declarations are not allowed.
      - 20-100 (and compounds): only one 20 marriage counts for the
        soloist; additional 40 or 20 declarations are not allowed.
      - All other contracts: no restriction.

    Returns ``"40"`` / ``"20"`` / ``None``.
    """
    comps = bid.components
    if COMP_40 in comps and COMP_100 in comps:
        return "40"
    if COMP_20 in comps and COMP_100 in comps:
        return "20"
    return None


# ---------------------------------------------------------------------------
#  AI heuristic
# ---------------------------------------------------------------------------


def ai_initial_bid(hand: list[Card]) -> tuple[Bid, list[Card]]:
    """AI's initial bid.  Always bids at least Passz.

    Returns (bid, discards_to_put_down).
    """
    # Discard 2 weakest cards.
    candidates = sorted(hand, key=lambda c: (c.points(), c.strength()))
    discards = candidates[:2]
    return BID_PASSZ, discards


def ai_should_pickup(hand: list[Card], auction: AuctionState) -> bool:
    """Should the AI pick up the talon?  (Conservative heuristic.)"""
    from trickster.games.ulti.cards import Rank

    aces = sum(1 for c in hand if c.rank == Rank.ACE)
    tens = sum(1 for c in hand if c.rank == Rank.TEN)
    strong = aces + tens
    return strong >= 5 and can_pickup(auction)


def ai_bid_after_pickup(hand: list[Card], auction: AuctionState) -> tuple[Bid, list[Card]]:
    """AI bids after picking up the talon.

    Returns (bid, discards).
    """
    candidates = sorted(hand, key=lambda c: (c.points(), c.strength()))
    discards = candidates[:2]
    bids = legal_bids(auction)
    if not bids:
        raise ValueError("No legal bids — should not have picked up")
    return bids[0], discards
