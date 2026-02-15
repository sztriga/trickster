"""Contract registry: maps bid ranks to contract models.

Adding a new contract:
  1. Train the play-phase model (e.g. scripts/train_20_100.py)
  2. Add an entry to CONTRACT_DEFS
  3. Add the bid ranks to BID_TO_CONTRACT
  4. That's it — bidding and training pick it up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractDef:
    """Definition of a playable contract type."""

    key: str              # unique identifier, e.g. "parti", "ulti"
    training_mode: str    # value passed to UltiGame.new_game()
    model_dir: str        # subdirectory under models/ (e.g. "parti")
    display_name: str     # human-readable, e.g. "Parti", "40-100"
    is_betli: bool = False  # betli has special rules (no trump, pick up talon)
    piros_only: bool = False  # can only be played as piros (Hearts trump)


# ---------------------------------------------------------------------------
#  All supported contracts
# ---------------------------------------------------------------------------

CONTRACT_DEFS: dict[str, ContractDef] = {
    "parti": ContractDef(
        key="parti",
        training_mode="simple",
        model_dir="parti",
        display_name="Parti",
        piros_only=True,  # standalone Parti can't be played; only Piros Parti
    ),
    "ulti": ContractDef(
        key="ulti",
        training_mode="ulti",
        model_dir="ulti",
        display_name="Ulti",
    ),
    "40-100": ContractDef(
        key="40-100",
        training_mode="40-100",
        model_dir="40-100",
        display_name="40-100",
    ),
    "betli": ContractDef(
        key="betli",
        training_mode="betli",
        model_dir="betli",
        display_name="Betli",
        is_betli=True,
    ),
    # Future:
    # "20-100": ContractDef(key="20-100", training_mode="20-100", ...),
    # "durchmars": ContractDef(key="durchmars", training_mode="durchmars", ...),
}


# ---------------------------------------------------------------------------
#  Bid rank → contract mapping
#
#  Each entry: bid_rank → (contract_key, is_piros)
#  Piros fixes trump to Hearts and doubles stakes.
#  Not every rank in the 38-bid table is listed — only those we can play.
# ---------------------------------------------------------------------------

BID_TO_CONTRACT: dict[int, tuple[str, bool]] = {
    1:  ("parti",  False),   # Passz
    2:  ("parti",  True),    # Piros passz
    3:  ("40-100", False),   # 40-100
    4:  ("ulti",   False),   # Ulti
    5:  ("betli",  False),   # Betli
    8:  ("40-100", True),    # Piros 40-100
    10: ("ulti",   True),    # Piros ulti
    11: ("betli",  True),    # Piros betli / Rebetli (10/10 pts)
}

# Sorted bid ranks we can play (ascending by strength)
SUPPORTED_BID_RANKS: list[int] = sorted(BID_TO_CONTRACT.keys())

# Max bid rank we can handle
MAX_SUPPORTED_RANK: int = max(SUPPORTED_BID_RANKS)
