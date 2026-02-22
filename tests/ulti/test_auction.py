"""Tests for the Ulti auction (talon-passing bidding) system."""

from __future__ import annotations

import pytest

from trickster.games.ulti.auction import (
    ALL_BIDS,
    AuctionState,
    Bid,
    BID_BY_RANK,
    BID_PASSZ,
    COMP_BETLI,
    COMP_DURCHMARS,
    COMP_PARTI,
    COMP_ULTI,
    ai_bid_after_pickup,
    ai_initial_bid,
    ai_should_pickup,
    can_pickup,
    component_value_map,
    contract_loss_value,
    contract_value,
    create_auction,
    kontrable_units,
    legal_bids,
    marriage_restriction,
    submit_bid,
    submit_pass,
    submit_pickup,
)
from trickster.games.ulti.cards import Card, Rank, Suit, make_deck


# ---------------------------------------------------------------------------
#  Bid table integrity
# ---------------------------------------------------------------------------


class TestBidTable:
    def test_38_entries(self):
        assert len(ALL_BIDS) == 38

    def test_ranks_are_sequential(self):
        for i, bid in enumerate(ALL_BIDS):
            assert bid.rank == i + 1

    def test_passz_is_first(self):
        assert ALL_BIDS[0].name == "Passz"
        assert ALL_BIDS[0].rank == 1

    def test_bid_by_rank_complete(self):
        assert len(BID_BY_RANK) == 38

    def test_passz_values(self):
        p = BID_PASSZ
        assert p.win_value == 1
        assert p.loss_value == 2

    def test_ulti_values(self):
        ulti = BID_BY_RANK[4]
        assert ulti.name == "Ulti"
        assert ulti.win_value == 5   # 4+1
        assert ulti.loss_value == 9  # 8+1

    def test_betli_values(self):
        betli = BID_BY_RANK[5]
        assert betli.name == "Betli"
        assert betli.win_value == 5
        assert betli.loss_value == 5
        assert betli.is_betli

    def test_durchmars_values(self):
        d = BID_BY_RANK[6]
        assert d.name == "Durchmars"
        assert d.win_value == 6
        assert d.loss_value == 12

    def test_durchmars_is_colorless(self):
        """Standalone Durchmars is always colorless (Betli ordering)."""
        d = BID_BY_RANK[6]
        assert d.is_colorless
        assert d.trump_mode == "none"
        assert COMP_PARTI not in d.components

    def test_teritett_durchmars_is_colorless(self):
        """Terített durchmars is also colorless."""
        td = BID_BY_RANK[16]
        assert td.name == "Terített durchmars"
        assert td.is_colorless
        assert td.is_open

    def test_redurchmars_is_colorless(self):
        rd = BID_BY_RANK[15]
        assert rd.name == "Redurchmars"
        assert rd.is_colorless

    def test_combined_durchmars_has_trump(self):
        """Combined Durchmars (with 40-100, Ulti) keeps trump."""
        d40 = BID_BY_RANK[12]  # 40-100 durchmars
        assert d40.name == "40-100 durchmars"
        assert d40.trump_mode == "choose"
        assert COMP_PARTI in d40.components

        du = BID_BY_RANK[13]  # Ulti durchmars
        assert du.name == "Ulti durchmars"
        assert du.trump_mode == "choose"

    def test_no_piros_durchmars_standalone(self):
        """No standalone 'Piros durchmars' in the bid table."""
        for b in ALL_BIDS:
            if b.name == "Piros durchmars":
                pytest.fail(f"Found standalone Piros durchmars at rank {b.rank}")

    def test_no_szintelen_durchmars(self):
        """No separate 'Színtelen durchmars' entry."""
        for b in ALL_BIDS:
            assert "Színtelen" not in b.name, f"Found Színtelen entry: {b.name}"

    def test_piros_betli_is_colorless(self):
        """Piros betli is still colorless (name is traditional)."""
        pb = BID_BY_RANK[11]
        assert pb.name == "Piros betli"
        assert pb.is_colorless
        assert pb.is_betli

    def test_highest_bid(self):
        top = ALL_BIDS[-1]
        assert top.rank == 38
        assert top.win_value == 48
        assert top.loss_value == 96


# ---------------------------------------------------------------------------
#  Display values (compound point breakdown)
# ---------------------------------------------------------------------------


class TestDisplayValues:
    def test_passz_simple(self):
        """Simple bids show total, no breakdown."""
        assert BID_PASSZ.display_win() == "1"
        assert BID_PASSZ.display_loss() == "2"

    def test_ulti_compound(self):
        """Ulti: 4+1 win, 8+1 loss (bukott ulti doubles)."""
        ulti = BID_BY_RANK[4]
        assert ulti.display_win() == "4+1"
        assert ulti.display_loss() == "8+1"

    def test_piros_ulti_compound(self):
        """Piros ulti: 8+2 win, 16+2 loss."""
        pu = BID_BY_RANK[10]
        assert pu.display_win() == "8+2"
        assert pu.display_loss() == "16+2"

    def test_40_100_simple(self):
        """40-100 is a single display value (no sub-components)."""
        b = BID_BY_RANK[3]
        assert b.display_win() == "4"
        assert b.display_loss() == "8"

    def test_40_100_ulti(self):
        b = BID_BY_RANK[7]  # 40-100 ulti
        assert b.display_win() == "4+4"
        assert b.display_loss() == "8+8"

    def test_40_100_durchmars(self):
        b = BID_BY_RANK[12]  # 40-100 durchmars
        assert b.display_win() == "4+6"
        assert b.display_loss() == "8+12"

    def test_ulti_durchmars(self):
        b = BID_BY_RANK[13]  # Ulti durchmars
        assert b.display_win() == "4+6"
        assert b.display_loss() == "8+12"

    def test_40_100_ulti_durchmars(self):
        b = BID_BY_RANK[17]  # 40-100 ulti durchmars
        assert b.display_win() == "4+4+6"
        assert b.display_loss() == "8+8+12"

    def test_teritett_durchmars_simple(self):
        """Terített durchmars is single: 12/24."""
        td = BID_BY_RANK[16]
        assert td.display_win() == "12"
        assert td.display_loss() == "24"

    def test_40_100_teritett_durchmars(self):
        """Terített doubles the durchmars component only: 4+12 / 8+24."""
        b = BID_BY_RANK[21]
        assert b.display_win() == "4+12"
        assert b.display_loss() == "8+24"

    def test_betli_simple(self):
        assert BID_BY_RANK[5].display_win() == "5"
        assert BID_BY_RANK[5].display_loss() == "5"

    def test_durchmars_standalone(self):
        assert BID_BY_RANK[6].display_win() == "6"
        assert BID_BY_RANK[6].display_loss() == "12"

    def test_display_sum_matches_total(self):
        """For compound bids (>1 part), the parts must sum to the total.

        Single-part bids use str(total) directly so the parts value
        doesn't need to match (e.g. Redurchmars = 12 but base duri = 6).
        """
        for b in ALL_BIDS:
            win_parts = b._component_parts(loss=False)
            loss_parts = b._component_parts(loss=True)
            if len(win_parts) > 1:
                assert sum(win_parts) == b.win_value, (
                    f"{b.name}: win parts {win_parts} sum to {sum(win_parts)}, "
                    f"expected {b.win_value}"
                )
            if len(loss_parts) > 1:
                assert sum(loss_parts) == b.loss_value, (
                    f"{b.name}: loss parts {loss_parts} sum to {sum(loss_parts)}, "
                    f"expected {b.loss_value}"
                )


class TestBid:
    def test_beats_by_rank(self):
        low = BID_BY_RANK[1]   # Passz
        high = BID_BY_RANK[3]  # 40-100
        assert high.beats(low)
        assert not low.beats(high)

    def test_equal_does_not_beat(self):
        a = BID_BY_RANK[1]
        b = BID_BY_RANK[1]
        assert not a.beats(b)

    def test_label(self):
        assert BID_BY_RANK[1].label() == "Passz"
        assert BID_BY_RANK[10].label() == "Piros ulti"

    def test_is_betli(self):
        assert BID_BY_RANK[5].is_betli
        assert BID_BY_RANK[11].is_betli  # Piros betli
        assert BID_BY_RANK[28].is_betli  # Terített betli
        assert not BID_BY_RANK[1].is_betli

    def test_is_red(self):
        assert BID_BY_RANK[2].is_red   # Piros passz
        assert not BID_BY_RANK[1].is_red

    def test_is_colorless(self):
        assert BID_BY_RANK[5].is_colorless   # Betli
        assert BID_BY_RANK[6].is_colorless   # Durchmars
        assert BID_BY_RANK[15].is_colorless  # Redurchmars
        assert not BID_BY_RANK[1].is_colorless

    def test_is_open(self):
        assert BID_BY_RANK[16].is_open   # Terített durchmars
        assert BID_BY_RANK[28].is_open   # Terített betli
        assert not BID_BY_RANK[6].is_open

    def test_has_ulti(self):
        assert BID_BY_RANK[4].has_ulti   # Ulti
        assert BID_BY_RANK[7].has_ulti   # 40-100 ulti
        assert not BID_BY_RANK[1].has_ulti

    def test_components_parti_in_adu(self):
        """All non-betli, non-colorless bids have COMP_PARTI."""
        for b in ALL_BIDS:
            if not b.is_betli and not b.is_colorless:
                assert COMP_PARTI in b.components, f"{b.name} missing parti"


# ---------------------------------------------------------------------------
#  Marriage restriction
# ---------------------------------------------------------------------------
#  Kontra helpers
# ---------------------------------------------------------------------------


class TestKontrableUnits:
    """Test kontrable_units() returns correct per-component unit labels."""

    def test_passz(self):
        assert kontrable_units(BID_PASSZ) == ["parti"]

    def test_ulti(self):
        bid = BID_BY_RANK[4]
        units = kontrable_units(bid)
        assert "ulti" in units
        assert "parti" in units
        assert len(units) == 2

    def test_40_100(self):
        bid = BID_BY_RANK[3]
        units = kontrable_units(bid)
        assert units == ["40-100"]

    def test_40_100_ulti(self):
        bid = BID_BY_RANK[7]
        units = kontrable_units(bid)
        assert "40-100" in units
        assert "ulti" in units
        assert "parti" not in units  # absorbed into 40-100

    def test_betli(self):
        bid = BID_BY_RANK[5]
        assert kontrable_units(bid) == ["betli"]

    def test_standalone_durchmars(self):
        bid = BID_BY_RANK[6]
        units = kontrable_units(bid)
        assert units == ["durchmars"]
        assert "parti" not in units

    def test_ulti_durchmars(self):
        bid = BID_BY_RANK[13]  # Ulti durchmars
        units = kontrable_units(bid)
        assert "ulti" in units
        assert "durchmars" in units
        assert "parti" not in units  # absorbed by durchmars

    def test_40_100_ulti_durchmars(self):
        bid = BID_BY_RANK[17]  # 40-100 ulti durchmars
        units = kontrable_units(bid)
        assert "40-100" in units
        assert "ulti" in units
        assert "durchmars" in units
        assert "parti" not in units


class TestComponentValueMap:
    """Test component_value_map() returns correct (win, loss) per unit."""

    def test_passz(self):
        vm = component_value_map(BID_PASSZ)
        assert vm == {"parti": (1, 2)}

    def test_ulti(self):
        bid = BID_BY_RANK[4]
        vm = component_value_map(bid)
        assert vm["ulti"] == (4, 8)
        assert vm["parti"] == (1, 1)  # compound parti

    def test_piros_ulti(self):
        bid = BID_BY_RANK[10]  # Piros ulti
        vm = component_value_map(bid)
        assert vm["ulti"] == (8, 16)
        assert vm["parti"] == (2, 2)

    def test_40_100(self):
        bid = BID_BY_RANK[3]
        vm = component_value_map(bid)
        assert vm == {"40-100": (4, 4)}

    def test_betli(self):
        bid = BID_BY_RANK[5]
        vm = component_value_map(bid)
        assert vm == {"betli": (bid.win_value, bid.loss_value)}

    def test_value_sum_matches_win_for_compounds(self):
        """Sum of component win values should equal bid.win_value for compounds."""
        for bid in ALL_BIDS:
            vm = component_value_map(bid)
            if len(vm) > 1:
                total = sum(w for w, _ in vm.values())
                assert total == bid.win_value, f"{bid.label()}: sum {total} != {bid.win_value}"


# ---------------------------------------------------------------------------


class TestMarriageRestriction:
    def test_passz_no_restriction(self):
        assert marriage_restriction(BID_PASSZ) is None

    def test_40_100_restricts_to_40(self):
        bid = BID_BY_RANK[3]  # 40-100
        assert marriage_restriction(bid) == "40"

    def test_20_100_restricts_to_20(self):
        bid = BID_BY_RANK[9]  # 20-100
        assert marriage_restriction(bid) == "20"

    def test_ulti_no_restriction(self):
        bid = BID_BY_RANK[4]  # Ulti
        assert marriage_restriction(bid) is None

    def test_40_100_ulti_restricts_to_40(self):
        bid = BID_BY_RANK[7]  # 40-100 ulti
        assert marriage_restriction(bid) == "40"

    def test_20_100_ulti_restricts_to_20(self):
        bid = BID_BY_RANK[14]  # 20-100 ulti
        assert marriage_restriction(bid) == "20"


# ---------------------------------------------------------------------------
#  Bid generation
# ---------------------------------------------------------------------------


class TestBidGeneration:
    def test_legal_bids_no_current(self):
        a = create_auction(first_bidder=0, talon=[])
        bids = legal_bids(a)
        assert len(bids) == 38  # all possible

    def test_legal_bids_after_highest(self):
        a = create_auction(first_bidder=0, talon=[])
        a.current_bid = ALL_BIDS[-1]  # highest
        bids = legal_bids(a)
        assert len(bids) == 0

    def test_legal_bids_after_passz(self):
        a = create_auction(first_bidder=0, talon=[])
        a.current_bid = BID_PASSZ
        bids = legal_bids(a)
        for b in bids:
            assert b.beats(a.current_bid)
        assert len(bids) == 37  # all except Passz


# ---------------------------------------------------------------------------
#  Auction flow
# ---------------------------------------------------------------------------


def _dummy_talon() -> list[Card]:
    return [Card(Suit.HEARTS, Rank.SEVEN), Card(Suit.HEARTS, Rank.EIGHT)]


def _dummy_discards() -> list[Card]:
    return [Card(Suit.BELLS, Rank.SEVEN), Card(Suit.BELLS, Rank.EIGHT)]


class TestAuctionFlow:
    def test_create_auction(self):
        talon = _dummy_talon()
        a = create_auction(first_bidder=1, talon=talon)
        assert a.first_bidder == 1
        assert a.turn == 1
        assert a.awaiting_bid
        assert not a.done

    def test_first_bid_sets_holder(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        bid = BID_PASSZ
        discards = _dummy_discards()
        submit_bid(a, 1, bid, discards)

        assert a.holder == 1
        assert a.current_bid == bid
        assert a.talon == discards
        assert not a.awaiting_bid
        assert a.turn == 2  # next player

    def test_all_pass_holder_wins(self):
        """After first bid, both opponents pass, holder stands."""
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        submit_bid(a, 1, BID_PASSZ, _dummy_discards())
        submit_pass(a, 2)
        assert not a.done
        submit_pass(a, 0)
        assert not a.done
        submit_pass(a, 1)
        assert a.done
        assert a.winner == 1

    def test_overbid(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        submit_bid(a, 1, BID_PASSZ, _dummy_discards())
        submit_pickup(a, 2)
        assert a.awaiting_bid
        ulti = BID_BY_RANK[4]  # Ulti
        new_discards = [Card(Suit.ACORNS, Rank.SEVEN), Card(Suit.ACORNS, Rank.EIGHT)]
        submit_bid(a, 2, ulti, new_discards)
        assert a.holder == 2
        assert a.turn == 0

    def test_holder_can_rebid(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        submit_bid(a, 1, BID_PASSZ, _dummy_discards())
        submit_pass(a, 2)
        submit_pass(a, 0)
        submit_pickup(a, 1)
        assert a.awaiting_bid
        ulti = BID_BY_RANK[4]
        new_discards = [Card(Suit.LEAVES, Rank.SEVEN), Card(Suit.LEAVES, Rank.EIGHT)]
        submit_bid(a, 1, ulti, new_discards)
        assert a.holder == 1
        assert a.current_bid == ulti
        assert not a.done

    def test_illegal_low_bid_raises(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        ulti = BID_BY_RANK[4]
        submit_bid(a, 1, ulti, _dummy_discards())
        submit_pickup(a, 2)
        with pytest.raises((ValueError, AssertionError)):
            submit_bid(a, 2, BID_PASSZ, _dummy_discards())

    def test_cannot_pickup_with_no_higher_bids(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        submit_bid(a, 1, ALL_BIDS[-1], _dummy_discards())
        assert not can_pickup(a)

    def test_can_pickup(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        submit_bid(a, 1, BID_PASSZ, _dummy_discards())
        assert can_pickup(a)

    def test_consecutive_passes_reset_on_bid(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        submit_bid(a, 1, BID_PASSZ, _dummy_discards())
        submit_pass(a, 2)
        assert a.consecutive_passes == 1
        submit_pickup(a, 0)
        ulti = BID_BY_RANK[4]
        submit_bid(a, 0, ulti, _dummy_discards())
        assert a.consecutive_passes == 0


# ---------------------------------------------------------------------------
#  Contract value & scoring
# ---------------------------------------------------------------------------


class TestContractValue:
    def test_no_kontra(self):
        assert contract_value(BID_PASSZ, 0) == 1

    def test_kontra_doubles(self):
        assert contract_value(BID_PASSZ, 1) == 2

    def test_rekontra_quadruples(self):
        assert contract_value(BID_PASSZ, 2) == 4

    def test_loss_value(self):
        assert contract_loss_value(BID_PASSZ, 0) == 2
        assert contract_loss_value(BID_PASSZ, 1) == 4

    def test_ulti_win_loss_asymmetric(self):
        ulti = BID_BY_RANK[4]
        assert contract_value(ulti, 0) == 5
        assert contract_loss_value(ulti, 0) == 9


# ---------------------------------------------------------------------------
#  AI heuristic (basic)
# ---------------------------------------------------------------------------


class TestAIBidding:
    def test_ai_initial_bid_returns_passz(self):
        hand = make_deck()[:12]
        bid, discards = ai_initial_bid(hand)
        assert bid.rank == 1  # Passz
        assert len(discards) == 2

    def test_ai_should_not_pickup_weak_hand(self):
        a = create_auction(first_bidder=1, talon=_dummy_talon())
        submit_bid(a, 1, BID_PASSZ, _dummy_discards())
        weak_hand = [
            Card(Suit.ACORNS, Rank.SEVEN),
            Card(Suit.ACORNS, Rank.EIGHT),
            Card(Suit.ACORNS, Rank.NINE),
            Card(Suit.BELLS, Rank.SEVEN),
            Card(Suit.BELLS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.EIGHT),
            Card(Suit.LEAVES, Rank.SEVEN),
            Card(Suit.LEAVES, Rank.EIGHT),
            Card(Suit.LEAVES, Rank.NINE),
        ]
        assert not ai_should_pickup(weak_hand, a)
