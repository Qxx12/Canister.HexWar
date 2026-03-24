"""Tests for unit generation."""

from hexwar.engine.types import AxialCoord, PlayerStats, Tile
from hexwar.engine.unit_generator import generate_units


def _make_tile(
    q: int,
    r: int,
    owner: str | None,
    units: int,
    newly_conquered: bool = False,
) -> tuple[str, Tile]:
    key = f"{q},{r}"
    t = Tile(
        coord=AxialCoord(q, r),
        owner=owner,
        units=units,
        is_start_tile=False,
        start_owner=None,
        terrain="plains",
        newly_conquered=newly_conquered,
    )
    return key, t


def _stats(*player_ids: str) -> dict[str, PlayerStats]:
    return {pid: PlayerStats(player_id=pid) for pid in player_ids}


class TestGenerateUnits:
    def test_neutral_tile_unchanged(self):
        """Neutral tiles are skipped entirely."""
        key, tile = _make_tile(0, 0, None, 0)
        board = {key: tile}
        stats: dict = {}
        generate_units(board, stats)
        assert tile.units == 0

    def test_normal_owned_tile_gains_unit(self):
        """A non-newly-conquered owned tile gains exactly 1 unit."""
        key, tile = _make_tile(0, 0, "p1", 3)
        board = {key: tile}
        s = _stats("p1")
        generate_units(board, s)
        assert tile.units == 4

    def test_units_generated_stat_incremented(self):
        """units_generated stat is incremented for each normal tile."""
        tiles = {f"{i},0": _make_tile(i, 0, "p1", 1)[1] for i in range(3)}
        s = _stats("p1")
        generate_units(tiles, s)
        assert s["p1"].units_generated == 3

    def test_newly_conquered_clears_flag_no_unit(self):
        """Newly conquered tile: flag cleared, no unit added."""
        key, tile = _make_tile(0, 0, "p1", 5, newly_conquered=True)
        board = {key: tile}
        s = _stats("p1")
        generate_units(board, s)
        assert tile.units == 5            # unchanged
        assert tile.newly_conquered is False

    def test_newly_conquered_no_stat_increment(self):
        """units_generated stat NOT incremented for newly-conquered tiles."""
        key, tile = _make_tile(0, 0, "p1", 5, newly_conquered=True)
        board = {key: tile}
        s = _stats("p1")
        generate_units(board, s)
        assert s["p1"].units_generated == 0

    def test_multiple_players_independent(self):
        """Each player's tiles are processed independently."""
        board = {
            "0,0": _make_tile(0, 0, "p1", 2)[1],
            "1,0": _make_tile(1, 0, "p2", 4)[1],
        }
        s = _stats("p1", "p2")
        generate_units(board, s)
        assert board["0,0"].units == 3
        assert board["1,0"].units == 5
        assert s["p1"].units_generated == 1
        assert s["p2"].units_generated == 1

    def test_missing_stats_entry_is_safe(self):
        """If a player has no stats entry, generate_units does not crash."""
        key, tile = _make_tile(0, 0, "p_unknown", 1)
        board = {key: tile}
        generate_units(board, {})   # empty stats
        assert tile.units == 2      # unit still generated

    def test_mixed_board(self):
        """Board with neutral, normal, and newly-conquered tiles."""
        board = {
            "0,0": _make_tile(0, 0, None, 0)[1],              # neutral
            "1,0": _make_tile(1, 0, "p1", 3)[1],              # normal
            "2,0": _make_tile(2, 0, "p1", 7, True)[1],        # newly conquered
        }
        s = _stats("p1")
        generate_units(board, s)
        assert board["0,0"].units == 0          # neutral unchanged
        assert board["1,0"].units == 4          # +1
        assert board["2,0"].units == 7          # unchanged
        assert board["2,0"].newly_conquered is False
        assert s["p1"].units_generated == 1     # only the normal tile
