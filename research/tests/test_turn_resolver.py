"""Tests for turn resolution including snapshot cap and win/elimination checks."""

from hexwar.engine.turn_resolver import resolve_player_turn
from hexwar.engine.types import (
    AxialCoord,
    MovementOrder,
    Player,
    PlayerStats,
    PlayerType,
    Tile,
)


def _tile(owner, units, q=0, r=0, is_start=False, start_owner=None, newly_conquered=False):
    return Tile(
        coord=AxialCoord(q, r),
        owner=owner,
        units=units,
        is_start_tile=is_start,
        start_owner=start_owner or owner,
        terrain="plains",
        newly_conquered=newly_conquered,
    )


def _player(pid, eliminated=False):
    return Player(
        id=pid, name=pid, color="#fff",
        type=PlayerType.AI, is_eliminated=eliminated,
    )


def _stats(*pids):
    return {pid: PlayerStats(player_id=pid) for pid in pids}


def _order(from_key, to_key, units):
    return MovementOrder(from_key=from_key, to_key=to_key, requested_units=units)


class TestBasicMoves:
    def test_move_to_neutral(self):
        """Moving to a neutral adjacent tile conquers it."""
        # (0,0) → (1,0) are adjacent
        board = {
            "0,0": _tile("p1", 3, q=0, r=0),
            "1,0": _tile(None, 0, q=1, r=0),
        }
        players = [_player("p1"), _player("p2")]
        orders = {"0,0": _order("0,0", "1,0", 2)}
        stats = _stats("p1", "p2")

        result = resolve_player_turn(board, players, orders, "p1", stats)

        assert result.board["1,0"].owner == "p1"
        assert result.board["1,0"].units == 2
        assert result.board["0,0"].units == 1  # 3 - 2

    def test_non_adjacent_order_ignored(self):
        """Orders to non-adjacent tiles are silently ignored."""
        board = {
            "0,0": _tile("p1", 5, q=0, r=0),
            "3,0": _tile(None, 0, q=3, r=0),  # not adjacent to 0,0
        }
        players = [_player("p1")]
        orders = {"0,0": _order("0,0", "3,0", 3)}
        stats = _stats("p1")

        result = resolve_player_turn(board, players, orders, "p1", stats)

        # Nothing changed
        assert result.board["0,0"].units == 5
        assert result.board["3,0"].owner is None

    def test_not_owned_tile_ignored(self):
        """Orders from tiles not owned by the player are ignored."""
        board = {
            "0,0": _tile("p2", 5, q=0, r=0),  # owned by p2, not p1
            "1,0": _tile(None, 0, q=1, r=0),
        }
        players = [_player("p1"), _player("p2")]
        orders = {"0,0": _order("0,0", "1,0", 3)}
        stats = _stats("p1", "p2")

        result = resolve_player_turn(board, players, orders, "p1", stats)
        assert result.board["1,0"].owner is None


class TestSnapshotCap:
    def test_chaining_prevented(self):
        """
        Snapshot cap: units that arrive this turn cannot be used for further moves.

        Setup: p1 has tile A (3 units) → B (neutral) → C (neutral)
               A moves 3 to B, then tries to order B→C.
               B had 0 units at turn start, so B→C is capped to 0 and ignored.
        """
        # A=(0,0), B=(1,0), C=(2,0) — all adjacent in chain
        board = {
            "0,0": _tile("p1", 3, q=0, r=0),   # A
            "1,0": _tile(None, 0, q=1, r=0),    # B
            "2,0": _tile(None, 0, q=2, r=0),    # C
        }
        players = [_player("p1")]
        # Two orders: A→B and B→C (B had 0 units at start)
        orders = {
            "0,0": _order("0,0", "1,0", 3),
            "1,0": _order("1,0", "2,0", 99),  # will be capped to 0
        }
        stats = _stats("p1")

        result = resolve_player_turn(board, players, orders, "p1", stats)

        # A→B should succeed
        assert result.board["1,0"].owner == "p1"
        # B→C should NOT succeed (B had 0 units at snapshot)
        assert result.board["2,0"].owner is None


class TestWinCondition:
    def test_win_detected(self):
        """
        Conquering the last enemy start tile triggers win detection.
        """
        # p1 already owns all start tiles except p2's; conquering it wins.
        board = {
            "0,0": _tile("p1", 5, q=0, r=0, is_start=True, start_owner="p1"),
            "1,0": _tile("p2", 1, q=1, r=0, is_start=True, start_owner="p2"),
        }
        players = [_player("p1"), _player("p2")]
        orders = {"0,0": _order("0,0", "1,0", 4)}
        stats = _stats("p1", "p2")

        result = resolve_player_turn(board, players, orders, "p1", stats)

        assert result.winner_id == "p1"


class TestElimination:
    def test_player_eliminated_when_no_tiles(self):
        """A player with no tiles remaining is marked eliminated."""
        board = {
            "0,0": _tile("p1", 5, q=0, r=0),
            "1,0": _tile("p2", 1, q=1, r=0),  # p2's only tile
        }
        players = [_player("p1"), _player("p2")]
        orders = {"0,0": _order("0,0", "1,0", 4)}
        stats = _stats("p1", "p2")

        result = resolve_player_turn(board, players, orders, "p1", stats)

        p2 = next(p for p in result.players if p.id == "p2")
        assert p2.is_eliminated is True


class TestStats:
    def test_kills_tracked(self):
        board = {
            "0,0": _tile("p1", 5, q=0, r=0),
            "1,0": _tile("p2", 2, q=1, r=0),
        }
        players = [_player("p1"), _player("p2")]
        orders = {"0,0": _order("0,0", "1,0", 5)}
        stats = _stats("p1", "p2")

        result = resolve_player_turn(board, players, orders, "p1", stats)

        assert result.stats["p1"].units_killed == 2
        assert result.stats["p1"].tiles_captured == 1
        assert result.stats["p2"].tiles_lost == 1
