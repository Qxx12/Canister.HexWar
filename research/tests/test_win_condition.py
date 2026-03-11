"""Tests for win condition and elimination checking."""

from hexwar.engine.types import AxialCoord, Player, PlayerType, Tile
from hexwar.engine.win_condition import check_eliminations, check_win


def _tile(owner, is_start=False, start_owner=None, q=0, r=0):
    return Tile(
        coord=AxialCoord(q, r),
        owner=owner,
        units=1,
        is_start_tile=is_start,
        start_owner=start_owner or owner,
        terrain="plains",
    )


def _player(pid, eliminated=False):
    return Player(id=pid, name=pid, color="#fff", type=PlayerType.AI, is_eliminated=eliminated)


class TestCheckWin:
    def test_no_winner_initially(self):
        board = {
            "0,0": _tile("p1", is_start=True, start_owner="p1"),
            "1,0": _tile("p2", is_start=True, start_owner="p2"),
        }
        players = [_player("p1"), _player("p2")]
        assert check_win(board, players) is None

    def test_all_start_tiles_owned(self):
        """p1 owns all start tiles → p1 wins."""
        board = {
            "0,0": _tile("p1", is_start=True, start_owner="p1"),
            "1,0": _tile("p1", is_start=True, start_owner="p2"),
        }
        players = [_player("p1"), _player("p2")]
        assert check_win(board, players) == "p1"

    def test_must_own_own_start_tile(self):
        """Player owns all start tiles except their own → no win."""
        board = {
            "0,0": _tile("p2", is_start=True, start_owner="p1"),  # p1's start owned by p2
            "1,0": _tile("p2", is_start=True, start_owner="p2"),
        }
        players = [_player("p1"), _player("p2")]
        # p2 owns all starts (including p1's), so p2 wins
        assert check_win(board, players) == "p2"

    def test_eliminated_player_ignored(self):
        board = {
            "0,0": _tile("p1", is_start=True, start_owner="p1"),
            "1,0": _tile("p1", is_start=True, start_owner="p2"),
        }
        players = [_player("p1"), _player("p2", eliminated=True)]
        assert check_win(board, players) == "p1"


class TestCheckEliminations:
    def test_player_with_no_tiles_eliminated(self):
        board = {
            "0,0": _tile("p1"),
        }
        players = [_player("p1"), _player("p2")]  # p2 has no tiles
        eliminated = check_eliminations(board, players)
        assert "p2" in eliminated
        assert "p1" not in eliminated

    def test_already_eliminated_not_re_eliminated(self):
        board = {"0,0": _tile("p1")}
        players = [_player("p1"), _player("p2", eliminated=True)]
        eliminated = check_eliminations(board, players)
        assert "p2" not in eliminated

    def test_no_eliminations(self):
        board = {
            "0,0": _tile("p1"),
            "1,0": _tile("p2"),
        }
        players = [_player("p1"), _player("p2")]
        assert check_eliminations(board, players) == []
