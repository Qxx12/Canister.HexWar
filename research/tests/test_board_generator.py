"""Tests for procedural board generation."""

from hexwar.engine.board_generator import _is_connected, generate_board
from hexwar.engine.types import PLAYER_IDS


class TestGenerateBoard:
    def test_returns_dict(self):
        board = generate_board(seed=42)
        assert isinstance(board, dict)
        assert len(board) > 0

    def test_all_tiles_have_coords(self):
        board = generate_board(seed=1)
        for key, tile in board.items():
            q, r = map(int, key.split(","))
            assert tile.coord.q == q
            assert tile.coord.r == r

    def test_start_tiles_assigned(self):
        board = generate_board(seed=42)
        start_tiles = [t for t in board.values() if t.is_start_tile]
        assert len(start_tiles) == len(PLAYER_IDS)

    def test_start_tiles_owned(self):
        board = generate_board(seed=42)
        for tile in board.values():
            if tile.is_start_tile:
                assert tile.owner is not None
                assert tile.owner == tile.start_owner
                assert tile.units == 1

    def test_non_start_tiles_neutral(self):
        board = generate_board(seed=42)
        for tile in board.values():
            if not tile.is_start_tile:
                assert tile.owner is None
                assert tile.units == 0

    def test_connectivity(self):
        """Board must be fully connected."""
        board = generate_board(seed=42)
        assert _is_connected(set(board.keys()))

    def test_reproducible(self):
        """Same seed → identical boards."""
        b1 = generate_board(seed=99)
        b2 = generate_board(seed=99)
        assert set(b1.keys()) == set(b2.keys())
        for k in b1:
            assert b1[k].owner == b2[k].owner
            assert b1[k].terrain == b2[k].terrain

    def test_different_seeds_differ(self):
        """Different seeds → different boards (with very high probability)."""
        b1 = generate_board(seed=1)
        b2 = generate_board(seed=2)
        # Keys may overlap but should not be identical
        # (statistically essentially guaranteed)
        assert set(b1.keys()) != set(b2.keys()) or any(
            b1[k].terrain != b2[k].terrain for k in set(b1.keys()) & set(b2.keys())
        )

    def test_terrain_values(self):
        valid = {"grassland", "plains", "desert", "tundra"}
        board = generate_board(seed=42)
        for tile in board.values():
            assert tile.terrain in valid

    def test_custom_player_ids(self):
        custom = ["a", "b"]
        board = generate_board(player_ids=custom, seed=0)
        start_tiles = [t for t in board.values() if t.is_start_tile]
        assert len(start_tiles) == 2
        owners = {t.owner for t in start_tiles}
        assert owners == {"a", "b"}
