"""Tests for hex coordinate utilities."""

import pytest

from hexwar.engine.hex_utils import (
    axial_to_pixel,
    hex_distance,
    hex_neighbor_keys,
    hex_neighbors,
    hex_to_key,
    key_to_hex,
)
from hexwar.engine.types import AxialCoord


class TestKeyCodec:
    def test_roundtrip(self):
        coord = AxialCoord(3, -2)
        assert key_to_hex(hex_to_key(coord)) == coord

    def test_zero(self):
        assert hex_to_key(AxialCoord(0, 0)) == "0,0"

    def test_negative(self):
        assert hex_to_key(AxialCoord(-3, 5)) == "-3,5"

    def test_parse(self):
        assert key_to_hex("-1,2") == AxialCoord(-1, 2)


class TestNeighbors:
    def test_count(self):
        """Every hex has exactly 6 neighbors."""
        assert len(hex_neighbors(AxialCoord(0, 0))) == 6

    def test_symmetry(self):
        """If B is a neighbor of A, then A is a neighbor of B."""
        center = AxialCoord(2, -1)
        for nb in hex_neighbors(center):
            assert center in hex_neighbors(nb)

    def test_keys(self):
        keys = hex_neighbor_keys(AxialCoord(0, 0))
        assert len(keys) == 6
        assert all(isinstance(k, str) for k in keys)


class TestDistance:
    def test_zero(self):
        assert hex_distance(AxialCoord(0, 0), AxialCoord(0, 0)) == 0

    def test_adjacent(self):
        """All neighbors are at distance 1."""
        center = AxialCoord(0, 0)
        for nb in hex_neighbors(center):
            assert hex_distance(center, nb) == 1

    def test_known(self):
        """Known distance: (0,0) to (3,0) = 3."""
        assert hex_distance(AxialCoord(0, 0), AxialCoord(3, 0)) == 3

    def test_symmetry(self):
        a, b = AxialCoord(1, 2), AxialCoord(-3, 4)
        assert hex_distance(a, b) == hex_distance(b, a)

    def test_diagonal(self):
        """(0,0) to (2,-2): dq=2, dr=-2, dq+dr=0 → max(2,2,0) = 2."""
        assert hex_distance(AxialCoord(0, 0), AxialCoord(2, -2)) == 2


class TestPixelConversion:
    def test_center_is_origin(self):
        x, y = axial_to_pixel(AxialCoord(0, 0), hex_size=1.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_positive_q(self):
        """Moving in +q direction shifts x positively."""
        x0, _ = axial_to_pixel(AxialCoord(0, 0), 1.0)
        x1, _ = axial_to_pixel(AxialCoord(1, 0), 1.0)
        assert x1 > x0


class TestHexCorners:
    def test_six_corners(self):
        from hexwar.engine.hex_utils import hex_corners
        corners = hex_corners(AxialCoord(0, 0), hex_size=1.0)
        assert len(corners) == 6

    def test_corners_are_tuples(self):
        from hexwar.engine.hex_utils import hex_corners
        corners = hex_corners(AxialCoord(0, 0), hex_size=1.0)
        assert all(len(c) == 2 for c in corners)

    def test_corners_equidistant_from_center(self):
        """All 6 corners are the same distance from the tile center."""
        import math

        from hexwar.engine.hex_utils import hex_corners
        hex_size = 2.0
        cx, cy = axial_to_pixel(AxialCoord(0, 0), hex_size)
        corners = hex_corners(AxialCoord(0, 0), hex_size)
        distances = [math.hypot(x - cx, y - cy) for x, y in corners]
        assert all(d == pytest.approx(hex_size, rel=1e-6) for d in distances)

    def test_corners_shift_with_tile(self):
        """Corners of an offset tile are displaced from the origin tile."""
        from hexwar.engine.hex_utils import hex_corners
        c0 = hex_corners(AxialCoord(0, 0), 1.0)
        c1 = hex_corners(AxialCoord(2, 0), 1.0)
        assert c0[0] != c1[0]
