"""
Axial-coordinate hex math — mirrors src/types/hex.ts exactly.

All functions are pure and stateless. The key insight for this codebase is
that we use the same "q,r" string keys as the JS engine so that board
serialization and test fixtures are interchangeable between the two.

Coordinate system: flat-top hexagons in axial (q, r) coordinates.
The 6 neighbor directions are the standard axial offsets.
"""

from __future__ import annotations

import math

from .types import AxialCoord

# ---------------------------------------------------------------------------
# Key codec
# ---------------------------------------------------------------------------

def hex_to_key(coord: AxialCoord) -> str:
    """Convert axial coord to string key.  Matches JS hexToKey()."""
    return f"{coord.q},{coord.r}"


def key_to_hex(key: str) -> AxialCoord:
    """Parse a "q,r" key back to an AxialCoord."""
    q, r = key.split(",")
    return AxialCoord(int(q), int(r))


# ---------------------------------------------------------------------------
# Neighbor directions (same order as JS hexNeighbors)
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 0), (1, -1), (0, -1),
    (-1, 0), (-1, 1), (0, 1),
)


def hex_neighbors(coord: AxialCoord) -> list[AxialCoord]:
    """Return the 6 axial-coordinate neighbors of a hex tile."""
    return [
        AxialCoord(coord.q + dq, coord.r + dr)
        for dq, dr in _NEIGHBOR_OFFSETS
    ]


def hex_neighbor_keys(coord: AxialCoord) -> list[str]:
    """Return string keys of the 6 neighbors."""
    return [hex_to_key(n) for n in hex_neighbors(coord)]


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def hex_distance(a: AxialCoord, b: AxialCoord) -> int:
    """Axial-coordinate hex distance.  Mirrors JS hexDistance()."""
    dq = a.q - b.q
    dr = a.r - b.r
    return max(abs(dq), abs(dr), abs(dq + dr))


# ---------------------------------------------------------------------------
# Pixel conversion (flat-top hex, used for board generation placement)
# ---------------------------------------------------------------------------

def axial_to_pixel(coord: AxialCoord, hex_size: float) -> tuple[float, float]:
    """
    Convert axial coord to pixel (x, y) for flat-top hexagons.
    Mirrors JS axialToPixel().
    """
    x = hex_size * (3.0 / 2.0) * coord.q
    y = hex_size * (math.sqrt(3) * (coord.r + coord.q / 2.0))
    return x, y


def hex_corners(coord: AxialCoord, hex_size: float) -> list[tuple[float, float]]:
    """
    Return the 6 corner pixel positions for a flat-top hex.
    Mirrors JS hexCorners().
    """
    cx, cy = axial_to_pixel(coord, hex_size)
    corners = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.radians(angle_deg)
        corners.append((
            cx + hex_size * math.cos(angle_rad),
            cy + hex_size * math.sin(angle_rad),
        ))
    return corners
