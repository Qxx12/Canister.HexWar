"""Tests for combat resolution — mirrors JS combat.test.ts expectations."""

from hexwar.engine.combat import apply_combat_result, resolve_combat
from hexwar.engine.types import AxialCoord, Board, Tile


def _tile(owner, units, q=0, r=0, is_start=False, start_owner=None) -> Tile:
    return Tile(
        coord=AxialCoord(q, r),
        owner=owner,
        units=units,
        is_start_tile=is_start,
        start_owner=start_owner,
        terrain="plains",
        newly_conquered=False,
    )


def _board(**kwargs) -> Board:
    """Build a minimal board: keyword args map key→Tile."""
    return dict(kwargs)


class TestResolveCombat:
    def test_neutral_tile_conquest(self):
        """Moving into a neutral tile: no casualties, conquered=True."""
        board = _board(
            src=_tile("p1", 5, q=0, r=0),
            dst=_tile(None, 0, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 3, "p1")
        assert result.conquered is True
        assert result.attacker_casualties == 0
        assert result.defender_casualties == 0
        assert result.units_sent == 3
        assert result.remaining_attackers == 3

    def test_friendly_tile_stack(self):
        """Moving into own tile: no casualties, conquered=False, units stack."""
        board = _board(
            src=_tile("p1", 5, q=0, r=0),
            dst=_tile("p1", 3, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 2, "p1")
        assert result.conquered is False
        assert result.attacker_casualties == 0
        assert result.units_sent == 2
        assert result.remaining_attackers == 2

    def test_hostile_win(self):
        """Attacker outnumbers defender: conquest."""
        board = _board(
            src=_tile("p1", 5, q=0, r=0),
            dst=_tile("p2", 2, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 5, "p1")
        assert result.conquered is True
        assert result.attacker_casualties == 2
        assert result.defender_casualties == 2
        assert result.remaining_attackers == 3

    def test_hostile_tie(self):
        """Equal units: no conquest, all attacker units lost."""
        board = _board(
            src=_tile("p1", 3, q=0, r=0),
            dst=_tile("p2", 3, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 3, "p1")
        assert result.conquered is False
        assert result.attacker_casualties == 3
        assert result.defender_casualties == 3
        assert result.remaining_attackers == 0

    def test_hostile_loss(self):
        """Attacker fewer units than defender: no conquest."""
        board = _board(
            src=_tile("p1", 2, q=0, r=0),
            dst=_tile("p2", 5, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 2, "p1")
        assert result.conquered is False
        assert result.attacker_casualties == 2
        assert result.defender_casualties == 2
        assert result.remaining_attackers == 0

    def test_clamping(self):
        """requested_units capped at from_tile.units."""
        board = _board(
            src=_tile("p1", 3, q=0, r=0),
            dst=_tile("p2", 1, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 99, "p1")
        assert result.units_sent == 3
        assert result.was_clamped is True


class TestApplyCombatResult:
    def test_neutral_conquest_mutates(self):
        """After neutral conquest: dest owned by attacker, units = sent."""
        board = _board(
            src=_tile("p1", 5, q=0, r=0),
            dst=_tile(None, 0, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 3, "p1")
        apply_combat_result(board, result)
        assert board["src"].units == 2      # 5 - 3
        assert board["dst"].owner == "p1"
        assert board["dst"].units == 3
        assert board["dst"].newly_conquered is True

    def test_friendly_stack(self):
        """Moving onto own tile: units stack, no ownership change."""
        board = _board(
            src=_tile("p1", 5, q=0, r=0),
            dst=_tile("p1", 3, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 2, "p1")
        apply_combat_result(board, result)
        assert board["src"].units == 3      # 5 - 2
        assert board["dst"].units == 5      # 3 + 2
        assert board["dst"].owner == "p1"

    def test_hostile_conquest(self):
        """Successful attack: dest changes owner, has remaining_attackers units."""
        board = _board(
            src=_tile("p1", 5, q=0, r=0),
            dst=_tile("p2", 2, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 5, "p1")
        apply_combat_result(board, result)
        assert board["src"].units == 0      # 5 - 5
        assert board["dst"].owner == "p1"
        assert board["dst"].units == 3      # 5 - 2 casualties
        assert board["dst"].newly_conquered is True

    def test_hostile_failed_attack(self):
        """Failed attack: dest loses defender_casualties, owner unchanged."""
        board = _board(
            src=_tile("p1", 2, q=0, r=0),
            dst=_tile("p2", 5, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 2, "p1")
        apply_combat_result(board, result)
        assert board["src"].units == 0      # 2 - 2 sent
        assert board["dst"].owner == "p2"   # unchanged
        assert board["dst"].units == 3      # 5 - 2 defender casualties

    def test_unit_zero_not_sent(self):
        """If units_sent == 0 after clamping, nothing changes."""
        board = _board(
            src=_tile("p1", 0, q=0, r=0),
            dst=_tile("p2", 3, q=1, r=0),
        )
        result = resolve_combat(board, "src", "dst", 0, "p1")
        # units_sent = min(0, 0) = 0 — should be caught by caller
        # We verify the result is sane
        assert result.units_sent == 0
        assert result.conquered is False
