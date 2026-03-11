"""
Deterministic combat resolution — faithful Python port of src/engine/combat.ts.

The HexWar combat model:

  Friendly or neutral tile (defender is None or same player):
    No casualties. Attacker simply moves in and units stack.
    conquered = (defender is None)  [neutral tile gets claimed]

  Hostile tile (different owner):
    casualties         = min(units_sent, defending_units)
    remaining_attackers = units_sent - casualties
    remaining_defenders = defending_units - casualties
    conquered          = remaining_attackers > 0 AND remaining_defenders == 0

No dice, no randomness. The attacker wins iff they outnumber the defender
(strictly more units than the defender has).
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Board

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass()
class CombatResult:
    from_key: str
    to_key: str
    units_sent: int
    attacker_id: str
    defender_id: str | None         # None for neutral tiles
    defending_units_present: int
    attacker_casualties: int
    defender_casualties: int
    conquered: bool
    remaining_attackers: int
    was_clamped: bool               # True if requestedUnits > available units


# ---------------------------------------------------------------------------
# Core resolution (pure — no board mutation)
# ---------------------------------------------------------------------------

def resolve_combat(
    board: Board,
    from_key: str,
    to_key: str,
    requested_units: int,
    attacker_id: str,
) -> CombatResult:
    """
    Compute the combat outcome without mutating any tile.

    Mirrors JS resolveCombat(board, order, attackingPlayerId).

    The `requested_units` is capped at what the source tile actually has
    (snapshot-capped by the caller — see turn_resolver.py).

    Args:
        board:           Current board snapshot.
        from_key:        Source tile key.
        to_key:          Destination tile key.
        requested_units: Units the attacker wants to commit (may be clamped).
        attacker_id:     The attacking player's ID.

    Returns:
        CombatResult describing casualties and conquest outcome.
    """
    from_tile = board[from_key]
    to_tile = board[to_key]

    actual_units = from_tile.units
    was_clamped = actual_units < requested_units
    units_sent = min(requested_units, actual_units)

    defender_id = to_tile.owner
    defending_units = to_tile.units

    # Friendly or neutral tile — no combat, units stack/claim
    if defender_id is None or defender_id == attacker_id:
        return CombatResult(
            from_key=from_key,
            to_key=to_key,
            units_sent=units_sent,
            attacker_id=attacker_id,
            defender_id=defender_id,
            defending_units_present=defending_units,
            attacker_casualties=0,
            defender_casualties=0,
            conquered=(defender_id != attacker_id),  # neutral → conquered
            remaining_attackers=units_sent,
            was_clamped=was_clamped,
        )

    # Hostile tile — mutual combat
    casualties = min(units_sent, defending_units)
    remaining_attackers = units_sent - casualties
    remaining_defenders = defending_units - casualties
    conquered = remaining_attackers > 0 and remaining_defenders == 0

    return CombatResult(
        from_key=from_key,
        to_key=to_key,
        units_sent=units_sent,
        attacker_id=attacker_id,
        defender_id=defender_id,
        defending_units_present=defending_units,
        attacker_casualties=casualties,
        defender_casualties=casualties,
        conquered=conquered,
        remaining_attackers=remaining_attackers,
        was_clamped=was_clamped,
    )


# ---------------------------------------------------------------------------
# Board mutation
# ---------------------------------------------------------------------------

def apply_combat_result(board: Board, result: CombatResult) -> None:
    """
    Apply a CombatResult to the board in-place.

    Mirrors JS applyCombatResult() logic:
      - Always deducts unitsSent from source tile.
      - On conquest: attacker takes tile with remainingAttackers.
      - On friendly/neutral non-conquest: units stack on dest tile.
      - On failed attack: only defender casualties are applied.

    Args:
        board:  The board dict (mutated in-place).
        result: CombatResult from resolve_combat().
    """
    from_tile = board[result.from_key]
    to_tile = board[result.to_key]

    # Always deduct sent units from source
    from_tile.units -= result.units_sent

    if result.conquered:
        # Attacker takes the tile
        to_tile.owner = result.attacker_id
        to_tile.units = result.remaining_attackers
        to_tile.newly_conquered = True
    elif result.defender_id is None or result.defender_id == result.attacker_id:
        # Move to friendly/neutral tile — stack units, take ownership if neutral
        to_tile.owner = result.attacker_id
        to_tile.units = to_tile.units + result.remaining_attackers
        # Don't reset newly_conquered — may have been conquered earlier this turn
    else:
        # Failed attack — apply defender casualties
        to_tile.units -= result.defender_casualties
