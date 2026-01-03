import type { Board } from '../types/board'
import type { MovementOrder } from '../types/orders'
import type { CombatResult } from '../types/combat'
import type { PlayerId } from '../types/player'

export function resolveCombat(
  board: Board,
  order: MovementOrder,
  attackingPlayerId: PlayerId,
): CombatResult {
  const fromTile = board.get(order.fromKey)!
  const toTile = board.get(order.toKey)!

  const actualUnits = fromTile.units
  const wasClamped = actualUnits < order.requestedUnits
  const unitsSent = Math.min(order.requestedUnits, actualUnits)

  const defendingPlayerId = toTile.owner
  const defendingUnitsPresent = toTile.units

  // Friendly or unconquered tile - no combat
  if (defendingPlayerId === null || defendingPlayerId === attackingPlayerId) {
    return {
      fromKey: order.fromKey,
      toKey: order.toKey,
      attackingPlayerId,
      unitsSent,
      defendingPlayerId,
      defendingUnitsPresent,
      attackerCasualties: 0,
      defenderCasualties: 0,
      conquered: defendingPlayerId !== attackingPlayerId, // unconquered tile gets conquered
      remainingAttackers: unitsSent,
      wasClamped,
    }
  }

  // Hostile tile - combat
  const casualties = Math.min(unitsSent, defendingUnitsPresent)
  const attackerCasualties = casualties
  const defenderCasualties = casualties
  const remainingAttackers = unitsSent - attackerCasualties
  const remainingDefenders = defendingUnitsPresent - defenderCasualties
  const conquered = remainingAttackers > 0 && remainingDefenders === 0

  return {
    fromKey: order.fromKey,
    toKey: order.toKey,
    attackingPlayerId,
    unitsSent,
    defendingPlayerId,
    defendingUnitsPresent,
    attackerCasualties,
    defenderCasualties,
    conquered,
    remainingAttackers,
    wasClamped,
  }
}

export function applyCombatResult(board: Board, result: CombatResult): Board {
  const newBoard = new Map(board)
  const fromTile = { ...newBoard.get(result.fromKey)! }
  const toTile = { ...newBoard.get(result.toKey)! }

  // Remove sent units from source
  fromTile.units -= result.unitsSent

  if (result.conquered) {
    // Attacker takes the tile
    toTile.owner = result.attackingPlayerId
    toTile.units = result.remainingAttackers
  } else if (result.defendingPlayerId === null || result.defendingPlayerId === result.attackingPlayerId) {
    // Move to friendly/unconquered tile
    toTile.owner = result.attackingPlayerId
    toTile.units = toTile.units + result.remainingAttackers
  } else {
    // Failed attack - both sides take casualties
    toTile.units -= result.defenderCasualties
  }

  newBoard.set(result.fromKey, fromTile)
  newBoard.set(result.toKey, toTile)
  return newBoard
}
