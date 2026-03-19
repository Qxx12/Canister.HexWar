import type { Board, PlayerId, Player, OrderMap, Tile } from '@hexwar/engine'
import { hexToKey } from '@hexwar/engine'
import type { TileConstraint } from '../types.ts'

const MIN_CAPITAL_GARRISON = 2

/**
 * Greedy tactical layer that operates within the operational constraints
 * produced by the High Command.
 *
 * For each source tile:
 *  - If crossBorderAllowed: score allowed enemy/neutral targets, pick best.
 *  - Otherwise: route units toward the nearest active front via interiorRoutes.
 *
 * After orders are built, enforces a minimum capital garrison.
 */
export function computeConstrainedOrders(
  board: Board,
  playerId: PlayerId,
  _players: Player[],
  constraints: Map<string, TileConstraint>,
  interiorRoutes: Map<string, string>,
): OrderMap {
  const orders: OrderMap = new Map()

  for (const [key, constraint] of constraints) {
    const tile = board.get(key)
    if (!tile || tile.units === 0) continue

    const maxUnits = Math.floor(tile.units * constraint.maxUnitsFraction)
    if (maxUnits === 0) continue

    if (constraint.crossBorderAllowed) {
      // Frontier tile — find best attack/expansion target
      let bestKey: string | null = null
      let bestScore = 0 // only act if score > 0

      for (const targetKey of constraint.allowedTargetKeys) {
        const targetTile = board.get(targetKey)
        if (!targetTile || targetTile.owner === playerId) continue

        const score = scoreTarget(maxUnits, targetTile)
        if (score > bestScore) {
          bestScore = score
          bestKey = targetKey
        }
      }

      if (bestKey !== null) {
        const target = board.get(bestKey)!
        orders.set(key, {
          fromKey: key,
          toKey: bestKey,
          requestedUnits: unitsToSend(maxUnits, target),
        })
      }
    } else {
      // Interior/hold tile — route toward nearest active front
      const next = interiorRoutes.get(key)
      if (next && constraint.allowedTargetKeys.includes(next)) {
        orders.set(key, {
          fromKey: key,
          toKey: next,
          requestedUnits: maxUnits,
        })
      }
    }
  }

  enforceCapitalGarrison(board, playerId, orders)
  return orders
}

function scoreTarget(myUnits: number, target: Tile): number {
  const startBonus = target.isStartTile ? 45 : 0

  if (target.owner === null) {
    return 35 + startBonus
  }

  const advantage = myUnits - target.units
  if (advantage <= 0) {
    return target.isStartTile ? 18 : -1
  }

  return 50 + advantage * 3 + startBonus
}

function unitsToSend(myUnits: number, target: Tile): number {
  if (target.owner === null) return myUnits
  if (myUnits <= target.units) return myUnits
  const minToConquer = target.units + 1
  return Math.min(myUnits, Math.max(minToConquer, Math.floor(myUnits * 0.85)))
}

/** Ensures the player's own start tile always keeps a minimum garrison. */
function enforceCapitalGarrison(board: Board, playerId: PlayerId, orders: OrderMap): void {
  for (const [key, tile] of board) {
    if (tile.owner !== playerId || !tile.isStartTile || tile.startOwner !== playerId) continue

    const order = orders.get(key)
    if (!order) continue

    const maxSend = tile.units - MIN_CAPITAL_GARRISON
    if (maxSend <= 0) {
      orders.delete(key)
      continue
    }

    const requested = order.requestedUnits === Infinity ? tile.units : order.requestedUnits
    if (requested > maxSend) {
      orders.set(key, { ...order, requestedUnits: maxSend })
    }
  }
}

// Re-export for consumers that need the key utility
export { hexToKey }
