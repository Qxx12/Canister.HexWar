import type { Board, PlayerId, Player, OrderMap } from '@hexwar/engine'
import { scoreTarget, unitsToSend } from '@hexwar/greedy'
import type { TileConstraint } from '../types.ts'

const MIN_CAPITAL_GARRISON = 1

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

    // Use ceil so the budget fraction is a soft limit: a 3-unit tile at 0.75
    // gets maxUnits=3 (ceil(2.25)) rather than 2 (floor), avoiding missed
    // attacks against same-strength enemies. maxUnitsFraction=0 (DETER) → 0.
    const maxUnits = Math.ceil(tile.units * constraint.maxUnitsFraction)
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

/**
 * Ensures the player's own start tile always keeps a minimum garrison against
 * enemy attack. Skipped for neutral targets — they cannot counter-attack, so
 * there is no defensive reason to hold units back.
 */
function enforceCapitalGarrison(board: Board, playerId: PlayerId, orders: OrderMap): void {
  for (const [key, tile] of board) {
    if (tile.owner !== playerId || !tile.isStartTile || tile.startOwner !== playerId) continue

    const order = orders.get(key)
    if (!order) continue

    // Neutral tiles have 0 defenders — no garrison needed
    const targetTile = board.get(order.toKey)
    if (targetTile && targetTile.owner === null) continue

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

