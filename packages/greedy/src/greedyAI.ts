import type { Board } from '@hexwar/engine'
import type { PlayerId, Player } from '@hexwar/engine'
import type { OrderMap } from '@hexwar/engine'
import type { AIStrategy } from './aiStrategy.ts'
import { hexNeighbors, hexToKey } from '@hexwar/engine'
import { scoreTarget, unitsToSend } from './scoring.ts'

export class GreedyAI implements AIStrategy {
  computeOrders(
    board: Board,
    playerId: PlayerId,
    _currentOrders: OrderMap,
    _allPlayers: Player[],
  ): OrderMap {
    const orders: OrderMap = new Map()

    // --- Step 1: Classify owned tiles into frontier vs interior ---
    const ownedKeys = new Set<string>()
    const frontierKeys = new Set<string>()

    for (const [key, tile] of board) {
      if (tile.owner !== playerId) continue
      ownedKeys.add(key)

      for (const n of hexNeighbors(tile.coord)) {
        const nTile = board.get(hexToKey(n))
        if (nTile && nTile.owner !== playerId) {
          frontierKeys.add(key)
          break
        }
      }
    }

    // --- Step 2: BFS from frontier tiles inward to build routing map ---
    // nextTowardFront[interiorKey] = the neighbor one step closer to the front
    const nextTowardFront = new Map<string, string>()
    const visited = new Set<string>(frontierKeys)
    const queue: string[] = [...frontierKeys]

    let qi = 0
    while (qi < queue.length) {
      const key = queue[qi++]
      const tile = board.get(key)!
      for (const n of hexNeighbors(tile.coord)) {
        const nKey = hexToKey(n)
        if (!ownedKeys.has(nKey) || visited.has(nKey)) continue
        visited.add(nKey)
        nextTowardFront.set(nKey, key)
        queue.push(nKey)
      }
    }

    // --- Step 3: Assign orders ---
    for (const key of ownedKeys) {
      const tile = board.get(key)!
      if (tile.units === 0) continue

      if (frontierKeys.has(key)) {
        // Front tile — evaluate and pick best attack target
        let bestKey: string | null = null
        let bestScore = 0 // only attack if score > 0

        for (const n of hexNeighbors(tile.coord)) {
          const nKey = hexToKey(n)
          const nTile = board.get(nKey)
          if (!nTile || nTile.owner === playerId) continue

          const score = scoreTarget(tile.units, nTile)
          if (score > bestScore) {
            bestScore = score
            bestKey = nKey
          }
        }

        if (bestKey !== null) {
          const target = board.get(bestKey)!
          orders.set(key, {
            fromKey: key,
            toKey: bestKey,
            requestedUnits: unitsToSend(tile.units, target),
          })
        }
      } else {
        // Interior tile — pipeline units toward the nearest front
        const next = nextTowardFront.get(key)
        if (next) {
          orders.set(key, {
            fromKey: key,
            toKey: next,
            requestedUnits: tile.units, // safe to send all through friendly territory
          })
        }
      }
    }

    return orders
  }
}
