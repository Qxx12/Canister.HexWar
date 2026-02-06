import type { Board, Tile } from '../types/board'
import type { PlayerId, Player } from '../types/player'
import type { OrderMap } from '../types/orders'
import type { AIStrategy } from './aiStrategy'
import { hexNeighbors, hexToKey } from '../types/hex'

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

function scoreTarget(myUnits: number, target: Tile): number {
  const startBonus = target.isStartTile ? 45 : 0

  if (target.owner === null) {
    // Neutral — free tile, always worth taking
    return 35 + startBonus
  }

  // Enemy tile
  const advantage = myUnits - target.units
  if (advantage <= 0) {
    // We'd tie or lose — only gamble on enemy start tiles
    return target.isStartTile ? 18 : -1
  }

  // We can win — prefer larger advantages and start tiles
  return 50 + advantage * 3 + startBonus
}

function unitsToSend(myUnits: number, target: Tile): number {
  if (target.owner === null) {
    // Neutral: send everything, no counter-attack risk
    return myUnits
  }

  if (myUnits <= target.units) {
    // Desperate attack on a start tile — commit everything
    return myUnits
  }

  // Send just enough to conquer with a small garrison, keep a buffer on source
  const minToConquer = target.units + 1
  const comfortable = Math.floor(myUnits * 0.85)
  return Math.min(myUnits, Math.max(minToConquer, comfortable))
}
