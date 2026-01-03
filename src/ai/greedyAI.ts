import type { Board } from '../types/board'
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

    for (const [key, tile] of board) {
      if (tile.owner !== playerId || tile.units === 0) continue

      const neighbors = hexNeighbors(tile.coord)
      let bestKey: string | null = null
      let bestScore = -Infinity

      for (const neighbor of neighbors) {
        const nKey = hexToKey(neighbor)
        const nTile = board.get(nKey)
        if (!nTile) continue

        let score: number
        if (nTile.owner === playerId) {
          score = -1 // friendly - skip
        } else if (nTile.owner === null) {
          score = 30 // unconquered
        } else {
          // hostile
          if (nTile.isStartTile) {
            score = 100
          } else if (nTile.units < tile.units) {
            score = 50
          } else {
            score = 20
          }
        }

        if (score > bestScore) {
          bestScore = score
          bestKey = nKey
        }
      }

      if (bestKey !== null && bestScore > 0) {
        const unitsSent = Math.max(1, Math.floor(tile.units * 0.8))
        orders.set(key, {
          fromKey: key,
          toKey: bestKey,
          requestedUnits: unitsSent,
        })
      }
    }

    return orders
  }
}
