import { describe, it, expect } from 'vitest'
import { computeAiOrders } from '../ai/aiController'
import type { Board, Tile } from '@hexwar/engine'
import type { Player } from '@hexwar/engine'

function makeBoard(tiles: Tile[]): Board {
  const board: Board = new Map()
  for (const tile of tiles) {
    board.set(`${tile.coord.q},${tile.coord.r}`, tile)
  }
  return board
}

const players: Player[] = [
  { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
  { id: 'p1', type: 'ai',    color: '#f00', name: 'P1', isEliminated: false },
]

describe('computeAiOrders', () => {
  it('returns an OrderMap', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = computeAiOrders(board, 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })

  it('issues orders when there are valid moves', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = computeAiOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBeGreaterThan(0)
  })

  it('issues no orders when player has no tiles', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = computeAiOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })
})
