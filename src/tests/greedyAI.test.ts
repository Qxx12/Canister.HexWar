import { describe, it, expect } from 'vitest'
import { GreedyAI } from '../ai/greedyAI'
import type { Board, Tile } from '../types/board'
import type { Player } from '../types/player'

function makeBoard(tiles: Tile[]): Board {
  const board: Board = new Map()
  for (const tile of tiles) {
    board.set(`${tile.coord.q},${tile.coord.r}`, tile)
  }
  return board
}

const players: Player[] = [
  { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
  { id: 'p1', type: 'ai', color: '#f00', name: 'P1', isEliminated: false },
]

const ai = new GreedyAI()

describe('GreedyAI', () => {
  it('issues no order when no valid neighbors', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })

  it('prefers unconquered tile over friendly', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 3, isStartTile: false, startOwner: null },
      { coord: { q: 0, r: 1 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    const order = orders.get('0,0')
    expect(order).toBeDefined()
    expect(order!.toKey).toBe('0,1') // unconquered preferred
  })

  it('prefers hostile start tile over unconquered', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 10, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 3, isStartTile: true, startOwner: 'p0' },
      { coord: { q: 0, r: 1 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    const order = orders.get('0,0')
    expect(order!.toKey).toBe('1,0') // hostile start tile preferred
  })

  it('sends 80% of units (floored)', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 10, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(8)
  })

  it('sends minimum 1 unit when tile has 1 unit', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 1, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(1)
  })

  it('does not issue order for tiles with 0 units', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 0, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })
})
