import { describe, it, expect } from 'vitest'
import { GreedyAI } from '../ai/greedyAI'
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
  { id: 'p1', type: 'ai', color: '#f00', name: 'P1', isEliminated: false },
]

const ai = new GreedyAI()

describe('GreedyAI', () => {
  it('issues no order when no valid neighbors', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })

  it('prefers unconquered tile over friendly', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 3, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 0, r: 1 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    const order = orders.get('0,0')
    expect(order).toBeDefined()
    expect(order!.toKey).toBe('0,1') // unconquered preferred
  })

  it('prefers hostile start tile over unconquered', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 10, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 3, isStartTile: true, startOwner: 'p0', terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 0, r: 1 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    const order = orders.get('0,0')
    expect(order!.toKey).toBe('1,0') // hostile start tile preferred
  })

  it('sends all units to a neutral tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 10, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(10)
  })

  it('sends minimum 1 unit when tile has 1 unit', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 1, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(1)
  })

  it('routes interior tile units toward the nearest frontier', () => {
    // (0,0) interior p1 → (1,0) frontier p1 → (2,0) enemy p0
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 2, r: 0 }, owner: 'p0', units: 2, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    // Interior tile (0,0) should pipeline toward frontier (1,0)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
    // Frontier tile (1,0) should attack enemy (2,0)
    expect(orders.get('1,0')!.toKey).toBe('2,0')
  })

  it('sends computed units when attacking an enemy tile with advantage', () => {
    // 10 units vs enemy 3 → max(minToConquer=4, floor(10*0.85)=8) = 8
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 10, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 3, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(8)
  })

  it('commits all units in a desperate attack on an enemy start tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 3, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 5, isStartTile: true,  startOwner: 'p0', terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(3)
  })

  it('does not attack a stronger non-start enemy tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 3, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })

  it('does not issue order for tiles with 0 units', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })
})
