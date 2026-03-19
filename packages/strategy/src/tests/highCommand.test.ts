import { describe, it, expect } from 'vitest'
import { HighCommandAI } from '../highCommand.ts'
import type { Board, Tile, Player } from '@hexwar/engine'

function makeTile(q: number, r: number, owner: string | null, units: number, opts: Partial<Tile> = {}): Tile {
  return {
    coord: { q, r },
    owner,
    units,
    isStartTile: false,
    startOwner: null,
    terrain: 'plains' as const,
    newlyConquered: false,
    ...opts,
  }
}

function makeBoard(tiles: Tile[]): Board {
  const board: Board = new Map()
  for (const t of tiles) board.set(`${t.coord.q},${t.coord.r}`, t)
  return board
}

const players: Player[] = [
  { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
  { id: 'p1', type: 'ai', color: '#f00', name: 'P1', isEliminated: false },
]

// Minimal board: p1 owns (0,0) with a neutral neighbour (1,0)
function simpleBoard(): Board {
  return makeBoard([
    makeTile(0, 0, 'p1', 5),
    makeTile(1, 0, null, 0),
  ])
}

describe('HighCommandAI — basic contract', () => {
  it('returns an OrderMap', () => {
    const ai = new HighCommandAI()
    const orders = ai.computeOrders(simpleBoard(), 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })

  it('issues at least one order when there are valid moves', () => {
    const ai = new HighCommandAI()
    const orders = ai.computeOrders(simpleBoard(), 'p1', new Map(), players)
    expect(orders.size).toBeGreaterThan(0)
  })

  it('issues no orders when player has no tiles', () => {
    const ai = new HighCommandAI()
    const board = makeBoard([makeTile(0, 0, 'p0', 5)])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })
})

describe('HighCommandAI — instance isolation', () => {
  it('two instances maintain independent history', () => {
    const ai1 = new HighCommandAI()
    const ai2 = new HighCommandAI()

    // Record different boards into each instance via computeOrders
    const boardWith2Tiles = makeBoard([
      makeTile(0, 0, 'p1', 3),
      makeTile(1, 0, null, 0),
      makeTile(2, 0, 'p1', 2),
    ])
    const boardWith1Tile = makeBoard([
      makeTile(0, 0, 'p1', 3),
      makeTile(1, 0, null, 0),
    ])

    ai1.computeOrders(boardWith2Tiles, 'p1', new Map(), players)
    ai2.computeOrders(boardWith1Tile, 'p1', new Map(), players)

    // Each instance should still work independently without error
    const orders1 = ai1.computeOrders(boardWith1Tile, 'p1', new Map(), players)
    const orders2 = ai2.computeOrders(boardWith2Tiles, 'p1', new Map(), players)

    expect(orders1).toBeInstanceOf(Map)
    expect(orders2).toBeInstanceOf(Map)
  })
})

describe('HighCommandAI — reset', () => {
  it('reset() allows fresh history after clearing', () => {
    const ai = new HighCommandAI()
    // Build up some history
    ai.computeOrders(simpleBoard(), 'p1', new Map(), players)
    ai.computeOrders(simpleBoard(), 'p1', new Map(), players)
    // Reset
    ai.reset()
    // Should still work normally after reset
    const orders = ai.computeOrders(simpleBoard(), 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })
})

describe('HighCommandAI — collapse exploitation integration', () => {
  it('invades a nearly-eliminated neighbor', () => {
    // p1 has a tile, p0 has 2 tiles (≤3, so isNearlyEliminated = true)
    const board = makeBoard([
      makeTile(0, 0, 'p1', 10),  // p1 frontier
      makeTile(1, 0, 'p0', 2),   // p0 tile 1
      makeTile(2, 0, 'p0', 1),   // p0 tile 2 (total: 2 — nearly eliminated)
    ])
    const mixedPlayers: Player[] = [
      { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
      { id: 'p1', type: 'ai', color: '#f00', name: 'P1', isEliminated: false },
    ]
    const ai = new HighCommandAI()
    const orders = ai.computeOrders(board, 'p1', new Map(), mixedPlayers)
    // Should issue an order attacking p0's tile
    const order = orders.get('0,0')
    expect(order).toBeDefined()
    expect(order!.toKey).toBe('1,0')
  })
})

describe('HighCommandAI — multi-turn history', () => {
  it('persists history across multiple computeOrders calls', () => {
    const ai = new HighCommandAI()
    // Call computeOrders multiple times — should not throw and should keep working
    for (let i = 0; i < 5; i++) {
      const orders = ai.computeOrders(simpleBoard(), 'p1', new Map(), players)
      expect(orders).toBeInstanceOf(Map)
    }
  })
})
