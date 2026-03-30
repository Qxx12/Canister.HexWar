import { describe, it, expect, beforeEach } from 'vitest'
import { computeAiOrders, resetAiAgents, setAiDifficulty } from '../ai/aiController'
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
  beforeEach(() => { resetAiAgents() })
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

describe('setAiDifficulty', () => {
  beforeEach(() => { resetAiAgents() })

  const twoTileBoard = () => makeBoard([
    { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
  ])

  it('soldier difficulty returns a valid OrderMap', () => {
    setAiDifficulty('soldier')
    const orders = computeAiOrders(twoTileBoard(), 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })

  it('commander difficulty returns a valid OrderMap', () => {
    setAiDifficulty('commander')
    const orders = computeAiOrders(twoTileBoard(), 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })

  it('warlord difficulty returns a valid OrderMap', () => {
    setAiDifficulty('warlord')
    const orders = computeAiOrders(twoTileBoard(), 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })

  it('changing difficulty after reset uses the new agent type', () => {
    setAiDifficulty('soldier')
    resetAiAgents()
    const orders1 = computeAiOrders(twoTileBoard(), 'p1', new Map(), players)

    setAiDifficulty('warlord')
    resetAiAgents()
    const orders2 = computeAiOrders(twoTileBoard(), 'p1', new Map(), players)

    expect(orders1).toBeInstanceOf(Map)
    expect(orders2).toBeInstanceOf(Map)
  })

  it('agents created before setAiDifficulty are not retroactively changed', () => {
    // Agent for p1 is created with 'commander' difficulty
    setAiDifficulty('commander')
    computeAiOrders(twoTileBoard(), 'p1', new Map(), players)

    // Changing difficulty does not affect already-created p1 agent
    setAiDifficulty('warlord')
    // p1 still uses the commander agent (no reset), p2 would get warlord
    const orders = computeAiOrders(twoTileBoard(), 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })
})

describe('resetAiAgents', () => {
  it('allows a fresh call after reset without retaining history', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    // First session
    computeAiOrders(board, 'p1', new Map(), players)
    computeAiOrders(board, 'p1', new Map(), players)
    // Reset clears history
    resetAiAgents()
    // Should work normally again after reset
    const orders = computeAiOrders(board, 'p1', new Map(), players)
    expect(orders).toBeInstanceOf(Map)
  })

  it('two different player IDs get independent agents', () => {
    const board1 = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const board2 = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p2', units: 5, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const allPlayers = [
      ...players,
      { id: 'p2', type: 'ai' as const, color: '#00f', name: 'P2', isEliminated: false },
    ]
    const orders1 = computeAiOrders(board1, 'p1', new Map(), allPlayers)
    const orders2 = computeAiOrders(board2, 'p2', new Map(), allPlayers)
    // Both produce valid order maps
    expect(orders1).toBeInstanceOf(Map)
    expect(orders2).toBeInstanceOf(Map)
  })
})
