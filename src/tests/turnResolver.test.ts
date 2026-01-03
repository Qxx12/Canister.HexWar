import { describe, it, expect } from 'vitest'
import { resolvePlayerTurn } from '../engine/turnResolver'
import type { Board, Tile } from '../types/board'
import type { Player } from '../types/player'
import type { OrderMap } from '../types/orders'
import type { PlayerStats } from '../types/stats'

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

function makeStats(): Map<string, PlayerStats> {
  return new Map([
    ['p0', { playerId: 'p0', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesAtEnd: 0 }],
    ['p1', { playerId: 'p1', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesAtEnd: 0 }],
  ])
}

describe('resolvePlayerTurn', () => {
  it('moves units to adjacent friendly tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 2, isStartTile: false, startOwner: null },
    ])
    const orders: OrderMap = new Map([
      ['0,0', { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }],
    ])
    const result = resolvePlayerTurn(board, players, orders, 'p0', makeStats())
    expect(result.board.get('0,0')!.units).toBe(2)
    expect(result.board.get('1,0')!.units).toBe(5)
    expect(result.animationEvents).toHaveLength(1)
    expect(result.animationEvents[0].kind).toBe('move')
  })

  it('conquers hostile tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 2, isStartTile: false, startOwner: null },
    ])
    const orders: OrderMap = new Map([
      ['0,0', { fromKey: '0,0', toKey: '1,0', requestedUnits: 5 }],
    ])
    const result = resolvePlayerTurn(board, players, orders, 'p0', makeStats())
    expect(result.board.get('1,0')!.owner).toBe('p0')
    expect(result.animationEvents[0].kind).toBe('conquer')
  })

  it('detects win when all start tiles captured', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: true, startOwner: 'p0' },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 2, isStartTile: true, startOwner: 'p1' },
    ])
    const orders: OrderMap = new Map([
      ['0,0', { fromKey: '0,0', toKey: '1,0', requestedUnits: 5 }],
    ])
    const result = resolvePlayerTurn(board, players, orders, 'p0', makeStats())
    expect(result.winnerId).toBe('p0')
  })

  it('skips order if tile not owned by player', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const orders: OrderMap = new Map([
      ['0,0', { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }],
    ])
    const result = resolvePlayerTurn(board, players, orders, 'p0', makeStats())
    expect(result.board.get('1,0')!.owner).toBeNull()
    expect(result.animationEvents).toHaveLength(0)
  })
})
