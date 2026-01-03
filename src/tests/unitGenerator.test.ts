import { describe, it, expect } from 'vitest'
import { generateUnits } from '../engine/unitGenerator'
import type { Board, Tile } from '../types/board'
import type { PlayerStats } from '../types/stats'

function makeBoard(tiles: Tile[]): Board {
  const board: Board = new Map()
  for (const tile of tiles) {
    board.set(`${tile.coord.q},${tile.coord.r}`, tile)
  }
  return board
}

describe('generateUnits', () => {
  it('generates 1 unit per owned tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 3, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 2, isStartTile: false, startOwner: null },
    ])
    const stats = new Map<string, PlayerStats>([
      ['p0', { playerId: 'p0', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesAtEnd: 0 }],
      ['p1', { playerId: 'p1', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesAtEnd: 0 }],
    ])
    const newBoard = generateUnits(board, stats)
    expect(newBoard.get('0,0')!.units).toBe(4)
    expect(newBoard.get('1,0')!.units).toBe(3)
  })

  it('does not generate units for unconquered tiles', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const stats = new Map()
    const newBoard = generateUnits(board, stats)
    expect(newBoard.get('0,0')!.units).toBe(0)
  })
})
