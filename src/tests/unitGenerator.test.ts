import { describe, it, expect } from 'vitest'
import { generateUnits } from '@hexwar/engine'
import type { Board, Tile } from '@hexwar/engine'
import type { PlayerStats } from '@hexwar/engine'

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
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 3, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 2, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const stats = new Map<string, PlayerStats>([
      ['p0', { playerId: 'p0', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesLost: 0, tilesAtEnd: 0 }],
      ['p1', { playerId: 'p1', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesLost: 0, tilesAtEnd: 0 }],
    ])
    const newBoard = generateUnits(board, stats)
    expect(newBoard.get('0,0')!.units).toBe(4)
    expect(newBoard.get('1,0')!.units).toBe(3)
  })

  it('does not generate units for unconquered tiles', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
    ])
    const stats = new Map()
    const newBoard = generateUnits(board, stats)
    expect(newBoard.get('0,0')!.units).toBe(0)
  })

  it('skips unit generation for newly conquered tiles and resets the flag', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 3, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: true },
    ])
    const stats = new Map<string, PlayerStats>([
      ['p0', { playerId: 'p0', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesLost: 0, tilesAtEnd: 0 }],
    ])
    const newBoard = generateUnits(board, stats)
    expect(newBoard.get('0,0')!.units).toBe(3)
    expect(newBoard.get('0,0')!.newlyConquered).toBe(false)
  })

  it('increments unitsGenerated stat for each owned tile that generates', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 2, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 1, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 2, r: 0 }, owner: 'p0', units: 4, isStartTile: false, startOwner: null, terrain: 'plains' as const, newlyConquered: true },
    ])
    const stats = new Map<string, PlayerStats>([
      ['p0', { playerId: 'p0', unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesLost: 0, tilesAtEnd: 0 }],
    ])
    generateUnits(board, stats)
    expect(stats.get('p0')!.unitsGenerated).toBe(2) // newly conquered tile excluded
  })
})
