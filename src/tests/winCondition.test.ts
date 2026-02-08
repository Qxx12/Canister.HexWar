import { describe, it, expect } from 'vitest'
import { checkWin, checkEliminations } from '../engine/winCondition'
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

describe('checkWin', () => {
  it('returns null when no one has won', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: true, startOwner: 'p0', terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 5, r: 0 }, owner: 'p1', units: 3, isStartTile: true, startOwner: 'p1', terrain: 'plains' as const, newlyConquered: false },
    ])
    expect(checkWin(board, players)).toBeNull()
  })

  it('returns winner when player owns all start tiles including their own', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: true, startOwner: 'p0', terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 5, r: 0 }, owner: 'p0', units: 3, isStartTile: true, startOwner: 'p1', terrain: 'plains' as const, newlyConquered: false },
    ])
    expect(checkWin(board, players)).toBe('p0')
  })

  it('does not declare win if player lost their own start tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p1', units: 5, isStartTile: true, startOwner: 'p0', terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 5, r: 0 }, owner: 'p0', units: 3, isStartTile: true, startOwner: 'p1', terrain: 'plains' as const, newlyConquered: false },
    ])
    expect(checkWin(board, players)).toBeNull()
  })
})

describe('checkEliminations', () => {
  it('returns player with no tiles', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: true, startOwner: 'p0', terrain: 'plains' as const, newlyConquered: false },
    ])
    expect(checkEliminations(board, players)).toContain('p1')
  })

  it('returns empty when all players have tiles', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: true, startOwner: 'p0', terrain: 'plains' as const, newlyConquered: false },
      { coord: { q: 5, r: 0 }, owner: 'p1', units: 3, isStartTile: true, startOwner: 'p1', terrain: 'plains' as const, newlyConquered: false },
    ])
    expect(checkEliminations(board, players)).toHaveLength(0)
  })
})
