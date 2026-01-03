import { describe, it, expect } from 'vitest'
import { resolveCombat, applyCombatResult } from '../engine/combat'
import type { Board, Tile } from '../types/board'

function makeBoard(tiles: Tile[]): Board {
  const board: Board = new Map()
  for (const tile of tiles) {
    board.set(`${tile.coord.q},${tile.coord.r}`, tile)
  }
  return board
}

describe('resolveCombat', () => {
  it('moves to friendly tile without casualties', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 2, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }, 'p0')
    expect(result.attackerCasualties).toBe(0)
    expect(result.defenderCasualties).toBe(0)
    expect(result.conquered).toBe(false)
    expect(result.remainingAttackers).toBe(3)
  })

  it('conquers unconquered tile without casualties', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }, 'p0')
    expect(result.conquered).toBe(true)
    expect(result.attackerCasualties).toBe(0)
    expect(result.remainingAttackers).toBe(3)
  })

  it('combat: attacker wins when sending more units', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 3, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 5 }, 'p0')
    expect(result.conquered).toBe(true)
    expect(result.remainingAttackers).toBe(2)
    expect(result.attackerCasualties).toBe(3)
    expect(result.defenderCasualties).toBe(3)
  })

  it('combat: attacker loses when sending fewer units', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 2, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 5, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 2 }, 'p0')
    expect(result.conquered).toBe(false)
    expect(result.remainingAttackers).toBe(0)
  })

  it('combat: tie - attacker loses', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 3, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 3, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }, 'p0')
    expect(result.conquered).toBe(false)
    expect(result.remainingAttackers).toBe(0)
  })

  it('clamps units to available', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 2, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: null, units: 0, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 10 }, 'p0')
    expect(result.unitsSent).toBe(2)
    expect(result.wasClamped).toBe(true)
  })
})

describe('applyCombatResult', () => {
  it('moves units to friendly tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p0', units: 2, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }, 'p0')
    const newBoard = applyCombatResult(board, result)
    expect(newBoard.get('0,0')!.units).toBe(2) // 5 - 3
    expect(newBoard.get('1,0')!.units).toBe(5) // 2 + 3
  })

  it('conquers hostile tile', () => {
    const board = makeBoard([
      { coord: { q: 0, r: 0 }, owner: 'p0', units: 5, isStartTile: false, startOwner: null },
      { coord: { q: 1, r: 0 }, owner: 'p1', units: 3, isStartTile: false, startOwner: null },
    ])
    const result = resolveCombat(board, { fromKey: '0,0', toKey: '1,0', requestedUnits: 5 }, 'p0')
    const newBoard = applyCombatResult(board, result)
    expect(newBoard.get('1,0')!.owner).toBe('p0')
    expect(newBoard.get('1,0')!.units).toBe(2) // 5 - 3
  })
})
