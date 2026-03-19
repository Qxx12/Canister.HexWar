import { describe, it, expect, beforeEach } from 'vitest'
import { HistoryTracker } from '../assessor/historyTracker.ts'
import type { Board, Tile, AxialCoord } from '@hexwar/engine'

function makeBoard(entries: Array<[string, string | null, number]>): Board {
  const board: Board = new Map()
  for (const [key, owner, units] of entries) {
    const [q, r] = key.split(',').map(Number)
    board.set(key, {
      coord: { q, r } as AxialCoord,
      owner,
      units,
      isStartTile: false,
      startOwner: null,
      terrain: 'grassland',
      newlyConquered: false,
    } as Tile)
  }
  return board
}

describe('HistoryTracker', () => {
  let tracker: HistoryTracker

  beforeEach(() => {
    tracker = new HistoryTracker()
  })

  it('returns 0 momentum when no history', () => {
    expect(tracker.getMomentumDelta('p1')).toBe(0)
  })

  it('returns 0 momentum with only one snapshot', () => {
    const board = makeBoard([['0,0', 'p1', 1], ['1,0', 'p1', 1]])
    tracker.recordTurn(board)
    expect(tracker.getMomentumDelta('p1')).toBe(0)
  })

  it('computes positive momentum when player grows', () => {
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1]]))
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1], ['1,0', 'p1', 1], ['2,0', 'p1', 1]]))
    expect(tracker.getMomentumDelta('p1')).toBe(2)
  })

  it('computes negative momentum when player shrinks', () => {
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1], ['1,0', 'p1', 1], ['2,0', 'p1', 1]]))
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1]]))
    expect(tracker.getMomentumDelta('p1')).toBe(-2)
  })

  it('caps history at 5 snapshots', () => {
    for (let i = 1; i <= 7; i++) {
      const entries: Array<[string, string | null, number]> = Array.from({ length: i }, (_, j) => [`${j},0`, 'p1', 1])
      tracker.recordTurn(makeBoard(entries))
    }
    // oldest snapshot in window is index 2 (3 tiles), newest is index 6 (7 tiles)
    expect(tracker.getMomentumDelta('p1')).toBe(4)
  })

  it('isLosingTiles returns true when delta ≤ threshold', () => {
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1], ['1,0', 'p1', 1]]))
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1]]))
    expect(tracker.isLosingTiles('p1', -1)).toBe(true)
  })

  it('isLosingTiles returns false when stable', () => {
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1]]))
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1]]))
    expect(tracker.isLosingTiles('p1', -1)).toBe(false)
  })

  it('reset clears all snapshots', () => {
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1]]))
    tracker.recordTurn(makeBoard([['0,0', 'p1', 1], ['1,0', 'p1', 1]]))
    tracker.reset()
    expect(tracker.getMomentumDelta('p1')).toBe(0)
  })
})
