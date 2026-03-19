import { describe, it, expect, beforeEach } from 'vitest'
import { buildSnapshot } from '../assessor/neighborAssessor.ts'
import { HistoryTracker } from '../assessor/historyTracker.ts'
import type { Board, Tile, Player } from '@hexwar/engine'

// hexNeighbors(q, r) uses directions: (+1,0),(+1,-1),(0,-1),(-1,0),(-1,+1),(0,+1)
// So (1,0) is adjacent to (0,0). (-1,0) is adjacent to (0,0) but NOT to (1,0).

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

function makePlayers(ids: string[], eliminatedIds: string[] = []): Player[] {
  return ids.map((id, i) => ({
    id,
    type: i === 0 ? 'human' : 'ai' as const,
    color: '#fff',
    name: id,
    isEliminated: eliminatedIds.includes(id),
  }))
}

describe('buildSnapshot — tile counts', () => {
  it('counts own tiles and units correctly', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(-1, 0, 'p0', 3),  // (-1,0) adjacent to (0,0) but not to (1,0)
      makeTile(1, 0, 'p1', 4),
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), new HistoryTracker())
    expect(snap.myTileCount).toBe(2)
    expect(snap.myTotalUnits).toBe(8)
  })

  it('excludes eliminated players from neighbor list', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),
    ])
    // p1 is eliminated
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1'], ['p1']), new HistoryTracker())
    expect(snap.neighbors).toHaveLength(0)
  })

  it('returns totalActivePlayers including self', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), new HistoryTracker())
    expect(snap.totalActivePlayers).toBe(2)
  })
})

describe('buildSnapshot — neighbor detection', () => {
  let tracker: HistoryTracker
  beforeEach(() => { tracker = new HistoryTracker() })

  it('identifies one neighbor on a two-player board', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), tracker)
    expect(snap.neighbors).toHaveLength(1)
    expect(snap.neighbors[0].playerId).toBe('p1')
  })

  it('does not include a non-adjacent player as a neighbor', () => {
    // p1 is two tiles away, separated by p0's tile — but NOT adjacent to p0
    // (-2,0) is NOT adjacent to (0,0) (distance = 2)
    const board2 = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(3, 0, 'p1', 4),  // not adjacent to (0,0)
    ])
    const snap = buildSnapshot(board2, 'p0', makePlayers(['p0', 'p1']), tracker)
    expect(snap.neighbors).toHaveLength(0)
  })

  it('identifies two neighbors on a three-player board', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),   // adjacent to (0,0)
      makeTile(0, 1, 'p2', 4),   // adjacent to (0,0)
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1', 'p2']), tracker)
    expect(snap.neighbors).toHaveLength(2)
    const ids = snap.neighbors.map(n => n.playerId).sort()
    expect(ids).toEqual(['p1', 'p2'])
  })
})

describe('buildSnapshot — border units', () => {
  let tracker: HistoryTracker
  beforeEach(() => { tracker = new HistoryTracker() })

  it('sums our border units correctly', () => {
    // (0,0) and (0,1) are both p0 tiles adjacent to (1,0) p1
    // hexNeighbors(0,1) includes (1,0) since (0+1, 1-1)=(1,0) — yes adjacent
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(0, 1, 'p0', 4),  // also adjacent to (1,0)
      makeTile(-1, 0, 'p0', 3), // NOT adjacent to (1,0)
      makeTile(1, 0, 'p1', 2),
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), tracker)
    const n = snap.neighbors[0]
    expect(n.ourBorderUnits).toBe(9)  // 5 + 4; (-1,0) is interior
  })

  it('sums their border units correctly', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), tracker)
    expect(snap.neighbors[0].theirBorderUnits).toBe(3)
  })

  it('relativeStrength is ourBorderUnits / theirBorderUnits', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 6),
      makeTile(1, 0, 'p1', 3),
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), tracker)
    expect(snap.neighbors[0].relativeStrength).toBeCloseTo(2.0)
  })

  it('relativeStrength is capped at 3', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 30),
      makeTile(1, 0, 'p1', 1),
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), tracker)
    expect(snap.neighbors[0].relativeStrength).toBe(3)
  })
})

describe('buildSnapshot — isNearlyEliminated & tileCount', () => {
  it('marks neighbor as nearly eliminated when they have ≤ 3 tiles', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 2),
      makeTile(2, 0, 'p1', 1),  // p1 has 2 tiles total
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), new HistoryTracker())
    expect(snap.neighbors[0].isNearlyEliminated).toBe(true)
    expect(snap.neighbors[0].tileCount).toBe(2)
  })

  it('does not mark neighbor as nearly eliminated with 4+ tiles', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 2),
      makeTile(2, 0, 'p1', 1),
      makeTile(2, -1, 'p1', 1),
      makeTile(3, 0, 'p1', 1), // p1 has 4 tiles
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), new HistoryTracker())
    expect(snap.neighbors[0].isNearlyEliminated).toBe(false)
  })
})

describe('buildSnapshot — isEngagedElsewhere', () => {
  it('marks neighbor as engaged elsewhere when they border a third player and are losing tiles', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),   // p1 adjacent to p0
      makeTile(2, 0, 'p2', 4),   // p2 adjacent to p1 — p1 fights on two fronts
    ])
    const tracker = new HistoryTracker()
    // Record two turns: p1 shrinking from 2 tiles to 1 tile
    tracker.recordTurn(makeBoard([makeTile(1, 0, 'p1', 3), makeTile(2, -1, 'p1', 2)]))
    tracker.recordTurn(makeBoard([makeTile(1, 0, 'p1', 3)]))
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1', 'p2']), tracker)
    const p1 = snap.neighbors.find(n => n.playerId === 'p1')!
    expect(p1.isEngagedElsewhere).toBe(true)
  })

  it('does not mark neighbor as engaged elsewhere if they are only adjacent to us', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),   // p1 only adjacent to p0
    ])
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), new HistoryTracker())
    expect(snap.neighbors[0].isEngagedElsewhere).toBe(false)
  })
})

describe('buildSnapshot — momentum', () => {
  it('reads momentum from history tracker', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 3),
    ])
    const tracker = new HistoryTracker()
    tracker.recordTurn(makeBoard([makeTile(1, 0, 'p1', 3), makeTile(2, 0, 'p1', 2)]))
    tracker.recordTurn(makeBoard([makeTile(1, 0, 'p1', 3)]))
    const snap = buildSnapshot(board, 'p0', makePlayers(['p0', 'p1']), tracker)
    expect(snap.neighbors[0].momentumDelta).toBe(-1)
  })
})
