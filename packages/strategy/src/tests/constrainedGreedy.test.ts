import { describe, it, expect } from 'vitest'
import { computeConstrainedOrders } from '../tactical/constrainedGreedy.ts'
import type { Board, Tile, Player } from '@hexwar/engine'
import type { TileConstraint } from '../types.ts'

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

function makeConstraint(sourceKey: string, allowedTargetKeys: string[], opts: Partial<TileConstraint> = {}): TileConstraint {
  return {
    sourceKey,
    allowedTargetKeys,
    maxUnitsFraction: 1.0,
    crossBorderAllowed: true,
    ...opts,
  }
}

const players: Player[] = [
  { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
  { id: 'p1', type: 'ai', color: '#f00', name: 'P1', isEliminated: false },
]

describe('computeConstrainedOrders — neutral target', () => {
  it('sends all units to a neutral tile', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, null, 0),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    const order = orders.get('0,0')!
    expect(order.toKey).toBe('1,0')
    expect(order.requestedUnits).toBe(5)
  })
})

describe('computeConstrainedOrders — enemy target', () => {
  it('sends enough to conquer (not all) when we have the advantage', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 10),
      makeTile(1, 0, 'p1', 3),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    const order = orders.get('0,0')!
    expect(order.toKey).toBe('1,0')
    // unitsToSend(10, enemy_3): max(4, floor(10*0.85)) = max(4, 8) = 8
    expect(order.requestedUnits).toBe(8)
  })

  it('skips an enemy tile we cannot conquer (negative score)', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 2),
      makeTile(1, 0, 'p1', 5),  // we'd lose
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    expect(orders.has('0,0')).toBe(false)
  })

  it('attacks an enemy start tile even at a disadvantage', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 2),
      makeTile(1, 0, 'p1', 5, { isStartTile: true }),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    expect(orders.has('0,0')).toBe(true)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
  })

  it('prefers start tiles over neutral tiles when both are in range', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 10),
      makeTile(1, 0, null, 0),            // neutral
      makeTile(0, 1, 'p1', 3, { isStartTile: true }),  // enemy start tile
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0', '0,1'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    // Enemy start tile scores 50 + (10-3)*3 + 45 = 50+21+45=116 vs neutral 35 → start tile wins
    expect(orders.get('0,0')!.toKey).toBe('0,1')
  })
})

describe('computeConstrainedOrders — unit budget cap', () => {
  it('respects maxUnitsFraction', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 10),
      makeTile(1, 0, null, 0),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'], { maxUnitsFraction: 0.5 })]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    // floor(10 * 0.5) = 5
    expect(orders.get('0,0')!.requestedUnits).toBe(5)
  })

  it('generates no order when maxUnitsFraction is 0', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, null, 0),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'], { maxUnitsFraction: 0 })]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    expect(orders.has('0,0')).toBe(false)
  })
})

describe('computeConstrainedOrders — interior routing', () => {
  it('routes an interior tile toward the front via interiorRoutes', () => {
    const board = makeBoard([
      makeTile(-1, 0, 'p0', 4),  // interior
      makeTile(0, 0, 'p0', 5),   // frontier
      makeTile(1, 0, 'p1', 2),
    ])
    // interior tile constraint: crossBorderAllowed=false, allowed target is (0,0)
    const constraints = new Map([
      ['-1,0', makeConstraint('-1,0', ['0,0'], { crossBorderAllowed: false })],
    ])
    const interiorRoutes = new Map([['-1,0', '0,0']])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, interiorRoutes)
    const order = orders.get('-1,0')!
    expect(order.toKey).toBe('0,0')
    expect(order.requestedUnits).toBe(4)
  })

  it('does not route when interior route is not in allowedTargetKeys', () => {
    const board = makeBoard([
      makeTile(-1, 0, 'p0', 4),
      makeTile(0, 0, 'p0', 5),
    ])
    // allowedTargetKeys doesn't include the route destination
    const constraints = new Map([
      ['-1,0', makeConstraint('-1,0', ['5,5'], { crossBorderAllowed: false })],
    ])
    const interiorRoutes = new Map([['-1,0', '0,0']])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, interiorRoutes)
    expect(orders.has('-1,0')).toBe(false)
  })
})

describe('computeConstrainedOrders — capital garrison', () => {
  it('keeps MIN_CAPITAL_GARRISON (2) units on own start tile', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5, { isStartTile: true, startOwner: 'p0' }),
      makeTile(1, 0, null, 0),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    // maxSend = 5 - 2 = 3
    expect(orders.get('0,0')!.requestedUnits).toBe(3)
  })

  it('deletes the order entirely when only MIN_CAPITAL_GARRISON units exist', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 2, { isStartTile: true, startOwner: 'p0' }),
      makeTile(1, 0, null, 0),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    expect(orders.has('0,0')).toBe(false)
  })

  it('does not apply garrison to a non-start tile', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 2, { isStartTile: false }),
      makeTile(1, 0, null, 0),
    ])
    const constraints = new Map([['0,0', makeConstraint('0,0', ['1,0'])]])
    const orders = computeConstrainedOrders(board, 'p0', players, constraints, new Map())
    expect(orders.get('0,0')!.requestedUnits).toBe(2)
  })
})
