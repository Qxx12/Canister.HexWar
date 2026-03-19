import { describe, it, expect } from 'vitest'
import { buildStrategicPlan } from '../strategies/registry.ts'
import { deterrenceWall } from '../strategies/threatGroup.ts'
import { collapseExploitation, woundedEnemy, vultureStrike } from '../strategies/opportunismGroup.ts'
import { turtleMode } from '../strategies/consolidationGroup.ts'
import { commonEnemy, dontFinishDecoy } from '../strategies/allianceGroup.ts'
import type { GeopoliticalSnapshot, NeighborAssessment } from '../types.ts'

function makeNeighbor(overrides: Partial<NeighborAssessment> & { playerId: string }): NeighborAssessment {
  return {
    sharedBorderTiles: ['0,0'],
    ourBorderUnits: 5,
    theirBorderUnits: 5,
    relativeStrength: 1.0,
    tileCount: 10,
    isEngagedElsewhere: false,
    momentumDelta: 0,
    isNearlyEliminated: false,
    ...overrides,
  }
}

function makeSnapshot(neighbors: NeighborAssessment[], overrides: Partial<GeopoliticalSnapshot> = {}): GeopoliticalSnapshot {
  return {
    myPlayerId: 'p0',
    myTileCount: 15,
    myTotalUnits: 30,
    myMomentumDelta: 0,
    totalActivePlayers: neighbors.length + 1,
    neighbors,
    ...overrides,
  }
}

// ─── Deterrence Wall ─────────────────────────────────────────────────────────

describe('deterrenceWall', () => {
  it('returns nothing with only one neighbor', () => {
    const snapshot = makeSnapshot([makeNeighbor({ playerId: 'p1' })])
    expect(deterrenceWall(snapshot)).toHaveLength(0)
  })

  it('deters strong neighbor and expands against weak', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', relativeStrength: 0.5, ourBorderUnits: 3, theirBorderUnits: 6 }),
      makeNeighbor({ playerId: 'p2', relativeStrength: 1.5, ourBorderUnits: 6, theirBorderUnits: 4 }),
    ])
    const directives = deterrenceWall(snapshot)
    const deter = directives.find(d => d.neighborId === 'p1')
    const expand = directives.find(d => d.neighborId === 'p2')
    expect(deter?.stance).toBe('DETER')
    expect(expand?.stance).toBe('EXPAND')
  })

  it('returns nothing when both neighbors are equally matched', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', relativeStrength: 1.0 }),
      makeNeighbor({ playerId: 'p2', relativeStrength: 1.0 }),
    ])
    expect(deterrenceWall(snapshot)).toHaveLength(0)
  })
})

// ─── Collapse Exploitation ────────────────────────────────────────────────────

describe('collapseExploitation', () => {
  it('invades a nearly-eliminated neighbor at highest priority', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', isNearlyEliminated: true, tileCount: 2 }),
    ])
    const directives = collapseExploitation(snapshot)
    expect(directives).toHaveLength(1)
    expect(directives[0].stance).toBe('INVADE')
    expect(directives[0].priority).toBe(10)
  })

  it('ignores neighbors that are not nearly eliminated', () => {
    const snapshot = makeSnapshot([makeNeighbor({ playerId: 'p1', tileCount: 10 })])
    expect(collapseExploitation(snapshot)).toHaveLength(0)
  })
})

// ─── Wounded Enemy ────────────────────────────────────────────────────────────

describe('woundedEnemy', () => {
  it('invades a rapidly shrinking neighbor', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', momentumDelta: -4, relativeStrength: 1.0 }),
    ])
    const directives = woundedEnemy(snapshot)
    expect(directives[0]?.stance).toBe('INVADE')
  })

  it('skips nearly-eliminated neighbors (handled by collapseExploitation)', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', momentumDelta: -5, isNearlyEliminated: true }),
    ])
    expect(woundedEnemy(snapshot)).toHaveLength(0)
  })
})

// ─── Vulture Strike ───────────────────────────────────────────────────────────

describe('vultureStrike', () => {
  it('holds when two neighbors are bleeding each other', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', isEngagedElsewhere: true, momentumDelta: -3 }),
      makeNeighbor({ playerId: 'p2', isEngagedElsewhere: true, momentumDelta: -2 }),
    ])
    const directives = vultureStrike(snapshot)
    expect(directives.every(d => d.stance === 'HOLD')).toBe(true)
  })

  it('returns empty with only one bleeding neighbor', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', isEngagedElsewhere: true, momentumDelta: -3 }),
      makeNeighbor({ playerId: 'p2', momentumDelta: 1 }),
    ])
    expect(vultureStrike(snapshot)).toHaveLength(0)
  })
})

// ─── Turtle Mode ─────────────────────────────────────────────────────────────

describe('turtleMode', () => {
  it('holds all fronts when losing badly with multiple weak borders', () => {
    const snapshot = makeSnapshot(
      [
        makeNeighbor({ playerId: 'p1', relativeStrength: 0.6 }),
        makeNeighbor({ playerId: 'p2', relativeStrength: 0.7 }),
      ],
      { myMomentumDelta: -4 },
    )
    const directives = turtleMode(snapshot)
    expect(directives.length).toBe(2)
    expect(directives.every(d => d.stance === 'HOLD')).toBe(true)
    expect(directives[0].priority).toBe(7)
  })

  it('does not trigger when momentum is healthy', () => {
    const snapshot = makeSnapshot(
      [
        makeNeighbor({ playerId: 'p1', relativeStrength: 0.6 }),
        makeNeighbor({ playerId: 'p2', relativeStrength: 0.7 }),
      ],
      { myMomentumDelta: 0 },
    )
    expect(turtleMode(snapshot)).toHaveLength(0)
  })
})

// ─── Common Enemy ─────────────────────────────────────────────────────────────

describe('commonEnemy', () => {
  it('focuses largest player and ignores smallest when imbalanced', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', tileCount: 40 }),
      makeNeighbor({ playerId: 'p2', tileCount: 8 }),
    ])
    const directives = commonEnemy(snapshot)
    const focus = directives.find(d => d.neighborId === 'p1')
    const ignore = directives.find(d => d.neighborId === 'p2')
    expect(focus?.stance).toBe('INVADE')
    expect(ignore?.stance).toBe('IGNORE')
  })

  it('returns empty when tile counts are balanced', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', tileCount: 12 }),
      makeNeighbor({ playerId: 'p2', tileCount: 10 }),
    ])
    expect(commonEnemy(snapshot)).toHaveLength(0)
  })
})

// ─── dontFinishDecoy edge cases ──────────────────────────────────────────────

describe('dontFinishDecoy', () => {
  it('ignores a nearly-eliminated neighbor that is also engaged elsewhere', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', isNearlyEliminated: true, isEngagedElsewhere: true }),
    ])
    const directives = dontFinishDecoy(snapshot)
    expect(directives[0]?.stance).toBe('IGNORE')
  })

  it('does not ignore a nearly-eliminated neighbor who is NOT engaged elsewhere', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', isNearlyEliminated: true, isEngagedElsewhere: false }),
    ])
    const directives = dontFinishDecoy(snapshot)
    expect(directives).toHaveLength(0)
  })
})

// ─── turtleMode threshold edge cases ─────────────────────────────────────────

describe('turtleMode — thresholds', () => {
  it('activates at exactly momentum -3 with 2 weak fronts', () => {
    const snapshot = makeSnapshot(
      [
        makeNeighbor({ playerId: 'p1', relativeStrength: 0.7 }),
        makeNeighbor({ playerId: 'p2', relativeStrength: 0.8 }),
      ],
      { myMomentumDelta: -3 },
    )
    expect(turtleMode(snapshot).length).toBe(2)
  })

  it('does not activate at momentum -2', () => {
    const snapshot = makeSnapshot(
      [
        makeNeighbor({ playerId: 'p1', relativeStrength: 0.7 }),
        makeNeighbor({ playerId: 'p2', relativeStrength: 0.8 }),
      ],
      { myMomentumDelta: -2 },
    )
    expect(turtleMode(snapshot)).toHaveLength(0)
  })

  it('does not activate with only 1 weak front', () => {
    const snapshot = makeSnapshot(
      [
        makeNeighbor({ playerId: 'p1', relativeStrength: 0.7 }),
        makeNeighbor({ playerId: 'p2', relativeStrength: 1.5 }),
      ],
      { myMomentumDelta: -4 },
    )
    expect(turtleMode(snapshot)).toHaveLength(0)
  })

  it('sets priority 7 on all directives when active', () => {
    const snapshot = makeSnapshot(
      [
        makeNeighbor({ playerId: 'p1', relativeStrength: 0.6 }),
        makeNeighbor({ playerId: 'p2', relativeStrength: 0.7 }),
        makeNeighbor({ playerId: 'p3', relativeStrength: 0.8 }),
      ],
      { myMomentumDelta: -5 },
    )
    const directives = turtleMode(snapshot)
    expect(directives.every(d => d.priority === 7)).toBe(true)
    expect(directives.every(d => d.stance === 'HOLD')).toBe(true)
  })
})

// ─── commonEnemy edge cases ───────────────────────────────────────────────────

describe('commonEnemy — edge cases', () => {
  it('does not ignore a nearly-eliminated smallest neighbor', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', tileCount: 40 }),
      makeNeighbor({ playerId: 'p2', tileCount: 2, isNearlyEliminated: true }),
    ])
    const directives = commonEnemy(snapshot)
    // p2 is nearly eliminated — we want them gone, don't ignore them
    expect(directives.find(d => d.neighborId === 'p2')?.stance).not.toBe('IGNORE')
  })

  it('returns empty with only one neighbor', () => {
    const snapshot = makeSnapshot([makeNeighbor({ playerId: 'p1', tileCount: 40 })])
    expect(commonEnemy(snapshot)).toHaveLength(0)
  })
})

// ─── Registry — two-front avoidance ──────────────────────────────────────────

describe('buildStrategicPlan — two-front cap', () => {
  it('caps offensive fronts to 2 in a 4-player game', () => {
    const snapshot = makeSnapshot([
      makeNeighbor({ playerId: 'p1', isNearlyEliminated: true, tileCount: 2 }),
      makeNeighbor({ playerId: 'p2', isNearlyEliminated: true, tileCount: 2 }),
      makeNeighbor({ playerId: 'p3', isNearlyEliminated: true, tileCount: 2 }),
    ])
    const plan = buildStrategicPlan(snapshot, [collapseExploitation])
    const offensive = [...plan.directives.values()].filter(
      d => d.stance === 'INVADE' || d.stance === 'EXPAND',
    )
    expect(offensive.length).toBeLessThanOrEqual(2)
  })

  it('allows 1 offensive front in a 2-player game', () => {
    const snapshot = makeSnapshot(
      [makeNeighbor({ playerId: 'p1', isNearlyEliminated: true, tileCount: 2 })],
      { totalActivePlayers: 2 },
    )
    const plan = buildStrategicPlan(snapshot, [collapseExploitation])
    const offensive = [...plan.directives.values()].filter(
      d => d.stance === 'INVADE' || d.stance === 'EXPAND',
    )
    expect(offensive.length).toBe(1)
  })
})
