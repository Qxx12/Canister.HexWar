import { describe, it, expect } from 'vitest'
import { buildFrontMask } from '../operational/frontMask.ts'
import { buildInteriorRoutes } from '../operational/interiorRouter.ts'
import type { Board, Tile } from '@hexwar/engine'
import type { StrategicPlan, FrontDirective } from '../types.ts'

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

function makePlan(directives: Record<string, Partial<FrontDirective>>, neutralExpansion = true): StrategicPlan {
  const map = new Map<string, FrontDirective>()
  for (const [id, partial] of Object.entries(directives)) {
    map.set(id, {
      neighborId: id,
      stance: 'HOLD',
      unitBudgetFraction: 0.7,
      priority: 1,
      rationale: 'test',
      ...partial,
    })
  }
  return { directives: map, neutralExpansionEnabled: neutralExpansion }
}

// ─── buildFrontMask ───────────────────────────────────────────────────────────

describe('buildFrontMask — INVADE neighbor', () => {
  it('allows crossing into enemy territory', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 2),
    ])
    const plan = makePlan({ p1: { stance: 'INVADE', unitBudgetFraction: 0.9 } })
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('0,0')!
    expect(c.crossBorderAllowed).toBe(true)
    expect(c.allowedTargetKeys).toContain('1,0')
    expect(c.maxUnitsFraction).toBeCloseTo(0.9)
  })
})

describe('buildFrontMask — EXPAND neighbor', () => {
  it('allows crossing into enemy territory with expansion budget', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 2),
    ])
    const plan = makePlan({ p1: { stance: 'EXPAND', unitBudgetFraction: 0.8 } })
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('0,0')!
    expect(c.crossBorderAllowed).toBe(true)
    expect(c.allowedTargetKeys).toContain('1,0')
    expect(c.maxUnitsFraction).toBeCloseTo(0.8)
  })
})

describe('buildFrontMask — DETER neighbor', () => {
  it('does not allow crossing, sets budget to 0 (units accumulate in place)', () => {
    // (-1,0) is a friendly interior tile adjacent to (0,0), so the constraint is created
    const board = makeBoard([
      makeTile(-1, 0, 'p0', 3),  // friendly interior — gives (0,0) a route entry
      makeTile(0, 0, 'p0', 5),   // frontier facing p1
      makeTile(1, 0, 'p1', 8),
    ])
    const plan = makePlan({ p1: { stance: 'DETER', unitBudgetFraction: 0.9 } })
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('0,0')!
    expect(c.crossBorderAllowed).toBe(false)
    expect(c.allowedTargetKeys).not.toContain('1,0')
    expect(c.maxUnitsFraction).toBe(0)
  })

  it('creates no constraint at all when a DETER tile has no friendly neighbours', () => {
    // (0,0) borders only an enemy — no routing possible, so skip entirely
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 8),
    ])
    const plan = makePlan({ p1: { stance: 'DETER', unitBudgetFraction: 0.9 } })
    const mask = buildFrontMask(board, 'p0', plan)
    expect(mask.has('0,0')).toBe(false)
  })
})

describe('buildFrontMask — HOLD neighbor', () => {
  it('does not allow crossing but keeps full interior routing budget', () => {
    const board = makeBoard([
      makeTile(-1, 0, 'p0', 3),  // friendly interior — ensures constraint is created
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 4),
    ])
    const plan = makePlan({ p1: { stance: 'HOLD', unitBudgetFraction: 0.7 } })
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('0,0')!
    expect(c.crossBorderAllowed).toBe(false)
    expect(c.allowedTargetKeys).not.toContain('1,0')
    expect(c.maxUnitsFraction).toBe(1.0)
  })
})

describe('buildFrontMask — IGNORE neighbor', () => {
  it('excludes the ignored neighbor tile from allowed targets entirely', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, 'p1', 4),
    ])
    const plan = makePlan({ p1: { stance: 'IGNORE', unitBudgetFraction: 0 } })
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('0,0')
    // No cross-border target → either no constraint or crossBorderAllowed = false
    if (c) {
      expect(c.crossBorderAllowed).toBe(false)
      expect(c.allowedTargetKeys).not.toContain('1,0')
    }
    // Either way, no attack order should be possible
  })
})

describe('buildFrontMask — neutral expansion', () => {
  it('includes neutral tiles when neutralExpansionEnabled', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, null, 0),  // neutral
    ])
    const plan = makePlan({}, true)
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('0,0')!
    expect(c.crossBorderAllowed).toBe(true)
    expect(c.allowedTargetKeys).toContain('1,0')
  })

  it('excludes neutral tiles when neutralExpansionEnabled is false', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(1, 0, null, 0),
    ])
    const plan = makePlan({}, false)
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('0,0')
    if (c) {
      expect(c.allowedTargetKeys).not.toContain('1,0')
    }
  })
})

describe('buildFrontMask — interior tile', () => {
  it('interior tile (no enemy neighbors) has crossBorderAllowed=false and full budget', () => {
    const board = makeBoard([
      makeTile(-1, 0, 'p0', 4),  // interior — only adjacent to (0,0)
      makeTile(0, 0, 'p0', 5),   // frontier — adjacent to p1
      makeTile(1, 0, 'p1', 3),
    ])
    const plan = makePlan({ p1: { stance: 'EXPAND', unitBudgetFraction: 0.8 } })
    const mask = buildFrontMask(board, 'p0', plan)
    const c = mask.get('-1,0')!
    expect(c.crossBorderAllowed).toBe(false)
    expect(c.maxUnitsFraction).toBe(1.0)
    expect(c.allowedTargetKeys).toContain('0,0')  // friendly neighbor for routing
  })
})

describe('buildFrontMask — zero-unit tiles', () => {
  it('excludes tiles with 0 units from the mask', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 0),  // no units — should be excluded
      makeTile(1, 0, 'p1', 3),
    ])
    const plan = makePlan({ p1: { stance: 'INVADE', unitBudgetFraction: 0.9 } })
    const mask = buildFrontMask(board, 'p0', plan)
    expect(mask.has('0,0')).toBe(false)
  })
})

// ─── buildInteriorRoutes ──────────────────────────────────────────────────────

describe('buildInteriorRoutes — basic routing', () => {
  it('routes an interior tile toward the active front tile', () => {
    // Layout: (-1,0) → (0,0) [front] → (1,0) [enemy]
    const board = makeBoard([
      makeTile(-1, 0, 'p0', 3),  // interior
      makeTile(0, 0, 'p0', 5),   // frontier
      makeTile(1, 0, 'p1', 2),   // enemy — not traversed
    ])
    const routes = buildInteriorRoutes(board, 'p0', new Set(['0,0']))
    // (-1,0) should route to (0,0)
    expect(routes.get('-1,0')).toBe('0,0')
    // (0,0) is in the front set — no route entry for it
    expect(routes.has('0,0')).toBe(false)
  })

  it('returns empty map when activeFrontTiles is empty', () => {
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),
      makeTile(-1, 0, 'p0', 3),
    ])
    const routes = buildInteriorRoutes(board, 'p0', new Set())
    expect(routes.size).toBe(0)
  })

  it('does not route through enemy tiles', () => {
    // (0,0) [p0 interior] — (1,0) [p1 enemy] — (2,0) [p0 front]
    // Route should not go through p1's tile
    const board = makeBoard([
      makeTile(0, 0, 'p0', 3),
      makeTile(1, 0, 'p1', 4),
      makeTile(2, 0, 'p0', 5),  // front
    ])
    const routes = buildInteriorRoutes(board, 'p0', new Set(['2,0']))
    // (0,0) has no path to (2,0) through friendly tiles, so no route
    expect(routes.has('0,0')).toBe(false)
  })

  it('routes multiple interior tiles toward the same front', () => {
    // Star: three interior tiles around one front tile
    // Front: (0,0). Interiors: (1,0), (0,1), (-1,1)
    // All adjacent to (0,0)
    const board = makeBoard([
      makeTile(0, 0, 'p0', 5),   // frontier (active)
      makeTile(1, 0, 'p0', 3),   // interior
      makeTile(0, 1, 'p0', 2),   // interior
      makeTile(-1, 1, 'p0', 4),  // interior
      makeTile(-1, 0, 'p1', 3),  // enemy creating the frontier
    ])
    const routes = buildInteriorRoutes(board, 'p0', new Set(['0,0']))
    expect(routes.get('1,0')).toBe('0,0')
    expect(routes.get('0,1')).toBe('0,0')
    expect(routes.get('-1,1')).toBe('0,0')
  })

  it('routes a 3-tile chain: deep interior → interior → front', () => {
    // (-2,0) → (-1,0) → (0,0) [front] ← (1,0) [enemy]
    const board = makeBoard([
      makeTile(-2, 0, 'p0', 2),  // deep interior
      makeTile(-1, 0, 'p0', 3),  // interior
      makeTile(0, 0, 'p0', 5),   // frontier (active)
      makeTile(1, 0, 'p1', 4),
    ])
    const routes = buildInteriorRoutes(board, 'p0', new Set(['0,0']))
    expect(routes.get('-1,0')).toBe('0,0')
    expect(routes.get('-2,0')).toBe('-1,0')
  })
})
