import { describe, it, expect } from 'vitest'
import { ConquerorAI } from '../index.ts'
import type { Board, Tile } from '@hexwar/engine'
import type { Player } from '@hexwar/engine'

// Hex neighbours of (q,r): (q+1,r),(q-1,r),(q,r+1),(q,r-1),(q+1,r-1),(q-1,r+1)

function makeBoard(tiles: Tile[]): Board {
  const board: Board = new Map()
  for (const tile of tiles) {
    board.set(`${tile.coord.q},${tile.coord.r}`, tile)
  }
  return board
}

function tile(
  q: number, r: number, owner: string | null, units: number,
  isStartTile = false,
): Tile {
  return {
    coord: { q, r },
    owner: owner as Tile['owner'],
    units,
    isStartTile,
    startOwner: isStartTile ? owner as Tile['owner'] : null,
    terrain: 'plains' as const,
    newlyConquered: false,
  }
}

/** Add enough extra p1 tiles (far away) to push myTileCount ≥ 20 (late phase). */
function withLatePhase(tiles: Tile[]): Tile[] {
  const extra: Tile[] = []
  for (let i = 0; i < 20; i++) extra.push(tile(1000 + i, 0, 'p1', 1))
  return [...tiles, ...extra]
}

const players: Player[] = [
  { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
  { id: 'p1', type: 'ai',   color: '#f00', name: 'P1', isEliminated: false },
  { id: 'p2', type: 'ai',   color: '#00f', name: 'P2', isEliminated: false },
]

const ai = new ConquerorAI()

describe('ConquerorAI', () => {
  // ── Basic sanity ─────────────────────────────────────────────────────────────

  it('issues no order when isolated (no neighbours)', () => {
    const board = makeBoard([tile(0, 0, 'p1', 5)])
    expect(ai.computeOrders(board, 'p1', new Map(), players).size).toBe(0)
  })

  it('issues no order for tiles with 0 units', () => {
    const board = makeBoard([tile(0, 0, 'p1', 0), tile(1, 0, null, 0)])
    expect(ai.computeOrders(board, 'p1', new Map(), players).size).toBe(0)
  })

  it('returns an OrderMap instance', () => {
    const board = makeBoard([tile(0, 0, 'p1', 5), tile(1, 0, null, 0)])
    expect(ai.computeOrders(board, 'p1', new Map(), players)).toBeInstanceOf(Map)
  })

  it('does not attack a stronger non-start enemy tile', () => {
    const board = makeBoard([tile(0, 0, 'p1', 3), tile(1, 0, 'p0', 5)])
    expect(ai.computeOrders(board, 'p1', new Map(), players).size).toBe(0)
  })

  // ── Elimination priority ─────────────────────────────────────────────────────

  it('prefers near-elimination player (+60 bonus) over equally-scored regular enemy', () => {
    // p0 has only 2 tiles → near-elim (+60). p2 has many tiles (no bonus).
    // Both enemies have 5 units, source has 10. Base score = 50+15=65.
    // p0: 65+60=125. p2: 65. Expected: attack p0 tile at (1,0).
    const p0ExtraTile = tile(5, 5, 'p0', 1)   // p0 total = 2 tiles → ≤ ELIM_THRESHOLD(4)
    const p2Tiles = Array.from({ length: 10 }, (_, i) => tile(100 + i, 0, 'p2', 1))
    const board = makeBoard([
      tile(0, 0, 'p1', 10),
      tile(1, 0, 'p0', 5),   // near-elim enemy
      tile(0, -1, 'p2', 5),  // not near-elim; (0,-1) not adj to (1,0)
      p0ExtraTile,
      ...p2Tiles,
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
  })

  it('gambles all units on a near-elimination start tile even when outnumbered', () => {
    // p0 has 2 tiles → near-elim. Start tile with more units → gamble.
    const board = makeBoard([
      tile(0, 0, 'p1', 3),
      tile(1, 0, 'p0', 5, true), // near-elim start tile
      tile(5, 5, 'p0', 1),       // p0 second tile
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(3)
  })

  // ── Path bonus (start-tile gradient) ─────────────────────────────────────────

  it('prefers enemy adjacent to unowned start tile (+20 path bonus)', () => {
    // Unowned start tile at (2,0). Enemy A at (1,0) is adjacent to it (dist=1) → +20.
    // Enemy B at (0,-1) is not adjacent to the start tile → no path bonus.
    // Both enemies have equal base score (50+15=65). Expected: attack (1,0).
    const board = makeBoard([
      tile(0, 0, 'p1', 10),
      tile(1, 0, 'p0', 5),       // adj to start tile at (2,0) → +20
      tile(0, -1, 'p0', 5),      // not adj to (2,0); not adj to (1,0) for neutral-adj bonus
      tile(2, 0, null, 0, true), // unowned start tile — gradient seed
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
  })

  it('neutral tile adjacent to unowned start tile gets path bonus', () => {
    // Neutral at (1,0) is adjacent to unowned start tile at (2,0) → score 55+20=75.
    // Neutral at (0,-1) has no path bonus → score 55.
    const board = makeBoard([
      tile(0, 0, 'p1', 5),
      tile(1, 0, null, 0),       // neutral adj to start tile
      tile(0, -1, null, 0),      // neutral not adj to start tile
      tile(2, 0, null, 0, true), // unowned start tile
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
  })

  // ── Garrison ─────────────────────────────────────────────────────────────────

  it('garrison is 0 in early phase regardless of adjacent enemy strength', () => {
    // Only 1 p1 tile → early phase. Source has 10 units, enemy adj has 15.
    // garrison = 0 → available = 10 → sends all to neutral.
    const board = makeBoard([
      tile(0, 0, 'p1', 10),
      tile(1, 0, 'p0', 15),  // strong enemy — sets garrison in late phase
      tile(0, -1, null, 0),  // neutral — only target if available > 0
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    // In early phase both neutral and enemy should score; with 10 vs 15 → skip enemy,
    // go neutral. Available = 10 → sends 10.
    expect(orders.get('0,0')!.requestedUnits).toBe(10)
  })

  it('late phase garrison = ceil(maxAdjEnemy × 0.6) for ordinary tiles', () => {
    // Late phase (20+ tiles). Source=10 units, max adj enemy=10.
    // garrison = ceil(10 × 0.6) = 6, available = 4 → sends 4 to neutral.
    const board = makeBoard(withLatePhase([
      tile(0, 0, 'p1', 10),
      tile(1, 0, 'p0', 10),  // garrison pressure
      tile(0, -1, null, 0),  // neutral; (0,-1) not adj to (1,0)
    ]))
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(4)
  })

  it('own start tile uses 1.0× garrison — never depleted for a neutral grab', () => {
    // Late phase. Our start tile (0,0) has 10 units, adj enemy has 10.
    // garrison = ceil(10 × 1.0) = 10 → available = 0 → no neutral attack.
    const board = makeBoard(withLatePhase([
      tile(0, 0, 'p1', 10, true), // our own start tile
      tile(1, 0, 'p0', 10),       // garrison pressure
      tile(0, -1, null, 0),       // neutral
    ]))
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    // available = 0, so no order is issued from our start tile
    expect(orders.get('0,0')).toBeUndefined()
  })

  // ── Coordinated attacks ───────────────────────────────────────────────────────

  it('coordinates two blocked tiles to conquer an enemy neither could beat alone', () => {
    // Both p1 tiles at (0,0) and (0,1) have 6 units; enemy at (1,0) has 8 units.
    // Neither can win solo (6 ≤ 8). Combined = 12 > 8 → joint attack.
    // After bleeder (smaller available) sends 6: enemy has 2 left.
    // Killer (6 units > 2) conquers.
    // hex neighbours of (0,0): (1,0),(−1,0),(0,1),(0,−1),(1,−1),(−1,1)
    // hex neighbours of (0,1): (1,1),(−1,1),(0,2),(0,0),(1,0),(−1,2)
    // Both (0,0) and (0,1) are adjacent to (1,0). ✓
    const board = makeBoard([
      tile(0, 0, 'p1', 6),  // blocked frontier — no solo target
      tile(0, 1, 'p1', 6),  // blocked frontier — no solo target
      tile(1, 0, 'p0', 8),  // too strong for either alone
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    // Both tiles should attack (1,0)
    expect(orders.get('0,0')?.toKey).toBe('1,0')
    expect(orders.get('0,1')?.toKey).toBe('1,0')
  })

  it('does not coordinate when combined force still insufficient', () => {
    // 4 + 4 = 8 ≤ 9 (enemy) → no joint attack possible
    const board = makeBoard([
      tile(0, 0, 'p1', 4),
      tile(0, 1, 'p1', 4),
      tile(1, 0, 'p0', 9),
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    // Neither attacks (1,0)
    expect(orders.get('0,0')?.toKey).not.toBe('1,0')
    expect(orders.get('0,1')?.toKey).not.toBe('1,0')
  })

  // ── Routing ──────────────────────────────────────────────────────────────────

  it('routes interior tile toward the nearest frontier', () => {
    // (0,0) interior → (1,0) frontier → (2,0) enemy
    const board = makeBoard([
      tile(0, 0, 'p1', 5),   // interior
      tile(1, 0, 'p1', 5),   // frontier
      tile(2, 0, 'p0', 2),   // enemy
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
    expect(orders.get('1,0')!.toKey).toBe('2,0')
  })

  // ── Unit commitment ───────────────────────────────────────────────────────────

  it('sends computed units when attacking a beatable enemy (ceil×0.85)', () => {
    // 10 units vs enemy 3 → max(4, ceil(10×0.85)=9) = 9
    const board = makeBoard([tile(0, 0, 'p1', 10), tile(1, 0, 'p0', 3)])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(9)
  })
})
