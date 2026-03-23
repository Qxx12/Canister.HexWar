import { describe, it, expect } from 'vitest'
import { WarlordAI } from '../index.ts'
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

function tile(q: number, r: number, owner: string | null, units: number, isStartTile = false): Tile {
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
  for (let i = 0; i < 20; i++) {
    extra.push(tile(1000 + i, 0, 'p1', 1))
  }
  return [...tiles, ...extra]
}

const players: Player[] = [
  { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
  { id: 'p1', type: 'ai',   color: '#f00', name: 'P1', isEliminated: false },
]

const ai = new WarlordAI()

describe('WarlordAI', () => {
  // ── Basic sanity ─────────────────────────────────────────────────────────────

  it('issues no order when isolated (no neighbours)', () => {
    const board = makeBoard([tile(0, 0, 'p1', 5)])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })

  it('issues no order for tiles with 0 units', () => {
    const board = makeBoard([
      tile(0, 0, 'p1', 0),
      tile(1, 0, null, 0),
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })

  it('returns an OrderMap instance', () => {
    const board = makeBoard([
      tile(0, 0, 'p1', 5),
      tile(1, 0, null, 0),
    ])
    expect(ai.computeOrders(board, 'p1', new Map(), players)).toBeInstanceOf(Map)
  })

  // ── Phase scoring ────────────────────────────────────────────────────────────

  it('early phase: prefers neutral (score 55) over barely-beatable enemy (score 53)', () => {
    // 1 p1 tile → earlyPhase=true. Neutral=55 > enemy adv=1 → 50+3=53.
    // Neutral is at (0,-1) which is NOT adjacent to the enemy at (1,0), so no +8 bonus applies.
    const board = makeBoard([
      tile(0, 0, 'p1', 5),       // source
      tile(1, 0, 'p0', 4),       // enemy: adv=1 → score 53
      tile(0, -1, null, 0),      // neutral → score 55 (not adjacent to enemy)
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('0,-1')
  })

  it('late phase: prefers beatable enemy (score 53) over neutral (score 35)', () => {
    // 20+ p1 tiles → earlyPhase=false. Neutral=35 < enemy adv=1 → 53.
    const board = makeBoard(withLatePhase([
      tile(0, 0, 'p1', 5),       // source
      tile(1, 0, 'p0', 4),       // enemy: adv=1 → score 53
      tile(0, 1, null, 0),       // neutral → score 35
    ]))
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
  })

  it('prefers hostile start tile over unconquered neutral', () => {
    const board = makeBoard([
      tile(0, 0, 'p1', 10),
      tile(1, 0, 'p0', 3, true), // enemy start tile: score 50+21+45=116
      tile(0, 1, null, 0),       // neutral: score 55
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('1,0')
  })

  it('gambles all units on an enemy start tile even when outnumbered', () => {
    const board = makeBoard([
      tile(0, 0, 'p1', 3),
      tile(1, 0, 'p0', 5, true), // stronger enemy start → score 18
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(3)
  })

  it('does not attack a stronger non-start enemy tile', () => {
    const board = makeBoard([
      tile(0, 0, 'p1', 3),
      tile(1, 0, 'p0', 5), // stronger, not a start tile → score -1
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.size).toBe(0)
  })

  // ── Neutral-adjacent enemy bonus ─────────────────────────────────────────────

  it('prefers enemy tile adjacent to neutral territory (+8 bonus per neutral neighbour)', () => {
    // Both enemies equally scored (adv=5 → 65), but only enemy B has a neutral neighbour.
    // Neutral at (0,2) is a neighbour of enemy B=(0,1) but NOT of enemy A=(1,0) or source=(0,0).
    const board = makeBoard([
      tile(0, 0, 'p1', 10),      // source
      tile(1, 0, 'p0', 5),       // enemy A: score 65, no neutral neighbours
      tile(0, 1, 'p0', 5),       // enemy B: score 65 + 8 = 73 (neutral at 0,2 is its neighbour)
      tile(0, 2, null, 0),       // neutral adjacent to enemy B only
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('0,1') // enemy B preferred
  })

  // ── Garrison ─────────────────────────────────────────────────────────────────

  it('sends all units to neutral when no adjacent enemy (garrison=0)', () => {
    // Interior p1 tile with only a neutral neighbour — no garrison pressure.
    const board = makeBoard(withLatePhase([
      tile(0, 0, 'p1', 10),
      tile(1, 0, null, 2),       // neutral — only neighbour
    ]))
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(10)
  })

  it('blocks neutral attack when strong adjacent enemy consumes all available units', () => {
    // source=10, enemy=15 (stronger) → garrison=15, available=0 → no neutral attack.
    const board = makeBoard(withLatePhase([
      tile(0, 0, 'p1', 10),
      tile(1, 0, 'p0', 15),      // too strong to beat, but sets garrison=15
      tile(0, 1, null, 0),       // neutral: only other option, but available=0
    ]))
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')).toBeUndefined()
  })

  // ── Routing ──────────────────────────────────────────────────────────────────

  it('routes interior tile toward the nearest frontier', () => {
    // (0,0) interior → (1,0) frontier → (2,0) enemy
    const board = makeBoard([
      tile(0, 0, 'p1', 5),       // interior
      tile(1, 0, 'p1', 5),       // frontier
      tile(2, 0, 'p0', 2),       // enemy
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.toKey).toBe('1,0')   // interior → frontier
    expect(orders.get('1,0')!.toKey).toBe('2,0')   // frontier → attack
  })

  // ── Enemy attack units ────────────────────────────────────────────────────────

  it('sends computed units when attacking an enemy with advantage (ceil×0.85)', () => {
    // 10 units vs enemy 3 → max(minToConquer=4, ceil(10×0.85)=9) = 9
    const board = makeBoard([
      tile(0, 0, 'p1', 10),
      tile(1, 0, 'p0', 3),
    ])
    const orders = ai.computeOrders(board, 'p1', new Map(), players)
    expect(orders.get('0,0')!.requestedUnits).toBe(9)
  })
})
