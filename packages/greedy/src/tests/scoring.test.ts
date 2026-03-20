import { describe, it, expect } from 'vitest'
import { scoreTarget, unitsToSend } from '../scoring.ts'
import type { Tile } from '@hexwar/engine'

function makeTile(owner: string | null, units: number, isStartTile = false): Tile {
  return {
    coord: { q: 0, r: 0 },
    owner,
    units,
    isStartTile,
    startOwner: isStartTile ? owner : null,
    terrain: 'plains' as const,
    newlyConquered: false,
  }
}

// ─── scoreTarget ──────────────────────────────────────────────────────────────

describe('scoreTarget — neutral tiles', () => {
  it('neutral plain tile scores 35', () => {
    expect(scoreTarget(5, makeTile(null, 0))).toBe(35)
  })

  it('neutral start tile scores 35 + 45 = 80', () => {
    expect(scoreTarget(5, makeTile(null, 0, true))).toBe(80)
  })
})

describe('scoreTarget — losing or tied fights', () => {
  it('scores -1 when attacker would lose (myUnits < target.units)', () => {
    expect(scoreTarget(3, makeTile('p1', 5))).toBe(-1)
  })

  it('scores -1 on a tie (myUnits === target.units)', () => {
    expect(scoreTarget(5, makeTile('p1', 5))).toBe(-1)
  })

  it('scores 18 on an enemy start tile even when losing (gamble value)', () => {
    expect(scoreTarget(3, makeTile('p1', 5, true))).toBe(18)
  })

  it('scores 18 on an enemy start tile when tied', () => {
    expect(scoreTarget(5, makeTile('p1', 5, true))).toBe(18)
  })
})

describe('scoreTarget — winning fights', () => {
  it('scores 50 + advantage*3 for a plain enemy tile', () => {
    // advantage = 10 - 2 = 8 → 50 + 24 = 74
    expect(scoreTarget(10, makeTile('p1', 2))).toBe(74)
  })

  it('scores 50 + advantage*3 + 45 for an enemy start tile', () => {
    // advantage = 10 - 2 = 8 → 50 + 24 + 45 = 119
    expect(scoreTarget(10, makeTile('p1', 2, true))).toBe(119)
  })

  it('scales linearly with advantage', () => {
    const base = makeTile('p1', 5)
    expect(scoreTarget(6,  base)).toBe(50 + 1 * 3)   // 53
    expect(scoreTarget(8,  base)).toBe(50 + 3 * 3)   // 59
    expect(scoreTarget(14, base)).toBe(50 + 9 * 3)   // 77
  })

  it('1 unit advantage over 0-unit enemy scores 50 + 3', () => {
    expect(scoreTarget(1, makeTile('p1', 0))).toBe(53)
  })
})

// ─── unitsToSend ──────────────────────────────────────────────────────────────

describe('unitsToSend — neutral targets', () => {
  it('sends all units to a neutral tile', () => {
    expect(unitsToSend(10, makeTile(null, 0))).toBe(10)
    expect(unitsToSend(1,  makeTile(null, 0))).toBe(1)
  })
})

describe('unitsToSend — enemy targets, winning', () => {
  it('sends max(minToConquer, floor(myUnits*0.85))', () => {
    // 10 vs 3: minToConquer=4, comfortable=8 → max(4,8)=8
    expect(unitsToSend(10, makeTile('p1', 3))).toBe(8)
    // 4 vs 3: minToConquer=4, comfortable=3 → max(4,3)=4
    expect(unitsToSend(4, makeTile('p1', 3))).toBe(4)
  })

  it('1 unit advantage over 0-unit enemy: sends 1', () => {
    // minToConquer=1, comfortable=floor(1*0.85)=0 → max(1,0)=1
    expect(unitsToSend(1, makeTile('p1', 0))).toBe(1)
  })

  it('large advantage: comfortable threshold dominates', () => {
    // 50 vs 1: minToConquer=2, comfortable=42 → max(2,42)=42
    expect(unitsToSend(50, makeTile('p1', 1))).toBe(42)
  })
})

describe('unitsToSend — enemy targets, losing or tied', () => {
  it('commits all units when tied (desperate attack)', () => {
    expect(unitsToSend(5, makeTile('p1', 5))).toBe(5)
  })

  it('commits all units when losing (desperate attack)', () => {
    expect(unitsToSend(3, makeTile('p1', 8))).toBe(3)
  })
})
