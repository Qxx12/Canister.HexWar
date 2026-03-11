import { describe, it, expect } from 'vitest'
import { hexNeighbors, hexDistance, hexEquals, hexToKey, keyToHex, axialToPixel, hexCorners } from '@hexwar/engine'

describe('hexNeighbors', () => {
  it('returns 6 neighbors', () => {
    expect(hexNeighbors({ q: 0, r: 0 })).toHaveLength(6)
  })

  it('returns correct neighbors for origin', () => {
    const neighbors = hexNeighbors({ q: 0, r: 0 })
    expect(neighbors).toContainEqual({ q: 1, r: 0 })
    expect(neighbors).toContainEqual({ q: -1, r: 0 })
    expect(neighbors).toContainEqual({ q: 0, r: 1 })
    expect(neighbors).toContainEqual({ q: 0, r: -1 })
  })
})

describe('hexDistance', () => {
  it('returns 0 for same tile', () => {
    expect(hexDistance({ q: 0, r: 0 }, { q: 0, r: 0 })).toBe(0)
  })

  it('returns 1 for adjacent tiles', () => {
    expect(hexDistance({ q: 0, r: 0 }, { q: 1, r: 0 })).toBe(1)
    expect(hexDistance({ q: 0, r: 0 }, { q: 0, r: 1 })).toBe(1)
  })

  it('returns correct distance', () => {
    expect(hexDistance({ q: 0, r: 0 }, { q: 3, r: -3 })).toBe(3)
    expect(hexDistance({ q: 0, r: 0 }, { q: 2, r: 1 })).toBe(3)
  })
})

describe('hexEquals', () => {
  it('returns true for same coord', () => {
    expect(hexEquals({ q: 1, r: 2 }, { q: 1, r: 2 })).toBe(true)
  })
  it('returns false for different coord', () => {
    expect(hexEquals({ q: 1, r: 2 }, { q: 1, r: 3 })).toBe(false)
  })
})

describe('hexToKey / keyToHex', () => {
  it('round-trips correctly', () => {
    const coord = { q: 3, r: -5 }
    expect(keyToHex(hexToKey(coord))).toEqual(coord)
  })
})

describe('axialToPixel', () => {
  it('returns origin for (0,0)', () => {
    const pos = axialToPixel({ q: 0, r: 0 }, 40)
    expect(pos.x).toBe(0)
    expect(pos.y).toBe(0)
  })
})

describe('hexCorners', () => {
  it('returns 6 corners', () => {
    expect(hexCorners(0, 0, 10)).toHaveLength(6)
  })

  it('each corner is at the correct distance from center', () => {
    const size = 20
    const corners = hexCorners(0, 0, size)
    for (const { x, y } of corners) {
      const dist = Math.sqrt(x * x + y * y)
      expect(dist).toBeCloseTo(size, 5)
    }
  })

  it('offsets corners by the given center', () => {
    const centered = hexCorners(0, 0, 10)
    const shifted  = hexCorners(100, 50, 10)
    for (let i = 0; i < 6; i++) {
      expect(shifted[i].x).toBeCloseTo(centered[i].x + 100, 5)
      expect(shifted[i].y).toBeCloseTo(centered[i].y + 50,  5)
    }
  })
})
