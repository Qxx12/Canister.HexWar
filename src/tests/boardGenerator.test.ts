import { describe, it, expect } from 'vitest'
import { generateBoard } from '@hexwar/engine'
import { hexNeighbors, hexToKey } from '@hexwar/engine'

const playerIds = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']

// Seeded RNG for deterministic tests
function seededRng(seed: number) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
}

describe('generateBoard', () => {
  it('generates a board with tiles', () => {
    const board = generateBoard(playerIds, seededRng(42))
    expect(board.size).toBeGreaterThan(50)
  })

  it('is connected', () => {
    const board = generateBoard(playerIds, seededRng(42))
    const keys = new Set(board.keys())
    const start = keys.values().next().value as string
    const visited = new Set<string>([start])
    const queue = [start]
    while (queue.length > 0) {
      const key = queue.shift()!
      const [q, r] = key.split(',').map(Number)
      for (const neighbor of hexNeighbors({ q, r })) {
        const nKey = hexToKey(neighbor)
        if (keys.has(nKey) && !visited.has(nKey)) {
          visited.add(nKey)
          queue.push(nKey)
        }
      }
    }
    expect(visited.size).toBe(keys.size)
  })

  it('places exactly 6 start tiles', () => {
    const board = generateBoard(playerIds, seededRng(42))
    const startTiles = [...board.values()].filter(t => t.isStartTile)
    expect(startTiles).toHaveLength(6)
  })

  it('each player has exactly 1 start tile', () => {
    const board = generateBoard(playerIds, seededRng(42))
    for (const id of playerIds) {
      const tiles = [...board.values()].filter(t => t.startOwner === id)
      expect(tiles).toHaveLength(1)
    }
  })

  it('start tiles have 1 unit and are owned by their player', () => {
    const board = generateBoard(playerIds, seededRng(42))
    const startTiles = [...board.values()].filter(t => t.isStartTile)
    for (const tile of startTiles) {
      expect(tile.units).toBe(1)
      expect(tile.owner).toBe(tile.startOwner)
    }
  })
})
