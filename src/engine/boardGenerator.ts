import type { Board, Tile, TerrainType } from '../types/board'
import type { AxialCoord } from '../types/hex'
import { hexToKey, hexNeighbors, hexDistance } from '../types/hex'
import type { PlayerId } from '../types/player'

const BOARD_SIZE = 12
const TARGET_TILES = Math.floor(BOARD_SIZE * BOARD_SIZE * 0.65) // ~166 tiles

function generateBlob(rng: () => number): Set<string> {
  const center: AxialCoord = { q: 0, r: 0 }
  const visited = new Set<string>()
  const queue: AxialCoord[] = [center]
  visited.add(hexToKey(center))

  while (visited.size < TARGET_TILES && queue.length > 0) {
    const idx = Math.floor(rng() * queue.length)
    const [current] = queue.splice(idx, 1)

    for (const neighbor of hexNeighbors(current)) {
      const key = hexToKey(neighbor)
      if (!visited.has(key) &&
          Math.abs(neighbor.q) <= BOARD_SIZE / 2 &&
          Math.abs(neighbor.r) <= BOARD_SIZE / 2 &&
          Math.abs(neighbor.q + neighbor.r) <= BOARD_SIZE / 2) {
        visited.add(key)
        queue.push(neighbor)
        if (visited.size >= TARGET_TILES) break
      }
    }
  }
  return visited
}

function isConnected(tileKeys: Set<string>): boolean {
  if (tileKeys.size === 0) return true
  const start = tileKeys.values().next().value as string
  const visited = new Set<string>([start])
  const queue = [start]
  while (queue.length > 0) {
    const key = queue.shift()!
    const [q, r] = key.split(',').map(Number)
    for (const neighbor of hexNeighbors({ q, r })) {
      const nKey = hexToKey(neighbor)
      if (tileKeys.has(nKey) && !visited.has(nKey)) {
        visited.add(nKey)
        queue.push(nKey)
      }
    }
  }
  return visited.size === tileKeys.size
}

function placeStartTiles(tileKeys: string[], playerIds: PlayerId[], rng: () => number): string[] {
  const count = playerIds.length
  // Try many random candidate sets, pick best by minimum pairwise distance
  let bestSet: string[] = []
  let bestMinDist = -1

  for (let attempt = 0; attempt < 200; attempt++) {
    // Shuffle and pick first `count` tiles from shuffled list
    const shuffled = [...tileKeys].sort(() => rng() - 0.5)
    const candidates = shuffled.slice(0, count)
    let minDist = Infinity
    for (let i = 0; i < candidates.length; i++) {
      for (let j = i + 1; j < candidates.length; j++) {
        const [aq, ar] = candidates[i].split(',').map(Number)
        const [bq, br] = candidates[j].split(',').map(Number)
        const d = hexDistance({ q: aq, r: ar }, { q: bq, r: br })
        if (d < minDist) minDist = d
      }
    }
    if (minDist > bestMinDist) {
      bestMinDist = minDist
      bestSet = candidates
    }
  }
  return bestSet
}

function terrainFor(q: number, r: number): TerrainType {
  // Latitude: 0 at equator, 1 at poles (based on r, with slight q-based noise)
  const lat = Math.abs(r) / (BOARD_SIZE / 2) + Math.sin(q * 0.8) * 0.08
  if (lat > 0.75) return 'tundra'
  if (lat > 0.45) return 'grassland'
  if (lat > 0.2)  return 'plains'
  return 'desert'
}

export function generateBoard(playerIds: PlayerId[], rng: () => number = Math.random): Board {
  const tileKeys = generateBlob(rng)
  const board: Board = new Map()

  for (const key of tileKeys) {
    const [q, r] = key.split(',').map(Number)
    const tile: Tile = {
      coord: { q, r },
      owner: null,
      units: 0,
      isStartTile: false,
      startOwner: null,
      terrain: terrainFor(q, r),
      newlyConquered: false,
    }
    board.set(key, tile)
  }

  const tileKeyList = Array.from(tileKeys)
  const startKeys = placeStartTiles(tileKeyList, playerIds, rng)

  startKeys.forEach((key, index) => {
    const tile = board.get(key)!
    board.set(key, {
      ...tile,
      isStartTile: true,
      startOwner: playerIds[index],
      owner: playerIds[index],
      units: 1,
    })
  })

  return board
}
