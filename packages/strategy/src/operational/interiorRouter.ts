import type { Board, PlayerId } from '@hexwar/engine'
import { hexNeighbors, hexToKey } from '@hexwar/engine'

/**
 * BFS outward from the set of active front tiles through friendly territory.
 *
 * Returns a routing table: interiorKey → nextHopKey one step closer to the
 * nearest active front. Interior tiles follow this table to pipeline units
 * toward where they're needed.
 *
 * If activeFrontTiles is empty (no active offensive), returns an empty map
 * (interior units stay put this turn).
 */
export function buildInteriorRoutes(
  board: Board,
  myPlayerId: PlayerId,
  activeFrontTiles: Set<string>,
): Map<string, string> {
  if (activeFrontTiles.size === 0) return new Map()

  const routes = new Map<string, string>()
  const visited = new Set<string>(activeFrontTiles)
  const queue: string[] = [...activeFrontTiles]
  let qi = 0

  while (qi < queue.length) {
    const key = queue[qi++]
    const tile = board.get(key)
    if (!tile) continue

    for (const nCoord of hexNeighbors(tile.coord)) {
      const nKey = hexToKey(nCoord)
      if (visited.has(nKey)) continue

      const nTile = board.get(nKey)
      if (!nTile || nTile.owner !== myPlayerId) continue

      visited.add(nKey)
      routes.set(nKey, key) // nKey → key brings us one step toward the front
      queue.push(nKey)
    }
  }

  return routes
}
