import type { Board, PlayerId, Player } from '@hexwar/engine'
import { hexNeighbors, hexToKey } from '@hexwar/engine'
import type { NeighborAssessment, GeopoliticalSnapshot } from '../types.ts'
import type { HistoryTracker } from './historyTracker.ts'

/**
 * Builds the full geopolitical snapshot for one player by scanning the board.
 *
 * Single-pass: computes tile counts, border units, and adjacency relationships
 * for all players simultaneously, then assembles per-neighbor assessments.
 */
export function buildSnapshot(
  board: Board,
  myPlayerId: PlayerId,
  players: Player[],
  history: HistoryTracker,
): GeopoliticalSnapshot {
  let myTileCount = 0
  let myTotalUnits = 0

  // our border tiles per neighbor: neighborId → our tile keys on that border
  const ourBorderTilesPerNeighbor = new Map<PlayerId, string[]>()
  // their border units facing us
  const theirBorderUnitsPerNeighbor = new Map<PlayerId, number>()
  // total tile count per player
  const tileCounts = new Map<PlayerId, number>()
  // which players are adjacent to which (for isEngagedElsewhere)
  const adjacencies = new Map<PlayerId, Set<PlayerId>>()

  for (const [key, tile] of board) {
    if (tile.owner === null) continue

    tileCounts.set(tile.owner, (tileCounts.get(tile.owner) ?? 0) + 1)

    if (tile.owner === myPlayerId) {
      myTileCount++
      myTotalUnits += tile.units
    }

    for (const nCoord of hexNeighbors(tile.coord)) {
      const nKey = hexToKey(nCoord)
      const nTile = board.get(nKey)
      if (!nTile || nTile.owner === null || nTile.owner === tile.owner) continue

      // Adjacency tracking (all players)
      if (!adjacencies.has(tile.owner)) adjacencies.set(tile.owner, new Set())
      adjacencies.get(tile.owner)!.add(nTile.owner)

      // Our border tiles facing each neighbor
      if (tile.owner === myPlayerId && nTile.owner !== myPlayerId) {
        if (!ourBorderTilesPerNeighbor.has(nTile.owner)) {
          ourBorderTilesPerNeighbor.set(nTile.owner, [])
        }
        const list = ourBorderTilesPerNeighbor.get(nTile.owner)!
        if (!list.includes(key)) list.push(key)
      }

      // Their border units facing us
      if (tile.owner !== myPlayerId && nTile.owner === myPlayerId) {
        theirBorderUnitsPerNeighbor.set(
          tile.owner,
          (theirBorderUnitsPerNeighbor.get(tile.owner) ?? 0) + tile.units,
        )
      }
    }
  }

  const myNeighborIds = adjacencies.get(myPlayerId) ?? new Set()
  const activePlayers = players.filter(p => !p.isEliminated)

  const neighbors: NeighborAssessment[] = []

  for (const neighborId of myNeighborIds) {
    const neighborPlayer = players.find(p => p.id === neighborId)
    if (!neighborPlayer || neighborPlayer.isEliminated) continue

    const sharedBorderTiles = ourBorderTilesPerNeighbor.get(neighborId) ?? []
    const ourBorderUnits = sharedBorderTiles.reduce(
      (sum, k) => sum + (board.get(k)?.units ?? 0), 0,
    )
    const theirBorderUnits = theirBorderUnitsPerNeighbor.get(neighborId) ?? 0
    const relativeStrength = Math.min(3, ourBorderUnits / Math.max(theirBorderUnits, 1))
    const tileCount = tileCounts.get(neighborId) ?? 0
    const momentumDelta = history.getMomentumDelta(neighborId)

    // isEngagedElsewhere: neighbor is adjacent to other players AND losing tiles
    const neighborAdj = adjacencies.get(neighborId) ?? new Set()
    const fightingOthers = [...neighborAdj].some(id => id !== myPlayerId)
    const isEngagedElsewhere = fightingOthers && history.isLosingTiles(neighborId, -1)

    neighbors.push({
      playerId: neighborId,
      sharedBorderTiles,
      ourBorderUnits,
      theirBorderUnits,
      relativeStrength,
      tileCount,
      isEngagedElsewhere,
      momentumDelta,
      isNearlyEliminated: tileCount <= 3,
    })
  }

  return {
    myPlayerId,
    myTileCount,
    myTotalUnits,
    myMomentumDelta: history.getMomentumDelta(myPlayerId),
    neighbors,
    totalActivePlayers: activePlayers.length,
  }
}
