import type { Board } from '../types/board'
import type { Player, PlayerId } from '../types/player'
import type { OrderMap } from '../types/orders'
import type { AnimationEvent } from '../types/animation'
import type { PlayerStats } from '../types/stats'
import { resolveCombat, applyCombatResult } from './combat'
import { checkWin, checkEliminations } from './winCondition'
import { hexToKey, hexNeighbors } from '../types/hex'

export interface TurnResolutionResult {
  board: Board
  players: Player[]
  animationEvents: AnimationEvent[]
  winnerId: PlayerId | null
  runningStats: Map<PlayerId, PlayerStats>
}

export function resolvePlayerTurn(
  board: Board,
  players: Player[],
  orders: OrderMap,
  playerId: PlayerId,
  runningStats: Map<PlayerId, PlayerStats>,
): TurnResolutionResult {
  let currentBoard = new Map(board)
  let currentPlayers = [...players]
  const animationEvents: AnimationEvent[] = []
  const updatedStats = new Map(runningStats)

  // Process each order for this player
  for (const [fromKey, order] of orders) {
    const fromTile = currentBoard.get(fromKey)
    // Skip if tile no longer owned by this player or has no units
    if (!fromTile || fromTile.owner !== playerId || fromTile.units === 0) continue

    // Validate destination is still adjacent
    const toTile = currentBoard.get(order.toKey)
    if (!toTile) continue

    // Verify adjacency
    const neighbors = hexNeighbors(fromTile.coord).map(hexToKey)
    if (!neighbors.includes(order.toKey)) continue

    const result = resolveCombat(currentBoard, order, playerId)
    if (result.unitsSent === 0) continue

    // Update attacker kills stats
    if (result.defenderCasualties > 0 && result.defendingPlayerId) {
      const attackerStats = updatedStats.get(playerId)
      if (attackerStats) {
        updatedStats.set(playerId, {
          ...attackerStats,
          unitsKilled: attackerStats.unitsKilled + result.defenderCasualties,
        })
      }
      if (result.conquered) {
        const aStats = updatedStats.get(playerId)
        if (aStats) {
          updatedStats.set(playerId, { ...aStats, tilesConquered: aStats.tilesConquered + 1 })
        }
      }
    } else if (result.conquered && result.defendingPlayerId === null) {
      const aStats = updatedStats.get(playerId)
      if (aStats) {
        updatedStats.set(playerId, { ...aStats, tilesConquered: aStats.tilesConquered + 1 })
      }
    }

    currentBoard = applyCombatResult(currentBoard, result)

    // Determine animation kind
    const kind = result.defenderCasualties > 0
      ? (result.conquered ? 'conquer' : 'fight')
      : 'move'

    animationEvents.push({
      kind,
      fromKey: order.fromKey,
      toKey: order.toKey,
      playerId,
      units: result.unitsSent,
      durationMs: kind === 'move' ? 220 : 380,
    })

    // Check win immediately after each conquer
    if (result.conquered) {
      const winnerId = checkWin(currentBoard, currentPlayers)
      if (winnerId) {
        return { board: currentBoard, players: currentPlayers, animationEvents, winnerId, runningStats: updatedStats }
      }
      // Check eliminations
      const newlyEliminated = checkEliminations(currentBoard, currentPlayers)
      if (newlyEliminated.length > 0) {
        currentPlayers = currentPlayers.map(p =>
          newlyEliminated.includes(p.id) ? { ...p, isEliminated: true } : p
        )
      }
    }
  }

  return { board: currentBoard, players: currentPlayers, animationEvents, winnerId: null, runningStats: updatedStats }
}
