import type { Board } from '../types/board'
import type { Player, PlayerId } from '../types/player'
import type { OrderMap } from '../types/orders'
import type { AnimationEvent } from '../types/animation'
import type { PlayerStats } from '../types/stats'
import { resolveCombat, applyCombatResult } from './combat'
import { checkWin, checkEliminations } from './winCondition'
import { hexToKey, hexNeighbors } from '../types/hex'

export interface TurnStep {
  event: AnimationEvent
  boardAfter: Board
  playersAfter: Player[]
  winnerAfter: PlayerId | null
  runningStatsAfter: Map<PlayerId, PlayerStats>
}

export interface TurnResolutionResult {
  board: Board
  players: Player[]
  steps: TurnStep[]
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
  const steps: TurnStep[] = []
  const updatedStats = new Map(runningStats)

  for (const [fromKey, order] of orders) {
    const fromTile = currentBoard.get(fromKey)
    if (!fromTile || fromTile.owner !== playerId || fromTile.units === 0) continue

    const toTile = currentBoard.get(order.toKey)
    if (!toTile) continue

    const neighbors = hexNeighbors(fromTile.coord).map(hexToKey)
    if (!neighbors.includes(order.toKey)) continue

    const result = resolveCombat(currentBoard, order, playerId)
    if (result.unitsSent === 0) continue

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

    const kind = result.defenderCasualties > 0
      ? (result.conquered ? 'conquer' : 'fight')
      : 'move'

    const event: AnimationEvent = {
      kind,
      fromKey: order.fromKey,
      toKey: order.toKey,
      playerId,
      units: result.unitsSent,
      durationMs: kind === 'move' ? 220 : 380,
    }
    animationEvents.push(event)

    // Check win/eliminations after applying this move
    let winnerAfter: PlayerId | null = null
    if (result.conquered) {
      winnerAfter = checkWin(currentBoard, currentPlayers)
      if (!winnerAfter) {
        const newlyEliminated = checkEliminations(currentBoard, currentPlayers)
        if (newlyEliminated.length > 0) {
          currentPlayers = currentPlayers.map(p =>
            newlyEliminated.includes(p.id) ? { ...p, isEliminated: true } : p
          )
        }
      }
    }

    steps.push({
      event,
      boardAfter: new Map(currentBoard),
      playersAfter: [...currentPlayers],
      winnerAfter,
      runningStatsAfter: new Map(updatedStats),
    })

    if (winnerAfter) {
      return { board: currentBoard, players: currentPlayers, steps, animationEvents, winnerId: winnerAfter, runningStats: updatedStats }
    }
  }

  return { board: currentBoard, players: currentPlayers, steps, animationEvents, winnerId: null, runningStats: updatedStats }
}
