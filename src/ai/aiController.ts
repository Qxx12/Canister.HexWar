import type { Board, PlayerId, Player, OrderMap } from '@hexwar/engine'
import { HighCommandAI } from '@hexwar/strategy'

// One HighCommandAI instance per player — maintains per-player history across turns.
const agents = new Map<PlayerId, HighCommandAI>()

function getAgent(playerId: PlayerId): HighCommandAI {
  if (!agents.has(playerId)) agents.set(playerId, new HighCommandAI())
  return agents.get(playerId)!
}

export function computeAiOrders(
  board: Board,
  playerId: PlayerId,
  currentOrders: OrderMap,
  allPlayers: Player[],
): OrderMap {
  return getAgent(playerId).computeOrders(board, playerId, currentOrders, allPlayers)
}

/** Call when the game resets so history does not bleed across sessions. */
export function resetAiAgents(): void {
  agents.clear()
}
