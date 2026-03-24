import type { Board, PlayerId, Player, OrderMap } from '@hexwar/engine'
import { ConquerorAI } from '@hexwar/conqueror'

// One ConquerorAI instance per player — maintains per-player state across turns.
const agents = new Map<PlayerId, ConquerorAI>()

function getAgent(playerId: PlayerId): ConquerorAI {
  if (!agents.has(playerId)) agents.set(playerId, new ConquerorAI())
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
