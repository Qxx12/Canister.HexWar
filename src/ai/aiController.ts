import type { Board, PlayerId, Player, OrderMap } from '@hexwar/engine'
import { WarlordAI } from '@hexwar/warlord'

// One WarlordAI instance per player — maintains per-player state across turns.
const agents = new Map<PlayerId, WarlordAI>()

function getAgent(playerId: PlayerId): WarlordAI {
  if (!agents.has(playerId)) agents.set(playerId, new WarlordAI())
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
