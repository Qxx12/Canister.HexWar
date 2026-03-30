import type { Board, PlayerId, Player, OrderMap } from '@hexwar/engine'
import { HighCommandAI } from '@hexwar/strategy'
import { ConquerorAI } from '@hexwar/conqueror'
import { WarlordAI } from '@hexwar/warlord'
import type { AiDifficulty } from '../types/settings'

type AiAgent = {
  computeOrders(board: Board, playerId: PlayerId, currentOrders: OrderMap, allPlayers: Player[]): OrderMap
}

// One agent instance per player — maintains per-player state across turns.
let _difficulty: AiDifficulty = 'commander'
const agents = new Map<PlayerId, AiAgent>()

function createAgent(): AiAgent {
  switch (_difficulty) {
    case 'soldier':   return new HighCommandAI()
    case 'commander': return new ConquerorAI()
    case 'warlord':   return new WarlordAI()
  }
}

function getAgent(playerId: PlayerId): AiAgent {
  if (!agents.has(playerId)) agents.set(playerId, createAgent())
  return agents.get(playerId)!
}

export function setAiDifficulty(difficulty: AiDifficulty): void {
  _difficulty = difficulty
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
