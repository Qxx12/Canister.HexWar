import type { Board } from '@hexwar/engine'
import type { PlayerId, Player } from '@hexwar/engine'
import type { OrderMap } from '@hexwar/engine'
import { GreedyAI } from './greedyAI'

const strategy = new GreedyAI()

export function computeAiOrders(
  board: Board,
  playerId: PlayerId,
  currentOrders: OrderMap,
  allPlayers: Player[],
): OrderMap {
  return strategy.computeOrders(board, playerId, currentOrders, allPlayers)
}
