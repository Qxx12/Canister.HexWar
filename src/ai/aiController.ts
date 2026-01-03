import type { Board } from '../types/board'
import type { PlayerId, Player } from '../types/player'
import type { OrderMap } from '../types/orders'
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
