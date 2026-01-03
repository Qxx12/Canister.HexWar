import type { Board } from '../types/board'
import type { PlayerId } from '../types/player'
import type { OrderMap } from '../types/orders'
import type { Player } from '../types/player'

export interface AIStrategy {
  computeOrders(
    board: Board,
    playerId: PlayerId,
    currentOrders: OrderMap,
    allPlayers: Player[],
  ): OrderMap
}
