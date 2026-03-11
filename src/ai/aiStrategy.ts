import type { Board } from '@hexwar/engine'
import type { PlayerId } from '@hexwar/engine'
import type { OrderMap } from '@hexwar/engine'
import type { Player } from '@hexwar/engine'

export interface AIStrategy {
  computeOrders(
    board: Board,
    playerId: PlayerId,
    currentOrders: OrderMap,
    allPlayers: Player[],
  ): OrderMap
}
