import type { PlayerId } from './player'

export interface MovementOrder {
  fromKey: string
  toKey: string
  requestedUnits: number
}

export type OrderMap = Map<string, MovementOrder>

export type AllOrders = Map<PlayerId, OrderMap>
