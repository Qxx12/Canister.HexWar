import type { PlayerId } from './player'

// Sentinel: move all units currently on the tile (only valid for standing orders)
export const UNITS_ALL = Infinity

export interface MovementOrder {
  fromKey: string
  toKey: string
  requestedUnits: number
}

export type OrderMap = Map<string, MovementOrder>

export type AllOrders = Map<PlayerId, OrderMap>
