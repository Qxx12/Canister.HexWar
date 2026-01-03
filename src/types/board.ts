import type { AxialCoord } from './hex'
import type { PlayerId } from './player'

export type TileOwner = PlayerId | null

export interface Tile {
  coord: AxialCoord
  owner: TileOwner
  units: number
  isStartTile: boolean
  startOwner: PlayerId | null
}

export type Board = Map<string, Tile>
