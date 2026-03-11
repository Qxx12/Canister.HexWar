import type { AxialCoord } from './hex'
import type { PlayerId } from './player'

export type TileOwner = PlayerId | null

export type TerrainType = 'grassland' | 'plains' | 'desert' | 'tundra'

export const TERRAIN_COLORS: Record<TerrainType, string> = {
  grassland: '#c4d8b8',
  plains:    '#ced8b8',
  desert:    '#e4dcc4',
  tundra:    '#e4e6e6',
}

export interface Tile {
  coord: AxialCoord
  owner: TileOwner
  units: number
  isStartTile: boolean
  startOwner: PlayerId | null
  terrain: TerrainType
  newlyConquered: boolean
}

export type Board = Map<string, Tile>
