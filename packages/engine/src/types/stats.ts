import type { PlayerId } from './player'

export interface PlayerStats {
  playerId: PlayerId
  unitsGenerated: number
  unitsKilled: number
  tilesConquered: number
  tilesLost: number
  tilesAtEnd: number
}

export type GameOutcome = 'win' | 'lose' | 'retire'

export interface EndGameStats {
  outcome: GameOutcome
  winnerId: PlayerId | null
  playerStats: PlayerStats[]
}
