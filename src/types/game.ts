import type { Board } from './board'
import type { Player, PlayerId } from './player'
import type { AllOrders, OrderMap } from './orders'
import type { EndGameStats, PlayerStats } from './stats'

export type GamePhase =
  | 'start'
  | 'playerTurn'
  | 'aiTurn'
  | 'animating'
  | 'end'

export interface TurnState {
  turnNumber: number
  activeAiIndex: number
}

export interface GameState {
  phase: GamePhase
  board: Board
  players: Player[]
  humanPlayerId: PlayerId
  orders: AllOrders
  humanStandingOrders: OrderMap
  turn: TurnState
  winner: PlayerId | null
  stats: EndGameStats | null
  runningStats: Map<PlayerId, PlayerStats>
}
