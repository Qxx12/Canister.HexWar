export type PlayerId = string

export type PlayerType = 'human' | 'ai'

export interface Player {
  id: PlayerId
  type: PlayerType
  color: string
  name: string
  isEliminated: boolean
}

// CivIV-inspired player colors: human + 5 AI
export const PLAYER_COLORS: string[] = [
  '#4A90D9', // human - blue
  '#C0392B', // AI 1 - red
  '#27AE60', // AI 2 - green
  '#E67E22', // AI 3 - orange
  '#8E44AD', // AI 4 - purple
  '#F1C40F', // AI 5 - gold
]

export const PLAYER_NAMES: string[] = [
  'You', 'Rome', 'Egypt', 'China', 'Aztec', 'Greece'
]
