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
  '#E84040', // human - red
  '#A855F7', // AI 1 - purple
  '#1ABC9C', // AI 2 - teal
  '#F472B6', // AI 3 - pink (China)
  '#F59E0B', // AI 4 - amber (Aztec)
  '#3B9EFF', // AI 5 - blue (Greece)
]

export const PLAYER_NAMES: string[] = [
  'You', 'Rome', 'Egypt', 'China', 'Aztec', 'Greece'
]
