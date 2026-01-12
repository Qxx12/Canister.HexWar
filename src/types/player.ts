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
  '#C0392B', // human - red
  '#7D3C98', // AI 1 - purple
  '#0d8c7a', // AI 2 - teal
  '#E07898', // AI 3 - light pink (China)
  '#D4882A', // AI 4 - amber (Aztec)
  '#1A78C2', // AI 5 - Santorini blue
]

export const PLAYER_NAMES: string[] = [
  'You', 'Rome', 'Egypt', 'China', 'Aztec', 'Greece'
]
