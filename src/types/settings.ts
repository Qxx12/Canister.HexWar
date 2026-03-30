export type AiDifficulty = 'soldier' | 'commander' | 'warlord'

export interface GameSettings {
  difficulty: AiDifficulty
}

export const DEFAULT_SETTINGS: GameSettings = {
  difficulty: 'commander',
}
