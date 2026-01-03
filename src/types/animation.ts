import type { PlayerId } from './player'

export type AnimationKind = 'move' | 'fight' | 'conquer'

export interface AnimationEvent {
  kind: AnimationKind
  fromKey: string
  toKey: string
  playerId: PlayerId
  units: number
  durationMs: number
}
