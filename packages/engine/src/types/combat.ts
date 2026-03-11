import type { PlayerId } from './player'

export interface CombatResult {
  fromKey: string
  toKey: string
  attackingPlayerId: PlayerId
  unitsSent: number
  defendingPlayerId: PlayerId | null
  defendingUnitsPresent: number
  attackerCasualties: number
  defenderCasualties: number
  conquered: boolean
  remainingAttackers: number
  wasClamped: boolean // true if available units < requestedUnits
}
