import type { Tile } from '@hexwar/engine'

export function scoreTarget(myUnits: number, target: Tile): number {
  const startBonus = target.isStartTile ? 45 : 0

  if (target.owner === null) {
    // Neutral — free tile, always worth taking
    return 35 + startBonus
  }

  // Enemy tile
  const advantage = myUnits - target.units
  if (advantage <= 0) {
    // We'd tie or lose — only gamble on enemy start tiles
    return target.isStartTile ? 18 : -1
  }

  // We can win — prefer larger advantages and start tiles
  return 50 + advantage * 3 + startBonus
}

export function unitsToSend(myUnits: number, target: Tile): number {
  if (target.owner === null) {
    // Neutral: send everything, no counter-attack risk
    return myUnits
  }

  if (myUnits <= target.units) {
    // Desperate attack on a start tile — commit everything
    return myUnits
  }

  // Send just enough to conquer with a small garrison, keep a buffer on source
  const minToConquer = target.units + 1
  const comfortable = Math.floor(myUnits * 0.85)
  return Math.min(myUnits, Math.max(minToConquer, comfortable))
}
