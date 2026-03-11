export interface AxialCoord {
  q: number
  r: number
}

// Flat-top hex directions
export const HEX_DIRECTIONS: AxialCoord[] = [
  { q: 1, r: 0 }, { q: 1, r: -1 }, { q: 0, r: -1 },
  { q: -1, r: 0 }, { q: -1, r: 1 }, { q: 0, r: 1 },
]

export function hexNeighbors(coord: AxialCoord): AxialCoord[] {
  return HEX_DIRECTIONS.map(d => ({ q: coord.q + d.q, r: coord.r + d.r }))
}

export function hexDistance(a: AxialCoord, b: AxialCoord): number {
  return Math.max(Math.abs(a.q - b.q), Math.abs(a.r - b.r), Math.abs((a.q + a.r) - (b.q + b.r)))
}

export function hexEquals(a: AxialCoord, b: AxialCoord): boolean {
  return a.q === b.q && a.r === b.r
}

export function hexToKey(coord: AxialCoord): string {
  return `${coord.q},${coord.r}`
}

export function keyToHex(key: string): AxialCoord {
  const [q, r] = key.split(',').map(Number)
  return { q, r }
}

// Pixel position for pointy-top hex (returns center coordinates)
export function axialToPixel(coord: AxialCoord, hexSize: number): { x: number; y: number } {
  return {
    x: hexSize * (Math.sqrt(3) * coord.q + Math.sqrt(3) / 2 * coord.r),
    y: hexSize * (3 / 2) * coord.r,
  }
}

// Get the 6 corner points of a pointy-top hex (for SVG polygon)
export function hexCorners(cx: number, cy: number, size: number): Array<{ x: number; y: number }> {
  return Array.from({ length: 6 }, (_, i) => {
    const angleDeg = 60 * i + 30
    const angleRad = (Math.PI / 180) * angleDeg
    return {
      x: cx + size * Math.cos(angleRad),
      y: cy + size * Math.sin(angleRad),
    }
  })
}
