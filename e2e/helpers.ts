import { expect, type Page } from '@playwright/test'

// Human player is always index 0, color #E84040
export const HUMAN_COLOR = '#E84040'
// Approximate screen-space distance between adjacent hex centers at zoom=1, hexSize=36
// sqrt(3) * 36 ≈ 62.4px; we use a generous range to accommodate minor zoom/float variance
export const ADJ_MIN = 50
export const ADJ_MAX = 75

export async function startGame(page: Page): Promise<void> {
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  // The start screen has a single ▶ button
  await page.getByRole('button').filter({ hasText: '▶' }).click()
  // Board is ready once hex tiles (SVG polygons) are visible
  await page.locator('svg polygon').first().waitFor({ state: 'visible' })
}

/**
 * Finds all hex tile groups in the SVG and returns their screen-space center
 * coordinates plus ownership info, using getBoundingClientRect() which already
 * accounts for CSS pan/zoom transforms applied to the board container.
 */
export async function getTileInfo(page: Page): Promise<
  { cx: number; cy: number; isHuman: boolean; units: number }[]
> {
  return page.evaluate((humanColor) => {
    const groups = Array.from(
      document.querySelectorAll('svg g[style*="cursor"]')
    ) as SVGGraphicsElement[]

    return groups.map(g => {
      const bbox = g.getBoundingClientRect()
      const cx = bbox.left + bbox.width / 2
      const cy = bbox.top + bbox.height / 2
      const humanText = g.querySelector(`text[fill="${humanColor}"]`)
      const isHuman = humanText !== null
      const units = humanText ? parseInt(humanText.textContent ?? '0') : 0
      return { cx, cy, isHuman, units }
    })
  }, HUMAN_COLOR)
}

/**
 * Finds a human-owned tile that has at least one board tile adjacent to it,
 * then clicks source → destination to open the Move Units order modal.
 * Returns whether the pair was found and clicked.
 */
export async function clickHumanTileAndNeighbor(page: Page): Promise<boolean> {
  const tiles = await getTileInfo(page)
  const humanTiles = tiles.filter(t => t.isHuman && t.units > 0)

  for (const src of humanTiles) {
    const neighbor = tiles.find(t => {
      if (t === src) return false
      const dx = t.cx - src.cx
      const dy = t.cy - src.cy
      const dist = Math.sqrt(dx * dx + dy * dy)
      return dist >= ADJ_MIN && dist <= ADJ_MAX
    })
    if (neighbor) {
      await page.mouse.click(src.cx, src.cy)
      await page.mouse.click(neighbor.cx, neighbor.cy)
      return true
    }
  }
  return false
}

/**
 * Advances turns (up to maxTurns), setting expansion orders each turn, until
 * two adjacent human-owned tiles are found. Returns the pair or null.
 */
export async function advanceUntilAdjacentHumanTiles(
  page: Page,
  maxTurns = 6,
): Promise<[{ cx: number; cy: number }, { cx: number; cy: number }] | null> {
  for (let i = 0; i < maxTurns; i++) {
    const tiles = await getTileInfo(page)
    const humanTiles = tiles.filter(t => t.isHuman && t.units > 0)
    for (const a of humanTiles) {
      const b = humanTiles.find(b => {
        if (b === a) return false
        const dx = b.cx - a.cx
        const dy = b.cy - a.cy
        return Math.sqrt(dx * dx + dy * dy) >= ADJ_MIN && Math.sqrt(dx * dx + dy * dy) <= ADJ_MAX
      })
      if (b) return [a, b]
    }
    // Not found yet — try to expand then end the turn
    await clickHumanTileAndNeighbor(page)
    if (await page.getByRole('heading', { name: 'Move Units' }).isVisible()) {
      await page.getByRole('button', { name: 'Confirm' }).click()
    }
    await page.getByTitle('End Turn').click()
    await expect(page.getByTitle('End Turn')).toBeEnabled({ timeout: 15_000 })
  }
  return null
}
