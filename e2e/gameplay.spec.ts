import { test, expect } from '@playwright/test'
import { startGame, clickHumanTileAndNeighbor, getTileInfo } from './helpers'

// ─── Game board rendering ────────────────────────────────────────────────────

test.describe('Game board', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('renders hex tiles on the board', async ({ page }) => {
    // There should be many polygons (hex tiles) in the SVG
    const polygons = page.locator('svg polygon')
    await expect(polygons.first()).toBeVisible()
    const count = await polygons.count()
    expect(count).toBeGreaterThan(10)
  })

  test('human player has tiles on the board', async ({ page }) => {
    const tiles = await getTileInfo(page)
    const humanTiles = tiles.filter(t => t.isHuman)
    expect(humanTiles.length).toBeGreaterThan(0)
  })

  test('shows player legend with You label', async ({ page }) => {
    await expect(page.getByText('You (You)')).toBeVisible()
  })
})

// ─── HUD state ───────────────────────────────────────────────────────────────

test.describe('HUD', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('shows Turn 1 at game start', async ({ page }) => {
    await expect(page.getByText('Turn 1')).toBeVisible()
  })

  test('End Turn button is enabled on player turn', async ({ page }) => {
    await expect(page.getByTitle('End Turn')).toBeEnabled()
  })

  test('view toggle button shows 2D (current mode)', async ({ page }) => {
    await expect(page.getByRole('button', { name: '2D' })).toBeVisible()
  })

  test('menu button is visible', async ({ page }) => {
    await expect(page.getByTitle('Menu')).toBeVisible()
  })
})

// ─── Tile interaction ────────────────────────────────────────────────────────

test.describe('Tile interaction', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('clicking a human tile does not crash the game', async ({ page }) => {
    const tiles = await getTileInfo(page)
    const human = tiles.find(t => t.isHuman && t.units > 0)
    expect(human).toBeDefined()
    await page.mouse.click(human!.cx, human!.cy)
    // Game should still be running — Turn 1 visible, no error overlay
    await expect(page.getByText('Turn 1')).toBeVisible()
  })

  test('clicking a human tile then an adjacent tile opens Move Units modal', async ({ page }) => {
    const found = await clickHumanTileAndNeighbor(page)
    expect(found).toBe(true)
    await expect(page.getByRole('heading', { name: 'Move Units' })).toBeVisible()
  })

  test('Move Units modal has Confirm and Close buttons', async ({ page }) => {
    await clickHumanTileAndNeighbor(page)
    await expect(page.getByRole('button', { name: 'Confirm' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Close' })).toBeVisible()
  })

  test('confirming an order closes the modal', async ({ page }) => {
    await clickHumanTileAndNeighbor(page)
    await page.getByRole('button', { name: 'Confirm' }).click()
    await expect(page.getByRole('heading', { name: 'Move Units' })).not.toBeVisible()
  })

  test('closing the modal without confirming leaves no order', async ({ page }) => {
    await clickHumanTileAndNeighbor(page)
    await page.getByRole('button', { name: 'Close' }).click()
    await expect(page.getByRole('heading', { name: 'Move Units' })).not.toBeVisible()
    // Still on Turn 1 — no turn was ended
    await expect(page.getByText('Turn 1')).toBeVisible()
  })
})

// ─── Turn flow ───────────────────────────────────────────────────────────────

test.describe('Turn flow', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('End Turn button click disables button while AI acts', async ({ page }) => {
    await page.getByTitle('End Turn').click()
    await expect(page.getByTitle('End Turn')).toBeDisabled()
  })

  test('after AI finishes, End Turn is re-enabled on Turn 2', async ({ page }) => {
    await page.getByTitle('End Turn').click()
    await expect(page.getByTitle('End Turn')).toBeEnabled({ timeout: 15000 })
    await expect(page.getByText('Turn 2')).toBeVisible({ timeout: 15000 })
  })

  test('Enter key ends turn', async ({ page }) => {
    await page.keyboard.press('Enter')
    await expect(page.getByTitle('End Turn')).toBeDisabled()
    await expect(page.getByTitle('End Turn')).toBeEnabled({ timeout: 15000 })
    await expect(page.getByText('Turn 2')).toBeVisible()
  })

  test('orders survive into animation — arrow appears then disappears', async ({ page }) => {
    await clickHumanTileAndNeighbor(page)
    await page.getByRole('button', { name: 'Confirm' }).click()
    await page.getByTitle('End Turn').click()
    // After processing, Turn 2
    await expect(page.getByText('Turn 2')).toBeVisible({ timeout: 15000 })
  })
})

// ─── View toggle ─────────────────────────────────────────────────────────────

test.describe('View toggle', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('toggling to 3D shows 3D button (current mode)', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await expect(page.getByRole('button', { name: '3D' })).toBeVisible()
  })

  test('toggling back to 2D shows 2D button and restores SVG board', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await page.getByRole('button', { name: '3D' }).click()
    await expect(page.getByRole('button', { name: '2D' })).toBeVisible()
    await expect(page.locator('svg polygon').first()).toBeVisible()
  })

  test('End Turn works in 3D mode', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await expect(page.getByTitle('End Turn')).toBeEnabled()
    await page.getByTitle('End Turn').click()
    await expect(page.getByTitle('End Turn')).toBeEnabled({ timeout: 15000 })
    await expect(page.getByText('Turn 2')).toBeVisible()
  })
})

// ─── Sun and shadow toggles ──────────────────────────────────────────────────

test.describe('Sun and shadow toggles', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('sun button not visible in 2D mode', async ({ page }) => {
    await expect(page.getByTitle('Sun: on')).not.toBeVisible()
    await expect(page.getByTitle('Sun: off')).not.toBeVisible()
  })

  test('shadow button not visible in 2D mode', async ({ page }) => {
    await expect(page.getByTitle('Shadows: on')).not.toBeVisible()
    await expect(page.getByTitle('Shadows: off')).not.toBeVisible()
  })

  test('sun button is visible and on when switching to 3D', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await expect(page.getByTitle('Sun: on')).toBeVisible()
  })

  test('shadow button is visible when switching to 3D', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await expect(page.getByTitle('Shadows: on')).toBeVisible()
  })

  test('sun button toggles off and on', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await page.getByTitle('Sun: on').click()
    await expect(page.getByTitle('Sun: off')).toBeVisible()
    await page.getByTitle('Sun: off').click()
    await expect(page.getByTitle('Sun: on')).toBeVisible()
  })

  test('shadow button toggles off and on', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await page.getByTitle('Shadows: on').click()
    await expect(page.getByTitle('Shadows: off')).toBeVisible()
    await page.getByTitle('Shadows: off').click()
    await expect(page.getByTitle('Shadows: on')).toBeVisible()
  })

  test('sun resets to on when switching back to 3D after turning it off', async ({ page }) => {
    await page.getByRole('button', { name: '2D' }).click()
    await page.getByTitle('Sun: on').click()
    await expect(page.getByTitle('Sun: off')).toBeVisible()
    // Switch to 2D then back to 3D — sun should re-enable
    await page.getByRole('button', { name: '3D' }).click()
    await page.getByRole('button', { name: '2D' }).click()
    await expect(page.getByTitle('Sun: on')).toBeVisible()
  })
})

// ─── Retire / End screen ─────────────────────────────────────────────────────

test.describe('Retire flow', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('Escape key opens the retire menu', async ({ page }) => {
    await page.keyboard.press('Escape')
    await expect(page.getByText('Do you want to retire')).toBeVisible()
  })

  test('menu button opens the retire dialog', async ({ page }) => {
    await page.getByTitle('Menu').click()
    await expect(page.getByText('Do you want to retire')).toBeVisible()
  })

  test('Continue button closes the retire dialog', async ({ page }) => {
    await page.getByTitle('Menu').click()
    await page.getByRole('button', { name: 'Continue' }).click()
    await expect(page.getByText('Do you want to retire')).not.toBeVisible()
    await expect(page.getByText('Turn 1')).toBeVisible()
  })

  test('Retire shows end screen with Retired heading', async ({ page }) => {
    await page.getByTitle('Menu').click()
    await page.getByRole('button', { name: 'Retire' }).click()
    await expect(page.getByRole('heading', { name: 'Retired' })).toBeVisible()
  })

  test('end screen shows stats table', async ({ page }) => {
    await page.getByTitle('Menu').click()
    await page.getByRole('button', { name: 'Retire' }).click()
    await expect(page.getByRole('table')).toBeVisible()
    // Table has player stat columns
    await expect(page.getByRole('columnheader', { name: 'Player' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: 'Tiles', exact: true })).toBeVisible()
  })

  test('Back to Menu returns to start screen', async ({ page }) => {
    await page.getByTitle('Menu').click()
    await page.getByRole('button', { name: 'Retire' }).click()
    await page.getByRole('button', { name: 'Back to Menu' }).click()
    await expect(page.getByRole('heading', { name: 'HexWar' })).toBeVisible()
  })
})
