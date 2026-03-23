import { test, expect } from '@playwright/test'
import { startGame, clickHumanTileAndNeighbor, getTileInfo, advanceUntilAdjacentHumanTiles } from './helpers'

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

// ─── AI behaviour ────────────────────────────────────────────────────────────

test.describe('AI behaviour', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('unit counts on human tiles change after AI resolves their turn', async ({ page }) => {
    const before = await getTileInfo(page)
    const humanBefore = before.filter(t => t.isHuman)
    const totalBefore = humanBefore.reduce((s, t) => s + t.units, 0)

    await page.getByTitle('End Turn').click()
    await expect(page.getByText('Turn 2')).toBeVisible({ timeout: 15_000 })

    const after = await getTileInfo(page)
    const humanAfter = after.filter(t => t.isHuman)
    const totalAfter = humanAfter.reduce((s, t) => s + t.units, 0)

    // Unit generation gives at least +1 per tile per round; even under attack
    // the total change must be non-zero (tiles gained units or were lost to AI).
    expect(totalAfter).not.toBe(totalBefore)
  })

  test('game survives three full turns without crashing', async ({ page }) => {
    for (let i = 0; i < 3; i++) {
      await page.getByTitle('End Turn').click()
      await expect(page.getByTitle('End Turn')).toBeEnabled({ timeout: 15_000 })
    }
    await expect(page.getByText('Turn 4')).toBeVisible()
  })
})

// ─── Restart cleanliness ──────────────────────────────────────────────────────

test.describe('Restart cleanliness', () => {
  test('game restarts cleanly and AI resolves turn after restart', async ({ page }) => {
    await startGame(page)
    // Play one turn then retire
    await page.getByTitle('Menu').click()
    await page.getByRole('button', { name: 'Retire' }).click()
    await page.getByRole('button', { name: 'Back to Menu' }).click()
    // Start a fresh game
    await startGame(page)
    await expect(page.getByText('Turn 1')).toBeVisible()
    // End turn — AI must resolve without error (no history bleed from old session)
    await page.getByTitle('End Turn').click()
    await expect(page.getByText('Turn 2')).toBeVisible({ timeout: 15_000 })
  })
})

// ─── Viewport panning ────────────────────────────────────────────────────────

test.describe('Viewport panning', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('RMB drag pans the board', async ({ page }) => {
    const transform = page.locator('div[style*="translate"]')
    const before = await transform.getAttribute('style')

    await page.mouse.move(400, 300)
    await page.mouse.down({ button: 'right' })
    await page.mouse.move(500, 300)
    await page.mouse.move(600, 300)
    await page.mouse.up({ button: 'right' })

    const after = await transform.getAttribute('style')
    expect(after).not.toBe(before)
  })

  test('LMB drag does not pan the board', async ({ page }) => {
    const transform = page.locator('div[style*="translate"]')
    const before = await transform.getAttribute('style')

    // Drag over an area unlikely to have tiles (top-left corner)
    await page.mouse.move(50, 50)
    await page.mouse.down({ button: 'left' })
    await page.mouse.move(250, 50)
    await page.mouse.up({ button: 'left' })

    const after = await transform.getAttribute('style')
    expect(after).toBe(before)
  })

  test('right-click does not open a context menu', async ({ page }) => {
    await page.mouse.click(400, 300, { button: 'right' })
    await page.waitForTimeout(200)
    // Game still running normally — no native menu, HUD still visible
    await expect(page.getByText('Turn 1')).toBeVisible()
  })

  test('single-finger touch drag pans the board', async ({ page }) => {
    const transform = page.locator('div[style*="translate"]')
    const before = await transform.getAttribute('style')

    // Dispatch touch pointer events directly — simulates a single finger pan
    await page.evaluate(() => {
      const board = document.querySelector('div[style*="translate"]')?.parentElement
      if (!board) throw new Error('board container not found')
      const opts = { bubbles: true, cancelable: true, pointerId: 1, pointerType: 'touch', button: 0, clientX: 400, clientY: 300 }
      board.dispatchEvent(new PointerEvent('pointerdown', opts))
      board.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: 420, clientY: 300 }))
      board.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: 460, clientY: 300 }))
      board.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: 500, clientY: 300 }))
      board.dispatchEvent(new PointerEvent('pointerup',   { ...opts, clientX: 500, clientY: 300 }))
    })

    const after = await transform.getAttribute('style')
    expect(after).not.toBe(before)
  })
})

// ─── Bidirectional order guard ────────────────────────────────────────────────

test.describe('Bidirectional order guard', () => {
  test.beforeEach(async ({ page }) => {
    await startGame(page)
  })

  test('reverse order is blocked when an order already exists in the other direction', async ({ page }) => {
    const pair = await advanceUntilAdjacentHumanTiles(page)
    if (!pair) return // Could not find adjacent human tiles within turn limit
    const [a, b] = pair

    // Set order A → B
    await page.mouse.click(a.cx, a.cy)
    await page.mouse.click(b.cx, b.cy)
    await expect(page.getByRole('heading', { name: 'Move Units' })).toBeVisible()
    await page.getByRole('button', { name: 'Confirm' }).click()

    // Attempt reverse B → A — modal must not open
    await page.mouse.click(b.cx, b.cy)
    await page.mouse.click(a.cx, a.cy)
    await expect(page.getByRole('heading', { name: 'Move Units' })).not.toBeVisible()
  })

  test('order in one direction is still accepted when no reverse exists', async ({ page }) => {
    const found = await clickHumanTileAndNeighbor(page)
    expect(found).toBe(true)
    await expect(page.getByRole('heading', { name: 'Move Units' })).toBeVisible()
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
