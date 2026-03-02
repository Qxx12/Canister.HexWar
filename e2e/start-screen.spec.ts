import { test, expect } from '@playwright/test'

test.describe('Start screen', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('shows game title', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'HexWar' })).toBeVisible()
  })

  test('shows rules list', async ({ page }) => {
    await expect(page.getByText('Command units across a hex grid battlefield')).toBeVisible()
    await expect(page.getByText('Battle multiple AI opponents simultaneously')).toBeVisible()
    await expect(page.getByText('Win by capturing all enemy capital tiles')).toBeVisible()
  })

  test('start button is visible and clickable', async ({ page }) => {
    const startBtn = page.getByRole('button').filter({ hasText: '▶' })
    await expect(startBtn).toBeVisible()
  })

  test('start button navigates to game board', async ({ page }) => {
    await page.getByRole('button').filter({ hasText: '▶' }).click()
    // Board SVG should appear; hex polygons are the first sign of the game board
    await expect(page.locator('svg polygon').first()).toBeVisible()
    // Start screen title is gone
    await expect(page.getByRole('heading', { name: 'HexWar' })).not.toBeVisible()
  })
})
