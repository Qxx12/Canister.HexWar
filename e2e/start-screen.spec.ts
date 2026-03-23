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

  test('shows copyright text', async ({ page }) => {
    await expect(page.getByText(/© \d{4} nonaction\.net/)).toBeVisible()
  })

  test('copyright links to nonaction.net', async ({ page }) => {
    const link = page.getByRole('link', { name: /nonaction\.net/ })
    await expect(link).toBeVisible()
    await expect(link).toHaveAttribute('href', 'https://nonaction.net')
  })

  test('copyright link opens in a new tab', async ({ page }) => {
    await expect(page.getByRole('link', { name: /nonaction\.net/ })).toHaveAttribute('target', '_blank')
  })
})

test.describe('Mobile meta tags', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('theme-color meta tag is present with correct value', async ({ page }) => {
    const content = await page.$eval(
      'meta[name="theme-color"]',
      (el: Element) => el.getAttribute('content'),
    )
    expect(content).toBe('#1c100a')
  })

  test('web manifest link is present', async ({ page }) => {
    const href = await page.$eval(
      'link[rel="manifest"]',
      (el: Element) => el.getAttribute('href'),
    )
    expect(href).toBe('/site.webmanifest')
  })
})
