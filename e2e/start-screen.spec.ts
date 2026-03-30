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

  test('start button navigates to settings screen', async ({ page }) => {
    await page.getByRole('button').filter({ hasText: '▶' }).click()
    // Settings screen shows the Opponent section
    await expect(page.getByRole('heading', { name: 'Opponent' })).toBeVisible()
    // HexWar title persists on the settings screen
    await expect(page.getByRole('heading', { name: 'HexWar' })).toBeVisible()
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

test.describe('Settings screen', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await page.getByRole('button').filter({ hasText: '▶' }).click()
    await expect(page.getByRole('heading', { name: 'Opponent' })).toBeVisible()
  })

  test('shows all three difficulty options', async ({ page }) => {
    await expect(page.getByRole('button', { name: /Soldier/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Commander/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Warlord/ })).toBeVisible()
  })

  test('Deploy button navigates to game board', async ({ page }) => {
    await page.getByRole('button', { name: 'Deploy' }).click()
    await expect(page.getByTitle('End Turn')).toBeVisible()
  })

  test('selecting Soldier then deploying starts game', async ({ page }) => {
    await page.getByRole('button', { name: /Soldier/ }).click()
    await page.getByRole('button', { name: 'Deploy' }).click()
    await expect(page.getByTitle('End Turn')).toBeVisible()
  })

  test('selecting Warlord then deploying starts game', async ({ page }) => {
    await page.getByRole('button', { name: /Warlord/ }).click()
    await page.getByRole('button', { name: 'Deploy' }).click()
    await expect(page.getByTitle('End Turn')).toBeVisible()
  })

  test('Back button returns to start screen', async ({ page }) => {
    await page.getByRole('button', { name: /Back/ }).click()
    await expect(page.getByRole('button').filter({ hasText: '▶' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Opponent' })).not.toBeVisible()
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
