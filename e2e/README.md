# End-to-End Tests (Playwright)

Playwright tests covering the full gameplay flow through a real browser.

## Running

```bash
# Run all e2e tests (headless)
npm run test:e2e

# Run with Playwright's interactive UI (shows browser, timeline, snapshots)
npm run test:e2e:ui

# Step-through debugging mode (opens browser, pauses on failures)
npm run test:e2e:debug
```

## How it works

The test command runs `npm run build && npm run preview` before executing tests.
The preview server serves the production bundle on `http://localhost:4173`.
This is intentional: using the build output (not the dev server) avoids Vite's
on-demand compilation lag, which caused the first requests to timeout under
parallel test workers.

## Structure

```
e2e/
  helpers.ts            # Shared utilities: startGame(), getTileInfo(), clickHumanTileAndNeighbor()
  start-screen.spec.ts  # Start screen rendering and navigation
  gameplay.spec.ts      # Board, HUD, tile interaction, turns, view toggle, retire flow
```

## Selectors and how tiles are found

The 2D board is an SVG. Hex tiles are `<g style="cursor: pointer">` elements each
containing a `<polygon>` and optionally a `<text>` showing the unit count.

Human player tiles are identified by text fill color `#E84040` (PLAYER_COLORS[0]).
There is no `data-testid` or accessible role on individual tiles — they are purely visual SVG.

### getTileInfo (helpers.ts)

Uses `page.evaluate()` to call `getBoundingClientRect()` on every hex group from
inside the browser. This returns actual screen (viewport-relative) coordinates that
already account for the CSS pan/zoom transform applied to the board container.

```ts
const groups = Array.from(document.querySelectorAll('svg g[style*="cursor"]'))
return groups.map(g => {
  const bbox = g.getBoundingClientRect()
  // cx/cy are usable directly with page.mouse.click(cx, cy)
  ...
})
```

### clickHumanTileAndNeighbor (helpers.ts)

Finds a human tile with units > 0, then looks for a tile whose screen-space center
is 50–75 px away (the expected adjacent-hex distance at zoom=1, hexSize=36 is
`sqrt(3) * 36 ≈ 62.4 px`). Clicks source then destination to trigger the order modal.

This distance range intentionally has some slack (~20%) to tolerate minor float
rounding and any minor zoom variance.

## Timeouts

| Setting | Value | Reason |
|---|---|---|
| `actionTimeout` | 15 s | Clicks on slowly-appearing elements |
| `expect.timeout` | 15 s | AI turn processing takes a few seconds |
| `webServer.timeout` | 60 s | `npm run build` needs time on first run |

## Adding new tests

1. Use `startGame(page)` from helpers to navigate past the start screen and the Settings screen (it clicks Deploy with default settings).
2. Use `getTileInfo(page)` when you need tile positions — prefer it over CSS class
   selectors, since CSS Modules mangle class names at build time.
3. Use role/title/text selectors for UI controls — they are stable:
   - `page.getByTitle('End Turn')` — end turn button
   - `page.getByTitle('Menu')` — hamburger menu
   - `page.getByRole('button', { name: '3D' })` — view toggle
   - `page.getByRole('heading', { name: 'Move Units' })` — order modal
4. For assertions that depend on AI finishing its turn, set an explicit timeout:
   ```ts
   await expect(page.getByText('Turn 2')).toBeVisible({ timeout: 15000 })
   ```
5. Run `npm run lint` — the `e2e/` directory is excluded from ESLint so Playwright
   imports won't conflict with the Vitest setup.
