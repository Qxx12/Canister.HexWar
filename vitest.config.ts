import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import path from 'path'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  plugins: [react() as never],
  resolve: {
    alias: {
      '@hexwar/engine': path.resolve(__dirname, 'packages/engine/src/index.ts'),
      '@hexwar/greedy': path.resolve(__dirname, 'packages/greedy/src/index.ts'),
      '@hexwar/strategy': path.resolve(__dirname, 'packages/strategy/src/index.ts'),
      '@hexwar/warlord': path.resolve(__dirname, 'packages/warlord/src/index.ts'),
      '@hexwar/conqueror': path.resolve(__dirname, 'packages/conqueror/src/index.ts'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test-setup.ts',
    exclude: ['**/node_modules/**', '**/e2e/**'],
    coverage: {
      provider: 'v8',
      include: [
        'src/**/*.{ts,tsx}',
        'packages/engine/src/**/*.ts',
        'packages/greedy/src/**/*.ts',
        'packages/strategy/src/**/*.ts',
        'packages/warlord/src/**/*.ts',
        'packages/conqueror/src/**/*.ts',
      ],
      exclude: [
        'src/main.tsx',
        'src/test-setup.ts',
        '**/*.d.ts',
        '**/scripts/**',
      ],
      reporter: ['text', 'html'],
      reportsDirectory: './coverage',
    },
  },
})
