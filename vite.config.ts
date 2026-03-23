import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import path from 'path'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@hexwar/engine': path.resolve(__dirname, 'packages/engine/src/index.ts'),
      '@hexwar/greedy': path.resolve(__dirname, 'packages/greedy/src/index.ts'),
      '@hexwar/strategy': path.resolve(__dirname, 'packages/strategy/src/index.ts'),
      '@hexwar/warlord': path.resolve(__dirname, 'packages/warlord/src/index.ts'),
    },
  },
})
