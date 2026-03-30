import { useState } from 'react'
import type { AiDifficulty, GameSettings } from '../../types/settings'
import { DEFAULT_SETTINGS } from '../../types/settings'
import { Button } from '../shared/Button'
import styles from './SettingsScreen.module.scss'

interface SettingsScreenProps {
  onConfirm: (settings: GameSettings) => void
  onBack: () => void
}

// Generates a pointy-top hex polygon string for a given center and size
function hexPoints(cx: number, cy: number, size: number): string {
  return Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 180) * (60 * i + 30)
    return `${cx + size * Math.cos(a)},${cy + size * Math.sin(a)}`
  }).join(' ')
}

function HexBackground() {
  const size = 36
  const w = size * Math.sqrt(3)
  const h = size * 1.5
  const screenW = window.innerWidth
  const screenH = window.innerHeight
  const cols = Math.ceil(screenW / w) + 2
  const rows = Math.ceil(screenH / h) + 2
  const hexes: { cx: number; cy: number }[] = []

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const cx = col * w + (row % 2 === 0 ? 0 : w / 2)
      const cy = row * h
      hexes.push({ cx, cy })
    }
  }

  return (
    <svg className={styles.hexBg} xmlns="http://www.w3.org/2000/svg">
      {hexes.map(({ cx, cy }, i) => (
        <polygon key={i} points={hexPoints(cx, cy, size - 1)} className={styles.hexBgCell} />
      ))}
    </svg>
  )
}

interface DifficultyOption {
  value: AiDifficulty
  label: string
  flavor: string
}

const DIFFICULTY_OPTIONS: DifficultyOption[] = [
  {
    value: 'soldier',
    label: 'Soldier',
    flavor: 'Predictable tactics. A worthy first engagement.',
  },
  {
    value: 'commander',
    label: 'Commander',
    flavor: 'Coordinates offensives. Will contest every border.',
  },
  {
    value: 'warlord',
    label: 'Warlord',
    flavor: 'Merciless expansion. Eliminates threats before they grow.',
  },
]

export function SettingsScreen({ onConfirm, onBack }: SettingsScreenProps) {
  const [settings, setSettings] = useState<GameSettings>(DEFAULT_SETTINGS)

  return (
    <div className={styles.screen}>
      <HexBackground />
      <div className={styles.content}>
        <h1 className={styles.title}>HexWar</h1>

        <div className={styles.panel}>
          <div className={styles.section}>
            <h2 className={styles.sectionLabel}>Opponent</h2>
            <div className={styles.difficultyRow}>
              {DIFFICULTY_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  className={`${styles.difficultyCard} ${settings.difficulty === opt.value ? styles.selected : ''}`}
                  onClick={() => setSettings(s => ({ ...s, difficulty: opt.value }))}
                >
                  <span className={styles.difficultyLabel}>{opt.label}</span>
                  <span className={styles.difficultyFlavor}>{opt.flavor}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className={styles.actions}>
          <Button variant="secondary" onClick={onBack}>← Back</Button>
          <Button variant="primary" onClick={() => onConfirm(settings)}>Deploy</Button>
        </div>
      </div>
    </div>
  )
}
