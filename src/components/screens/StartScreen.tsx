import styles from './StartScreen.module.scss'

interface StartScreenProps {
  onStart: () => void
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

export function StartScreen({ onStart }: StartScreenProps) {
  return (
    <div className={styles.screen}>
      <HexBackground />
      <div className={styles.content}>
        <h1 className={styles.title}>HexWar</h1>
        <ul className={styles.rules}>
          <li>Command units across a hex grid battlefield</li>
          <li>Battle multiple AI opponents simultaneously</li>
          <li>Win by capturing all enemy capital tiles</li>
        </ul>
        <div className={styles.startWrap}>
          <button className={styles.startBtn} onClick={onStart}>
            ▶
          </button>
        </div>
      </div>
    </div>
  )
}
