import { Button } from '../shared/Button'
import styles from './StartScreen.module.scss'

interface StartScreenProps {
  onStart: () => void
}

export function StartScreen({ onStart }: StartScreenProps) {
  return (
    <div className={styles.screen}>
      <div className={styles.content}>
        <h1 className={styles.title}>HexWar</h1>
        <p className={styles.subtitle}>
          A turn-based hex strategy game. Conquer all enemy capitals while defending your own.
        </p>
        <div className={styles.rules}>
          <p>&#x2022; Command your units across a hex grid battlefield</p>
          <p>&#x2022; Battle 5 AI opponents simultaneously</p>
          <p>&#x2022; Movement orders persist across turns</p>
          <p>&#x2022; Win by capturing all enemy capital tiles</p>
        </div>
        <Button onClick={onStart} variant="primary">
          Start Game
        </Button>
      </div>
    </div>
  )
}
