import { useState } from 'react'
import type { GameState } from '../../types/game'
import { Modal } from '../shared/Modal'
import { Button } from '../shared/Button'
import styles from './GameHUD.module.scss'

interface GameHUDProps {
  gameState: GameState
  onEndTurn: () => void
  onRetire: () => void
  isAnimating: boolean
}

export function GameHUD({ gameState, onEndTurn, onRetire, isAnimating }: GameHUDProps) {
  const [showRetireConfirm, setShowRetireConfirm] = useState(false)
  const { phase, players, humanPlayerId, turn } = gameState
  const isPlayerTurn = phase === 'playerTurn' && !isAnimating

  return (
    <>
      <div className={styles.hud}>
        {/* Top bar: burger menu left, player list right */}
        <div className={styles.topBar}>
          <div className={styles.legend}>
            {players.filter(p => !p.isEliminated).map(p => (
              <div key={p.id} className={styles.playerEntry}>
                <div className={styles.colorDot} style={{ background: p.color }} />
                <span className={`${styles.playerName} ${p.id === humanPlayerId ? styles.you : ''}`}>
                  {p.name}{p.id === humanPlayerId ? ' (You)' : ''}
                </span>
              </div>
            ))}
          </div>
          <button className={styles.menuBtn} onClick={() => setShowRetireConfirm(true)} title="Menu">
            &#9776;
          </button>
        </div>

        {/* Bottom-left: zoom, end turn button, turn number */}
        <div className={styles.bottomLeft}>
          <div className={`${styles.endTurnWrap} ${!isPlayerTurn ? styles.waitingWrap : ''}`}>
            <button
              className={`${styles.endTurnBtn} ${!isPlayerTurn ? styles.waiting : ''}`}
              onClick={isPlayerTurn ? onEndTurn : undefined}
              disabled={!isPlayerTurn}
              title="End Turn"
            >
              {!isPlayerTurn ? <span className={styles.spinner} /> : <span className={styles.playIcon}>▶</span>}
            </button>
          </div>
          <span className={styles.turnNumber}>Turn {turn.turnNumber}</span>
        </div>
      </div>

      {showRetireConfirm && (
        <Modal title="Menu" onClose={() => setShowRetireConfirm(false)}>
          <div className={styles.menuContent}>
            <p>Do you want to retire and return to the start screen?</p>
            <div className={styles.menuActions}>
              <Button onClick={onRetire} variant="danger">Retire</Button>
              <Button onClick={() => setShowRetireConfirm(false)} variant="secondary">Continue</Button>
            </div>
          </div>
        </Modal>
      )}
    </>
  )
}
