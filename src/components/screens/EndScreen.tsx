import type { EndGameStats } from '../../types/stats'
import type { Player } from '../../types/player'
import { Button } from '../shared/Button'
import styles from './EndScreen.module.scss'

interface EndScreenProps {
  stats: EndGameStats
  players: Player[]
  onRestart: () => void
}

export function EndScreen({ stats, players, onRestart }: EndScreenProps) {
  const title =
    stats.outcome === 'win' ? 'Victory!' :
    stats.outcome === 'lose' ? 'Defeated' :
    'Retired'

  const subtitle =
    stats.outcome === 'win' ? 'You conquered the world!' :
    stats.outcome === 'lose' ? 'Your civilization has fallen.' :
    'Better luck next time.'

  const sortedStats = [...stats.playerStats].sort((a, b) => b.tilesAtEnd - a.tilesAtEnd)

  return (
    <div className={styles.screen}>
      <div className={styles.content}>
        <h1 className={`${styles.title} ${styles[stats.outcome]}`}>{title}</h1>
        <p className={styles.subtitle}>{subtitle}</p>

        <table className={styles.table}>
          <thead>
            <tr>
              <th>Player</th>
              <th>Tiles</th>
              <th>Tiles Captured</th>
              <th>Tiles Lost</th>
              <th>Units Killed</th>
            </tr>
          </thead>
          <tbody>
            {sortedStats.map(s => {
              const player = players.find(p => p.id === s.playerId)
              if (!player) return null
              const isWinner = s.playerId === stats.winnerId
              return (
                <tr key={s.playerId} className={isWinner ? styles.winner : ''}>
                  <td>
                    <div className={styles.playerCell}>
                      <div className={styles.dot} style={{ background: player.color }} />
                      {player.name}
                      {isWinner && <span className={styles.crown}>&#9819;</span>}
                    </div>
                  </td>
                  <td>{s.tilesAtEnd}</td>
                  <td>{s.tilesConquered}</td>
                  <td>{s.tilesLost}</td>
                  <td>{s.unitsKilled}</td>
                </tr>
              )
            })}
          </tbody>
        </table>

        <Button onClick={onRestart} variant="primary">
          Back to Menu
        </Button>
      </div>
    </div>
  )
}
