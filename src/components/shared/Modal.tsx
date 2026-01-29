import { useEffect } from 'react'
import styles from './Modal.module.scss'

interface ModalProps {
  title: string
  onClose: () => void
  children: React.ReactNode
  maxWidth?: number
}

export function Modal({ title, onClose, children, maxWidth }: ModalProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.dialog} style={maxWidth ? { maxWidth } : undefined} onClick={e => e.stopPropagation()}>
        <h3 className={styles.title}>{title}</h3>
        {children}
      </div>
    </div>
  )
}
