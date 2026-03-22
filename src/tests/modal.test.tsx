// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { Modal } from '../components/shared/Modal'

afterEach(cleanup)

describe('Modal', () => {
  it('renders the title', () => {
    render(<Modal title="Test Title" onClose={() => {}}>content</Modal>)
    expect(screen.getByRole('heading', { name: 'Test Title' })).toBeDefined()
  })

  it('renders children', () => {
    render(<Modal title="T" onClose={() => {}}><span>child text</span></Modal>)
    expect(screen.getByText('child text')).toBeDefined()
  })

  it('calls onClose when the overlay is clicked', () => {
    const onClose = vi.fn()
    render(<Modal title="T" onClose={onClose}>content</Modal>)
    // overlay is grandparent of the h3 heading: overlay > dialog > h3
    const overlay = screen.getByRole('heading').parentElement!.parentElement!
    fireEvent.click(overlay)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('does not call onClose when the dialog itself is clicked', () => {
    const onClose = vi.fn()
    render(<Modal title="T" onClose={onClose}><button>Inside</button></Modal>)
    fireEvent.click(screen.getByText('Inside'))
    expect(onClose).not.toHaveBeenCalled()
  })

  it('calls onClose when Escape key is pressed', () => {
    const onClose = vi.fn()
    render(<Modal title="T" onClose={onClose}>content</Modal>)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('does not call onClose for non-Escape keys', () => {
    const onClose = vi.fn()
    render(<Modal title="T" onClose={onClose}>content</Modal>)
    fireEvent.keyDown(window, { key: 'Enter' })
    fireEvent.keyDown(window, { key: 'Space' })
    expect(onClose).not.toHaveBeenCalled()
  })

  it('applies maxWidth style to the dialog', () => {
    render(<Modal title="T" onClose={() => {}} maxWidth={320}>content</Modal>)
    const dialog = screen.getByRole('heading').parentElement!
    expect(dialog.style.maxWidth).toBe('320px')
  })

  it('does not set a style attribute when maxWidth is omitted', () => {
    render(<Modal title="T" onClose={() => {}}>content</Modal>)
    const dialog = screen.getByRole('heading').parentElement!
    expect(dialog.getAttribute('style')).toBeNull()
  })

  it('removes the keyboard listener on unmount', () => {
    const onClose = vi.fn()
    const { unmount } = render(<Modal title="T" onClose={onClose}>content</Modal>)
    unmount()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
  })
})
