import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HubPickerFrame } from './hub-picker-frame'

const PRIMARY = 'https://hermes-agent.nousresearch.com'
const MIRROR = 'https://nousresearch.github.io'
const openExternal = vi.hoisted(() => vi.fn())
vi.mock('@/lib/external-link', () => ({ openExternalLink: openExternal }))

function mount() {
  const frameRef = createRef<HTMLIFrameElement>()

  const view = render(
    <div style={{ position: 'relative', height: 400 }}>
      <HubPickerFrame
        frameRef={frameRef}
        sandbox="allow-scripts allow-same-origin"
        style={{ width: '100%', height: '100%' }}
        title="Skills Hub"
      />
    </div>
  )

  return { ...view, frameRef }
}

function ready(frame: HTMLIFrameElement, origin = PRIMARY, source: Window | null = frame.contentWindow) {
  act(() => {
    window.dispatchEvent(new MessageEvent('message', { origin, source, data: { type: 'hermes-hub-ready' } }))
  })
}

const advance = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(8_000)
  })

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }))
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('bounded embedded catalog readiness', () => {
  it('accepts primary hydration, not HTTP success alone', async () => {
    const { frameRef } = mount()
    expect(frameRef.current?.style.visibility).toBe('hidden')
    ready(frameRef.current!)
    expect(frameRef.current?.style.visibility).toBe('visible')
    await advance()
    await advance()
    expect(frameRef.current?.src).toContain(PRIMARY)
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it.each([
    ['untrusted origin', 'https://evil.example', true],
    ['unrelated window', PRIMARY, false],
    ['wrong candidate origin', MIRROR, true]
  ])('rejects readiness from %s', async (_label, origin, ownFrame) => {
    const { frameRef } = mount()
    ready(frameRef.current!, String(origin), ownFrame ? frameRef.current!.contentWindow : window)
    await advance()
    expect(frameRef.current?.src).toContain(MIRROR)
    expect(frameRef.current?.style.visibility).toBe('hidden')
    await advance()
    expect(screen.getByRole('alert')).toBeTruthy()
  })

  it('accepts mirror readiness only after switching to it', async () => {
    const { frameRef } = mount()
    await advance()
    // The standalone Pages build has its own project-prefixed asset base;
    // /hermes-agent/docs/skills serves the primary site's /docs/ artifact.
    expect(frameRef.current?.src).toBe(`${MIRROR}/hermes-agent/picker/skills/?embed=picker`)
    ready(frameRef.current!, MIRROR)
    await advance()
    expect(frameRef.current?.style.visibility).toBe('visible')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('bounds a hanging HEAD probe and opens only the canonical external URL on user click', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => {}))
    )
    const { frameRef } = mount()
    fireEvent.load(frameRef.current!)
    await advance()
    fireEvent.load(frameRef.current!)
    await advance()
    expect(frameRef.current).toBeNull()
    expect(openExternal).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))
    expect(openExternal).toHaveBeenCalledExactlyOnceWith(`${PRIMARY}/docs/skills`)
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(frameRef.current?.src).toContain(PRIMARY)
    expect(frameRef.current?.getAttribute('sandbox')).toBe('allow-scripts allow-same-origin')
  })

  it('advances on frame errors and cleans up its timer on unmount', async () => {
    const { frameRef, unmount } = mount()
    fireEvent.error(frameRef.current!)
    expect(frameRef.current?.src).toContain(MIRROR)
    fireEvent.error(frameRef.current!)
    expect(screen.getByRole('alert')).toBeTruthy()
    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('ignores a late failed HEAD response after valid hydration', async () => {
    let resolve!: (response: { ok: boolean }) => void
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise(r => {
            resolve = r
          })
      )
    )
    const { frameRef } = mount()
    ready(frameRef.current!)
    await act(async () => {
      resolve({ ok: false })
    })
    await advance()
    expect(frameRef.current?.src).toContain(PRIMARY)
    expect(frameRef.current?.style.visibility).toBe('visible')
  })
})
