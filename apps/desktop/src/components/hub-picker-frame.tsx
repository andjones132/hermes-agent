import { type CSSProperties, type RefObject, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'

const HUB_URL = 'https://hermes-agent.nousresearch.com/docs/skills'

const PICKER_URLS = [`${HUB_URL}?embed=picker`, 'https://nousresearch.github.io/hermes-agent/picker/skills/?embed=picker']

const READY_TIMEOUT_MS = 8_000

interface HubPickerFrameProps {
  allow?: string
  frameRef: RefObject<HTMLIFrameElement | null>
  sandbox: string
  style: CSSProperties
  title: string
}

/** HTTP 200 and iframe load both occur for blocked/error documents. Only a
 * hydrated picker can acknowledge readiness. Never override framing headers. */
export function HubPickerFrame({ allow, frameRef, sandbox, style, title }: HubPickerFrameProps) {
  const { t } = useI18n()
  const [attempt, setAttempt] = useState(0)
  const [ready, setReady] = useState(false)

  const src = PICKER_URLS[attempt]

  useEffect(() => {
    if (!src || ready) {
      return
    }
    let active = true

    const fail = () => {
      if (active) {
        setAttempt(value => value + 1)
      }
      active = false
    }

    const timer = setTimeout(fail, READY_TIMEOUT_MS)

    const onMessage = (event: MessageEvent) => {
      if (
        active &&
        event.origin === new URL(src).origin &&
        frameRef.current &&
        event.source === frameRef.current.contentWindow &&
        event.data?.type === 'hermes-hub-ready'
      ) {
        active = false
        clearTimeout(timer)
        setReady(true)
      }
    }

    window.addEventListener('message', onMessage)
    const frame = frameRef.current
    frame?.addEventListener('error', fail)

    // Keep the fast WAF/offline fallback, but never equate HEAD success with
    // a rendered catalog (CORS also hides CSP/X-Frame-Options response headers).
    if (attempt === 0) {
      void fetch(src, { method: 'HEAD', signal: AbortSignal.timeout(READY_TIMEOUT_MS) })
        .then(response => {
          if (!response.ok) {
            fail()
          }
        })
        .catch(fail)
    }

    return () => {
      active = false
      clearTimeout(timer)
      window.removeEventListener('message', onMessage)
      frame?.removeEventListener('error', fail)
    }
  }, [attempt, frameRef, ready, src])

  return (
    <>
      {src && (
        <iframe
          allow={allow}
          ref={frameRef}
          sandbox={sandbox}
          src={src}
          style={{ ...style, visibility: ready ? 'visible' : 'hidden' }}
          title={title}
        />
      )}
      {!ready && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-3 text-sm text-(--ui-text-secondary)">
          <p role={src ? 'status' : 'alert'}>{src ? t.common.loading : t.skills.hub.loadFailed}</p>
          <div className="flex gap-2">
            {!src && (
              <Button
                onClick={() => {
                  setReady(false)
                  setAttempt(0)
                }}
                size="sm"
                variant="secondary"
              >
                {t.common.retry}
              </Button>
            )}
            <Button onClick={() => openExternalLink(HUB_URL)} size="sm" variant="secondary">
              {t.connectors.openInBrowser}
            </Button>
          </div>
        </div>
      )}
    </>
  )
}
