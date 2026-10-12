import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'
import { HERMES_HUB_FALLBACK_ORIGIN, HERMES_HUB_ORIGIN } from './hub-iframe-policy'

// Execute the production functions and registered callback, not a copied policy.
// main.ts cannot be imported without booting the app. AST extraction isolates
// this boundary while leaving its behavior untouched. This is not a live
// Electron event-delivery test: the fixture supplies the identified subframe.
const source = ts.createSourceFile('main.ts', readFileSync(new URL('./main.ts', import.meta.url), 'utf8'),
  ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
const names = new Set(['isHermesHubPickerUrl', 'findFrameByIdentifier', 'safeFrameOrigin'])
const helpers = source.statements.filter(statement => ts.isFunctionDeclaration(statement)
  && statement.name && names.has(statement.name.text))
const handlers: ts.CallExpression[] = []
function visit(node: ts.Node) {
  if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression)
    && node.expression.name.text === 'on' && node.arguments[0]
    && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text === 'will-frame-navigate') {
    handlers.push(node)
  }
  ts.forEachChild(node, visit)
}
visit(source)
if (helpers.length !== names.size || handlers.length !== 1) {
  throw new Error('Cannot isolate the production Hub navigation boundary')
}
const compiled = ts.transpileModule([...helpers, ...handlers].map(node => node.getText(source)).join('\n'), {
  compilerOptions: { target: ts.ScriptTarget.ES2023 }
}).outputText

function harness(origin = HERMES_HUB_ORIGIN) {
  const frame = { origin, processId: 1, routingId: 2, reload: vi.fn() }
  let navigate!: (event: { preventDefault: () => void }, url: string, main: boolean, process: number, routing: number) => void
  const context = {
    URL, HERMES_HUB_ORIGIN, HERMES_HUB_FALLBACK_ORIGIN,
    win: { webContents: { frames: [frame], on: (_event: string, handler: typeof navigate) => { navigate = handler } } },
    isHermesHubPickerUrl: undefined as unknown as (url: string) => boolean
  }
  runInNewContext(compiled, context)
  return {
    accepts: context.isHermesHubPickerUrl,
    navigate(url: string, main = false, process = 1, routing = 2) {
      const preventDefault = vi.fn()
      frame.reload.mockClear()
      navigate({ preventDefault }, url, main, process, routing)
      return { prevented: preventDefault.mock.calls.length, reloaded: frame.reload.mock.calls.length }
    }
  }
}

const allowed = [
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/picker/skills/?embed=picker`,
  `${HERMES_HUB_ORIGIN}/docs/skills`,
  `${HERMES_HUB_ORIGIN}/docs/skills/?embed=picker`,
  // Deliberate compatibility with the previous project-prefixed docs mirror.
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/docs/skills?embed=picker`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/docs/skills/`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/picker/skills?embed=picker`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/picker/skills/?embed=picker&page=2#search`
]
const denied = [
  'not a URL',
  'https://evil.example/hermes-agent/picker/skills/?embed=picker',
  'https://nousresearch.github.io.evil.example/hermes-agent/picker/skills/',
  'https://nousresearch.github.io@evil.example/hermes-agent/picker/skills/',
  'http://nousresearch.github.io/hermes-agent/picker/skills/',
  'https://nousresearch.github.io:444/hermes-agent/picker/skills/',
  `${HERMES_HUB_ORIGIN}/hermes-agent/picker/skills/`,
  `${HERMES_HUB_ORIGIN}/hermes-agent/docs/skills`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/docs/skills`,
  `${HERMES_HUB_ORIGIN}/docs/plugins`,
  `${HERMES_HUB_ORIGIN}/docs/skills/apple-notes`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/picker/skills/other`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/picker/skills//`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/picker/skills-evil`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/hermes-agent/picker/%73kills/`,
  `${HERMES_HUB_FALLBACK_ORIGIN}/other-project/picker/skills/`
]

describe('actual Electron Hub picker navigation boundary', () => {
  it('accepts only the exact deployment paths with an optional single trailing slash', () => {
    const { accepts } = harness()
    for (const url of allowed) expect(accepts(url), url).toBe(true)
    for (const url of denied) expect(accepts(url), url).toBe(false)
  })

  it.each([HERMES_HUB_ORIGIN, HERMES_HUB_FALLBACK_ORIGIN])(
    'allows picker self-navigation and prevents/reloads escape from %s', origin => {
      const { navigate } = harness(origin)
      for (const url of allowed) expect(navigate(url), url).toEqual({ prevented: 0, reloaded: 0 })
      for (const url of denied) expect(navigate(url), url).toEqual({ prevented: 1, reloaded: 1 })
    }
  )

  it('does not apply the Hub subframe guard to top-level, unknown, or unrelated frames', () => {
    const { navigate } = harness()
    expect(navigate(denied[1], true)).toEqual({ prevented: 0, reloaded: 0 })
    expect(navigate(denied[1], false, 9, 9)).toEqual({ prevented: 0, reloaded: 0 })
    expect(harness('https://unrelated.example').navigate(denied[1])).toEqual({ prevented: 0, reloaded: 0 })
  })
})
