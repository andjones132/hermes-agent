// Check the emitted artifact, not config text: a 200 HTML page with root-relative
// /docs/assets URLs still cannot hydrate on the GitHub Pages project mirror.
// Usage: node scripts/check-picker-build.mjs <build-dir> <public-base-path>
import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const [buildDir, basePath] = process.argv.slice(2);
assert(buildDir && basePath?.startsWith('/') && basePath.endsWith('/'),
  'Pass a build directory and an absolute base path ending in /');
const html = await readFile(resolve(buildDir, 'skills/index.html'), 'utf8');
const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)].map(m => m[1]);
assert(assets.some(url => url.endsWith('.js')), 'Picker must emit JavaScript');
assert(assets.some(url => url.endsWith('.css')), 'Picker must emit styles');
for (const url of assets) {
  assert(url.startsWith(basePath), `${url} is outside the deployed base ${basePath}`);
  assert((await stat(resolve(buildDir, url.slice(basePath.length)))).isFile(), `${url} is missing`);
}
// Check the shared cross-navigation, not the navbar (which has separate links).
// Canonical locales must remain internal/same-tab; only the standalone picker
// lacks a Plugins route and therefore needs the English canonical external link.
const standalone = basePath === '/hermes-agent/picker/';
const expectedPluginsHref = standalone
  ? 'https://hermes-agent.nousresearch.com/docs/plugins'
  : `${basePath}plugins`;
const crossNav = html.match(/<nav\b[^>]*aria-label="Catalog pages"[^>]*>([\s\S]*?)<\/nav>/)?.[1];
assert(crossNav, 'Skills page must emit catalog cross-navigation');
const pluginsAnchors = [...crossNav.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/g)]
  .filter(match => match[2].replace(/<[^>]*>/g, '').trim() === 'Plugins');
assert.equal(pluginsAnchors.length, 1, 'Expected one Plugins cross-navigation anchor');
const attributes = Object.fromEntries([...pluginsAnchors[0][1].matchAll(/([\w-]+)="([^"]*)"/g)]
  .map(match => [match[1], match[2]]));
assert.equal(attributes.href, expectedPluginsHref, 'Plugins must retain the build/locale destination');
assert.equal(attributes.target, standalone ? '_blank' : undefined,
  'Only the standalone Plugins link should open an external tab');
if (standalone) {
  const rel = new Set((attributes.rel ?? '').split(/\s+/));
  assert(rel.has('noopener') && rel.has('noreferrer'), 'External Plugins link must be protected');
} else {
  assert.equal(attributes.rel, undefined, 'Canonical Plugins link must remain internal');
}
const skills = JSON.parse(await readFile(resolve(buildDir, 'api/skills.json'), 'utf8'));
assert(Array.isArray(skills) && skills.length > 0, 'Picker needs a populated catalog');
console.log(JSON.stringify({ basePath, assetCount: assets.length, skillCount: skills.length, status: 'pass' }));
