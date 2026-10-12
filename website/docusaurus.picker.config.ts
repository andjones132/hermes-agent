import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// The normal /docs/ artifact also backs the canonical domain through its proxy.
// Do not change that base to fix direct GitHub Pages embedding: publish a small,
// separate picker at /hermes-agent/picker/ with its own router and asset base.
const config: Config = {
  title: 'Hermes Skills Hub',
  url: 'https://nousresearch.github.io',
  baseUrl: '/hermes-agent/picker/',
  trailingSlash: true,
  customFields: {standalonePicker: true},
  favicon: 'img/favicon.ico',
  organizationName: 'NousResearch',
  projectName: 'hermes-agent',
  onBrokenLinks: 'throw',
  i18n: {defaultLocale: 'en', locales: ['en']},
  presets: [
    ['classic', {
      docs: false,
      blog: false,
      pages: {include: ['skills/index.tsx']},
      theme: {customCss: './src/css/custom.css'},
    } satisfies Preset.Options],
  ],
  themeConfig: {
    colorMode: {defaultMode: 'dark', respectPrefersColorScheme: true},
    navbar: {
      title: 'Hermes Skills Hub',
      logo: {src: 'img/logo.png', href: 'https://hermes-agent.nousresearch.com/docs/skills'},
      items: [],
    },
    footer: {links: []},
  } satisfies Preset.ThemeConfig,
};

export default config;
