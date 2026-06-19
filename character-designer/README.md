# Character Designer (Open-Source, Browser-Only)

Free open-source character designer for AI roleplay chats.

## What it supports
- **SillyTavern**: export JSON in V1 and V2 (`chara_card_v2`) structures
- **SpicyChat**: export/import JSON format
- **Universal card format**: PNG character cards with embedded metadata (`tEXt` chunk, `chara` key)
- **Generic**: plain text system prompt and generic JSON export

## Features
- Character profile editor (name, nickname, age, gender, species/race)
- Personality tags with quick suggestions
- Description, lore, scenario, greeting, and example dialogue fields
- Avatar upload (base64) or URL
- Import: SillyTavern JSON/PNG cards, SpicyChat JSON
- Export: ST V1, ST V2, SpicyChat JSON, generic JSON, system prompt TXT, PNG card
- Template library: warrior, wizard, romance, villain
- Dark/light mode
- Live system prompt preview while typing

## Usage
1. Open `character-designer/public/index.html` in a browser.
2. Build your character profile.
3. Import existing cards from the import panel or export using the export panel.

## Standards references
- SillyTavern character cards (community SPEC_V2 / `chara_card_v2`)
- Chub / Pygmalion ecosystem card interoperability conventions

## Hosting
This is frontend-only and can be hosted for free on:
- GitHub Pages
- Vercel

Suggested repo topics: `roleplay`, `character-card`, `sillytavern`, `spicychat`, `ai-characters`
