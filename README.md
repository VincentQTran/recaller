# Recaller

A CLI tool that syncs notes from Notion to Anki with semantic deduplication and AI-generated flashcards.

## How It Works

```
Notion "Current" page → Fetch notes → Generate embeddings → Detect similar notes
→ Interactive merge confirmation → Generate flashcards (LLM) → Export to Anki
```

Recaller reads notes from your Notion workspace, uses semantic similarity to find and merge duplicate concepts, generates high-quality flashcards using an LLM, and exports them directly to Anki via AnkiConnect.

## Notion Setup

### Page Structure

Create a parent page in Notion (e.g., "Recaller") and share it with your Notion integration. Recaller will automatically create the following structure:

```
Recaller (parent page)
├── Current      ← Add your notes here as subpages
├── Archive      ← Processed notes are moved here
└── Notes Database  ← Deduplicated notes stored here
```

### Writing Notes

Each note is a subpage under "Current". Only the **title** and **content** are required.

You can optionally add metadata flags in the **first 3 lines** of the note content:

| Flag | Description | Example |
|------|-------------|---------|
| `Category:` | Tag for organizing flashcards | `Category: Cooking` |
| `Source:` | Where the information came from | `Source: MIT OCW` |
| `Flashcard:` | Set to any value to skip flashcard generation | `Flashcard: skip` |

**Example note content:**
```
Category: Programming
Source: Effective Python
Flashcard: skip

Python's GIL (Global Interpreter Lock) prevents multiple native threads
from executing Python bytecode simultaneously...
```

Notes with `Flashcard:` specified to any value (e.g. skip) will be processed and stored but won't generate flashcards.

## Installation

### Prerequisites

- Python 3.10+
- [Anki](https://apps.ankiweb.net/) with [AnkiConnect](https://ankiweb.net/shared/info/2055492159) add-on
- A Notion integration token
- Access to an Ollama-compatible LLM API

### Install

```bash
# Clone and install
git clone https://github.com/VincentQTran/recaller.git
cd recaller

# Create environment (optional but recommended)
conda create -n recaller python=3.11
conda activate recaller

# Install package
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Configuration

Copy the example config and fill in your credentials:

```bash
cp .env.example .env
```

### Required Settings

```bash
# Notion - Get from https://www.notion.so/my-integrations
RECALLER_NOTION_TOKEN=secret_xxx
RECALLER_NOTION_PAGE_ID=your-recaller-page-id  # ID of the parent page

# LLM API
RECALLER_OLLAMA_HOST=https://your-ollama-host.com
RECALLER_OLLAMA_MODEL=your-model-name
RECALLER_OLLAMA_API_KEY=your-api-key
```

### Optional Settings

```bash
# Similarity detection (0.5-0.95, higher = stricter matching)
RECALLER_SIMILARITY_THRESHOLD=0.78

# Flashcards per note
RECALLER_CARDS_PER_NOTE_MIN=1
RECALLER_CARDS_PER_NOTE_MAX=3

# AnkiConnect (default: http://localhost:8765)
RECALLER_ANKICONNECT_URL=http://localhost:8765

# Database location
RECALLER_DATABASE_PATH=data/recaller.db
```

## Usage

### Main Commands

```bash
# Run the full sync pipeline
recaller sync

# Preview without making changes
recaller sync --dry-run

# Sync without archiving processed notes
recaller sync --no-archive

# Show database statistics
recaller status

# Re-export pending flashcards to Anki
recaller export

# Show current configuration
recaller config
```

### Testing & Debugging

```bash
# Test Notion connection and view notes
recaller test-notion

# Show note content
recaller test-notion --content

# View archive instead of current
recaller test-notion --archive

# Sync embeddings for database notes
recaller sync-embeddings
```

## Sync Pipeline

When you run `recaller sync`:

1. **Fetch**: Retrieves all notes from the "Current" page in Notion
2. **Embed**: Generates semantic embeddings for each note using sentence-transformers
3. **Deduplicate**: Finds similar notes (above threshold) and prompts you to merge them
4. **Store**: Adds notes to the Notion "Notes Database" for long-term storage
5. **Generate**: Creates 1-3 flashcards per note using the configured LLM
6. **Archive**: Moves processed notes from "Current" to "Archive" (skip with `--no-archive`)
7. **Export**: Sends flashcards to Anki via AnkiConnect

Notes with `Flashcard: skip` are processed through steps 1-4 but skipped for flashcard generation.

## Flashcard Output

Flashcards are exported to Anki with:
- **Deck**: `Recaller::MM-DD-YYYY` (dated by sync)
- **Tags**: `recaller`, `category::your_category`, `source::your_source`
- **Types**: Basic (Q&A) or Cloze deletion

## Development

```bash
# Run tests
pytest

# Run unit tests only
pytest tests/unit/

# Linting
ruff check src/
ruff check --fix src/

# Type checking
mypy src/recaller/
```

## License

MIT
