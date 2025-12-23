"""CLI interface for Recaller."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from recaller import __version__
from recaller.config import get_settings, Settings
from recaller.database.repository import Repository

app = typer.Typer(
    name="recaller",
    help="Notion-Anki integration tool with semantic deduplication and AI-generated flashcards.",
    no_args_is_help=True,
)
console = Console()


def get_repository(settings: Settings) -> Repository:
    """Get repository instance, ensuring data directory exists."""
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    return Repository(settings.database_url)


@app.command()
def sync(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Preview changes without modifying anything"
    ),
):
    """
    Sync notes from Notion and generate flashcards.

    This is the main pipeline that:
    1. Fetches new notes from Notion
    2. Generates embeddings
    3. Detects and merges similar notes (with confirmation)
    4. Generates flashcards using Gemini
    5. Exports to Anki via AnkiConnect
    """
    settings = get_settings()

    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]\n")

    console.print("[bold blue]Starting Recaller sync...[/bold blue]")
    console.print(f"  Notion page: {settings.notion_random_page_id}")
    console.print(f"  Similarity threshold: {settings.similarity_threshold}")
    console.print(f"  Anki deck: {settings.anki_deck_name}")
    console.print()

    # TODO: Implement pipeline in Milestone 2+
    console.print("[yellow]Pipeline not yet implemented. Coming in Milestone 2![/yellow]")


@app.command()
def status():
    """Show database statistics and sync status."""
    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Make sure you have a .env file with required settings.")
        raise typer.Exit(1)

    if not settings.database_path.exists():
        console.print("[yellow]Database not yet initialized. Run 'recaller sync' first.[/yellow]")
        raise typer.Exit(0)

    repo = get_repository(settings)
    stats = repo.get_stats()

    table = Table(title="Recaller Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Notes", str(stats["total_notes"]))
    table.add_row("  New", str(stats["new_notes"]))
    table.add_row("  Processed", str(stats["processed_notes"]))
    table.add_row("  Merged", str(stats["merged_notes"]))
    table.add_row("", "")
    table.add_row("Total Flashcards", str(stats["total_flashcards"]))
    table.add_row("  Pending Export", str(stats["pending_flashcards"]))
    table.add_row("  Exported", str(stats["exported_flashcards"]))

    console.print(table)


@app.command()
def export():
    """Re-export pending flashcards to Anki."""
    settings = get_settings()

    if not settings.database_path.exists():
        console.print("[yellow]Database not yet initialized. Run 'recaller sync' first.[/yellow]")
        raise typer.Exit(0)

    repo = get_repository(settings)
    from recaller.models.flashcard import ExportStatus

    pending = repo.get_flashcards_by_status(ExportStatus.PENDING)

    if not pending:
        console.print("[green]No pending flashcards to export.[/green]")
        raise typer.Exit(0)

    console.print(f"[blue]Found {len(pending)} pending flashcards to export.[/blue]")

    # TODO: Implement Anki export in Milestone 6
    console.print("[yellow]Export not yet implemented. Coming in Milestone 6![/yellow]")


@app.command()
def config():
    """Show current configuration."""
    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("\nMake sure you have a .env file. See .env.example for reference.")
        raise typer.Exit(1)

    table = Table(title="Recaller Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Mask sensitive values
    notion_token_masked = settings.notion_token[:10] + "..." if len(settings.notion_token) > 10 else "***"
    gemini_key_masked = settings.gemini_api_key[:10] + "..." if len(settings.gemini_api_key) > 10 else "***"

    table.add_row("Notion Token", notion_token_masked)
    table.add_row("Notion Page ID", settings.notion_random_page_id)
    table.add_row("Gemini API Key", gemini_key_masked)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Similarity Threshold", str(settings.similarity_threshold))
    table.add_row("Database Path", str(settings.database_path))
    table.add_row("Anki Deck Name", settings.anki_deck_name)
    table.add_row("AnkiConnect URL", settings.ankiconnect_url)
    table.add_row("Cards per Note", f"{settings.cards_per_note_min}-{settings.cards_per_note_max}")

    console.print(table)


@app.command()
def version():
    """Show version information."""
    console.print(f"Recaller v{__version__}")


@app.callback()
def main():
    """
    Recaller - Notion-Anki integration tool.

    Syncs notes from Notion, detects semantic duplicates,
    generates AI flashcards, and exports to Anki.
    """
    pass


if __name__ == "__main__":
    app()
