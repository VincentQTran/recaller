"""CLI interface for Recaller."""


import typer
from rich.console import Console
from rich.table import Table

from recaller import __version__
from recaller.config import Settings, get_settings
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
    4. Generates flashcards using Ollama
    5. Exports to Anki via AnkiConnect
    """
    settings = get_settings()

    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]\n")

    console.print("[bold blue]Starting Recaller sync...[/bold blue]")
    console.print(f"  Notion page: {settings.notion_page_id}")
    console.print(f"  Ollama model: {settings.ollama_model}")
    console.print(f"  Similarity threshold: {settings.similarity_threshold}")
    console.print(f"  Anki deck: {settings.anki_deck_name}")
    console.print()

    # Step 1: Fetch notes from Notion
    console.print("[bold]Step 1:[/bold] Fetching notes from Notion...")
    from recaller.services.notion_client import NotionService

    notion = NotionService(
        token=settings.notion_token,
        recaller_page_id=settings.notion_page_id,
    )

    try:
        notion.ensure_page_structure()
        notes = notion.fetch_current_notes()
    except Exception as e:
        console.print(f"[red]Error fetching notes: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"  Found [green]{len(notes)}[/green] notes in Current\n")

    if not notes:
        console.print("[yellow]No notes to process.[/yellow]")
        raise typer.Exit(0)

    # Step 2: Generate embeddings
    console.print("[bold]Step 2:[/bold] Generating embeddings...")
    from recaller.services.embedding_service import EmbeddingService

    embedding_service = EmbeddingService(model_name=settings.embedding_model)

    with console.status("[yellow]Loading embedding model...[/yellow]"):
        for note in notes:
            note.embedding = embedding_service.generate_note_embedding(note)

    console.print(f"  Generated embeddings for [green]{len(notes)}[/green] notes\n")

    # Step 3: Find similar notes
    console.print("[bold]Step 3:[/bold] Finding similar notes...")
    from recaller.services.similarity_engine import SimilarityEngine

    similarity_engine = SimilarityEngine(threshold=settings.similarity_threshold)
    similar_pairs = similarity_engine.find_similar_pairs(
        notes, same_category_only=True
    )

    if not similar_pairs:
        console.print("  [green]No similar notes found[/green]\n")
    else:
        console.print(
            f"  Found [yellow]{len(similar_pairs)}[/yellow] similar note pairs\n"
        )

        # Display similar pairs
        console.print("[bold]Proposed merges:[/bold]")
        for i, pair in enumerate(similar_pairs, 1):
            console.print(f"\n[cyan]─── Pair {i} ───[/cyan]")
            console.print(f"  [bold]Note A:[/bold] {pair.note_a.title}")
            console.print(f"  [bold]Note B:[/bold] {pair.note_b.title}")
            console.print(f"  [bold]Category:[/bold] {pair.note_a.category}")
            console.print(f"  [bold]Similarity:[/bold] {pair.similarity:.2%}")

    if dry_run:
        console.print("\n[yellow]Dry run complete. No changes were made.[/yellow]")
        raise typer.Exit(0)

    # Initialize Ollama service for merge and flashcard generation
    from recaller.services.ollama_service import OllamaService

    ollama = OllamaService(
        model_name=settings.ollama_model,
        host=settings.ollama_host,
        api_key=settings.ollama_api_key or None,
    )

    # Check Ollama connection
    if not ollama.check_connection():
        console.print(
            f"[red]Cannot connect to Ollama at {settings.ollama_host}. "
            "Check your API key and connection.[/red]"
        )
        raise typer.Exit(1)
    console.print(f"  [green]✓[/green] Connected to Ollama ({settings.ollama_model})\n")

    # Step 4: Interactive merge confirmation
    if similar_pairs:
        console.print("[bold]Step 4:[/bold] Interactive merge confirmation...")

        from recaller.services.merge_engine import MergeEngine

        merge_engine = MergeEngine(ollama_service=ollama)

        # Create merge proposals from pairs
        proposals = merge_engine.create_proposals_from_pairs(similar_pairs)
        console.print(f"  Created [cyan]{len(proposals)}[/cyan] merge proposal(s)\n")

        merged_notes: list = []
        skipped_proposals = 0

        for i, proposal in enumerate(proposals, 1):
            console.print(
                f"\n[bold cyan]═══ Merge Proposal {i}/{len(proposals)} ═══[/bold cyan]"
            )

            # Display notes in this proposal
            console.print(f"[bold]Notes to merge ({len(proposal.notes)}):[/bold]")
            for j, note in enumerate(proposal.notes, 1):
                console.print(f"  {j}. {note.title}")
                if note.source:
                    console.print(f"     [dim]Source: {note.source}[/dim]")

            # Show similarity info
            if proposal.similarity_scores:
                avg_sim = (
                    sum(proposal.similarity_scores) / len(proposal.similarity_scores)
                )
                console.print(f"\n[bold]Average similarity:[/bold] {avg_sim:.2%}")

            # Generate suggested title
            with console.status("[yellow]Generating combined title...[/yellow]"):
                try:
                    suggested_title = merge_engine.generate_title_for_proposal(proposal)
                except Exception as e:
                    console.print(f"[red]Error generating title: {e}[/red]")
                    suggested_title = proposal.primary_note.title

            console.print(f"\n[bold]Suggested title:[/bold] {suggested_title}")

            # Prompt for action
            console.print("\n[bold]Options:[/bold]")
            console.print("  [green]y[/green] - Accept merge with suggested title")
            console.print("  [yellow]e[/yellow] - Edit title before merging")
            console.print("  [red]n[/red] - Skip this merge")
            console.print("  [red]q[/red] - Quit merging (skip remaining)")

            choice = typer.prompt("Choice", default="y").lower().strip()

            if choice == "q":
                console.print("[yellow]Skipping remaining merges.[/yellow]")
                skipped_proposals += len(proposals) - i
                break
            elif choice == "n":
                console.print("[yellow]Skipping this merge.[/yellow]")
                skipped_proposals += 1
                continue
            elif choice == "e":
                final_title = typer.prompt("Enter new title", default=suggested_title)
            else:  # 'y' or default
                final_title = suggested_title

            # Execute the merge
            try:
                result = merge_engine.execute_merge(proposal, final_title)
                merged_notes.append(result)
                console.print(f"[green]✓ Merged into:[/green] {final_title}")
            except Exception as e:
                console.print(f"[red]Error executing merge: {e}[/red]")
                skipped_proposals += 1

        # Summary
        console.print("\n[bold]Merge summary:[/bold]")
        console.print(f"  Merged: [green]{len(merged_notes)}[/green] proposal(s)")
        console.print(f"  Skipped: [yellow]{skipped_proposals}[/yellow] proposal(s)")

        # Update notes list with merged results
        if merged_notes:
            # Replace merged notes with their merged versions
            merged_page_ids = set()
            for result in merged_notes:
                for original in result.original_notes[1:]:  # Skip primary
                    merged_page_ids.add(original.notion_page_id)
                # Update primary note with merged content
                for i, note in enumerate(notes):
                    if note.notion_page_id == result.merged_note.notion_page_id:
                        notes[i] = result.merged_note
                        break

            # Remove merged secondary notes from processing
            notes = [n for n in notes if n.notion_page_id not in merged_page_ids]
            console.print(
                f"  [green]Updated notes list:[/green] {len(notes)} notes to process\n"
            )
    else:
        console.print("[bold]Step 4:[/bold] No merges needed - skipping.\n")

    # Step 5: Generate flashcards
    console.print("[bold]Step 5:[/bold] Generating flashcards...")

    from recaller.services.flashcard_generator import FlashcardGenerator

    flashcard_gen = FlashcardGenerator(
        ollama_service=ollama,
        cards_per_note_min=settings.cards_per_note_min,
        cards_per_note_max=settings.cards_per_note_max,
        deck_name=settings.anki_deck_name,
    )

    all_flashcards: list = []
    failed_notes = 0
    processed_notes: list = []  # Track successfully processed notes for archiving

    with console.status("[yellow]Generating flashcards...[/yellow]") as status:
        for i, note in enumerate(notes, 1):
            status.update(
                f"[yellow]Generating flashcards for note {i}/{len(notes)}...[/yellow]"
            )
            try:
                flashcards = flashcard_gen.generate_flashcards(note)
                all_flashcards.extend(flashcards)
                processed_notes.append(note)
                console.print(
                    f"  [green]✓[/green] {note.title}: {len(flashcards)} card(s)"
                )
            except Exception as e:
                console.print(f"  [red]✗[/red] {note.title}: {e}")
                failed_notes += 1

    console.print("\n[bold]Flashcard summary:[/bold]")
    console.print(f"  Generated: [green]{len(all_flashcards)}[/green] flashcard(s)")
    console.print(f"  Failed notes: [yellow]{failed_notes}[/yellow]")

    if all_flashcards:
        # Preview generated flashcards
        console.print("\n[bold]Preview of generated flashcards:[/bold]")
        for i, card in enumerate(all_flashcards[:5], 1):
            console.print(f"\n[cyan]─── Card {i} ───[/cyan]")
            front_preview = card.front[:100] + "..." if len(card.front) > 100 else card.front
            back_preview = card.back[:100] + "..." if len(card.back) > 100 else card.back
            console.print(f"  [bold]Q:[/bold] {front_preview}")
            console.print(f"  [bold]A:[/bold] {back_preview}")
            console.print(f"  [dim]Tags: {', '.join(card.tags)}[/dim]")

        if len(all_flashcards) > 5:
            console.print(f"\n[dim]... and {len(all_flashcards) - 5} more flashcard(s)[/dim]")

    # Step 6: Archive processed notes
    if processed_notes:
        console.print("\n[bold]Step 6:[/bold] Archiving processed notes...")

        archive_ids = [note.notion_page_id for note in processed_notes]
        archive_results = notion.archive_notes(archive_ids)

        archived_count = sum(1 for success in archive_results.values() if success)
        failed_archive = len(archive_results) - archived_count

        console.print(f"  Archived: [green]{archived_count}[/green] note(s)")
        if failed_archive > 0:
            console.print(f"  Failed to archive: [red]{failed_archive}[/red] note(s)")
    else:
        console.print("\n[bold]Step 6:[/bold] No notes to archive.\n")

    # Step 7: Export to Anki
    if all_flashcards:
        console.print("\n[bold]Step 7:[/bold] Exporting to Anki...")

        from recaller.services.anki_exporter import AnkiConnectError, AnkiExporter

        anki = AnkiExporter(url=settings.ankiconnect_url)

        # Check connection
        if not anki.check_connection():
            console.print(
                "[red]Cannot connect to AnkiConnect. "
                "Make sure Anki is running with AnkiConnect add-on installed.[/red]"
            )
            console.print(
                "[yellow]Flashcards were generated but not exported. "
                "Run 'recaller export' later to export them.[/yellow]"
            )
        else:
            console.print("  [green]✓[/green] Connected to AnkiConnect")

            # Ensure deck exists
            if anki.ensure_deck_exists(settings.anki_deck_name):
                console.print(f"  [green]✓[/green] Deck '{settings.anki_deck_name}' ready")
            else:
                console.print("  [yellow]Warning: Could not verify deck[/yellow]")

            # Export flashcards
            try:
                results = anki.add_notes(all_flashcards)

                exported = sum(1 for r in results if r.success)
                failed = len(results) - exported

                console.print("\n[bold]Export summary:[/bold]")
                console.print(f"  Exported: [green]{exported}[/green] flashcard(s)")
                if failed > 0:
                    console.print(f"  Failed: [red]{failed}[/red] flashcard(s)")
                    # Show first few errors
                    errors = [r for r in results if not r.success][:3]
                    for err in errors:
                        console.print(f"    [dim]- {err.error}[/dim]")

                # Offer to sync with AnkiWeb
                if exported > 0:
                    console.print("\n[green]✓ Flashcards exported successfully![/green]")

            except AnkiConnectError as e:
                console.print(f"[red]Export failed: {e}[/red]")
    else:
        console.print("\n[bold]Step 7:[/bold] No flashcards to export.\n")

    console.print("\n[bold blue]Sync complete![/bold blue]")


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

    from recaller.services.anki_exporter import AnkiConnectError, AnkiExporter

    anki = AnkiExporter(url=settings.ankiconnect_url)

    # Check connection
    if not anki.check_connection():
        console.print(
            "[red]Cannot connect to AnkiConnect. "
            "Make sure Anki is running with AnkiConnect add-on installed.[/red]"
        )
        raise typer.Exit(1)

    console.print("[green]✓[/green] Connected to AnkiConnect")

    # Ensure deck exists
    if anki.ensure_deck_exists(settings.anki_deck_name):
        console.print(f"[green]✓[/green] Deck '{settings.anki_deck_name}' ready")

    # Export flashcards
    try:
        results = anki.add_notes(pending)

        exported = sum(1 for r in results if r.success)
        failed = len(results) - exported

        console.print("\n[bold]Export summary:[/bold]")
        console.print(f"  Exported: [green]{exported}[/green] flashcard(s)")
        if failed > 0:
            console.print(f"  Failed: [red]{failed}[/red] flashcard(s)")

        if exported > 0:
            console.print("\n[green]✓ Export complete![/green]")

    except AnkiConnectError as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(1)


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
    notion_token_masked = (
        settings.notion_token[:10] + "..." if len(settings.notion_token) > 10 else "***"
    )

    # Mask Ollama API key
    ollama_key_masked = (
        settings.ollama_api_key[:10] + "..." if len(settings.ollama_api_key) > 10 else "***"
    ) if settings.ollama_api_key else "(from env)"

    table.add_row("Notion Token", notion_token_masked)
    table.add_row("Notion Page ID", settings.notion_page_id)
    table.add_row("Ollama Host", settings.ollama_host)
    table.add_row("Ollama Model", settings.ollama_model)
    table.add_row("Ollama API Key", ollama_key_masked)
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


@app.command("test-notion")
def test_notion(
    limit: int = typer.Option(5, "--limit", "-l", help="Max notes to fetch (0 for all)"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show note content"),
    archive: bool = typer.Option(False, "--archive", "-a", help="Show archive notes"),
):
    """Test Notion connection and fetch notes from Current or Archive."""
    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print(
            "\nMake sure you have a .env file with "
            "RECALLER_NOTION_TOKEN and RECALLER_NOTION_PAGE_ID."
        )
        raise typer.Exit(1)

    console.print("[bold blue]Testing Notion connection...[/bold blue]")
    console.print(f"  Recaller Page ID: {settings.notion_page_id}\n")

    try:
        from recaller.services.notion_client import NotionService

        service = NotionService(
            token=settings.notion_token,
            recaller_page_id=settings.notion_page_id,
        )

        # Ensure page structure exists
        console.print("[yellow]Checking page structure...[/yellow]")
        current_id, archive_id = service.ensure_page_structure()
        console.print(f"  [green]✓[/green] Current page: {current_id[:8]}...")
        console.print(f"  [green]✓[/green] Archive page: {archive_id[:8]}...")
        console.print()

        # Fetch notes from appropriate page
        source = "Archive" if archive else "Current"
        console.print(f"[yellow]Fetching notes from {source}...[/yellow]")

        if archive:
            notes = service.fetch_archive_notes()
        else:
            notes = service.fetch_current_notes()

        console.print(f"\n[green]Successfully fetched {len(notes)} notes from {source}![/green]\n")

        if not notes:
            console.print(f"[yellow]No notes found in {source}. Add some note pages![/yellow]")
            return

        display_notes = notes[:limit] if limit > 0 else notes

        for i, note in enumerate(display_notes, 1):
            console.print(f"[bold cyan]─── Note {i} ───[/bold cyan]")
            console.print(f"  [bold]Title:[/bold] {note.title}")
            console.print(f"  [bold]Category:[/bold] {note.category or '(none)'}")
            console.print(f"  [bold]Source:[/bold] {note.source or '(none)'}")
            console.print(f"  [bold]Notion ID:[/bold] {note.notion_page_id}")
            if note.notion_last_edited:
                console.print(f"  [bold]Last Edited:[/bold] {note.notion_last_edited}")

            if show_content:
                content_preview = (
                    note.content[:500] + "..." if len(note.content) > 500 else note.content
                )
                console.print(f"  [bold]Content:[/bold]\n{content_preview}")

            console.print()

        if limit > 0 and len(notes) > limit:
            console.print(
                f"[dim]Showing {limit} of {len(notes)} notes. Use --limit 0 to see all.[/dim]"
            )

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("\n[yellow]Troubleshooting tips:[/yellow]")
        console.print("  1. Verify your RECALLER_NOTION_TOKEN is correct")
        console.print("  2. Verify your RECALLER_NOTION_PAGE_ID is the Recaller parent page")
        console.print("  3. Make sure the Notion integration has access to the page")
        raise typer.Exit(1)


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
