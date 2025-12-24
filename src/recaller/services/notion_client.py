"""Notion API client for fetching notes."""

import re
from datetime import datetime
from typing import Any, Optional, cast

from notion_client import Client
from notion_client.errors import APIResponseError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from recaller.models.note import Note, NoteStatus


class NotionService:
    """Service for interacting with Notion API.

    Manages notes in a Recaller page structure:
        Recaller (parent page)
        ├── Current (notes for this week)
        └── Archive (archived notes)
    """

    CURRENT_PAGE_NAME = "Current"
    ARCHIVE_PAGE_NAME = "Archive"

    def __init__(self, token: str, recaller_page_id: str):
        """Initialize Notion client.

        Args:
            token: Notion integration token
            recaller_page_id: ID of the "Recaller" parent page
        """
        self.client = Client(auth=token)
        self.recaller_page_id = recaller_page_id
        self._current_page_id: Optional[str] = None
        self._archive_page_id: Optional[str] = None

    def ensure_page_structure(self) -> tuple[str, str]:
        """Ensure Current and Archive pages exist, creating them if needed.

        Returns:
            Tuple of (current_page_id, archive_page_id)
        """
        current_id = self._find_or_create_subpage(self.CURRENT_PAGE_NAME)
        archive_id = self._find_or_create_subpage(self.ARCHIVE_PAGE_NAME)

        self._current_page_id = current_id
        self._archive_page_id = archive_id

        return current_id, archive_id

    def get_current_page_id(self) -> str:
        """Get the Current page ID, ensuring it exists.

        Returns:
            Current page ID
        """
        if not self._current_page_id:
            self._current_page_id = self._find_or_create_subpage(self.CURRENT_PAGE_NAME)
        return self._current_page_id

    def get_archive_page_id(self) -> str:
        """Get the Archive page ID, ensuring it exists.

        Returns:
            Archive page ID
        """
        if not self._archive_page_id:
            self._archive_page_id = self._find_or_create_subpage(self.ARCHIVE_PAGE_NAME)
        return self._archive_page_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIResponseError),
    )
    def _find_or_create_subpage(self, page_name: str) -> str:
        """Find a subpage by name or create it if it doesn't exist.

        Args:
            page_name: Name of the subpage to find/create

        Returns:
            Page ID of the found or created page
        """
        # Search for existing page
        children = self._get_all_children(self.recaller_page_id)

        for child in children:
            if child.get("type") == "child_page":
                child_title = child.get("child_page", {}).get("title", "")
                if child_title == page_name:
                    return child["id"]

        # Page not found, create it
        return self._create_subpage(page_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIResponseError),
    )
    def _create_subpage(self, page_name: str) -> str:
        """Create a new subpage under the Recaller page.

        Args:
            page_name: Name for the new page

        Returns:
            ID of the created page
        """
        response: dict[str, Any] = cast(
            dict[str, Any],
            self.client.pages.create(
                parent={"page_id": self.recaller_page_id},
                properties={
                    "title": {
                        "title": [{"text": {"content": page_name}}]
                    }
                },
            ),
        )
        return response["id"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIResponseError),
    )
    def fetch_current_notes(self) -> list[Note]:
        """Fetch all notes from the Current page.

        Returns:
            List of Note objects from the Current page
        """
        current_page_id = self.get_current_page_id()
        return self._fetch_notes_from_page(current_page_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIResponseError),
    )
    def fetch_archive_notes(self) -> list[Note]:
        """Fetch all notes from the Archive page.

        Returns:
            List of Note objects from the Archive page
        """
        archive_page_id = self.get_archive_page_id()
        return self._fetch_notes_from_page(archive_page_id)

    def _fetch_notes_from_page(self, page_id: str) -> list[Note]:
        """Fetch all note subpages from a given page.

        Args:
            page_id: Parent page ID to fetch notes from

        Returns:
            List of Note objects
        """
        notes = []
        children = self._get_all_children(page_id)

        for child in children:
            if child.get("type") == "child_page":
                note_page_id = child["id"]
                note = self._fetch_page_as_note(note_page_id)
                if note:
                    notes.append(note)

        return notes

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIResponseError),
    )
    def archive_note(self, note_page_id: str) -> bool:
        """Move a note from Current to Archive.

        Args:
            note_page_id: ID of the note page to archive

        Returns:
            True if successful, False otherwise
        """
        archive_page_id = self.get_archive_page_id()

        try:
            # Update the page's parent to move it to Archive
            self.client.pages.update(
                page_id=note_page_id,
                parent={"page_id": archive_page_id},
            )
            return True
        except APIResponseError:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIResponseError),
    )
    def archive_notes(self, note_page_ids: list[str]) -> dict[str, bool]:
        """Move multiple notes from Current to Archive.

        Args:
            note_page_ids: List of note page IDs to archive

        Returns:
            Dict mapping page_id to success status
        """
        results = {}
        for page_id in note_page_ids:
            results[page_id] = self.archive_note(page_id)
        return results

    # Legacy method for backwards compatibility
    def fetch_notes(self) -> list[Note]:
        """Fetch all notes from Current page (legacy method).

        Returns:
            List of Note objects from Notion
        """
        return self.fetch_current_notes()

    def _get_all_children(self, block_id: str) -> list[dict[str, Any]]:
        """Get all child blocks with pagination.

        Args:
            block_id: Parent block ID

        Returns:
            List of child block objects
        """
        children: list[dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            response: dict[str, Any] = cast(
                dict[str, Any],
                self.client.blocks.children.list(
                    block_id=block_id,
                    start_cursor=cursor,
                ),
            )
            children.extend(response["results"])

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        return children

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIResponseError),
    )
    def _fetch_page_as_note(self, page_id: str) -> Optional[Note]:
        """Fetch a single page and convert to Note.

        Args:
            page_id: Notion page ID

        Returns:
            Note object or None if page cannot be parsed
        """
        try:
            # Fetch page properties
            page: dict[str, Any] = cast(
                dict[str, Any], self.client.pages.retrieve(page_id=page_id)
            )

            # Parse properties
            title = self._extract_title(page)
            if not title:
                return None

            # Try to get category/source from page properties first
            category = self._extract_property(page, "Category", "select")
            source = self._extract_property(page, "Source", "url") or \
                     self._extract_property(page, "Source", "rich_text")

            # Parse last edited time
            last_edited = None
            if page.get("last_edited_time"):
                last_edited = datetime.fromisoformat(
                    page["last_edited_time"].replace("Z", "+00:00")
                )

            # Fetch content blocks
            content = self._fetch_content(page_id)

            # If category/source not in properties, try to extract from content
            if not category:
                category = self._extract_metadata_from_content(content, "category")
            if not source:
                source = self._extract_metadata_from_content(content, "source")

            return Note(
                notion_page_id=page_id,
                title=title,
                category=category or "",
                source=source or "",
                content=content,
                notion_last_edited=last_edited,
                status=NoteStatus.NEW,
            )

        except APIResponseError:
            raise
        except Exception:
            return None

    def _extract_metadata_from_content(
        self, content: str, field: str
    ) -> Optional[str]:
        """Extract metadata (category/source) from content text.

        Looks for patterns like "Category: value" or "source: value" (case-insensitive).

        Args:
            content: The note content as markdown
            field: The field name to look for ("category" or "source")

        Returns:
            The extracted value or None if not found
        """
        # Pattern matches "field: value" at start of line, case-insensitive
        # Uses [ \t]* instead of \s* to avoid matching newlines
        # Captures everything after the colon until end of line
        pattern = rf"^{field}[ \t]*:[ \t]*([^\n]*)"
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)

        if match:
            value = match.group(1).strip()
            # Return None if the value is empty or just whitespace
            return value if value else None

        return None

    def _extract_title(self, page: dict[str, Any]) -> Optional[str]:
        """Extract title from page properties.

        Args:
            page: Notion page object

        Returns:
            Title string or None
        """
        properties = page.get("properties", {})

        # Try common title property names
        for prop_name in ["title", "Title", "Name", "name"]:
            prop = properties.get(prop_name)
            if prop and prop.get("type") == "title":
                title_parts = prop.get("title", [])
                if title_parts:
                    return "".join(t.get("plain_text", "") for t in title_parts)

        # Fallback: find any title-type property
        for prop in properties.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                if title_parts:
                    return "".join(t.get("plain_text", "") for t in title_parts)

        return None

    def _extract_property(
        self, page: dict[str, Any], prop_name: str, prop_type: str
    ) -> Optional[str]:
        """Extract a property value from page.

        Args:
            page: Notion page object
            prop_name: Property name to look for
            prop_type: Expected property type

        Returns:
            Property value as string or None
        """
        properties = page.get("properties", {})
        prop = properties.get(prop_name)

        if not prop:
            return None

        if prop.get("type") == "select" and prop_type == "select":
            select_val = prop.get("select")
            return select_val.get("name") if select_val else None

        if prop.get("type") == "url" and prop_type == "url":
            url_val: Optional[str] = prop.get("url")
            return url_val

        if prop.get("type") == "rich_text" and prop_type == "rich_text":
            texts = prop.get("rich_text", [])
            if texts:
                return "".join(t.get("plain_text", "") for t in texts)

        if prop.get("type") == "multi_select" and prop_type == "multi_select":
            options = prop.get("multi_select", [])
            return ", ".join(o.get("name", "") for o in options)

        return None

    def _fetch_content(self, page_id: str) -> str:
        """Fetch all content blocks from a page as markdown.

        Args:
            page_id: Notion page ID

        Returns:
            Content as markdown string
        """
        blocks = self._get_all_children(page_id)
        return self._blocks_to_markdown(blocks)

    def _blocks_to_markdown(self, blocks: list[dict[str, Any]], indent: int = 0) -> str:
        """Convert Notion blocks to markdown.

        Args:
            blocks: List of Notion block objects
            indent: Current indentation level

        Returns:
            Markdown string
        """
        lines = []
        indent_str = "  " * indent

        for block in blocks:
            block_type = block.get("type")

            if block_type == "paragraph":
                text = self._rich_text_to_markdown(
                    block.get("paragraph", {}).get("rich_text", [])
                )
                if text:
                    lines.append(f"{indent_str}{text}")
                else:
                    lines.append("")

            elif block_type == "heading_1":
                text = self._rich_text_to_markdown(
                    block.get("heading_1", {}).get("rich_text", [])
                )
                lines.append(f"{indent_str}# {text}")

            elif block_type == "heading_2":
                text = self._rich_text_to_markdown(
                    block.get("heading_2", {}).get("rich_text", [])
                )
                lines.append(f"{indent_str}## {text}")

            elif block_type == "heading_3":
                text = self._rich_text_to_markdown(
                    block.get("heading_3", {}).get("rich_text", [])
                )
                lines.append(f"{indent_str}### {text}")

            elif block_type == "bulleted_list_item":
                text = self._rich_text_to_markdown(
                    block.get("bulleted_list_item", {}).get("rich_text", [])
                )
                lines.append(f"{indent_str}- {text}")
                # Handle nested children
                if block.get("has_children"):
                    children = self._get_all_children(block["id"])
                    nested = self._blocks_to_markdown(children, indent + 1)
                    if nested:
                        lines.append(nested)

            elif block_type == "numbered_list_item":
                text = self._rich_text_to_markdown(
                    block.get("numbered_list_item", {}).get("rich_text", [])
                )
                lines.append(f"{indent_str}1. {text}")
                if block.get("has_children"):
                    children = self._get_all_children(block["id"])
                    nested = self._blocks_to_markdown(children, indent + 1)
                    if nested:
                        lines.append(nested)

            elif block_type == "to_do":
                text = self._rich_text_to_markdown(
                    block.get("to_do", {}).get("rich_text", [])
                )
                checked = block.get("to_do", {}).get("checked", False)
                checkbox = "[x]" if checked else "[ ]"
                lines.append(f"{indent_str}- {checkbox} {text}")

            elif block_type == "toggle":
                text = self._rich_text_to_markdown(
                    block.get("toggle", {}).get("rich_text", [])
                )
                lines.append(f"{indent_str}<details>")
                lines.append(f"{indent_str}<summary>{text}</summary>")
                if block.get("has_children"):
                    children = self._get_all_children(block["id"])
                    nested = self._blocks_to_markdown(children, indent)
                    if nested:
                        lines.append(nested)
                lines.append(f"{indent_str}</details>")

            elif block_type == "code":
                code_block = block.get("code", {})
                language = code_block.get("language", "")
                text = self._rich_text_to_markdown(code_block.get("rich_text", []))
                lines.append(f"{indent_str}```{language}")
                lines.append(f"{indent_str}{text}")
                lines.append(f"{indent_str}```")

            elif block_type == "quote":
                text = self._rich_text_to_markdown(
                    block.get("quote", {}).get("rich_text", [])
                )
                for line in text.split("\n"):
                    lines.append(f"{indent_str}> {line}")

            elif block_type == "callout":
                callout = block.get("callout", {})
                icon = callout.get("icon", {})
                emoji = icon.get("emoji", "") if icon.get("type") == "emoji" else ""
                text = self._rich_text_to_markdown(callout.get("rich_text", []))
                lines.append(f"{indent_str}> {emoji} {text}")

            elif block_type == "divider":
                lines.append(f"{indent_str}---")

            elif block_type == "bookmark":
                url = block.get("bookmark", {}).get("url", "")
                lines.append(f"{indent_str}[Bookmark]({url})")

            elif block_type == "image":
                image = block.get("image", {})
                url = ""
                if image.get("type") == "external":
                    url = image.get("external", {}).get("url", "")
                elif image.get("type") == "file":
                    url = image.get("file", {}).get("url", "")
                caption = self._rich_text_to_markdown(image.get("caption", []))
                lines.append(f"{indent_str}![{caption}]({url})")

            elif block_type == "child_page":
                # Skip child pages - we handle them at the top level
                pass

            elif block_type == "child_database":
                # Skip child databases
                pass

        return "\n".join(lines)

    def _rich_text_to_markdown(self, rich_text: list[dict[str, Any]]) -> str:
        """Convert Notion rich text to markdown.

        Args:
            rich_text: List of rich text objects

        Returns:
            Markdown string
        """
        parts = []

        for rt in rich_text:
            text = rt.get("plain_text", "")
            annotations = rt.get("annotations", {})
            href = rt.get("href")

            # Apply formatting
            if annotations.get("bold"):
                text = f"**{text}**"
            if annotations.get("italic"):
                text = f"*{text}*"
            if annotations.get("strikethrough"):
                text = f"~~{text}~~"
            if annotations.get("code"):
                text = f"`{text}`"
            if href:
                text = f"[{text}]({href})"

            parts.append(text)

        return "".join(parts)
