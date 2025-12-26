"""Notion API client for fetching notes.

Uses the Notion REST API directly via requests library.
"""

import re
from datetime import datetime
from typing import Any, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from recaller.models.note import Note, NoteStatus


class NotionAPIError(Exception):
    """Exception raised for Notion API errors.

    Args:
        message (str): Error message
        status_code (int): HTTP status code
        code (str): Notion error code

    Attributes:
        message (str): Error message
        status_code (int): HTTP status code
        code (str): Notion error code
    """

    def __init__(self, message: str, status_code: int = 0, code: str = ""):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(self.message)


class NotionService:
    """Service for interacting with Notion API.

    Uses raw HTTP requests to the Notion REST API.

    Manages notes in a Recaller page structure:
        Recaller (parent page)
        ├── Current (notes for this week)
        ├── Archive (archived notes)
        └── Notes Database (full page database)

    Args:
        token (str): Notion integration token
        recaller_page_id (str): ID of the "Recaller" parent page

    Attributes:
        token (str): Notion integration token
        api_version (str): Notion API version
        base_url (str): Notion API base URL
        recaller_page_id (str): ID of the "Recaller" parent page
    """

    CURRENT_PAGE_NAME = "Current"
    ARCHIVE_PAGE_NAME = "Archive"
    DATABASE_NAME = "Notes Database"

    def __init__(self, token: str, recaller_page_id: str):
        """Initialize Notion client.

        Args:
            token: Notion integration token
            recaller_page_id: ID of the "Recaller" parent page
        """
        self.token = token
        self.api_version = "2025-09-03"
        self.base_url = "https://api.notion.com/v1"
        self.recaller_page_id = recaller_page_id
        self._current_page_id: Optional[str] = None
        self._archive_page_id: Optional[str] = None
        self._database_id: Optional[str] = None
        self._data_source_id: Optional[str] = None

    def _get_headers(self) -> dict[str, str]:
        """Get standard headers for Notion API requests.

        Returns:
            dict[str, str]: Headers dict with Authorization, Content-Type, and Notion-Version
        """
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Notion-Version": self.api_version,
        }

    def _handle_response(self, response: requests.Response) -> dict[str, Any]:
        """Handle API response and raise NotionAPIError on failure.

        Args:
            response (requests.Response): Response from requests library

        Returns:
            dict[str, Any]: Parsed JSON response

        Raises:
            NotionAPIError: If the API returns an error status
        """
        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise NotionAPIError(
                    message=error_data.get("message", "Unknown error"),
                    status_code=response.status_code,
                    code=error_data.get("code", ""),
                )
            except (ValueError, KeyError):
                raise NotionAPIError(
                    message=response.text or "Unknown error",
                    status_code=response.status_code,
                )
        return response.json()

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request to the Notion API.

        Args:
            endpoint (str): API endpoint path (e.g., "/pages")
            payload (dict[str, Any]): JSON payload to send

        Returns:
            dict[str, Any]: Parsed JSON response
        """
        response = requests.post(
            f"{self.base_url}{endpoint}",
            headers=self._get_headers(),
            json=payload,
        )
        return self._handle_response(response)

    def _get(self, endpoint: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Make a GET request to the Notion API.

        Args:
            endpoint (str): API endpoint path (e.g., "/pages/{page_id}")
            params (dict[str, Any], optional): Query parameters

        Returns:
            dict[str, Any]: Parsed JSON response
        """
        response = requests.get(
            f"{self.base_url}{endpoint}",
            headers=self._get_headers(),
            params=params,
        )
        return self._handle_response(response)

    def _patch(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a PATCH request to the Notion API.

        Args:
            endpoint (str): API endpoint path (e.g., "/pages/{page_id}")
            payload (dict[str, Any]): JSON payload to send

        Returns:
            dict[str, Any]: Parsed JSON response
        """
        response = requests.patch(
            f"{self.base_url}{endpoint}",
            headers=self._get_headers(),
            json=payload,
        )
        return self._handle_response(response)

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

    def get_database_id(self) -> str:
        """Get the Notes Database ID, ensuring it exists.

        Returns:
            Database ID
        """
        if not self._database_id:
            self._database_id = self._find_or_create_database()
        return self._database_id

    def get_data_source_id(self) -> str:
        """Get the Notes Database data source ID, ensuring it exists.

        Returns:
            Data source ID
        """
        if not self._data_source_id:
            database_id = self.get_database_id()
            self._data_source_id = self._get_data_source_id_from_database(database_id)
        return self._data_source_id

    def _find_or_create_database(self) -> str:
        """Find the Notes Database or create it if it doesn't exist.

        Returns:
            Database ID
        """
        # Search for existing database
        children = self._get_all_children(self.recaller_page_id)

        for child in children:
            if child.get("type") == "child_database":
                db_title = child.get("child_database", {}).get("title", "")
                if db_title == self.DATABASE_NAME:
                    return child["id"]

        # Database not found, create it
        return self._create_database()

    def _create_database(self) -> str:
        """Create the Notes Database with required columns.

        Uses the initial_data_source format for API version 2025-09-03.

        Returns:
            Database ID
        """
        payload = {
            "parent": {"type": "page_id", "page_id": self.recaller_page_id},
            "title": [{"type": "text", "text": {"content": self.DATABASE_NAME}}],
            "initial_data_source": {
                "properties": {
                    "Title": {"title": {}},
                    "Category": {"select": {"options": []}},
                    "Source": {"rich_text": {}},
                    "Date Imported": {"date": {}},
                }
            },
        }
        result = self._post("/databases", payload)
        # Extract data source ID from response
        if "data_sources" in result and len(result["data_sources"]) > 0:
            self._data_source_id = result["data_sources"][0]["id"]
        return result["id"]

    def _get_data_source_id_from_database(self, database_id: str) -> str:
        """Get the data source ID from a database.

        Args:
            database_id (str): Database ID

        Returns:
            str: Data source ID (first data source if multiple exist)

        Raises:
            ValueError: If no data sources found for the database
        """
        result = self._get(f"/databases/{database_id}")
        if "data_sources" in result and len(result["data_sources"]) > 0:
            return result["data_sources"][0]["id"]
        raise ValueError(f"No data sources found for database {database_id}")

    def fetch_database_notes(self) -> list[Note]:
        """Fetch all notes from the Notes Database.

        Returns:
            List of Note objects from the database
        """
        database_id = self.get_database_id()
        notes = []

        # Use search API to find all pages in the database
        cursor: Optional[str] = None
        while True:
            payload: dict[str, Any] = {
                "filter": {
                    "property": "object",
                    "value": "page",
                },
                "page_size": 100,
            }
            if cursor:
                payload["start_cursor"] = cursor

            response = self._post("/search", payload)

            for page in response.get("results", []):
                # Filter to only pages in our database/data source
                parent = page.get("parent", {})
                parent_type = parent.get("type")

                # Handle both database_id (legacy) and data_source_id (2025-09-03)
                if parent_type == "database_id":
                    parent_db_id = parent.get("database_id", "").replace("-", "")
                    if parent_db_id == database_id.replace("-", ""):
                        note = self._database_page_to_note(page)
                        if note:
                            notes.append(note)
                elif parent_type == "data_source_id":
                    # Check if this data source belongs to our database
                    parent_data_source_id = parent.get("data_source_id", "")
                    try:
                        our_data_source_id = self.get_data_source_id()
                        if parent_data_source_id == our_data_source_id:
                            note = self._database_page_to_note(page)
                            if note:
                                notes.append(note)
                    except ValueError:
                        # If we can't get data source ID, skip this page
                        pass

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        return notes

    def _database_page_to_note(self, page: dict[str, Any]) -> Optional[Note]:
        """Convert a database page to a Note object.

        Args:
            page (dict[str, Any]): Notion database page object

        Returns:
            Optional[Note]: Note object or None if conversion fails
        """
        try:
            page_id = page["id"]
            properties = page.get("properties", {})

            # Extract title
            title_prop = properties.get("Title", {})
            title_parts = title_prop.get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_parts) if title_parts else ""

            if not title:
                return None

            # Extract category
            category_prop = properties.get("Category", {})
            category_select = category_prop.get("select")
            category = category_select.get("name", "") if category_select else ""

            # Extract source (rich_text)
            source_prop = properties.get("Source", {})
            source_texts = source_prop.get("rich_text", [])
            source = "".join(t.get("plain_text", "") for t in source_texts) if source_texts else ""

            # Fetch content
            content = self._fetch_content(page_id)

            # Parse last edited time
            last_edited = None
            if page.get("last_edited_time"):
                last_edited = datetime.fromisoformat(
                    page["last_edited_time"].replace("Z", "+00:00")
                )

            return Note(
                notion_page_id=page_id,
                title=title,
                category=category,
                source=source,
                content=content,
                notion_last_edited=last_edited,
                status=NoteStatus.NEW,
            )
        except Exception:
            return None

    def add_note_to_database(self, note: Note) -> Optional[str]:
        """Add a note to the Notes Database.

        Args:
            note (Note): Note to add

        Returns:
            Optional[str]: Page ID of created database entry, or None if failed
        """
        data_source_id = self.get_data_source_id()

        try:
            # Build properties
            properties: dict[str, Any] = {
                "Title": {"title": [{"text": {"content": note.title}}]},
                "Date Imported": {"date": {"start": datetime.now().isoformat()}},
            }

            if note.category:
                properties["Category"] = {"select": {"name": note.category}}

            if note.source:
                properties["Source"] = {"rich_text": [{"text": {"content": note.source}}]}

            # Create the database page using data_source_id
            payload = {
                "parent": {"type": "data_source_id", "data_source_id": data_source_id},
                "properties": properties,
            }
            response = self._post("/pages", payload)

            new_page_id = response["id"]

            # Copy content blocks from original note (excluding metadata blocks)
            if note.notion_page_id:
                blocks = self._get_all_children(note.notion_page_id)
                if blocks:
                    # Filter out metadata blocks from the beginning
                    blocks = self._filter_metadata_blocks(blocks)
                    if blocks:
                        self._copy_blocks_to_page(new_page_id, blocks)

            return new_page_id
        except NotionAPIError as e:
            print(f"Failed to add note to database: {e}")
            return None

    def add_merged_note_to_database(
        self, note: Note, source_page_ids: list[str]
    ) -> Optional[str]:
        """Add a merged note to the Notes Database, copying content from multiple sources.

        Args:
            note (Note): The merged note to add
            source_page_ids (list[str]): List of Notion page IDs to copy content from

        Returns:
            Optional[str]: Page ID of created database entry, or None if failed
        """
        data_source_id = self.get_data_source_id()

        try:
            # Build properties
            properties: dict[str, Any] = {
                "Title": {"title": [{"text": {"content": note.title}}]},
                "Date Imported": {"date": {"start": datetime.now().isoformat()}},
            }

            if note.category:
                properties["Category"] = {"select": {"name": note.category}}

            if note.source:
                properties["Source"] = {"rich_text": [{"text": {"content": note.source}}]}

            # Create the database page using data_source_id
            payload = {
                "parent": {"type": "data_source_id", "data_source_id": data_source_id},
                "properties": properties,
            }
            response = self._post("/pages", payload)

            new_page_id = response["id"]

            # Collect all blocks from all source pages first
            all_children: list[dict[str, Any]] = []

            for i, source_page_id in enumerate(source_page_ids):
                blocks = self._get_all_children(source_page_id)
                if blocks:
                    # Filter out metadata blocks from the beginning
                    blocks = self._filter_metadata_blocks(blocks)
                    if blocks:
                        # Add divider between merged notes (except before the first one)
                        if i > 0 and all_children:
                            all_children.append({"type": "divider", "divider": {}})

                        # Convert blocks - limit nesting to 2 levels (Notion API limitation)
                        for block in blocks:
                            converted = self._convert_block_for_copy(block)
                            if converted:
                                # Handle nested children (level 1)
                                if block.get("has_children"):
                                    nested_blocks = self._get_all_children(block["id"])
                                    nested_children = []
                                    for nested_block in nested_blocks:
                                        nested_converted = self._convert_block_for_copy(nested_block)
                                        if nested_converted:
                                            # Handle level 2 children (max depth for API)
                                            if nested_block.get("has_children"):
                                                level2_blocks = self._get_all_children(nested_block["id"])
                                                level2_children = []
                                                for level2_block in level2_blocks:
                                                    level2_converted = self._convert_block_for_copy(level2_block)
                                                    if level2_converted:
                                                        # Don't include deeper children - API limit
                                                        level2_children.append(level2_converted)
                                                if level2_children:
                                                    nested_type = nested_converted["type"]
                                                    nested_converted[nested_type]["children"] = level2_children
                                            nested_children.append(nested_converted)
                                    if nested_children:
                                        block_type = converted["type"]
                                        converted[block_type]["children"] = nested_children
                                all_children.append(converted)

            # Copy all blocks in batches
            if all_children:
                for i in range(0, len(all_children), 100):
                    batch = all_children[i : i + 100]
                    self._patch(f"/blocks/{new_page_id}/children", {"children": batch})

            return new_page_id
        except NotionAPIError as e:
            print(f"Failed to add merged note to database: {e}")
            return None

    def add_notes_to_database(self, notes: list[Note]) -> dict[str, Optional[str]]:
        """Add multiple notes to the Notes Database.

        Args:
            notes (list[Note]): List of notes to add

        Returns:
            dict[str, Optional[str]]: Dict mapping original notion_page_id to new database page ID
        """
        results = {}
        for note in notes:
            results[note.notion_page_id] = self.add_note_to_database(note)
        return results

    def append_to_database_page(
        self, database_page_id: str, note: Note, source_page_ids: Optional[list[str]] = None
    ) -> bool:
        """Append content from a note to an existing database page.

        Adds a divider and the note's content blocks to the end of the existing page.

        Args:
            database_page_id (str): ID of the existing database page
            note (Note): Note whose content will be appended
            source_page_ids (Optional[list[str]]): For merged notes, list of all source page IDs

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine source pages to copy from
            if source_page_ids:
                pages_to_copy = source_page_ids
            elif note.notion_page_id:
                pages_to_copy = [note.notion_page_id]
            else:
                return False

            # Collect all blocks from source pages
            all_children: list[dict[str, Any]] = []

            for i, source_page_id in enumerate(pages_to_copy):
                blocks = self._get_all_children(source_page_id)
                if blocks:
                    blocks = self._filter_metadata_blocks(blocks)
                    if blocks:
                        # Add divider between sources (except before first)
                        if i > 0 and all_children:
                            all_children.append({"type": "divider", "divider": {}})

                        # Convert blocks with 2-level nesting limit
                        for block in blocks:
                            converted = self._convert_block_for_copy(block)
                            if converted:
                                if block.get("has_children"):
                                    nested_blocks = self._get_all_children(block["id"])
                                    nested_children = []
                                    for nested_block in nested_blocks:
                                        nested_converted = self._convert_block_for_copy(nested_block)
                                        if nested_converted:
                                            if nested_block.get("has_children"):
                                                level2_blocks = self._get_all_children(nested_block["id"])
                                                level2_children = []
                                                for level2_block in level2_blocks:
                                                    level2_converted = self._convert_block_for_copy(level2_block)
                                                    if level2_converted:
                                                        level2_children.append(level2_converted)
                                                if level2_children:
                                                    nested_type = nested_converted["type"]
                                                    nested_converted[nested_type]["children"] = level2_children
                                            nested_children.append(nested_converted)
                                    if nested_children:
                                        block_type = converted["type"]
                                        converted[block_type]["children"] = nested_children
                                all_children.append(converted)

            if not all_children:
                return True  # Nothing to append

            # Add a divider before the appended content
            self._patch(
                f"/blocks/{database_page_id}/children",
                {"children": [{"type": "divider", "divider": {}}]},
            )

            # Copy all blocks in batches
            for i in range(0, len(all_children), 100):
                batch = all_children[i : i + 100]
                self._patch(f"/blocks/{database_page_id}/children", {"children": batch})

            return True
        except NotionAPIError as e:
            print(f"Failed to append to database page: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NotionAPIError),
    )
    def _find_or_create_subpage(self, page_name: str) -> str:
        """Find a subpage by name or create it if it doesn't exist.

        Args:
            page_name (str): Name of the subpage to find/create

        Returns:
            str: Page ID of the found or created page
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
        retry=retry_if_exception_type(NotionAPIError),
    )
    def _create_subpage(self, page_name: str) -> str:
        """Create a new subpage under the Recaller page.

        Args:
            page_name (str): Name for the new page

        Returns:
            str: ID of the created page
        """
        payload = {
            "parent": {"page_id": self.recaller_page_id},
            "properties": {
                "title": {"title": [{"text": {"content": page_name}}]}
            },
        }
        response = self._post("/pages", payload)
        return response["id"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NotionAPIError),
    )
    def fetch_current_notes(self) -> list[Note]:
        """Fetch all notes from the Current page.

        Returns:
            list[Note]: List of Note objects from the Current page
        """
        current_page_id = self.get_current_page_id()
        return self._fetch_notes_from_page(current_page_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NotionAPIError),
    )
    def fetch_archive_notes(self) -> list[Note]:
        """Fetch all notes from the Archive page.

        Returns:
            list[Note]: List of Note objects from the Archive page
        """
        archive_page_id = self.get_archive_page_id()
        return self._fetch_notes_from_page(archive_page_id)

    def _fetch_notes_from_page(self, page_id: str) -> list[Note]:
        """Fetch all note subpages from a given page.

        Args:
            page_id (str): Parent page ID to fetch notes from

        Returns:
            list[Note]: List of Note objects
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

    def archive_note(self, note_page_id: str) -> bool:
        """Move a note from Current to Archive.

        Since the Notion API doesn't support moving pages between parents,
        this creates a copy in Archive and then trashes the original.

        Args:
            note_page_id (str): ID of the note page to archive

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 1. Fetch the original page
            page = self._get(f"/pages/{note_page_id}")

            # 2. Extract title
            title = self._extract_title(page)
            if not title:
                return False

            # 3. Fetch content blocks
            blocks = self._get_all_children(note_page_id)

            # 4. Create new page in Archive
            archive_page_id = self.get_archive_page_id()
            payload = {
                "parent": {"page_id": archive_page_id},
                "properties": {
                    "title": {"title": [{"text": {"content": title}}]}
                },
            }
            new_page = self._post("/pages", payload)

            # 5. Copy content blocks to new page
            if blocks:
                self._copy_blocks_to_page(new_page["id"], blocks)

            # 6. Trash the original page
            self._patch(f"/pages/{note_page_id}", {"archived": True})

            return True
        except NotionAPIError as e:
            print(f"Failed to archive note {note_page_id}: {e}")
            return False

    def _copy_blocks_to_page(self, page_id: str, blocks: list[dict[str, Any]]) -> None:
        """Copy blocks to a page, including nested children.

        Args:
            page_id (str): Target page ID
            blocks (list[dict[str, Any]]): List of block objects to copy
        """
        children = []
        for block in blocks:
            converted = self._convert_block_for_copy(block)
            if converted:
                # Handle nested children for blocks that support them
                if block.get("has_children"):
                    nested_blocks = self._get_all_children(block["id"])
                    nested_children = []
                    for nested_block in nested_blocks:
                        nested_converted = self._convert_block_for_copy(nested_block)
                        if nested_converted:
                            # Recursively handle deeply nested children
                            if nested_block.get("has_children"):
                                nested_converted = self._convert_block_with_children(nested_block)
                            nested_children.append(nested_converted)
                    if nested_children:
                        block_type = converted["type"]
                        converted[block_type]["children"] = nested_children
                children.append(converted)

        if children:
            # Notion API allows max 100 blocks per request
            for i in range(0, len(children), 100):
                batch = children[i : i + 100]
                self._patch(f"/blocks/{page_id}/children", {"children": batch})

    def _convert_block_with_children(self, block: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Convert a block including its nested children recursively.

        Args:
            block (dict[str, Any]): Original block object

        Returns:
            Optional[dict[str, Any]]: Block object with children included, or None if unsupported
        """
        converted = self._convert_block_for_copy(block)
        if not converted:
            return None

        if block.get("has_children"):
            nested_blocks = self._get_all_children(block["id"])
            nested_children = []
            for nested_block in nested_blocks:
                nested_converted = self._convert_block_with_children(nested_block)
                if nested_converted:
                    nested_children.append(nested_converted)
            if nested_children:
                block_type = converted["type"]
                converted[block_type]["children"] = nested_children

        return converted

    def _sanitize_rich_text(self, rich_text: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize rich_text by filtering out or fixing problematic elements.

        Mentions can cause issues when copied if they have incomplete data.

        Args:
            rich_text (list[dict[str, Any]]): Original rich_text array

        Returns:
            list[dict[str, Any]]: Sanitized rich_text array
        """
        sanitized = []
        for item in rich_text:
            item_type = item.get("type")

            if item_type == "mention":
                mention = item.get("mention", {})
                mention_type = mention.get("type")

                # Only keep mentions with valid, complete data
                if mention_type == "date" and mention.get("date"):
                    sanitized.append(item)
                elif mention_type == "page" and mention.get("page", {}).get("id"):
                    sanitized.append(item)
                elif mention_type == "database" and mention.get("database", {}).get("id"):
                    sanitized.append(item)
                elif mention_type == "user" and mention.get("user", {}).get("id"):
                    sanitized.append(item)
                else:
                    # Convert invalid mention to plain text
                    plain_text = item.get("plain_text", "")
                    if plain_text:
                        sanitized.append({
                            "type": "text",
                            "text": {"content": plain_text}
                        })
            else:
                # Keep non-mention items as-is
                sanitized.append(item)

        return sanitized

    def _convert_block_for_copy(self, block: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Convert a block to the format needed for creating a copy.

        Args:
            block (dict[str, Any]): Original block object

        Returns:
            Optional[dict[str, Any]]: Block object suitable for creation, or None if unsupported
        """
        block_type = block.get("type")
        if not block_type or block_type in ("child_page", "child_database"):
            return None

        block_data = block.get(block_type, {})
        rich_text = self._sanitize_rich_text(block_data.get("rich_text", []))

        # Handle different block types
        if block_type == "paragraph":
            return {"type": "paragraph", "paragraph": {"rich_text": rich_text}}
        elif block_type == "heading_1":
            return {"type": "heading_1", "heading_1": {"rich_text": rich_text}}
        elif block_type == "heading_2":
            return {"type": "heading_2", "heading_2": {"rich_text": rich_text}}
        elif block_type == "heading_3":
            return {"type": "heading_3", "heading_3": {"rich_text": rich_text}}
        elif block_type == "bulleted_list_item":
            return {
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": rich_text},
            }
        elif block_type == "numbered_list_item":
            return {
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": rich_text},
            }
        elif block_type == "to_do":
            return {
                "type": "to_do",
                "to_do": {
                    "rich_text": rich_text,
                    "checked": block_data.get("checked", False),
                },
            }
        elif block_type == "toggle":
            return {"type": "toggle", "toggle": {"rich_text": rich_text}}
        elif block_type == "code":
            return {
                "type": "code",
                "code": {
                    "rich_text": rich_text,
                    "language": block_data.get("language", "plain text"),
                },
            }
        elif block_type == "quote":
            return {"type": "quote", "quote": {"rich_text": rich_text}}
        elif block_type == "callout":
            result: dict[str, Any] = {
                "type": "callout",
                "callout": {"rich_text": rich_text},
            }
            if block_data.get("icon"):
                result["callout"]["icon"] = block_data["icon"]
            return result
        elif block_type == "divider":
            return {"type": "divider", "divider": {}}
        elif block_type == "bookmark":
            return {"type": "bookmark", "bookmark": {"url": block_data.get("url", "")}}
        elif block_type == "image":
            # Get the image URL (external or Notion-hosted file)
            image_type = block_data.get("type")
            url = ""
            if image_type == "external":
                url = block_data.get("external", {}).get("url", "")
            elif image_type == "file":
                # Notion-hosted images have temporary signed URLs
                url = block_data.get("file", {}).get("url", "")

            if url:
                # Create as external image (Notion API only allows external for creation)
                result = {
                    "type": "image",
                    "image": {"type": "external", "external": {"url": url}},
                }
                # Preserve caption if present
                if block_data.get("caption"):
                    result["image"]["caption"] = block_data["caption"]
                return result
            return None

        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NotionAPIError),
    )
    def archive_notes(self, note_page_ids: list[str]) -> dict[str, bool]:
        """Move multiple notes from Current to Archive.

        Args:
            note_page_ids (list[str]): List of note page IDs to archive

        Returns:
            dict[str, bool]: Dict mapping page_id to success status
        """
        results = {}
        for page_id in note_page_ids:
            results[page_id] = self.archive_note(page_id)
        return results

    # Legacy method for backwards compatibility
    def fetch_notes(self) -> list[Note]:
        """Fetch all notes from Current page (legacy method).

        Returns:
            list[Note]: List of Note objects from Notion
        """
        return self.fetch_current_notes()

    def _get_all_children(self, block_id: str) -> list[dict[str, Any]]:
        """Get all child blocks with pagination.

        Args:
            block_id (str): Parent block ID

        Returns:
            list[dict[str, Any]]: List of child block objects
        """
        children: list[dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["start_cursor"] = cursor

            response = self._get(f"/blocks/{block_id}/children", params=params)
            children.extend(response["results"])

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        return children

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NotionAPIError),
    )
    def _fetch_page_as_note(self, page_id: str) -> Optional[Note]:
        """Fetch a single page and convert to Note.

        Args:
            page_id (str): Notion page ID

        Returns:
            Optional[Note]: Note object or None if page cannot be parsed
        """
        try:
            # Fetch page properties
            page = self._get(f"/pages/{page_id}")

            # Parse properties
            title = self._extract_title(page)
            if not title:
                return None

            # Try to get category/source from page properties first
            category = self._extract_property(page, "Category", "select")
            source = self._extract_property(page, "Source", "url") or self._extract_property(
                page, "Source", "rich_text"
            )

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

            # Check for flashcard skip flag - if any value is set, disable flashcards
            flashcard_value = self._extract_metadata_from_content(content, "flashcard")
            flashcard_enabled = flashcard_value is None  # Disabled if any value is set

            return Note(
                notion_page_id=page_id,
                title=title,
                category=category or "",
                source=source or "",
                content=content,
                notion_last_edited=last_edited,
                status=NoteStatus.NEW,
                flashcard=flashcard_enabled,
            )

        except NotionAPIError:
            raise
        except Exception:
            return None

    def _is_metadata_block(self, block: dict[str, Any]) -> bool:
        """Check if a block contains metadata flags that should be filtered.

        Looks for patterns like "Category:" or "Flashcard:" at the start of paragraph text.
        Note: "Source:" is NOT filtered - it's kept with the content.

        Args:
            block (dict[str, Any]): Notion block object

        Returns:
            bool: True if the block is a metadata block that should be filtered
        """
        block_type = block.get("type")

        # Only check paragraph blocks
        if block_type != "paragraph":
            return False

        # Get the text content
        rich_text = block.get("paragraph", {}).get("rich_text", [])
        if not rich_text:
            return False

        text = "".join(t.get("plain_text", "") for t in rich_text).strip()

        # Check for metadata patterns (case-insensitive)
        # Note: "source" is intentionally excluded - we keep it with the content
        metadata_patterns = [r"^category\s*:", r"^flashcard\s*:"]
        for pattern in metadata_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False

    def _filter_metadata_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out metadata blocks from the beginning of a block list.

        Only checks and removes metadata blocks from the first few blocks,
        since metadata is expected at the start of the note.

        Args:
            blocks (list[dict[str, Any]]): List of Notion block objects

        Returns:
            list[dict[str, Any]]: Filtered list with leading metadata blocks removed
        """
        # Only check the first 5 blocks for metadata
        max_check = min(5, len(blocks))
        first_content_idx = 0

        for i in range(max_check):
            block = blocks[i]
            block_type = block.get("type")

            # Skip empty paragraphs at the start
            if block_type == "paragraph":
                rich_text = block.get("paragraph", {}).get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich_text).strip()
                if not text:
                    first_content_idx = i + 1
                    continue

            # Check if it's a metadata block
            if self._is_metadata_block(block):
                first_content_idx = i + 1
            else:
                # Stop at first non-metadata, non-empty block
                break

        return blocks[first_content_idx:]

    def _extract_metadata_from_content(self, content: str, field: str) -> Optional[str]:
        """Extract metadata (category/source) from content text.

        Looks for patterns like "Category: value" or "source: value" (case-insensitive).

        Args:
            content (str): The note content as markdown
            field (str): The field name to look for ("category" or "source")

        Returns:
            Optional[str]: The extracted value or None if not found
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
            page (dict[str, Any]): Notion page object

        Returns:
            Optional[str]: Title string or None
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

    def _extract_property(self, page: dict[str, Any], prop_name: str, prop_type: str) -> Optional[str]:
        """Extract a property value from page.

        Args:
            page (dict[str, Any]): Notion page object
            prop_name (str): Property name to look for
            prop_type (str): Expected property type

        Returns:
            Optional[str]: Property value as string or None
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
            page_id (str): Notion page ID

        Returns:
            str: Content as markdown string
        """
        blocks = self._get_all_children(page_id)
        return self._blocks_to_markdown(blocks)

    def _blocks_to_markdown(self, blocks: list[dict[str, Any]], indent: int = 0) -> str:
        """Convert Notion blocks to markdown.

        Args:
            blocks (list[dict[str, Any]]): List of Notion block objects
            indent (int): Current indentation level

        Returns:
            str: Markdown string
        """
        lines = []
        indent_str = "  " * indent

        for block in blocks:
            block_type = block.get("type")

            if block_type == "paragraph":
                text = self._rich_text_to_markdown(block.get("paragraph", {}).get("rich_text", []))
                if text:
                    lines.append(f"{indent_str}{text}")
                else:
                    lines.append("")

            elif block_type == "heading_1":
                text = self._rich_text_to_markdown(block.get("heading_1", {}).get("rich_text", []))
                lines.append(f"{indent_str}# {text}")

            elif block_type == "heading_2":
                text = self._rich_text_to_markdown(block.get("heading_2", {}).get("rich_text", []))
                lines.append(f"{indent_str}## {text}")

            elif block_type == "heading_3":
                text = self._rich_text_to_markdown(block.get("heading_3", {}).get("rich_text", []))
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
                text = self._rich_text_to_markdown(block.get("to_do", {}).get("rich_text", []))
                checked = block.get("to_do", {}).get("checked", False)
                checkbox = "[x]" if checked else "[ ]"
                lines.append(f"{indent_str}- {checkbox} {text}")

            elif block_type == "toggle":
                text = self._rich_text_to_markdown(block.get("toggle", {}).get("rich_text", []))
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
                text = self._rich_text_to_markdown(block.get("quote", {}).get("rich_text", []))
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
            rich_text (list[dict[str, Any]]): List of rich text objects

        Returns:
            str: Markdown string
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
