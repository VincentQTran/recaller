"""Unit tests for NotionService."""

from unittest.mock import MagicMock, patch

import pytest

from recaller.models.note import NoteStatus
from recaller.services.notion_client import NotionAPIError, NotionService


@pytest.fixture
def mock_requests():
    """Create mocked requests module."""
    with patch("recaller.services.notion_client.requests") as mock_req:
        # Create mock response object
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        mock_req.get.return_value = mock_response
        mock_req.post.return_value = mock_response
        mock_req.patch.return_value = mock_response

        yield mock_req


@pytest.fixture
def notion_service(mock_requests):  # noqa: ARG001
    """Create NotionService with mocked requests."""
    service = NotionService(token="test-token", recaller_page_id="test-page-id")
    return service


class TestNotionServiceInit:
    """Tests for NotionService initialization."""

    def test_init_creates_service(self):
        """Test that initialization creates a NotionService correctly."""
        service = NotionService(token="my-token", recaller_page_id="page-123")
        assert service.token == "my-token"
        assert service.recaller_page_id == "page-123"
        assert service.api_version == "2025-09-03"
        assert service.base_url == "https://api.notion.com/v1"


class TestNotionAPIError:
    """Tests for NotionAPIError exception."""

    def test_error_with_all_fields(self):
        """Test error with all fields."""
        error = NotionAPIError("Test message", status_code=400, code="invalid_request")
        assert error.message == "Test message"
        assert error.status_code == 400
        assert error.code == "invalid_request"
        assert str(error) == "Test message"

    def test_error_with_defaults(self):
        """Test error with default values."""
        error = NotionAPIError("Test message")
        assert error.message == "Test message"
        assert error.status_code == 0
        assert error.code == ""


class TestRichTextToMarkdown:
    """Tests for rich text to markdown conversion."""

    def test_plain_text(self, notion_service):
        """Test plain text conversion."""
        rich_text = [{"plain_text": "Hello world", "annotations": {}}]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "Hello world"

    def test_bold_text(self, notion_service):
        """Test bold text conversion."""
        rich_text = [{"plain_text": "bold", "annotations": {"bold": True}}]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "**bold**"

    def test_italic_text(self, notion_service):
        """Test italic text conversion."""
        rich_text = [{"plain_text": "italic", "annotations": {"italic": True}}]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "*italic*"

    def test_strikethrough_text(self, notion_service):
        """Test strikethrough text conversion."""
        rich_text = [{"plain_text": "strike", "annotations": {"strikethrough": True}}]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "~~strike~~"

    def test_code_text(self, notion_service):
        """Test inline code conversion."""
        rich_text = [{"plain_text": "code", "annotations": {"code": True}}]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "`code`"

    def test_link_text(self, notion_service):
        """Test link conversion."""
        rich_text = [
            {"plain_text": "link", "annotations": {}, "href": "https://example.com"}
        ]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "[link](https://example.com)"

    def test_combined_formatting(self, notion_service):
        """Test combined bold and italic."""
        rich_text = [
            {"plain_text": "text", "annotations": {"bold": True, "italic": True}}
        ]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "***text***"

    def test_multiple_segments(self, notion_service):
        """Test multiple text segments."""
        rich_text = [
            {"plain_text": "Hello ", "annotations": {}},
            {"plain_text": "world", "annotations": {"bold": True}},
        ]
        result = notion_service._rich_text_to_markdown(rich_text)
        assert result == "Hello **world**"

    def test_empty_rich_text(self, notion_service):
        """Test empty rich text list."""
        result = notion_service._rich_text_to_markdown([])
        assert result == ""


class TestBlocksToMarkdown:
    """Tests for blocks to markdown conversion."""

    def test_paragraph_block(self, notion_service):
        """Test paragraph block conversion."""
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Test paragraph", "annotations": {}}]},
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "Test paragraph"

    def test_heading_blocks(self, notion_service):
        """Test heading block conversions."""
        blocks = [
            {
                "type": "heading_1",
                "heading_1": {"rich_text": [{"plain_text": "H1", "annotations": {}}]},
            },
            {
                "type": "heading_2",
                "heading_2": {"rich_text": [{"plain_text": "H2", "annotations": {}}]},
            },
            {
                "type": "heading_3",
                "heading_3": {"rich_text": [{"plain_text": "H3", "annotations": {}}]},
            },
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert "# H1" in result
        assert "## H2" in result
        assert "### H3" in result

    def test_bulleted_list(self, notion_service):
        """Test bulleted list conversion."""
        blocks = [
            {
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"plain_text": "Item 1", "annotations": {}}]},
                "has_children": False,
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "- Item 1"

    def test_numbered_list(self, notion_service):
        """Test numbered list conversion."""
        blocks = [
            {
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [{"plain_text": "First", "annotations": {}}]},
                "has_children": False,
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "1. First"

    def test_code_block(self, notion_service):
        """Test code block conversion."""
        blocks = [
            {
                "type": "code",
                "code": {
                    "language": "python",
                    "rich_text": [{"plain_text": "print('hello')", "annotations": {}}],
                },
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert "```python" in result
        assert "print('hello')" in result
        assert "```" in result

    def test_quote_block(self, notion_service):
        """Test quote block conversion."""
        blocks = [
            {
                "type": "quote",
                "quote": {"rich_text": [{"plain_text": "A quote", "annotations": {}}]},
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "> A quote"

    def test_divider_block(self, notion_service):
        """Test divider block conversion."""
        blocks = [{"type": "divider"}]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "---"

    def test_todo_unchecked(self, notion_service):
        """Test unchecked todo item."""
        blocks = [
            {
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"plain_text": "Task", "annotations": {}}],
                    "checked": False,
                },
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "- [ ] Task"

    def test_todo_checked(self, notion_service):
        """Test checked todo item."""
        blocks = [
            {
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"plain_text": "Done", "annotations": {}}],
                    "checked": True,
                },
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "- [x] Done"

    def test_callout_block(self, notion_service):
        """Test callout block conversion."""
        blocks = [
            {
                "type": "callout",
                "callout": {
                    "icon": {"type": "emoji", "emoji": "💡"},
                    "rich_text": [{"plain_text": "Tip", "annotations": {}}],
                },
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert "> 💡 Tip" in result

    def test_bookmark_block(self, notion_service):
        """Test bookmark block conversion."""
        blocks = [
            {
                "type": "bookmark",
                "bookmark": {"url": "https://example.com"},
            }
        ]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == "[Bookmark](https://example.com)"

    def test_empty_paragraph(self, notion_service):
        """Test empty paragraph creates blank line."""
        blocks = [{"type": "paragraph", "paragraph": {"rich_text": []}}]
        result = notion_service._blocks_to_markdown(blocks)
        assert result == ""


class TestExtractTitle:
    """Tests for title extraction."""

    def test_extract_title_from_title_property(self, notion_service):
        """Test extracting title from 'title' property."""
        page = {
            "properties": {
                "title": {
                    "type": "title",
                    "title": [{"plain_text": "My Note Title"}],
                }
            }
        }
        result = notion_service._extract_title(page)
        assert result == "My Note Title"

    def test_extract_title_from_name_property(self, notion_service):
        """Test extracting title from 'Name' property."""
        page = {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Named Note"}],
                }
            }
        }
        result = notion_service._extract_title(page)
        assert result == "Named Note"

    def test_extract_title_multiple_parts(self, notion_service):
        """Test extracting title with multiple text parts."""
        page = {
            "properties": {
                "title": {
                    "type": "title",
                    "title": [
                        {"plain_text": "Part 1 "},
                        {"plain_text": "Part 2"},
                    ],
                }
            }
        }
        result = notion_service._extract_title(page)
        assert result == "Part 1 Part 2"

    def test_extract_title_empty(self, notion_service):
        """Test extracting from page with no title."""
        page = {"properties": {}}
        result = notion_service._extract_title(page)
        assert result is None


class TestExtractProperty:
    """Tests for property extraction."""

    def test_extract_select_property(self, notion_service):
        """Test extracting select property."""
        page = {
            "properties": {
                "Category": {
                    "type": "select",
                    "select": {"name": "Programming"},
                }
            }
        }
        result = notion_service._extract_property(page, "Category", "select")
        assert result == "Programming"

    def test_extract_url_property(self, notion_service):
        """Test extracting URL property."""
        page = {
            "properties": {
                "Source": {
                    "type": "url",
                    "url": "https://example.com/article",
                }
            }
        }
        result = notion_service._extract_property(page, "Source", "url")
        assert result == "https://example.com/article"

    def test_extract_rich_text_property(self, notion_service):
        """Test extracting rich text property."""
        page = {
            "properties": {
                "Source": {
                    "type": "rich_text",
                    "rich_text": [{"plain_text": "Book: Clean Code"}],
                }
            }
        }
        result = notion_service._extract_property(page, "Source", "rich_text")
        assert result == "Book: Clean Code"

    def test_extract_missing_property(self, notion_service):
        """Test extracting non-existent property."""
        page = {"properties": {}}
        result = notion_service._extract_property(page, "Missing", "select")
        assert result is None

    def test_extract_empty_select(self, notion_service):
        """Test extracting empty select property."""
        page = {
            "properties": {
                "Category": {
                    "type": "select",
                    "select": None,
                }
            }
        }
        result = notion_service._extract_property(page, "Category", "select")
        assert result is None


class TestFetchNotesFromPage:
    """Tests for fetching notes from a specific page."""

    def test_fetch_notes_from_page_success(self, notion_service, mock_requests):
        """Test successful note fetching from a page."""
        # Create mock responses for different API calls
        children_response = MagicMock()
        children_response.status_code = 200
        children_response.json.return_value = {
            "results": [
                {"type": "child_page", "id": "page-1"},
                {"type": "child_page", "id": "page-2"},
            ],
            "has_more": False,
        }

        page_response = MagicMock()
        page_response.status_code = 200
        page_response.json.return_value = {
            "id": "page-1",
            "last_edited_time": "2025-01-15T10:00:00.000Z",
            "properties": {
                "title": {
                    "type": "title",
                    "title": [{"plain_text": "Test Note"}],
                },
                "Category": {
                    "type": "select",
                    "select": {"name": "Learning"},
                },
                "Source": {
                    "type": "url",
                    "url": "https://example.com",
                },
            },
        }

        content_response = MagicMock()
        content_response.status_code = 200
        content_response.json.return_value = {
            "results": [],
            "has_more": False,
        }

        # Set up mock to return different responses for different calls
        mock_requests.get.side_effect = [
            children_response,  # First call: get children of parent page
            page_response,  # Second call: get page-1 details
            content_response,  # Third call: get page-1 content
            page_response,  # Fourth call: get page-2 details
            content_response,  # Fifth call: get page-2 content
        ]

        notes = notion_service._fetch_notes_from_page("some-page-id")

        assert len(notes) == 2
        assert notes[0].title == "Test Note"
        assert notes[0].category == "Learning"
        assert notes[0].source == "https://example.com"
        assert notes[0].status == NoteStatus.NEW

    def test_fetch_notes_skips_non_pages(self, notion_service, mock_requests):
        """Test that non-page blocks are skipped."""
        children_response = MagicMock()
        children_response.status_code = 200
        children_response.json.return_value = {
            "results": [
                {"type": "paragraph", "id": "para-1"},
                {"type": "child_page", "id": "page-1"},
                {"type": "heading_1", "id": "h1-1"},
            ],
            "has_more": False,
        }

        page_response = MagicMock()
        page_response.status_code = 200
        page_response.json.return_value = {
            "id": "page-1",
            "properties": {
                "title": {"type": "title", "title": [{"plain_text": "Note"}]},
            },
        }

        content_response = MagicMock()
        content_response.status_code = 200
        content_response.json.return_value = {
            "results": [],
            "has_more": False,
        }

        mock_requests.get.side_effect = [
            children_response,
            page_response,
            content_response,
        ]

        notes = notion_service._fetch_notes_from_page("some-page-id")

        # Should only have 1 note (the child_page)
        assert len(notes) == 1


class TestExtractMetadataFromContent:
    """Tests for extracting category/source from content."""

    def test_extract_category_from_content(self, notion_service):
        """Test extracting category from content."""
        content = "Category: Programming\n\nSome note content here."
        result = notion_service._extract_metadata_from_content(content, "category")
        assert result == "Programming"

    def test_extract_source_from_content(self, notion_service):
        """Test extracting source from content."""
        content = "Source: https://example.com\n\nSome note content."
        result = notion_service._extract_metadata_from_content(content, "source")
        assert result == "https://example.com"

    def test_extract_case_insensitive(self, notion_service):
        """Test that extraction is case-insensitive."""
        content = "CATEGORY: Tech\nsource: Book"
        assert notion_service._extract_metadata_from_content(content, "category") == "Tech"
        assert notion_service._extract_metadata_from_content(content, "source") == "Book"

    def test_extract_with_extra_whitespace(self, notion_service):
        """Test extraction with extra whitespace around colon."""
        content = "Category  :   Machine Learning  \n\nContent here."
        result = notion_service._extract_metadata_from_content(content, "category")
        assert result == "Machine Learning"

    def test_extract_not_found(self, notion_service):
        """Test when field is not in content."""
        content = "Just some regular content without metadata."
        result = notion_service._extract_metadata_from_content(content, "category")
        assert result is None

    def test_extract_empty_value(self, notion_service):
        """Test when field has empty value."""
        content = "Category: \n\nSome content."
        result = notion_service._extract_metadata_from_content(content, "category")
        assert result is None

    def test_extract_from_middle_of_content(self, notion_service):
        """Test extraction when metadata is not at the start."""
        content = "# Title\n\nSome intro text.\n\nCategory: Design\n\nMore content."
        result = notion_service._extract_metadata_from_content(content, "category")
        assert result == "Design"

    def test_extract_both_category_and_source(self, notion_service):
        """Test extracting both category and source from same content."""
        content = "Category: Programming\nSource: Clean Code book\n\nNote content here."
        assert notion_service._extract_metadata_from_content(content, "category") == "Programming"
        assert notion_service._extract_metadata_from_content(content, "source") == "Clean Code book"


class TestPageStructure:
    """Tests for page structure management."""

    def test_find_existing_subpage(self, notion_service, mock_requests):
        """Test finding an existing subpage."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "type": "child_page",
                    "id": "current-page-id",
                    "child_page": {"title": "Current"},
                },
                {
                    "type": "child_page",
                    "id": "archive-page-id",
                    "child_page": {"title": "Archive"},
                },
            ],
            "has_more": False,
        }
        mock_requests.get.return_value = mock_response

        current_id = notion_service._find_or_create_subpage("Current")
        assert current_id == "current-page-id"

        # post should not be called since page exists
        mock_requests.post.assert_not_called()

    def test_create_missing_subpage(self, notion_service, mock_requests):
        """Test creating a subpage when it doesn't exist."""
        children_response = MagicMock()
        children_response.status_code = 200
        children_response.json.return_value = {
            "results": [],
            "has_more": False,
        }

        create_response = MagicMock()
        create_response.status_code = 200
        create_response.json.return_value = {
            "id": "new-page-id",
        }

        mock_requests.get.return_value = children_response
        mock_requests.post.return_value = create_response

        page_id = notion_service._find_or_create_subpage("Current")

        assert page_id == "new-page-id"
        mock_requests.post.assert_called_once()

    def test_ensure_page_structure(self, notion_service, mock_requests):
        """Test ensuring both Current and Archive pages exist."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "type": "child_page",
                    "id": "current-id",
                    "child_page": {"title": "Current"},
                },
                {
                    "type": "child_page",
                    "id": "archive-id",
                    "child_page": {"title": "Archive"},
                },
            ],
            "has_more": False,
        }
        mock_requests.get.return_value = mock_response

        current_id, archive_id = notion_service.ensure_page_structure()

        assert current_id == "current-id"
        assert archive_id == "archive-id"
        assert notion_service._current_page_id == "current-id"
        assert notion_service._archive_page_id == "archive-id"


class TestAPIHelpers:
    """Tests for API helper methods."""

    def test_get_headers(self, notion_service):
        """Test that headers are correctly constructed."""
        headers = notion_service._get_headers()
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"
        assert headers["Notion-Version"] == "2025-09-03"

    def test_handle_response_success(self, notion_service):
        """Test handling successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "test-id"}

        result = notion_service._handle_response(mock_response)
        assert result == {"id": "test-id"}

    def test_handle_response_error(self, notion_service):
        """Test handling error response."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "message": "Invalid request",
            "code": "invalid_request",
        }

        with pytest.raises(NotionAPIError) as exc_info:
            notion_service._handle_response(mock_response)

        assert exc_info.value.message == "Invalid request"
        assert exc_info.value.status_code == 400
        assert exc_info.value.code == "invalid_request"

    def test_handle_response_error_non_json(self, notion_service):
        """Test handling error response with non-JSON body."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.text = "Internal Server Error"

        with pytest.raises(NotionAPIError) as exc_info:
            notion_service._handle_response(mock_response)

        assert exc_info.value.message == "Internal Server Error"
        assert exc_info.value.status_code == 500
