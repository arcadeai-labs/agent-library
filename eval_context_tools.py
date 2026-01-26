"""
Evaluation suite for Librarian MCP tools.

This module defines comprehensive test cases to evaluate how well LLMs
use the agent library tools correctly.

Run with:
    arcade evals . -p openai
    arcade evals . -p anthropic
    arcade evals . --details  # For detailed critic feedback
"""

from datetime import datetime, timedelta

import arcade_evals
from arcade_evals import (
    BinaryCritic,
    DatetimeCritic,
    EvalRubric,
    EvalSuite,
    ExpectedToolCall,
    NumericCritic,
    SimilarityCritic,
    tool_eval,
)

# Use ExpectedToolCall instead of ExpectedMCPToolCall (API change)
ExpectedMCPToolCall = ExpectedToolCall


@tool_eval()
async def search_tools_eval() -> EvalSuite:
    """Evaluate search tool usage and query understanding."""
    suite = EvalSuite(
        name="Library Search Tools",
        system_message=(
            "You are a helpful assistant with access to a personal knowledge library. "
            "Use the library tools to store, search, and retrieve information. "
            "The library persists across sessions and contains notes, documents, and knowledge."
        ),
        rubric=EvalRubric(fail_threshold=0.75, warn_threshold=0.85),
    )

    # Load tools from the MCP server
    await suite.add_mcp_stdio_server(
        command=["uv", "run", "librarian/server.py", "stdio"],
    )

    # ==========================================================================
    # Basic Search Queries
    # ==========================================================================

    suite.add_case(
        name="Simple topic search",
        user_message="Find my notes about Python programming",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "Python programming", "limit": 10},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.8),
            NumericCritic(critic_field="limit", value_range=(5, 15), weight=0.2),
        ],
    )

    suite.add_case(
        name="Search with specific limit",
        user_message="Show me the top 5 documents about machine learning from my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "machine learning", "limit": 5},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.6),
            BinaryCritic(critic_field="limit", weight=0.4),
        ],
    )

    suite.add_case(
        name="Search for meeting notes",
        user_message="Find all my meeting notes from the project kickoff in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "meeting notes project kickoff"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
    )

    # ==========================================================================
    # Timeframe-based Search
    # ==========================================================================

    suite.add_case(
        name="Search with today timeframe",
        user_message="What did I add to my library today?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "notes", "timeframe": "today"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="timeframe", weight=0.8),
            SimilarityCritic(critic_field="query", weight=0.2),
        ],
    )

    suite.add_case(
        name="Search this week",
        user_message="Show me everything I stored this week about the API design",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "API design", "timeframe": "this_week"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.5),
            BinaryCritic(critic_field="timeframe", weight=0.5),
        ],
    )

    suite.add_case(
        name="Search last 7 days",
        user_message="Find recent notes from the past week about database migrations",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "database migrations", "timeframe": "last_7_days"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.5),
            BinaryCritic(critic_field="timeframe", weight=0.5),
        ],
    )

    suite.add_case(
        name="Search last month",
        user_message="What were my notes from last month about the product roadmap?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "product roadmap", "timeframe": "last_month"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.5),
            BinaryCritic(critic_field="timeframe", weight=0.5),
        ],
    )

    # ==========================================================================
    # Specific Date Range Search
    # ==========================================================================

    # Calculate realistic dates for test cases
    today = datetime.now()
    last_week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    last_week_end = today.strftime("%Y-%m-%d")

    suite.add_case(
        name="Search with specific date range",
        user_message=(
            f"Find notes about the sprint review between {last_week_start} and {last_week_end}"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibraryByDates",
                {
                    "query": "sprint review",
                    "start_date": last_week_start,
                    "end_date": last_week_end,
                },
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            DatetimeCritic(critic_field="start_date", tolerance=timedelta(days=1), weight=0.3),
            DatetimeCritic(critic_field="end_date", tolerance=timedelta(days=1), weight=0.3),
        ],
    )

    suite.add_case(
        name="Search Q4 2025",
        user_message="Find all documentation I stored in Q4 2025 about authentication",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibraryByDates",
                {
                    "query": "authentication",
                    "start_date": "2025-10-01",
                    "end_date": "2025-12-31",
                },
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            DatetimeCritic(critic_field="start_date", tolerance=timedelta(days=3), weight=0.3),
            DatetimeCritic(critic_field="end_date", tolerance=timedelta(days=3), weight=0.3),
        ],
    )

    # ==========================================================================
    # Semantic vs Keyword Search
    # ==========================================================================

    suite.add_case(
        name="Semantic search request",
        user_message=(
            "Find information in my library that is conceptually related to "
            "containerization and Docker"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SemanticSearchLibrary",
                {"query": "containerization Docker"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
    )

    suite.add_case(
        name="Exact keyword search",
        user_message="Search for the exact term 'JIRA-1234' in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_KeywordSearchLibrary",
                {"query": "JIRA-1234"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="query", weight=1.0),
        ],
    )

    return suite


@tool_eval()
async def document_management_eval() -> EvalSuite:
    """Evaluate document creation, reading, and management tools."""
    suite = EvalSuite(
        name="Library Management",
        system_message=(
            "You are a helpful assistant with access to a personal knowledge library. "
            "You can add, read, update, and remove information from the library. "
            "Use the library to store and retrieve notes, documents, and any useful information."
        ),
        rubric=EvalRubric(fail_threshold=0.75, warn_threshold=0.85),
    )

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "librarian/server.py", "stdio"],
    )

    # ==========================================================================
    # Adding to Library
    # ==========================================================================

    suite.add_case(
        name="Store simple note",
        user_message=(
            "Save a note called 'meeting-notes' with the content: "
            "'# Team Standup\n\nDiscussed sprint goals.'"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_AddToLibrary",
                {
                    "title": "meeting-notes",
                    "content": "# Team Standup\n\nDiscussed sprint goals.",
                },
            )
        ],
        critics=[
            BinaryCritic(critic_field="title", weight=0.5),
            SimilarityCritic(critic_field="content", weight=0.5),
        ],
    )

    suite.add_case(
        name="Store note with tags",
        user_message=(
            "Add to my library a document called 'project-plan' with tags 'planning' and 'q1' "
            "about the project timeline"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_AddToLibrary",
                {
                    "title": "project-plan",
                    "content": "project timeline",
                    "tags": ["planning", "q1"],
                },
            )
        ],
        critics=[
            BinaryCritic(critic_field="title", weight=0.4),
            SimilarityCritic(critic_field="content", weight=0.3),
            SimilarityCritic(critic_field="tags", weight=0.3),
        ],
    )

    # ==========================================================================
    # Reading from Library
    # ==========================================================================

    suite.add_case(
        name="Read specific document",
        user_message="Show me the full contents of /documents/readme.md from my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_ReadFromLibrary",
                {"path": "/documents/readme.md"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=1.0),
        ],
    )

    suite.add_case(
        name="List library contents",
        user_message="Show me everything in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_ListLibraryContents",
                {},
            )
        ],
        critics=[],  # No parameters to validate
    )

    # ==========================================================================
    # Updating Library Content
    # ==========================================================================

    suite.add_case(
        name="Update document content",
        user_message=(
            "Update the content at /notes/todo.md with: '# Updated Todo\n\n- [ ] New task'"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_UpdateLibraryDoc",
                {
                    "path": "/notes/todo.md",
                    "content": "# Updated Todo\n\n- [ ] New task",
                },
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=0.5),
            SimilarityCritic(critic_field="content", weight=0.5),
        ],
    )

    # ==========================================================================
    # Removing from Library
    # ==========================================================================

    suite.add_case(
        name="Remove from index only",
        user_message=(
            "Remove /old/archive.md from my library search but keep the file on disk"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_RemoveFromLibrary",
                {"path": "/old/archive.md", "delete_file": False},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=0.6),
            BinaryCritic(critic_field="delete_file", weight=0.4),
        ],
    )

    suite.add_case(
        name="Permanently delete",
        user_message="Permanently delete /temp/scratch.md from my library and disk",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_RemoveFromLibrary",
                {"path": "/temp/scratch.md", "delete_file": True},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=0.6),
            BinaryCritic(critic_field="delete_file", weight=0.4),
        ],
    )

    return suite


@tool_eval()
async def ingestion_eval() -> EvalSuite:
    """Evaluate document ingestion tools."""
    suite = EvalSuite(
        name="Library Ingestion",
        system_message=(
            "You are a helpful assistant with access to a personal knowledge library. "
            "You can add entire directories of files to the library for indexing and search."
        ),
        rubric=EvalRubric(fail_threshold=0.7, warn_threshold=0.85),
    )

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "librarian/server.py", "stdio"],
    )

    # ==========================================================================
    # Directory Ingestion
    # ==========================================================================

    suite.add_case(
        name="Index default directory",
        user_message="Add all my documents to the library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"recursive": True},
            )
        ],
        critics=[
            BinaryCritic(critic_field="recursive", weight=1.0),
        ],
    )

    suite.add_case(
        name="Index specific directory",
        user_message="Add all files from /projects/documentation to my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"directory": "/projects/documentation", "recursive": True},
            )
        ],
        critics=[
            BinaryCritic(critic_field="directory", weight=0.6),
            BinaryCritic(critic_field="recursive", weight=0.4),
        ],
    )

    suite.add_case(
        name="Force reindex",
        user_message="Re-index everything in /notes even if already in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"directory": "/notes", "force_reindex": True},
            )
        ],
        critics=[
            BinaryCritic(critic_field="directory", weight=0.5),
            BinaryCritic(critic_field="force_reindex", weight=0.5),
        ],
    )

    suite.add_case(
        name="Non-recursive ingestion",
        user_message=(
            "Add only the top-level files in /archive to my library, not subdirectories"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"directory": "/archive", "recursive": False},
            )
        ],
        critics=[
            BinaryCritic(critic_field="directory", weight=0.5),
            BinaryCritic(critic_field="recursive", weight=0.5),
        ],
    )

    # ==========================================================================
    # Stats
    # ==========================================================================

    suite.add_case(
        name="Get library statistics",
        user_message="How many documents do I have in my library?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_GetLibraryStats",
                {},
            )
        ],
        critics=[],  # No parameters to validate
    )

    suite.add_case(
        name="Get library sources",
        user_message="What sources are in my library and how many documents from each?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_GetLibrarySources",
                {},
            )
        ],
        critics=[],  # No parameters to validate
    )

    return suite


@tool_eval()
async def complex_workflows_eval() -> EvalSuite:
    """Evaluate complex multi-step workflows."""
    suite = EvalSuite(
        name="Complex Library Workflows",
        system_message=(
            "You are a helpful assistant with access to a personal knowledge library. "
            "You can store, search, and manage information in the library. "
            "Perform multi-step operations when needed to help the user."
        ),
        rubric=EvalRubric(fail_threshold=0.7, warn_threshold=0.85),
    )

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "librarian/server.py", "stdio"],
    )

    # ==========================================================================
    # Multi-step Operations
    # ==========================================================================

    suite.add_case(
        name="Search then read",
        user_message="Find my notes about the budget and show me the most relevant one",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "budget", "limit": 1},
            ),
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.7),
            NumericCritic(critic_field="limit", value_range=(1, 5), weight=0.3),
        ],
    )

    suite.add_case(
        name="Diverse results request",
        user_message=(
            "Search my library for diverse perspectives on the architecture decision"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "architecture decision", "use_mmr": True},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.6),
            BinaryCritic(critic_field="use_mmr", weight=0.4),
        ],
    )

    suite.add_case(
        name="Keyword-heavy search",
        user_message="Search my library for exact keyword matches, not similar concepts",
        additional_messages=[
            {"role": "user", "content": "Search for 'kubernetes deployment yaml'"},
        ],
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "kubernetes deployment yaml", "hybrid_alpha": 0.0},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.6),
            NumericCritic(critic_field="hybrid_alpha", value_range=(0.0, 0.3), weight=0.4),
        ],
    )

    return suite


@tool_eval()
async def multimodal_eval() -> EvalSuite:
    """Evaluate multi-modal asset type handling."""
    suite = EvalSuite(
        name="Multi-Modal Library Support",
        system_message=(
            "You are a helpful assistant with access to a personal knowledge library. "
            "The library supports multiple asset types: text, code, PDFs, and images. "
            "Search results include asset_type to distinguish file types."
        ),
        rubric=EvalRubric(fail_threshold=0.7, warn_threshold=0.85),
    )

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "librarian/server.py", "stdio"],
    )

    # ==========================================================================
    # Multi-Modal Search
    # ==========================================================================

    suite.add_case(
        name="Search returns asset_type",
        user_message="Search my library for calculator",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "calculator"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
        # Note: Critics don't validate response structure, but the tool
        # should return asset_type field in results
    )

    suite.add_case(
        name="Semantic search returns asset_type",
        user_message="Find conceptually similar content about data structures",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SemanticSearchLibrary",
                {"query": "data structures"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
    )

    suite.add_case(
        name="Keyword search returns asset_type",
        user_message="Search for the exact keyword 'Calculator' in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_KeywordSearchLibrary",
                {"query": "Calculator"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="query", weight=1.0),
        ],
    )

    suite.add_case(
        name="Index code directory",
        user_message="Add all files from /projects/api-server to my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"directory": "/projects/api-server", "recursive": True},
            )
        ],
        critics=[
            BinaryCritic(critic_field="directory", weight=0.6),
            BinaryCritic(critic_field="recursive", weight=0.4),
        ],
    )

    suite.add_case(
        name="Search with code-specific query",
        user_message="Find authentication functions in my code",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "authentication functions"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
    )

    return suite
