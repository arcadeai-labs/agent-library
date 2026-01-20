"""
Evaluation suite for Librarian MCP tools.

This module defines comprehensive test cases to evaluate how well LLMs
use the document management tools correctly.

Run with:
    arcade evals . -p openai
    arcade evals . -p anthropic
    arcade evals . --details  # For detailed critic feedback
"""

from datetime import datetime, timedelta

from arcade_evals import (
    BinaryCritic,
    DatetimeCritic,
    EvalRubric,
    EvalSuite,
    ExpectedMCPToolCall,
    NumericCritic,
    SimilarityCritic,
    tool_eval,
)


@tool_eval()
async def search_tools_eval() -> EvalSuite:
    """Evaluate search tool usage and query understanding."""
    suite = EvalSuite(
        name="Search Tools",
        system_message=(
            "You are a helpful assistant that helps users search and manage their "
            "markdown documents. Use the available tools to find relevant information."
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
                "Librarian_Search",
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
        user_message="Show me the top 5 documents about machine learning",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_Search",
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
        user_message="Find all my meeting notes from the project kickoff",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_Search",
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
        user_message="What did I write today?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_Search",
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
        user_message="Show me everything I documented this week about the API design",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_Search",
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
                "Librarian_Search",
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
                "Librarian_Search",
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
                "Librarian_FindRelevantContextWithinSpecificDates",
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
        user_message="Find all documentation I wrote in Q4 2025 about authentication",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_FindRelevantContextWithinSpecificDates",
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
        user_message="Find documents that are conceptually related to containerization and Docker",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_VectorSearch",
                {"query": "containerization Docker"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
    )

    suite.add_case(
        name="Exact keyword search",
        user_message="Search for the exact term 'JIRA-1234' in my notes",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_KeywordSearch",
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
        name="Document Management",
        system_message=(
            "You are a helpful assistant that manages markdown documents. "
            "You can create, read, update, and delete documents. "
            "Always use appropriate filenames with .md extension."
        ),
        rubric=EvalRubric(fail_threshold=0.75, warn_threshold=0.85),
    )

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "librarian/server.py", "stdio"],
    )

    # ==========================================================================
    # Document Creation
    # ==========================================================================

    suite.add_case(
        name="Create simple note",
        user_message=(
            "Create a new note called 'meeting-notes' with the content: "
            "'# Team Standup\n\nDiscussed sprint goals.'"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_AddDocument",
                {
                    "filename": "meeting-notes.md",
                    "content": "# Team Standup\n\nDiscussed sprint goals.",
                },
            )
        ],
        critics=[
            BinaryCritic(critic_field="filename", weight=0.5),
            SimilarityCritic(critic_field="content", weight=0.5),
        ],
    )

    suite.add_case(
        name="Create note with metadata",
        user_message=(
            "Create a document called 'project-plan' with tags 'planning' and 'q1' "
            "and content about the project timeline"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_AddDocument",
                {
                    "filename": "project-plan.md",
                    "content": "project timeline",
                    "metadata": {"tags": ["planning", "q1"]},
                },
            )
        ],
        critics=[
            BinaryCritic(critic_field="filename", weight=0.4),
            SimilarityCritic(critic_field="content", weight=0.3),
            SimilarityCritic(critic_field="metadata", weight=0.3),
        ],
    )

    # ==========================================================================
    # Document Reading
    # ==========================================================================

    suite.add_case(
        name="Read specific document",
        user_message="Show me the contents of the file at /documents/readme.md",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_ReadDocument",
                {"path": "/documents/readme.md"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=1.0),
        ],
    )

    suite.add_case(
        name="List all documents",
        user_message="Show me all my indexed documents",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_ListDocuments",
                {},
            )
        ],
        critics=[],  # No parameters to validate
    )

    # ==========================================================================
    # Document Updates
    # ==========================================================================

    suite.add_case(
        name="Update document content",
        user_message=(
            "Update the file at /notes/todo.md with new content: '# Updated Todo\n\n- [ ] New task'"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_UpdateDocument",
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
    # Document Deletion
    # ==========================================================================

    suite.add_case(
        name="Delete document from index only",
        user_message=(
            "Remove the document at /old/archive.md from the search index but keep the file"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_DeleteDocument",
                {"path": "/old/archive.md", "delete_file": False},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=0.6),
            BinaryCritic(critic_field="delete_file", weight=0.4),
        ],
    )

    suite.add_case(
        name="Delete document completely",
        user_message="Permanently delete the file /temp/scratch.md",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_DeleteDocument",
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
        name="Document Ingestion",
        system_message=(
            "You are a helpful assistant that indexes markdown documents for search. "
            "You can ingest entire directories of markdown files."
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
        name="Ingest default directory",
        user_message="Index all my documents",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IngestDirectory",
                {"recursive": True},
            )
        ],
        critics=[
            BinaryCritic(critic_field="recursive", weight=1.0),
        ],
    )

    suite.add_case(
        name="Ingest specific directory",
        user_message="Index all markdown files in /projects/documentation",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IngestDirectory",
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
        user_message="Re-index all documents in /notes, even ones already indexed",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IngestDirectory",
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
            "Index only the top-level markdown files in /archive, don't go into subdirectories"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IngestDirectory",
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
        name="Get index statistics",
        user_message="How many documents do I have indexed?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_GetStats",
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
        name="Complex Workflows",
        system_message=(
            "You are a helpful assistant that manages and searches markdown documents. "
            "You can perform multi-step operations when needed."
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
                "Librarian_Search",
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
            "Find diverse perspectives on the architecture decision, avoid repetitive results"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_Search",
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
        user_message="I need results that exactly match the keywords, not just similar concepts",
        additional_messages=[
            {"role": "user", "content": "Search for 'kubernetes deployment yaml'"},
        ],
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_Search",
                {"query": "kubernetes deployment yaml", "hybrid_alpha": 0.0},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.6),
            NumericCritic(critic_field="hybrid_alpha", value_range=(0.0, 0.3), weight=0.4),
        ],
    )

    return suite
