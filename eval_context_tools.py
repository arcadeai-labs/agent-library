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
        command=["uv", "run", "python", "-m", "librarian.server", "stdio"],
        env={"LIBRARIAN_ENABLE_OPTIONAL_TOOLS": "false"},
    )

    # ==========================================================================
    # Basic Search Queries (all use unified SearchLibrary)
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
    # Timeframe-based Search (uses timeframe enum parameter)
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
    # Specific Date Range Search (uses start_date/end_date parameters)
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
                "Librarian_SearchLibrary",
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
                "Librarian_SearchLibrary",
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
    # Search Mode Selection (semantic vs keyword via mode parameter)
    # ==========================================================================

    suite.add_case(
        name="Semantic search request",
        user_message=(
            "Find information in my library that is conceptually related to "
            "containerization and Docker"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "containerization Docker", "mode": "semantic"},
            )
        ],
        critics=[
            # The mode choice is the actual subject of this case; the query
            # text is incidental, so mode gets the heavier weight.
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
        ],
    )

    suite.add_case(
        name="Exact keyword search",
        user_message="Search for the exact term 'JIRA-1234' in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "JIRA-1234", "mode": "keyword"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
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
        command=["uv", "run", "python", "-m", "librarian.server", "stdio"],
        env={"LIBRARIAN_ENABLE_OPTIONAL_TOOLS": "false"},
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
        user_message=("Remove /old/archive.md from my library search but keep the file on disk"),
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

    # Store to an explicit directory — covers a path the model must lift from
    # the user message rather than defaulting.
    suite.add_case(
        name="Store note to explicit directory",
        user_message=(
            "Save a note called 'deploy-plan' into /Users/me/work-notes/deploys "
            "with the content: '# Deploy Plan\n\nStage 1: canary'"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_AddToLibrary",
                {
                    "title": "deploy-plan",
                    "content": "# Deploy Plan\n\nStage 1: canary",
                    "directory": "/Users/me/work-notes/deploys",
                },
            )
        ],
        critics=[
            BinaryCritic(critic_field="title", weight=0.3),
            SimilarityCritic(critic_field="content", weight=0.2),
            BinaryCritic(critic_field="directory", weight=0.5),
        ],
    )

    # Remove without mentioning file deletion — model should leave delete_file
    # at the default (False), so we assert the tool + path and do NOT assert
    # delete_file (the auto-NoneCritic will cover the unchecked field).
    suite.add_case(
        name="Remove without mentioning file deletion",
        user_message="Take /archive/old-spec.md out of my library index",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_RemoveFromLibrary",
                {"path": "/archive/old-spec.md"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=1.0),
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

    # This suite tests the optional GetLibraryOverview tool alongside the core
    # IndexDirectoryToLibrary tool, so optional tools must be enabled here
    # (the other suites disable them to keep tool-selection unambiguous
    # between direct actions and workflow helpers).
    await suite.add_mcp_stdio_server(
        command=["uv", "run", "python", "-m", "librarian.server", "stdio"],
        env={"LIBRARIAN_ENABLE_OPTIONAL_TOOLS": "true"},
    )

    # ==========================================================================
    # Directory Ingestion
    # ==========================================================================

    suite.add_case(
        name="Index specific directory",
        user_message="Add all files from /projects/documentation to my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"directory": "/projects/documentation"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="directory", weight=1.0),
        ],
    )

    suite.add_case(
        name="Index notes directory",
        user_message="Index everything in /notes into my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"directory": "/notes"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="directory", weight=1.0),
        ],
    )

    # ==========================================================================
    # Library overview (consolidated stats / sections / tree)
    # ==========================================================================

    suite.add_case(
        name="Get library statistics (view=stats)",
        user_message="How many documents do I have in my library?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_GetLibraryOverview",
                {"view": "stats"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="view", weight=1.0),
        ],
    )

    suite.add_case(
        name="Show library tree (view=tree)",
        user_message="Show me the folder layout of my library so I can see how it's organized",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_GetLibraryOverview",
                {"view": "tree"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="view", weight=1.0),
        ],
    )

    return suite


@tool_eval()
async def location_workflow_eval() -> EvalSuite:
    """Evaluate the 'where should I put this?' workflow helpers.

    GetLibraryOverview (default view='sections') is the canonical pre-flight
    call before AddToLibrary, and SuggestLibraryLocation is the smart-default
    for content the user hasn't filed by hand. These cases verify the model
    picks each one in the right context.
    """
    suite = EvalSuite(
        name="Library Location Workflow",
        system_message=(
            "You are a helpful assistant with access to a personal knowledge library. "
            "Before storing new content, figure out where it belongs. "
            "Use GetLibraryOverview to enumerate writable locations, "
            "or SuggestLibraryLocation to get ranked recommendations based on "
            "title and content."
        ),
        rubric=EvalRubric(fail_threshold=0.7, warn_threshold=0.85),
    )

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "python", "-m", "librarian.server", "stdio"],
        env={"LIBRARIAN_ENABLE_OPTIONAL_TOOLS": "true"},
    )

    suite.add_case(
        name="Enumerate available sections",
        user_message="Where can I save things in my library? Show me the sections.",
        expected_tool_calls=[
            # Default view is 'sections', so passing nothing — or view='sections' —
            # are both correct.
            ExpectedMCPToolCall("Librarian_GetLibraryOverview", {}),
        ],
        critics=[],  # Tool selection alone carries the signal.
    )

    suite.add_case(
        name="Ask for placement recommendation",
        user_message=(
            "I've got some notes on rate limiting strategies — "
            "where should I file this in my library?"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SuggestLibraryLocation",
                {"title": "rate limiting strategies"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="title", weight=1.0),
        ],
    )

    suite.add_case(
        name="Suggestion with content summary",
        user_message=(
            "Help me find the best place to put a doc titled 'Q2 OKRs' "
            "about engineering team objectives for next quarter"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SuggestLibraryLocation",
                {
                    "title": "Q2 OKRs",
                    "content_summary": "engineering team objectives for next quarter",
                },
            )
        ],
        critics=[
            SimilarityCritic(critic_field="title", weight=0.5),
            SimilarityCritic(
                critic_field="content_summary", weight=0.5, similarity_threshold=0.5
            ),
        ],
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
        command=["uv", "run", "python", "-m", "librarian.server", "stdio"],
        env={"LIBRARIAN_ENABLE_OPTIONAL_TOOLS": "false"},
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
        name="Search code assets",
        user_message="Find authentication code in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "authentication", "asset_type": "code"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.6),
            BinaryCritic(critic_field="asset_type", weight=0.4),
        ],
    )

    suite.add_case(
        name="Search PDFs only",
        user_message="Search my PDF documents for information about the contract terms",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "contract terms", "asset_type": "pdf"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.6),
            BinaryCritic(critic_field="asset_type", weight=0.4),
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
            "Use the SearchLibrary tool with the 'mode' parameter for semantic or keyword "
            "search, and 'asset_type' to filter by content type."
        ),
        rubric=EvalRubric(fail_threshold=0.7, warn_threshold=0.85),
    )

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "python", "-m", "librarian.server", "stdio"],
        env={"LIBRARIAN_ENABLE_OPTIONAL_TOOLS": "false"},
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
    )

    suite.add_case(
        name="Semantic search via mode parameter",
        user_message="Find conceptually similar content about data structures in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "data structures", "mode": "semantic"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
        ],
    )

    suite.add_case(
        name="Keyword search via mode parameter",
        user_message="Search for the exact keyword 'Calculator' in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "Calculator", "mode": "keyword"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
        ],
    )

    suite.add_case(
        name="Index code directory",
        user_message="Add all files from /projects/api-server to my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_IndexDirectoryToLibrary",
                {"directory": "/projects/api-server"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="directory", weight=1.0),
        ],
    )

    suite.add_case(
        name="Search with code asset type filter",
        user_message="Find authentication functions in my code files",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "authentication functions", "asset_type": "code"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.6),
            BinaryCritic(critic_field="asset_type", weight=0.4),
        ],
    )

    return suite


@tool_eval()
async def adversarial_eval() -> EvalSuite:
    """Adversarial coverage: ambiguous phrasings, boundary values, mode/asset-type
    inference, date-range parsing, and tool-selection traps.

    Looser rubric (fail=0.6, warn=0.75) is intentional: these cases are designed
    to expose weak spots, not to enforce 100% pass rates.
    """
    suite = EvalSuite(
        name="Library Adversarial Coverage",
        system_message=(
            "You are a helpful assistant with access to a personal knowledge library. "
            "Choose the right tool and arguments for each request. Prefer direct "
            "action tools over discovery/workflow tools when the user's intent is "
            "clear."
        ),
        rubric=EvalRubric(fail_threshold=0.6, warn_threshold=0.75),
    )

    # Keep the tool surface constrained to the 7 core tools so tool selection
    # isn't diluted by the optional workflow helpers.
    await suite.add_mcp_stdio_server(
        command=["uv", "run", "python", "-m", "librarian.server", "stdio"],
        env={"LIBRARIAN_ENABLE_OPTIONAL_TOOLS": "false"},
    )

    # ==========================================================================
    # Block A — Tool selection under ambiguity
    # ==========================================================================

    suite.add_case(
        name="List everything I have",
        user_message="List everything I have in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall("Librarian_ListLibraryContents", {}),
        ],
        critics=[],  # Tool-selection correctness is enforced by the rubric.
    )

    # Realistic phrasing of "do I have notes on X?" — the answer comes from
    # search, not from list/read. This replaces an earlier "Is /tmp/foo.md in
    # my library?" case that didn't match how users actually ask.
    suite.add_case(
        name="Existential search query",
        user_message="Do I have any notes on Kubernetes networking?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "Kubernetes networking"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
    )

    suite.add_case(
        name="Search inside an indexed folder",
        user_message=(
            "I want to see what's in the api-server folder I indexed "
            "— anything about rate limiting?"
        ),
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "rate limiting api-server"},
            )
        ],
        critics=[
            # Threshold loosened; phrasing of the query will vary.
            SimilarityCritic(
                critic_field="query", weight=1.0, similarity_threshold=0.5
            ),
        ],
    )

    suite.add_case(
        name="Unindex jargon",
        user_message="Unindex /old/archive.md from my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_RemoveFromLibrary",
                {"path": "/old/archive.md", "delete_file": False},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=0.5),
            BinaryCritic(critic_field="delete_file", weight=0.5),
        ],
    )

    suite.add_case(
        name="Permanent-delete slang",
        user_message="Nuke /temp/scratch.md from orbit",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_RemoveFromLibrary",
                {"path": "/temp/scratch.md", "delete_file": True},
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=0.5),
            BinaryCritic(critic_field="delete_file", weight=0.5),
        ],
    )

    # ==========================================================================
    # Block B — Mode inference
    # ==========================================================================

    suite.add_case(
        name="Semantic phrasing — conceptually similar",
        user_message="Find notes conceptually similar to 'distributed consensus'",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "distributed consensus", "mode": "semantic"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
        ],
    )

    suite.add_case(
        name="Keyword phrasing — literal string",
        user_message="Find notes that literally contain the string 'TODO(spartee)'",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "TODO(spartee)", "mode": "keyword"},
            )
        ],
        critics=[
            # Exact-string — BinaryCritic, not SimilarityCritic, because the
            # whole point is that the model preserves the exact token.
            BinaryCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
        ],
    )

    suite.add_case(
        name="Hybrid default — best overall match",
        user_message="Search my library for budget forecasting — best overall match",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "budget forecasting"},
            )
        ],
        critics=[
            # Intentionally no mode critic: either omitting mode (defaults to
            # hybrid) or passing mode="hybrid" is acceptable.
            SimilarityCritic(critic_field="query", weight=1.0),
        ],
    )

    suite.add_case(
        name="Keyword phrasing — exact phrase",
        user_message="I know the exact phrase — find 'eventual consistency' in my library",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "eventual consistency", "mode": "keyword"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
        ],
    )

    # ==========================================================================
    # Block C — Asset-type inference
    # ==========================================================================

    suite.add_case(
        name="Infer asset_type=code from language cue",
        user_message="Find Python code dealing with JWT parsing",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "JWT parsing", "asset_type": "code"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="asset_type", weight=0.6),
        ],
    )

    # Prior context establishes the user just indexed image-format diagrams,
    # so "diagrams" should now resolve to asset_type=image rather than text.
    # This is a realistic agent scenario — the tool catalog plus conversation
    # history together carry the signal.
    suite.add_case(
        name="Infer asset_type=image from 'diagrams' (with context)",
        user_message="Pull up the diagrams about the auth flow",
        additional_messages=[
            {
                "role": "user",
                "content": (
                    "I just indexed our architecture folder — it's mostly PNG "
                    "diagrams alongside the docs."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Got it — your library now includes the architecture "
                    "diagrams alongside the existing documentation."
                ),
            },
        ],
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "auth flow", "asset_type": "image"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="asset_type", weight=0.6),
        ],
    )

    suite.add_case(
        name="Infer asset_type=pdf from 'PDFs'",
        user_message="Look through my PDFs for the SLA clause",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "SLA clause", "asset_type": "pdf"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="asset_type", weight=0.6),
        ],
    )

    # ==========================================================================
    # Block D — Timeframe / date-range parsing
    # ==========================================================================

    suite.add_case(
        name="Timeframe: today",
        user_message="Show me any notes from today",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "notes", "timeframe": "today"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.3),
            BinaryCritic(critic_field="timeframe", weight=0.7),
        ],
    )

    suite.add_case(
        name="Custom date range — explicit",
        user_message="What did I write between March 1 and March 15, 2026?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {
                    "query": "notes",
                    "start_date": "2026-03-01",
                    "end_date": "2026-03-15",
                },
            )
        ],
        critics=[
            # The query text will vary ('notes', 'things', 'wrote' etc.) —
            # low weight. The dates carry the signal.
            SimilarityCritic(
                critic_field="query", weight=0.2, similarity_threshold=0.3
            ),
            DatetimeCritic(
                critic_field="start_date", weight=0.4, tolerance=timedelta(hours=1)
            ),
            DatetimeCritic(
                critic_field="end_date", weight=0.4, tolerance=timedelta(hours=1)
            ),
        ],
    )

    suite.add_case(
        name="Timeframe: last_30_days",
        user_message="Anything from the last 30 days about onboarding?",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "onboarding", "timeframe": "last_30_days"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.3),
            BinaryCritic(critic_field="timeframe", weight=0.7),
        ],
    )

    # 'Last Thursday' has no clean enum value — the model should synthesize
    # a single-day custom range, or (acceptable fallback) use `last_7_days`.
    # We only critic the query; tool selection alone is the primary signal.
    suite.add_case(
        name="Ambiguous relative date — last Thursday",
        user_message="Show me notes from last Thursday about the deploy",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "deploy"},
            )
        ],
        critics=[
            SimilarityCritic(
                critic_field="query", weight=1.0, similarity_threshold=0.5
            ),
        ],
    )

    # ==========================================================================
    # Block E — Limit / boundary values (NumericCritic)
    # ==========================================================================

    suite.add_case(
        name="Limit: top result only",
        user_message="Just the top result for 'retry policy'",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "retry policy", "limit": 1},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.5),
            NumericCritic(critic_field="limit", weight=0.5, value_range=(1, 2)),
        ],
    )

    suite.add_case(
        name="Limit: explicit large number",
        user_message="Give me all 50 hits for 'deploy'",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "deploy", "limit": 50},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.3),
            NumericCritic(critic_field="limit", weight=0.7, value_range=(40, 50)),
        ],
    )

    suite.add_case(
        name="Limit: fuzzy quantifier ('a few')",
        user_message="Show me a few articles about k8s",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "k8s", "limit": 5},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            # 'A few' has wide tolerance; anything in 3-10 is reasonable.
            NumericCritic(critic_field="limit", weight=0.6, value_range=(3, 10)),
        ],
    )

    # ==========================================================================
    # Block F — Multi-turn / context-aware reasoning (additional_messages)
    # ==========================================================================
    #
    # These cases exercise the agent's ability to integrate prior conversation
    # turns. The user's final message is intentionally terse; the model must
    # lift the missing arguments from earlier context. Inlining short threads
    # keeps the cases readable; longer threads can be moved to JSON fixtures
    # under `evals/threads/` and loaded via `json.load`.

    suite.add_case(
        name="Multi-turn add: content lifted from prior turn",
        user_message="Yeah just save it as a quick placeholder note — nothing fancy",
        additional_messages=[
            # 'titled' makes the prior message unambiguous — without it the
            # model can plausibly read 'foobar' as the content and invent a
            # title, which we'd rather not penalize.
            {"role": "user", "content": "Add a note titled 'foobar' to my library"},
            {
                "role": "assistant",
                "content": "Sure, what content should the note hold?",
            },
        ],
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_AddToLibrary",
                # title comes from turn 1, content comes from final turn.
                {"title": "foobar", "content": "placeholder note"},
            )
        ],
        critics=[
            BinaryCritic(critic_field="title", weight=0.6),
            SimilarityCritic(
                critic_field="content", weight=0.4, similarity_threshold=0.3
            ),
        ],
    )

    suite.add_case(
        name="Multi-turn add: directory lifted from prior overview",
        user_message=(
            "Save my deploy notes to the work section: '# Deploy Plan\\n\\nStage 1: canary'"
        ),
        additional_messages=[
            {"role": "user", "content": "Where can I save things in my library?"},
            {
                "role": "assistant",
                "content": (
                    "You have these top-level sections: "
                    "/Users/me/notes/work, /Users/me/notes/personal, "
                    "/Users/me/notes/research."
                ),
            },
        ],
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_AddToLibrary",
                {
                    "title": "deploy notes",
                    "content": "# Deploy Plan\n\nStage 1: canary",
                    "directory": "/Users/me/notes/work",
                },
            )
        ],
        critics=[
            SimilarityCritic(critic_field="title", weight=0.25),
            SimilarityCritic(critic_field="content", weight=0.25),
            BinaryCritic(critic_field="directory", weight=0.5),
        ],
    )

    suite.add_case(
        name="Multi-turn search: keyword fallback after empty hybrid",
        user_message="Try a keyword search instead — the exact phrase is 'CRDT consistency'",
        additional_messages=[
            {
                "role": "user",
                "content": "Find docs about CRDT consistency in my library",
            },
            {
                "role": "assistant",
                "content": (
                    "I searched but didn't find anything matching that. "
                    "Want me to try a different mode?"
                ),
            },
        ],
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_SearchLibrary",
                {"query": "CRDT consistency", "mode": "keyword"},
            )
        ],
        critics=[
            SimilarityCritic(critic_field="query", weight=0.4),
            BinaryCritic(critic_field="mode", weight=0.6),
        ],
    )

    suite.add_case(
        name="Multi-turn update: path lifted from prior search result",
        user_message=(
            # 'Replace the contents of that doc' makes intent UNAMBIGUOUSLY an
            # update, not an add. The earlier draft used 'Add a note about ...'
            # which the model reasonably routed to AddToLibrary.
            "Replace the contents of that doc with: "
            "'Token rotation policy: 30 day expiry'"
        ),
        additional_messages=[
            {"role": "user", "content": "Find docs about JWT in my library"},
            {
                "role": "assistant",
                "content": (
                    "I found 1 doc: /Users/me/notes/work/JWT_Best_Practices.md "
                    "(score 0.94)."
                ),
            },
        ],
        expected_tool_calls=[
            ExpectedMCPToolCall(
                "Librarian_UpdateLibraryDoc",
                {
                    "path": "/Users/me/notes/work/JWT_Best_Practices.md",
                    "content": "Token rotation policy: 30 day expiry",
                },
            )
        ],
        critics=[
            BinaryCritic(critic_field="path", weight=0.6),
            SimilarityCritic(
                critic_field="content", weight=0.4, similarity_threshold=0.5
            ),
        ],
    )

    return suite
