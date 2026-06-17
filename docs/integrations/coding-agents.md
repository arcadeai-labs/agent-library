# Coding Agents

Agent Library is most useful for coding agents when it acts as **long-term retrieval memory**:

- a place to look up durable context from prior sessions
- a place to keep user-specific knowledge that is expensive to recreate
- not a place to dump every response, plan, log, or transient thought

The library already gives agents raw tools. This guide adds the missing usage pattern: when to search, when to save, when to update, and when to leave memory alone.

## What to use it for

Use Agent Library for knowledge that is likely to matter again in a later session:

- project context that is not obvious from the current repo
- recurring user preferences or workflow constraints
- durable notes, design decisions, and operating runbooks
- external docs or references you intentionally indexed
- artifacts an agent may need to revisit and refine over time

Do **not** treat it like a transcript store. Most chat turns, command output, and temporary reasoning should never be written to memory.

## When to search first

Search the library before answering or acting when the task may depend on prior work or stored knowledge.

Good times to search first:

- the user references earlier decisions, previous sessions, or existing notes
- the task depends on recurring preferences, conventions, or constraints
- the answer may already exist in indexed docs, plans, or project notes
- the agent is about to repeat research or rediscover context
- the task spans multiple sessions or long-running workflows

Suggested flow:

1. Call `search_library` with the user task or the likely concept.
2. If a hit looks relevant, call `read_from_library` before relying on it.
3. Use the full document to ground the next action or answer.

## When to write

Write to the library only when the information is:

- durable
- user- or project-specific
- likely to help in a future session
- hard to reconstruct from the repo, shell history, or public docs

Good write candidates:

- stable user preferences that affect future work
- distilled findings from a long debugging or research session
- project-specific operating notes or handoff context
- reusable summaries of external material the user wants retained

Bad write candidates:

- one-off command output
- low-value summaries of the current chat
- temporary plans that will be obsolete soon
- facts that are easy to re-derive from the current codebase
- verbose logs, stack traces, or scratch notes with no future reuse

## Update vs create vs skip

Prefer `update_library_doc` when you are refining an existing artifact:

- extending an existing project note
- revising a known plan or runbook
- replacing stale content in a durable document

Prefer `add_to_library` only when the knowledge is genuinely new and deserves its own durable document.

Prefer doing nothing when the information is ephemeral or low-value. A quiet library is usually a more useful library.

Before writing:

1. Call `get_library_overview` to inspect where durable notes already live.
2. Call `suggest_library_location` if the right place is unclear.
3. If a near-match already exists, update it instead of creating a duplicate.

## Copy-paste prompt for coding agents

Paste this into your agent instructions, project rules, or system prompt:

```text
When Agent Library is available, treat it as long-term retrieval memory, not a transcript dump.

Search first when the task may depend on prior sessions, indexed docs, recurring user preferences, or durable project context. Use search_library first, then read_from_library before relying on a result.

Write only durable, reusable knowledge that is likely to help in a future session. Do not save every response, plan, debug log, or transient thought.

Before writing, inspect the library with get_library_overview and use suggest_library_location if placement is unclear.

Prefer update_library_doc over add_to_library when refining an existing note or artifact. Avoid creating near-duplicate documents.

Skip memory entirely for ephemeral chat, one-off command output, facts that are easy to re-derive from the current repo, and low-value summaries.
```

## Where this helps most

This pattern is especially useful for coding agents such as Codex, Claude Code, Cursor, and similar MCP-capable tools. It helps them use Agent Library consistently:

- search when recall matters
- save only what is worth keeping
- update instead of duplicating
- avoid noisy writes that make later retrieval worse

If you are still setting up a specific client, start here:

- [Claude Code](claude-code.md)
- [Cursor](cursor.md)
- [Claude Desktop](claude-desktop.md)
