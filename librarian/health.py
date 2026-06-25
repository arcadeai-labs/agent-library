"""Read-only library health checks.

The health scanner reports retrieval risk signals for a user's existing
library. It does not claim retrieval accuracy because arbitrary user data has
no ground truth.
"""

# ruff: noqa: S608

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from librarian.storage.database import Database, get_database


@dataclass
class HealthIssue:
    """One health issue found in the indexed library."""

    severity: str
    code: str
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregate health report for an indexed library."""

    generated_at: str
    database_path: str
    document_count: int
    chunk_count: int
    embedding_count: int
    fts_count: int
    embedding_coverage: float
    fts_coverage: float
    document_asset_counts: dict[str, int]
    chunk_asset_counts: dict[str, int]
    chunk_modality_counts: dict[str, int]
    embedding_counts: dict[str, int]
    issue_counts: dict[str, int]
    issues: list[HealthIssue]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _safe_count(db: Database, table_name: str) -> int:
    """Count rows in a table, returning 0 if the table is unavailable."""
    try:
        with db.connection() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    except Exception:
        return 0


def _counter_from_rows(rows: list[Any], key: str, value_key: str = "count") -> dict[str, int]:
    """Convert SQLite rows with a grouping key and count into a plain dict."""
    return {str(row[key] or "unknown"): int(row[value_key]) for row in rows}


def _source_prefix(
    source: str | None,
    sources: list[dict[str, Any]] | None,
) -> str | None:
    """Resolve a source name/path filter to a path prefix."""
    if not source:
        return None

    for entry in sources or []:
        if entry.get("name") == source or entry.get("path") == source:
            return str(Path(entry["path"]).expanduser().resolve())

    return str(Path(source).expanduser().resolve())


def _source_entries(
    source: str | None,
    sources: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return source entries relevant to this run."""
    if not sources:
        return []
    if not source:
        return sources

    prefix = _source_prefix(source, sources)
    return [
        entry
        for entry in sources
        if entry.get("name") == source
        or entry.get("path") == source
        or (prefix is not None and str(entry.get("path", "")).startswith(prefix))
    ]


def _json_metadata(raw: str | None) -> dict[str, Any]:
    """Parse metadata JSON stored by the database."""
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _where_clause(base: str = "", extra: str = "") -> str:
    """Build a SQL WHERE clause from optional base and extra predicates."""
    predicates = [part for part in (base, extra) if part]
    return f"WHERE {' AND '.join(predicates)}" if predicates else ""


def _has_ocr_text(metadata: dict[str, Any]) -> bool:
    """Return True when image metadata contains non-empty OCR text."""
    value = metadata.get("ocr_text")
    return isinstance(value, str) and bool(value.strip())


def run_health_check(
    db: Database | None = None,
    sources: list[dict[str, Any]] | None = None,
    source: str | None = None,
    sample_limit: int = 20,
) -> HealthReport:
    """Run read-only health checks against the current library index."""
    db = db or get_database()
    sample_limit = max(1, sample_limit)

    prefix = _source_prefix(source, sources)
    doc_filter = ""
    chunk_filter = ""
    params: tuple[str, ...] = ()
    if prefix:
        doc_filter = "path LIKE ?"
        chunk_filter = "d.path LIKE ?"
        params = (f"{prefix}%",)
    doc_where = _where_clause(doc_filter)
    chunk_where = _where_clause(chunk_filter)

    issues: list[HealthIssue] = []

    with db.connection() as conn:
        document_count = int(
            conn.execute(f"SELECT COUNT(*) FROM documents {doc_where}", params).fetchone()[0]
        )
        chunk_count = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                {chunk_where}
                """,
                params,
            ).fetchone()[0]
        )

        document_asset_counts = _counter_from_rows(
            conn.execute(
                f"""
                SELECT COALESCE(asset_type, 'unknown') AS asset_type, COUNT(*) AS count
                FROM documents
                {doc_where}
                GROUP BY COALESCE(asset_type, 'unknown')
                """,
                params,
            ).fetchall(),
            "asset_type",
        )
        chunk_asset_counts = _counter_from_rows(
            conn.execute(
                f"""
                SELECT COALESCE(c.asset_type, 'unknown') AS asset_type, COUNT(*) AS count
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                {chunk_where}
                GROUP BY COALESCE(c.asset_type, 'unknown')
                """,
                params,
            ).fetchall(),
            "asset_type",
        )
        chunk_modality_counts = _counter_from_rows(
            conn.execute(
                f"""
                SELECT COALESCE(c.modality, 'unknown') AS modality, COUNT(*) AS count
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                {chunk_where}
                GROUP BY COALESCE(c.modality, 'unknown')
                """,
                params,
            ).fetchall(),
            "modality",
        )

        embedding_counts = {
            "text": _safe_count(db, "chunk_embeddings"),
            "code": _safe_count(db, "vec_chunks_code"),
            "vision": _safe_count(db, "vec_chunks_vision"),
        }

        if prefix:
            embedding_counts = {
                "text": int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM chunk_embeddings e
                        JOIN chunks c ON e.chunk_id = c.id
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.path LIKE ?
                        """,
                        params,
                    ).fetchone()[0]
                ),
                "code": int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM vec_chunks_code e
                        JOIN chunks c ON e.chunk_id = c.id
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.path LIKE ?
                        """,
                        params,
                    ).fetchone()[0]
                ),
                "vision": int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM vec_chunks_vision e
                        JOIN chunks c ON e.chunk_id = c.id
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.path LIKE ?
                        """,
                        params,
                    ).fetchone()[0]
                ),
            }

        embedding_count = sum(embedding_counts.values())

        fts_count = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM chunks_fts f
                JOIN chunks c ON f.rowid = c.id
                JOIN documents d ON c.document_id = d.id
                {chunk_where}
                """,
                params,
            ).fetchone()[0]
        )

        missing_embedding_rows = conn.execute(
            f"""
            SELECT c.id, c.asset_type, d.path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            LEFT JOIN chunk_embeddings te ON c.id = te.chunk_id
            LEFT JOIN vec_chunks_code ce ON c.id = ce.chunk_id
            LEFT JOIN vec_chunks_vision ve ON c.id = ve.chunk_id
            {_where_clause(chunk_filter, "te.chunk_id IS NULL AND ce.chunk_id IS NULL AND ve.chunk_id IS NULL")}
            LIMIT ?
            """,
            (*params, sample_limit) if params else (sample_limit,),
        ).fetchall()

        missing_embedding_count = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                LEFT JOIN chunk_embeddings te ON c.id = te.chunk_id
                LEFT JOIN vec_chunks_code ce ON c.id = ce.chunk_id
                LEFT JOIN vec_chunks_vision ve ON c.id = ve.chunk_id
                {_where_clause(chunk_filter, "te.chunk_id IS NULL AND ce.chunk_id IS NULL AND ve.chunk_id IS NULL")}
                """,
                params,
            ).fetchone()[0]
        )

        zero_chunk_rows = conn.execute(
            f"""
            SELECT d.path
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            {doc_where}
            GROUP BY d.id
            HAVING COUNT(c.id) = 0
            LIMIT ?
            """,
            (*params, sample_limit) if params else (sample_limit,),
        ).fetchall()

        short_doc_rows = conn.execute(
            f"""
            SELECT path, asset_type, LENGTH(TRIM(content)) AS content_length
            FROM documents
            {_where_clause(doc_filter, "LENGTH(TRIM(content)) < 100")}
            LIMIT ?
            """,
            (*params, sample_limit) if params else (sample_limit,),
        ).fetchall()

        short_pdf_rows = conn.execute(
            f"""
            SELECT path, LENGTH(TRIM(content)) AS content_length
            FROM documents
            {_where_clause(doc_filter, "asset_type = 'pdf' AND LENGTH(TRIM(content)) < 500")}
            LIMIT ?
            """,
            (*params, sample_limit) if params else (sample_limit,),
        ).fetchall()

        chunk_quality = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                AVG(LENGTH(c.content)) AS avg_length,
                SUM(CASE WHEN LENGTH(TRIM(c.content)) = 0 THEN 1 ELSE 0 END) AS empty_count,
                SUM(CASE WHEN LENGTH(TRIM(c.content)) < 50 THEN 1 ELSE 0 END) AS small_count,
                SUM(CASE WHEN LENGTH(c.content) > 4000 THEN 1 ELSE 0 END) AS large_count
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            {chunk_where}
            """,
            params,
        ).fetchone()

        duplicate_chunk_rows = conn.execute(
            f"""
            SELECT c.content, COUNT(*) AS count
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            {chunk_where}
            GROUP BY c.content
            HAVING COUNT(*) > 1 AND LENGTH(TRIM(c.content)) > 0
            ORDER BY count DESC
            LIMIT ?
            """,
            (*params, sample_limit) if params else (sample_limit,),
        ).fetchall()

        docs = conn.execute(
            f"""
            SELECT id, path, asset_type, metadata, content, file_mtime
            FROM documents
            {doc_where}
            """,
            params,
        ).fetchall()

    embedding_coverage = (embedding_count / chunk_count) if chunk_count else 1.0
    fts_coverage = (fts_count / chunk_count) if chunk_count else 1.0

    if document_count == 0:
        issues.append(
            HealthIssue(
                severity="high",
                code="empty_index",
                message="No indexed documents found. Run `libr add <path>` first.",
            )
        )

    if missing_embedding_count:
        issues.append(
            HealthIssue(
                severity="high",
                code="missing_embeddings",
                message=f"{missing_embedding_count} chunks have no vector embedding.",
                details={
                    "count": missing_embedding_count,
                    "samples": [dict(row) for row in missing_embedding_rows],
                },
            )
        )

    if chunk_count and fts_count != chunk_count:
        issues.append(
            HealthIssue(
                severity="high",
                code="fts_count_mismatch",
                message=f"FTS index has {fts_count} rows for {chunk_count} chunks.",
                details={"chunks": chunk_count, "fts_rows": fts_count},
            )
        )

    for row in zero_chunk_rows:
        issues.append(
            HealthIssue(
                severity="high",
                code="zero_chunks",
                message="Indexed document has no chunks.",
                path=row["path"],
            )
        )

    for row in short_doc_rows:
        issues.append(
            HealthIssue(
                severity="medium",
                code="short_extracted_content",
                message="Document has very little extracted text.",
                path=row["path"],
                details={"asset_type": row["asset_type"], "content_length": row["content_length"]},
            )
        )

    for row in short_pdf_rows:
        issues.append(
            HealthIssue(
                severity="high",
                code="pdf_text_too_short",
                message="PDF produced very little extracted text; it may be scanned or image-only.",
                path=row["path"],
                details={"content_length": row["content_length"]},
            )
        )

    empty_chunks = int(chunk_quality["empty_count"] or 0)
    small_chunks = int(chunk_quality["small_count"] or 0)
    large_chunks = int(chunk_quality["large_count"] or 0)
    avg_length = float(chunk_quality["avg_length"] or 0.0)

    if empty_chunks:
        issues.append(
            HealthIssue(
                severity="high",
                code="empty_chunks",
                message=f"{empty_chunks} chunks are empty.",
                details={"count": empty_chunks},
            )
        )
    if small_chunks:
        issues.append(
            HealthIssue(
                severity="low",
                code="small_chunks",
                message=f"{small_chunks} chunks are under 50 characters.",
                details={"count": small_chunks, "avg_chunk_length": round(avg_length, 1)},
            )
        )
    if large_chunks:
        issues.append(
            HealthIssue(
                severity="medium",
                code="large_chunks",
                message=f"{large_chunks} chunks are over 4000 characters.",
                details={"count": large_chunks, "avg_chunk_length": round(avg_length, 1)},
            )
        )
    if duplicate_chunk_rows:
        duplicate_total = sum(int(row["count"]) - 1 for row in duplicate_chunk_rows)
        issues.append(
            HealthIssue(
                severity="low",
                code="duplicate_chunks",
                message=f"Found at least {duplicate_total} duplicate chunk copies.",
                details={
                    "samples": [
                        {
                            "count": int(row["count"]),
                            "preview": str(row["content"])[:120],
                        }
                        for row in duplicate_chunk_rows[:5]
                    ]
                },
            )
        )

    for doc in docs:
        path = str(doc["path"])
        asset_type = str(doc["asset_type"] or "text")
        metadata = _json_metadata(doc["metadata"])

        if asset_type == "image" and not _has_ocr_text(metadata):
            issues.append(
                HealthIssue(
                    severity="medium",
                    code="image_no_ocr_text",
                    message="Image has no OCR text; search may rely only on metadata or vision embeddings.",
                    path=path,
                )
            )

        if asset_type == "code":
            symbols = metadata.get("symbols")
            if isinstance(symbols, list) and not symbols:
                issues.append(
                    HealthIssue(
                        severity="medium",
                        code="code_no_symbols",
                        message="Code file has no detected symbols.",
                        path=path,
                    )
                )

        file_path = Path(path)
        if not file_path.exists():
            issues.append(
                HealthIssue(
                    severity="medium",
                    code="indexed_file_missing",
                    message="Indexed file no longer exists on disk.",
                    path=path,
                )
            )
            continue

        stored_mtime = doc["file_mtime"]
        if stored_mtime is not None:
            try:
                current_mtime = file_path.stat().st_mtime
            except OSError:
                continue
            if current_mtime > float(stored_mtime):
                issues.append(
                    HealthIssue(
                        severity="medium",
                        code="indexed_file_changed",
                        message="File changed on disk after it was indexed.",
                        path=path,
                    )
                )

    for entry in _source_entries(source, sources):
        raw_path = entry.get("path")
        if raw_path and not Path(str(raw_path)).exists():
            issues.append(
                HealthIssue(
                    severity="medium",
                    code="source_path_missing",
                    message="Registered source path no longer exists.",
                    path=str(raw_path),
                    details={"source": entry.get("name")},
                )
            )

    issue_counts = dict(Counter(issue.severity for issue in issues))

    return HealthReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        database_path=db.db_path,
        document_count=document_count,
        chunk_count=chunk_count,
        embedding_count=embedding_count,
        fts_count=fts_count,
        embedding_coverage=round(embedding_coverage, 4),
        fts_coverage=round(fts_coverage, 4),
        document_asset_counts=document_asset_counts,
        chunk_asset_counts=chunk_asset_counts,
        chunk_modality_counts=chunk_modality_counts,
        embedding_counts=embedding_counts,
        issue_counts=issue_counts,
        issues=issues,
    )
