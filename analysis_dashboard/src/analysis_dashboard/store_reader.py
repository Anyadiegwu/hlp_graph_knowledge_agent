# analysis_dashboard/src/analysis_dashboard/store_reader.py
#
# Read-only adapter that lets the analysis_dashboard process access
# the HLPLogStore written by agent_client without importing from
# that package (maintaining process isolation).
#
# This avoids cross-package imports by re-implementing only the
# read operations the analysis agent needs, pointing at the same DB file.

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("analysis_dashboard.store_reader")

# ─────────────────────────────────────────────────────────────
# Cosine similarity (pure Python — no numpy required here)
# ─────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────
# Embedding (Gemini, same as log_store.py)
# ─────────────────────────────────────────────────────────────

class _EmbeddingProvider:
    def __init__(self) -> None:
        self._model  = None
        self._stub   = False
        self._dim    = 768

    def _load(self) -> None:
        if self._model or self._stub:
            return
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            self._stub = True
            return
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )
        except Exception:
            self._stub = True

    def embed(self, text: str) -> list[float]:
        self._load()
        if self._stub:
            return [0.0] * self._dim
        try:
            return self._model.embed_query(text)
        except Exception:
            return [0.0] * self._dim


_embedder = _EmbeddingProvider()


# ─────────────────────────────────────────────────────────────
# SharedLogStoreReader
# ─────────────────────────────────────────────────────────────

class SharedLogStoreReader:
    """
    Read-only view over an HLPLogStore SQLite database.
    Safe to open from a separate process (SQLite WAL mode or read-only connection).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock   = threading.Lock()
        self._conn: sqlite3.Connection | None = None

        if not self.db_path.exists():
            logger.warning(
                "Log DB not found at %s — search/stats will return empty results. "
                "Run the agent_client first to populate it.",
                self.db_path,
            )

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            uri = f"file:{self.db_path}?mode=ro"
            try:
                self._conn = sqlite3.connect(
                    uri, uri=True, check_same_thread=False, timeout=10
                )
            except Exception:
                # DB doesn't exist yet — open in create mode but never write
                self._conn = sqlite3.connect(
                    str(self.db_path), check_same_thread=False, timeout=10
                )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _safe_rows(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        try:
            with self._lock:
                conn = self._get_conn()
                return conn.execute(query, params).fetchall()
        except Exception as exc:
            logger.warning("DB read error: %s", exc)
            return []

    def get_all(self, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._safe_rows(
            "SELECT * FROM log_entries ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.get("metadata_json", "{}"))
            results.append(d)
        return results

    def get_stats(self) -> dict[str, Any]:
        try:
            with self._lock:
                conn = self._get_conn()
                total     = conn.execute("SELECT COUNT(*) FROM log_entries").fetchone()[0]
                sessions  = conn.execute("SELECT COUNT(DISTINCT session_id) FROM log_entries").fetchone()[0]
                by_type   = conn.execute(
                    "SELECT interaction_type, COUNT(*) as cnt FROM log_entries GROUP BY interaction_type"
                ).fetchall()
                by_ns     = conn.execute(
                    "SELECT namespace, COUNT(*) as cnt FROM log_entries GROUP BY namespace ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
                avg_lat   = conn.execute(
                    "SELECT AVG(latency_ms) FROM log_entries WHERE latency_ms IS NOT NULL"
                ).fetchone()[0]
                errors    = conn.execute(
                    "SELECT COUNT(*) FROM log_entries WHERE interaction_type = 'error'"
                ).fetchone()[0]
                tool_lats = conn.execute(
                    """SELECT tool_name, AVG(latency_ms) as avg_lat, COUNT(*) as cnt
                       FROM log_entries
                       WHERE tool_name IS NOT NULL AND latency_ms IS NOT NULL
                       GROUP BY tool_name"""
                ).fetchall()

            return {
                "total_entries":       total,
                "total_sessions":      sessions,
                "by_interaction_type": {r[0]: r[1] for r in by_type},
                "by_namespace":        {r[0]: r[1] for r in by_ns},
                "avg_latency_ms":      avg_lat,
                "error_count":         errors,
                "tool_latencies":      [dict(r) for r in tool_lats],
            }
        except Exception as exc:
            logger.warning("get_stats error: %s", exc)
            return {
                "total_entries": 0, "total_sessions": 0,
                "by_interaction_type": {}, "by_namespace": {},
                "avg_latency_ms": None, "error_count": 0, "tool_latencies": [],
            }

    def get_sessions(self) -> list[str]:
        rows = self._safe_rows(
            "SELECT DISTINCT session_id FROM log_entries ORDER BY MIN(timestamp) ASC"
        )
        return [r[0] for r in rows]

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._safe_rows(
            "SELECT * FROM log_entries WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )
        return [dict(r) for r in rows]

    def search(
        self,
        query: str,
        k: int = 10,
        namespace_prefix: str | None = None,
        session_id: str | None = None,
        interaction_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vec = _embedder.embed(query)
        is_stub   = all(v == 0.0 for v in query_vec)

        clauses: list[str] = []
        params:  list[Any] = []
        if namespace_prefix:
            clauses.append("namespace LIKE ?")
            params.append(f"{namespace_prefix}%")
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if interaction_type:
            clauses.append("interaction_type = ?")
            params.append(interaction_type)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows  = self._safe_rows(
            f"SELECT * FROM log_entries {where} ORDER BY timestamp DESC LIMIT 500",
            tuple(params),
        )

        results = []
        for row in rows:
            d = dict(row)
            if is_stub:
                score = float(query.lower() in d.get("content", "").lower())
            else:
                emb_raw = d.get("embedding_json")
                if not emb_raw:
                    score = 0.0
                else:
                    stored = json.loads(emb_raw)
                    score  = _cosine(query_vec, stored)
            d["_score"] = score
            d["metadata"] = json.loads(d.get("metadata_json", "{}"))
            results.append(d)

        results.sort(key=lambda r: r["_score"], reverse=True)
        return results[:k]

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# ─────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────

_reader_instance: SharedLogStoreReader | None = None


def get_shared_log_store(db_path: str | Path | None = None) -> SharedLogStoreReader:
    global _reader_instance
    if _reader_instance is None:
        if db_path is None or str(db_path) == "":
            # __file__ = <repo_root>/analysis_dashboard/src/analysis_dashboard/store_reader.py
            # parents[0] = analysis_dashboard/  (package dir)
            # parents[1] = src/
            # parents[2] = analysis_dashboard/  (package root)
            # parents[3] = repo root            (where mcp_agent_log.db lives)
            _repo_root = Path(__file__).resolve().parents[3]
            db_path = _repo_root / "mcp_agent_log.db"
        _reader_instance = SharedLogStoreReader(db_path)
        logger.info("SharedLogStoreReader initialised at %s", db_path)
    return _reader_instance
