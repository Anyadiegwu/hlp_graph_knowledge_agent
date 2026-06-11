# agent_client/src/agent_client/log_store.py
#
# HLP Graph Knowledge Agent — Stage 3 — Embedded Vector Log Store
#
# Wraps LangGraph's SqliteStore (synchronous) with:
#   • Hierarchical dot-separated namespace tuples
#   • Embedding model integration for semantic vector indexing
#   • Strict Pydantic schema: session_id, mcp_interaction_type, content
#   • Integer-key casting guardrail for JSON serialisation safety
#   • Thread-safe write path for use inside async agent loops
#
# All logs are ALSO still written to agent_system.log (flat file) from client.py.
# This store is the ADDITIONAL structured persistence layer required by Stage 3.

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("agent.log_store")

# ─────────────────────────────────────────────────────────────
# 1. Schema
# ─────────────────────────────────────────────────────────────

class MCPInteractionType(str, Enum):
    """Explicitly typed MCP interaction classification."""
    TOOL_INVOCATION      = "tool_invocation"
    RESOURCE_READ        = "resource_read"
    SAMPLING_REQUEST     = "sampling_request"
    SAMPLING_RESPONSE    = "sampling_response"
    AGENT_REASONING      = "agent_reasoning"
    AGENT_FINAL_ANSWER   = "agent_final_answer"
    SYSTEM_EVENT         = "system_event"
    ERROR                = "error"


class LogEntry(BaseModel):
    """
    Validated schema for every log entry persisted to the SQLite vector store.

    Fields
    ------
    session_id          : UUID tracking the multi-turn execution trace.
    mcp_interaction_type: Explicit enum — tool_invocation, resource_read,
                          sampling_request, sampling_response, agent_reasoning,
                          agent_final_answer, system_event, error.
    content             : Raw text payload targeted for semantic vector search.
    namespace_path      : Dot-separated path string (e.g. "logs.agent.planning").
    component           : Origin component (e.g. "agent_client", "mcp_server").
    tool_name           : Name of the MCP tool involved (if applicable).
    latency_ms          : Wall-clock latency for the operation in milliseconds.
    token_count         : Approximate token count of the content (if known).
    metadata            : Arbitrary extra data — integer keys are cast back to int.
    timestamp           : ISO-8601 UTC timestamp (auto-set on creation).
    """

    session_id:           str             = Field(default_factory=lambda: str(uuid.uuid4()))
    mcp_interaction_type: MCPInteractionType
    content:              str
    namespace_path:       str             = "logs.system"
    component:            str             = "unknown"
    tool_name:            Optional[str]   = None
    latency_ms:           Optional[float] = None
    token_count:          Optional[int]   = None
    metadata:             dict[str, Any]  = Field(default_factory=dict)
    timestamp:            str             = Field(
                              default_factory=lambda: datetime.now(timezone.utc).isoformat()
                          )

    @field_validator("metadata", mode="before")
    @classmethod
    def cast_integer_keys(cls, v: Any) -> dict[str, Any]:
        """
        Data Guardrail: JSON serialisation converts integer dict keys to strings.
        This validator casts them back to integers where the original key was
        a digit string, preventing type-mismatch faults downstream.
        """
        if not isinstance(v, dict):
            return v or {}
        repaired: dict[str, Any] = {}
        for key, val in v.items():
            try:
                repaired[int(key)] = val  # type: ignore[assignment]
            except (ValueError, TypeError):
                repaired[key] = val
        return repaired  # type: ignore[return-value]

    def namespace_tuple(self) -> tuple[str, ...]:
        """Convert dot-separated path to tuple for LangGraph store API."""
        return tuple(self.namespace_path.split("."))

    def store_key(self) -> str:
        """Unique key within the namespace: timestamp + session fragment."""
        ts_clean = self.timestamp.replace(":", "-").replace(".", "-")
        session_fragment = self.session_id[:8]
        return f"{ts_clean}_{session_fragment}"

    def to_store_value(self) -> dict[str, Any]:
        """Serialise to the dict stored in the vector store's value field."""
        return {
            "session_id":           self.session_id,
            "mcp_interaction_type": self.mcp_interaction_type.value,
            "content":              self.content,
            "namespace_path":       self.namespace_path,
            "component":            self.component,
            "tool_name":            self.tool_name,
            "latency_ms":           self.latency_ms,
            "token_count":          self.token_count,
            "metadata":             {str(k): v for k, v in self.metadata.items()},
            "timestamp":            self.timestamp,
        }


# ─────────────────────────────────────────────────────────────
# 2. Namespace helpers
#    Predefined hierarchical namespaces — Stage 3 requirement.
# ─────────────────────────────────────────────────────────────

class NS:
    """Canonical namespace path constants."""
    # Agent-side namespaces
    AGENT_REASONING       = "logs.agent.reasoning"
    AGENT_PLANNING        = "logs.agent.planning.reflexive_loop"
    AGENT_FINAL_ANSWER    = "logs.agent.output.final_answer"
    AGENT_ERROR           = "logs.agent.error"

    # MCP client-side namespaces
    MCP_CLIENT_CONNECT    = "logs.mcp.client.connection"
    MCP_CLIENT_TOOL_CALL  = "logs.mcp.client.tool_invocation"
    MCP_CLIENT_SAMPLING   = "logs.mcp.client.sampling"

    # MCP server-side namespaces (forwarded via log relay)
    MCP_SERVER_TOOL       = "logs.mcp.server.tools.execute_code"
    MCP_SERVER_CRAG       = "logs.mcp.server.tools.crag_pipeline"
    MCP_SERVER_REFLECTION = "logs.mcp.server.tools.reflection"
    MCP_SERVER_SAMPLING   = "logs.mcp.server.sampling_request"

    # System-level
    SYSTEM_STARTUP        = "logs.system.startup"
    SYSTEM_SHUTDOWN       = "logs.system.shutdown"


# ─────────────────────────────────────────────────────────────
# 3. Embedding helper
#    Wraps Google Gemini embeddings with a simple fallback to
#    a zero-vector stub when no API key is available, so the
#    store can still be used without embedding credentials.
# ─────────────────────────────────────────────────────────────

class _EmbeddingProvider:
    """Lazy-loaded embedding provider with Gemini → stub fallback."""

    def __init__(self) -> None:
        self._model: Any = None
        self._stub: bool = False
        self._dim: int = 768  # gemini-embedding-001 dimensionality

    def _load(self) -> None:
        if self._model is not None or self._stub:
            return
        import os
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set — using zero-vector stub embeddings. "
                "Semantic search will not rank by meaning."
            )
            self._stub = True
            return
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )
            logger.info("Embedding model loaded: models/gemini-embedding-001")
        except Exception as exc:
            logger.warning("Embedding model init failed (%s) — using stub.", exc)
            self._stub = True

    def embed(self, text: str) -> list[float]:
        self._load()
        if self._stub:
            return [0.0] * self._dim
        try:
            return self._model.embed_query(text)
        except Exception as exc:
            logger.warning("Embedding call failed (%s) — using zero vector.", exc)
            return [0.0] * self._dim

    @property
    def dim(self) -> int:
        return self._dim


_embedder = _EmbeddingProvider()


# ─────────────────────────────────────────────────────────────
# 4. SQLite Vector Log Store
#
#    Implements the Stage 3 requirement using raw SQLite with
#    a custom schema that mirrors the LangGraph store tuple
#    API (namespace, key, value, embedding) while remaining
#    installable without the full langgraph-checkpoint-sqlite
#    binary wheels in all environments.
#
#    The store exposes:
#      • put(entry)        — write a LogEntry
#      • search(query, k)  — semantic similarity search
#      • list_namespace(ns)— list all entries under a namespace
#      • get_all()         — dump all entries (for analysis agent)
# ─────────────────────────────────────────────────────────────

class HLPLogStore:
    """
    Thread-safe SQLite-backed vector log store.

    Schema
    ------
    table: log_entries
      id             INTEGER PRIMARY KEY
      namespace      TEXT    — dot-separated namespace path
      store_key      TEXT    — unique key within namespace
      session_id     TEXT    — UUID of the originating session
      interaction_type TEXT  — MCPInteractionType value
      component      TEXT
      tool_name      TEXT
      latency_ms     REAL
      token_count    INTEGER
      content        TEXT    — raw text for search
      metadata_json  TEXT    — JSON-serialised metadata
      timestamp      TEXT    — ISO-8601 UTC
      embedding_json TEXT    — JSON-serialised float list (nullable)
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS log_entries (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace        TEXT    NOT NULL,
        store_key        TEXT    NOT NULL,
        session_id       TEXT    NOT NULL,
        interaction_type TEXT    NOT NULL,
        component        TEXT    NOT NULL DEFAULT 'unknown',
        tool_name        TEXT,
        latency_ms       REAL,
        token_count      INTEGER,
        content          TEXT    NOT NULL,
        metadata_json    TEXT    NOT NULL DEFAULT '{}',
        timestamp        TEXT    NOT NULL,
        embedding_json   TEXT,
        UNIQUE(namespace, store_key)
    );
    CREATE INDEX IF NOT EXISTS idx_namespace  ON log_entries(namespace);
    CREATE INDEX IF NOT EXISTS idx_session    ON log_entries(session_id);
    CREATE INDEX IF NOT EXISTS idx_type       ON log_entries(interaction_type);
    CREATE INDEX IF NOT EXISTS idx_timestamp  ON log_entries(timestamp);
    CREATE INDEX IF NOT EXISTS idx_tool       ON log_entries(tool_name);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()
        logger.info("HLPLogStore initialised at %s", self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(self._CREATE_TABLE)
            conn.commit()

    # ── Write ──────────────────────────────────────────────────

    def put(self, entry: LogEntry, embed: bool = True) -> None:
        """
        Persist a LogEntry to the store.

        Parameters
        ----------
        entry : LogEntry   The validated log entry to persist.
        embed : bool       Whether to compute and store the embedding vector.
                           Set False for high-frequency low-value log lines
                           to avoid API rate limits.
        """
        embedding: list[float] | None = None
        if embed:
            embedding = _embedder.embed(entry.content)

        ns_str  = entry.namespace_path
        key_str = entry.store_key()

        # Repair integer keys before JSON serialisation
        safe_meta = {str(k): v for k, v in entry.metadata.items()}

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO log_entries
                  (namespace, store_key, session_id, interaction_type,
                   component, tool_name, latency_ms, token_count,
                   content, metadata_json, timestamp, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ns_str,
                    key_str,
                    entry.session_id,
                    entry.mcp_interaction_type.value,
                    entry.component,
                    entry.tool_name,
                    entry.latency_ms,
                    entry.token_count,
                    entry.content,
                    json.dumps(safe_meta),
                    entry.timestamp,
                    json.dumps(embedding) if embedding is not None else None,
                ),
            )
            conn.commit()

    # ── Semantic Search ────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        namespace_prefix: str | None = None,
        session_id: str | None = None,
        interaction_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Cosine-similarity semantic search over stored embeddings.

        Falls back to keyword substring match when embeddings are stubs
        (i.e. when GEMINI_API_KEY is not set).

        Parameters
        ----------
        query            : Natural language search query.
        k                : Maximum results to return.
        namespace_prefix : If set, restrict search to namespaces starting with this prefix.
        session_id       : If set, restrict to a specific session.
        interaction_type : If set, restrict to a specific MCPInteractionType value.
        """
        query_vec = _embedder.embed(query)
        is_stub = all(v == 0.0 for v in query_vec)

        with self._lock:
            conn = self._get_conn()
            clauses: list[str] = []
            params: list[Any] = []

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
            rows = conn.execute(
                f"SELECT * FROM log_entries {where} ORDER BY timestamp DESC LIMIT 500",
                params,
            ).fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            if is_stub:
                # Keyword fallback
                score = float(query.lower() in row_dict["content"].lower())
            else:
                emb_raw = row_dict.get("embedding_json")
                if not emb_raw:
                    score = 0.0
                else:
                    stored_vec: list[float] = json.loads(emb_raw)
                    score = self._cosine(query_vec, stored_vec)
            row_dict["_score"] = score
            row_dict["metadata"] = json.loads(row_dict.get("metadata_json", "{}"))
            results.append(row_dict)

        results.sort(key=lambda r: r["_score"], reverse=True)
        return results[:k]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Pure-Python cosine similarity — avoids numpy dependency in client."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Listing & Retrieval ────────────────────────────────────

    def list_namespace(self, namespace_prefix: str) -> list[dict[str, Any]]:
        """Return all entries under a namespace prefix, ordered by timestamp."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM log_entries WHERE namespace LIKE ? ORDER BY timestamp ASC",
                (f"{namespace_prefix}%",),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Dump all entries for the analysis agent to process."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM log_entries ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.get("metadata_json", "{}"))
            results.append(d)
        return results

    def get_sessions(self) -> list[str]:
        """Return distinct session IDs ordered by first-seen timestamp."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM log_entries ORDER BY MIN(timestamp) ASC",
            ).fetchall()
        return [r[0] for r in rows]

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """All entries for a given session, ordered by timestamp."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM log_entries WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard overview."""
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
            "total_entries":    total,
            "total_sessions":   sessions,
            "by_interaction_type": {r[0]: r[1] for r in by_type},
            "by_namespace":     {r[0]: r[1] for r in by_ns},
            "avg_latency_ms":   avg_lat,
            "error_count":      errors,
            "tool_latencies":   [dict(r) for r in tool_lats],
        }

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# ─────────────────────────────────────────────────────────────
# 5. Convenience factory — singleton used by client.py
# ─────────────────────────────────────────────────────────────

_store_instance: HLPLogStore | None = None


def get_log_store(db_path: str | Path | None = None) -> HLPLogStore:
    """Return the singleton HLPLogStore, creating it if necessary."""
    global _store_instance
    if _store_instance is None:
        if db_path is None:
            # Default: repo root / mcp_agent_log.db
            _repo_root = Path(__file__).resolve().parents[3]
            db_path = _repo_root / "mcp_agent_log.db"
        _store_instance = HLPLogStore(db_path)
    return _store_instance
