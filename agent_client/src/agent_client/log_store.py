from __future__ import annotations

import os
import json
import logging
import asyncpg
from pgvector.asyncpg import register_vector
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv(find_dotenv(".env"))

logger = logging.getLogger("agent.log_store")

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
                repaired[int(key)] = val
            except (ValueError, TypeError):
                repaired[key] = val
        return repaired 

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


class _EmbeddingProvider:
    """Lazy-loaded embedding provider with Gemini → stub fallback."""

    def __init__(self) -> None:
        self._model: Any = None
        self._stub: bool = False
        self._dim: int = 768 

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
                output_dimensionality=768,
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

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None
    
    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn, 
                min_size=2, 
                max_size=10,
                init=register_vector,
                statement_cache_size=0,
            )
            logger.info("HLPLogStore connected to SUpabase (pooled)")
        return self._pool
    
    async def _get_or_create_session_row(
            self, pool: asyncpg.Pool, session_id: str, component: str
    ) -> Any:
        """Ensure a sessions row exists for this app-level session_id; return its internal uuid id."""
        row = await pool.fetchrow(
            """
            INSERT INTO  sessions (session_id, component)
            VALUES ($1, $2)
            ON CONFLICT (session_id) DO UPDATE
                SET component = EXCLUDED.component
            RETURNING id
            """, 
            session_id, 
            component
        )
        return row['id']

    async def put(self, entry: LogEntry, embed: bool = True) -> None:
        """
        Persist a LogEntry to Supabase (sessions + log_entries).

        Parameters
        ----------
        entry : LogEntry   The validated log entry to persist.
        embed : bool       Whether to compute and store the embedding vector.
        """
        embedding: list[float] | None = None
        if embed:
            embedding = _embedder.embed(entry.content)
        
        safe_meta = {str(k): v for k, v in entry.metadata.items()}

        pool =  await self._get_pool()
        session_row_id = await self._get_or_create_session_row(
            pool, entry.session_id, entry.component
        )

        await pool.execute(
            """
            INSERT INTO log_entries
                (session_id, namespace, store_key, interaction_type,
                 component, tool_name, latency_ms, token_count,
                 content, metadata, timestamp, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (namespace, store_key) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """,
            session_row_id,
            entry.namespace_path,
            entry.store_key(),
            entry.mcp_interaction_type.value,
            entry.component,
            entry.tool_name,
            entry.latency_ms,
            entry.token_count,
            entry.content,
            json.dumps(safe_meta),
            datetime.fromisoformat(entry.timestamp),
            embedding if embedding is not None else None
        )

    async def search(
        self,
        query: str,
        k: int = 10,
        namespace_prefix: str | None = None,
        session_id: str | None = None,
        interaction_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Vector similarity search stored embeddings using pgvector's 
        cosine-distance operator (<=>), backed by the hnsw index on 
        log_entries.embedding.

        Falls back to keyword substring match when embeddings are stubs
        (i.e. when GEMINI_API_KEY is not set).
        """
        query_vec = _embedder.embed(query)
        is_stub = all(v == 0.0 for v in query_vec)

        pool = await self._get_pool()

        clauses: list[str] = []
        params: list[Any] = []

        if namespace_prefix:
            params.append(f"{namespace_prefix}%")
            clauses.append(f"namespace LIKE ${len(params)}")
        if session_id:
            params.append(session_id)
            clauses.append(f"session_id = (SELECT id FROM sessions WHERE session_id = ${len(params)})")
        if interaction_type:
            params.append(interaction_type)
            clauses.append(f"interaction_type = ${len(params)}")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        
        if is_stub:
            params.append(f"%{query.lower()}%")
            clauses_kw = where + (" AND " if where else "WHERE ") + f"LOWER(content) LIKE ${len(params)}"
            rows = await pool.fetch(
                f"SELECT *, 1.0 AS score FROM log_entries {clauses_kw} "
                f"ORDER BY timestamp DESC LIMIT {k}",
                *params
            )
        else:
            params.append(embedding_query := query_vec)
            emb_param = f"${len(params)}"
            rows = await pool.fetch(
                f"""
                SELECT *, 1 - (embedding <=> {emb_param}) AS score
                FROM log_entries
                {where}
                ORDER BY embedding <=> {emb_param}
                LIMIT {k}
                """,
                *params,    
            )

        results = []
        for row in rows:
            row_dict = dict(row)
            row_dict["_score"] = row_dict.pop("score")
            results.append(row_dict)
        return results

    async def list_namespace(self, namespace_prefix: str) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM log_entries WHERE namespace LIKE $1 ORDER BY timestamp ASC",
            (f"{namespace_prefix}%",),
        )
        return [dict(r) for r in rows]
    
    async def get_all(self, limit: int = 1000) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM log_entries ORDER BY timestamp DESC LIMIT $1",
            limit,
        )
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            results.append(d)
        return results

    async def get_sessions(self) -> list[str]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT session_id 
            FROM sessions 
            ORDER BY started_at ASC
            """
        )
        return [r["session_id"] for r in rows]

    async def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM log_entries le
            JOIN sessions s ON s.id = le.session_id
            WHERE s.session_id = $1 
            ORDER BY le.timestamp ASC
            """,
            session_id,
        )
        return [dict(r) for r in rows]

    async def get_stats(self) -> dict[str, Any]:
        pool = await self._get_pool()
        total    = await pool.fetchval("SELECT COUNT(*) FROM log_entries")
        sessions = await pool.fetchval("SELECT COUNT(*) FROM sessions")
        by_type  = await pool.fetch(
            "SELECT interaction_type, COUNT(*) as cnt FROM log_entries GROUP BY interaction_type"
        )
        by_ns    = await pool.fetch(
            "SELECT namespace, COUNT(*) as cnt FROM log_entries GROUP BY namespace ORDER BY cnt DESC LIMIT 10"
        )
        avg_lat  = await pool.fetchval(
            "SELECT AVG(latency_ms) FROM log_entries WHERE latency_ms IS NOT NULL"
        )
        errors   = await pool.fetchval(
            "SELECT COUNT(*) FROM log_entries WHERE interaction_type = 'error'"
        )
        tool_lats = await pool.fetch(
            """SELECT tool_name, AVG(latency_ms) as avg_lat, COUNT(*) as cnt
               FROM log_entries
               WHERE tool_name IS NOT NULL AND latency_ms IS NOT NULL
               GROUP BY tool_name"""
        )
        return {
            "total_entries":    total,
            "total_sessions":   sessions,
            "by_interaction_type": {r["interaction_type"]: r["cnt"] for r in by_type},
            "by_namespace":     {r["namespace"]: r["cnt"] for r in by_ns},
            "avg_latency_ms":   avg_lat,
            "error_count":      errors,
            "tool_latencies":   [dict(r) for r in tool_lats],
        }

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("HLPLogStore connection pool closed")


_store_instance: HLPLogStore | None = None


def get_log_store(dsn: str | None = None) -> HLPLogStore:
    """Return the singleton HLPLogStore, creating it if necessary."""
    global _store_instance
    if _store_instance is None:
        dsn = dsn or os.getenv("SUPABASE_DB_URL")
        if not dsn:
            raise RuntimeError(
                "SUPABASE_DB_URL environment variable not set. "
                )
        _store_instance = HLPLogStore(dsn)
    return _store_instance