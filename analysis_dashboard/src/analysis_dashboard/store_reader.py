from __future__ import annotations

import json
import logging
import os
import asyncpg
import asyncio
from pgvector.asyncpg import register_vector
from typing import Any
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env"))
logger = logging.getLogger("analysis_dashboard.store_reader")

class _EmbeddingProvider:
    def __init__(self) -> None:
        self._model = None
        self._stub  = False
        self._dim   = 768

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
                output_dimensionality=768,
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

class SharedLogStoreReader:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._pool_loop: asyncio.AbstractEventLoop | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        current_loop = asyncio.get_running_loop()
        if self._pool is None or self._pool_loop is not current_loop:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=1,
                max_size=5,
                init=register_vector,
                statement_cache_size=0,
            )
            self._pool_loop = current_loop
        return self._pool
            
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

    async def get_stats(self) -> dict[str, Any]:
        try:
            pool = await self._get_pool()
            total    = await pool.fetchval("SELECT COUNT(*) FROM log_entries")
            sessions = await pool.fetchval("SELECT COUNT(DISTINCT session_id) FROM log_entries")
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
                "total_entries":       total,
                "total_sessions":      sessions,
                "by_interaction_type": {r["interaction_type"]: r["cnt"] for r in by_type},
                "by_namespace":        {r["namespace"]: r["cnt"] for r in by_ns},
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

    async def search(
        self,
        query: str,
        k: int = 10,
        namespace_prefix: str | None = None,
        session_id: str | None = None,
        interaction_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vec = _embedder.embed(query)
        is_stub   = all(v == 0.0 for v in query_vec)

        pool = await self._get_pool()

        clauses: list[str] = []
        params:  list[Any] = []

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
    
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None


_reader_instance: SharedLogStoreReader | None = None


def get_shared_log_store(dsn: str | None = None) -> SharedLogStoreReader:
    global _reader_instance
    if _reader_instance is None:
        dsn = dsn or os.getenv("SUPABASE_DB_URL")
        if not dsn:
            raise RuntimeError("SUPABASE_DB_URL environment variable is not set.")
        _reader_instance = SharedLogStoreReader(dsn)
    return _reader_instance