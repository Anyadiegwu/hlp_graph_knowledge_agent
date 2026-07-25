"""
Stage 6: True Cosine Semantic Multi-Index CRAG.

  Tier 1 — Redis (hot path):
    Three separate RedisVL `SearchIndex` objects — one each for domains,
    sections, and leaf chunks — every one declared with
    `distance_metric: "cosine"`. Retrieval issues a real `VectorQuery`
    (HNSW/FLAT ANN search) instead of iterating in Python.

  Tier 2 — Supabase / pgvector (durable path):
    A single `crag_kb_vectors` table with a `vector(768)` column and a
    cosine-ops ANN index (`vector_cosine_ops`), queried via the `<=>`
    cosine-distance operator. Used when Redis is unavailable, and as the
    system of record for rebuilding the Redis tier after a flush.

If both vector tiers return nothing above threshold, `server.py` escalates
to the Tavily web fallback — that policy lives in server.py, not here.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

logger = logging.getLogger(__name__)

EMBED_DIM = 768


def _schema_for(bucket: str) -> IndexSchema:
    return IndexSchema.from_dict({
        "index": {
            "name": f"hlp_crag_{bucket}",
            "prefix": f"hlp:crag6:{bucket}",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "item_id", "type": "tag"},
            {"name": "domain", "type": "tag"},
            {"name": "section", "type": "tag"},
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": EMBED_DIM,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
        ],
    })


@dataclass
class ScoredItem:
    bucket: str
    item_id: str
    similarity: float  # 1 - cosine_distance, so higher == more similar
    domain: str = ""
    section: str = ""


class RedisCosineMultiIndex:
    """Hot-path tier: one real cosine-ANN RedisVL index per hierarchy level."""

    BUCKETS = ("domains", "sections", "chunks")

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "")
        self._indexes: dict[str, SearchIndex] = {}
        self._available = False
        if not self.redis_url:
            logger.warning("REDIS_URL not set — Redis CRAG tier disabled, will rely on Supabase tier.")
            return
        try:
            for bucket in self.BUCKETS:
                idx = SearchIndex(_schema_for(bucket), redis_url=self.redis_url)
                idx.create(overwrite=False)
                self._indexes[bucket] = idx
            self._available = True
        except Exception as exc:
            logger.error("Redis CRAG multi-index init failed (%s) — falling back to Supabase-only.", exc)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def upsert(self, bucket: str, item_id: str, vector: list[float], domain: str = "", section: str = "") -> None:
        if not self._available:
            return
        idx = self._indexes[bucket]
        idx.load([
            {
                "item_id": item_id,
                "domain": domain,
                "section": section,
                "vector": np.array(vector, dtype=np.float32).tobytes(),
            }
        ], keys=[f"{idx.schema.index.prefix}:{item_id}"])

    def query(self, bucket: str, query_vector: list[float], top_k: int = 5) -> list[ScoredItem]:
        """Runs a real cosine-distance ANN query via RedisVL's VectorQuery
        and converts the returned distance to a similarity score
        (similarity = 1 - distance), rather than recomputing cosine by
        hand in Python."""
        if not self._available:
            return []
        idx = self._indexes[bucket]
        vq = VectorQuery(
            vector=np.array(query_vector, dtype=np.float32).tobytes(),
            vector_field_name="vector",
            return_fields=["item_id", "domain", "section"],
            num_results=top_k,
        )
        try:
            results = idx.query(vq)
        except Exception as exc:
            logger.error("RedisVL cosine query failed on bucket=%s: %s", bucket, exc)
            return []

        scored: list[ScoredItem] = []
        for r in results:
            distance = float(r.get("vector_distance", 1.0))
            similarity = 1.0 - distance
            scored.append(ScoredItem(
                bucket=bucket,
                item_id=r.get("item_id", ""),
                similarity=similarity,
                domain=r.get("domain", ""),
                section=r.get("section", ""),
            ))
        return scored


class SupabaseCosineStore:
    """Durable-path tier: pgvector table queried with the native `<=>`
    cosine-distance operator (standard cosine space — no custom metric)."""

    TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS crag_kb_vectors (
            id          BIGSERIAL PRIMARY KEY,
            bucket      TEXT NOT NULL,
            item_id     TEXT NOT NULL,
            domain      TEXT,
            section     TEXT,
            embedding   vector(768) NOT NULL,
            UNIQUE (bucket, item_id)
        );
        CREATE INDEX IF NOT EXISTS crag_kb_vectors_cosine_idx
            ON crag_kb_vectors USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("SUPABASE_DB_URL", "")
        self._pool = None

    @property
    def configured(self) -> bool:
        return bool(self.dsn)

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            from pgvector.asyncpg import register_vector
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn, min_size=1, max_size=5,
                init=register_vector, statement_cache_size=0,
            )
            async with self._pool.acquire() as conn:
                await conn.execute(self.TABLE_DDL)
        return self._pool

    async def upsert(self, bucket: str, item_id: str, vector: list[float], domain: str = "", section: str = "") -> None:
        if not self.configured:
            return
        try:
            pool = await self._get_pool()
            await pool.execute(
                """
                INSERT INTO crag_kb_vectors (bucket, item_id, domain, section, embedding)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (bucket, item_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding, domain = EXCLUDED.domain, section = EXCLUDED.section
                """,
                bucket, item_id, domain, section, vector,
            )
        except Exception as exc:
            logger.error("Supabase CRAG upsert failed for %s/%s: %s", bucket, item_id, exc)

    async def query(self, bucket: str, query_vector: list[float], top_k: int = 5) -> list[ScoredItem]:
        """`embedding <=> $1` is pgvector's native cosine-distance operator
        (0 = identical, 2 = opposite) — we convert to a similarity score so
        callers treat both tiers identically."""
        if not self.configured:
            return []
        try:
            pool = await self._get_pool()
            rows = await pool.fetch(
                """
                SELECT item_id, domain, section, (embedding <=> $2) AS distance
                FROM crag_kb_vectors
                WHERE bucket = $1
                ORDER BY embedding <=> $2
                LIMIT $3
                """,
                bucket, query_vector, top_k,
            )
        except Exception as exc:
            logger.error("Supabase CRAG cosine query failed on bucket=%s: %s", bucket, exc)
            return []

        return [
            ScoredItem(
                bucket=bucket,
                item_id=row["item_id"],
                similarity=1.0 - float(row["distance"]),
                domain=row["domain"] or "",
                section=row["section"] or "",
            )
            for row in rows
        ]


class CragMultiIndex:
    """Facade the rest of the server talks to: tries the Redis hot tier
    first, and only pays the Supabase round-trip when Redis is unavailable
    or empty for that bucket."""

    def __init__(self) -> None:
        self.redis_tier = RedisCosineMultiIndex()
        self.supabase_tier = SupabaseCosineStore()

    async def upsert_everywhere(self, bucket: str, item_id: str, vector: list[float], domain: str = "", section: str = "") -> None:
        self.redis_tier.upsert(bucket, item_id, vector, domain, section)
        await self.supabase_tier.upsert(bucket, item_id, vector, domain, section)

    async def cosine_query(self, bucket: str, query_vector: list[float], top_k: int = 5) -> list[ScoredItem]:
        if self.redis_tier.available:
            results = self.redis_tier.query(bucket, query_vector, top_k)
            if results:
                return results
            logger.info("Redis CRAG tier returned 0 results for bucket=%s — trying Supabase tier.", bucket)
        return await self.supabase_tier.query(bucket, query_vector, top_k)
