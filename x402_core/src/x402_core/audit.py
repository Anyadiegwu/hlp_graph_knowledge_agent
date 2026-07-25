"""
Shared audit trail for the x402 payment lifecycle — both the MCP server's
paywall (mcp_server/paywall.py) and the agent client's reciprocal sampling
wall (agent_client/client.py) append to this SAME Redis list, so the
FinOps dashboard can render one chronological stream covering challenges
issued, payments verified, and settlements confirmed on both sides of the
protocol, regardless of which process wrote them.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))
logger = logging.getLogger("x402_core.audit")

AUDIT_KEY = "hlp:finops:audit"
AUDIT_MAX_ENTRIES = 500

_redis_client = None
_redis_tried = False


def _get_redis():
    global _redis_client, _redis_tried
    if _redis_tried:
        return _redis_client
    _redis_tried = True
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return None
    try:
        import redis as redis_lib
        _redis_client = redis_lib.from_url(redis_url)
    except Exception as exc:
        logger.warning("Audit trail: Redis unavailable (%s) — entries will be log-only.", exc)
        _redis_client = None
    return _redis_client


def append_audit_entry(event: str, **fields: Any) -> dict[str, Any]:
    entry = {"event": event, **fields, "recorded_at": time.time()}
    logger.info("[FINOPS_AUDIT] %s", json.dumps(entry, default=str))
    r = _get_redis()
    if r is not None:
        try:
            r.rpush(AUDIT_KEY, json.dumps(entry, default=str))
            r.ltrim(AUDIT_KEY, -AUDIT_MAX_ENTRIES, -1)
        except Exception as exc:
            logger.warning("Could not append to %s: %s", AUDIT_KEY, exc)
    return entry


def read_audit_trail(limit: int = 200) -> list[dict[str, Any]]:
    r = _get_redis()
    if r is None:
        return []
    try:
        raw_entries = r.lrange(AUDIT_KEY, -limit, -1)
    except Exception as exc:
        logger.warning("Could not read %s: %s", AUDIT_KEY, exc)
        return []
    parsed: list[dict[str, Any]] = []
    for raw in raw_entries:
        try:
            parsed.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    return parsed
