from __future__ import annotations

import json
import time
import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Optional
import operator
import asyncio
from dotenv import find_dotenv
import hashlib
import redis as redis_client_lib
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT        = Path(__file__).resolve().parents[3]
ANALYSIS_LOG_FILE = _REPO_ROOT / "analysis_agent.log"
LOG_FORMAT        = "[%(asctime)s] [ANALYSIS_AGENT] [%(levelname)s] %(message)s"
DATE_FORMAT       = "%Y-%m-%d %H:%M:%S"

_handler_console = logging.StreamHandler(sys.stdout)
_handler_file    = logging.FileHandler(ANALYSIS_LOG_FILE, encoding="utf-8")
for _h in (_handler_console, _handler_file):
    _h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

agent_logger = logging.getLogger("analysis_dashboard.agent")
agent_logger.setLevel(logging.DEBUG)
agent_logger.propagate = False
if not agent_logger.handlers:
    agent_logger.addHandler(_handler_console)
    agent_logger.addHandler(_handler_file)

agent_logger.info("Analysis Agent log: %s", ANALYSIS_LOG_FILE)


class AnalysisSettings(BaseSettings):
    groq_api_key:      SecretStr | None = None
    groq_model_name:   str              = "llama-3.3-70b-versatile"
    gemini_api_key:    SecretStr | None = None
    gemini_model_name: str              = "gemini-2.5-flash"
    model_temperature: float            = 0.0
    neo4j_uri:         str              = ""
    neo4j_username:    str              = "neo4j"
    neo4j_password:    SecretStr | None = None
    redis_url:         SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = AnalysisSettings()

_tier2_redis: "redis_client_lib.Redis | None" = None

def _tier2_cache_client():
    global _tier2_redis
    if _tier2_redis is None:
        if not settings.redis_url:
            return None
        _tier2_redis = redis_client_lib.from_url(settings.redis_url.get_secret_value())
    return _tier2_redis

def _tier2_get_or_compute(cache_key: str, compute_fn, ttl_seconds: int = 3600) -> str:
    """
    Exact-match cache for Tier 2 (Neo4j subgraphs + SHAP/LIME results).
    Returns a JSON string, either from Redis or freshly computed via compute_fn().
    """
    client = _tier2_cache_client()
    full_key = f"hlp:cache:tier2:{cache_key}"

    if client is not None:
        start = time.perf_counter()
        cached = client.get(full_key)
        if cached is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            client.incr("hlp:cache:tier2:hits")
            client.incrbyfloat("hlp:cache:tier2:latency_sum_hit", elapsed_ms)
            client.incr("hlp:cache:tier2:latency_count_hit")
            agent_logger.info("Tier 2 cache HIT: %s", cache_key)
            return cached.decode("utf-8")
        client.incr("hlp:cache:tier2:misses")

    start = time.perf_counter()
    result = compute_fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if client is not None:
        client.setex(full_key, ttl_seconds, result)
        client.incrbyfloat("hlp:cache:tier2:latency_sum_miss", elapsed_ms)
        client.incr("hlp:cache:tier2:latency_count_miss")
        
    return result

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
    
def _get_log_store():
    from analysis_dashboard.store_reader import get_shared_log_store
    return get_shared_log_store()


def _get_graph_client():
    from analysis_dashboard.graph_client import Neo4jGraphClient
    if not settings.neo4j_uri:
        agent_logger.warning("NEO4J_URI not set — graph operations will be no-ops.")
        return None
    client = Neo4jGraphClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=(
            settings.neo4j_password.get_secret_value()
            if settings.neo4j_password else ""
        ),
    )
    connected = client.connect()
    if not connected:
        agent_logger.warning("Neo4j connection failed — graph tools will report errors.")
    return client


_graph_client_cache = None
_model_cache: BaseChatModel | None = None


def _graph():
    global _graph_client_cache
    if _graph_client_cache is None:
        _graph_client_cache = _get_graph_client()
    return _graph_client_cache


def _model() -> BaseChatModel:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _model_cache = ChatGoogleGenerativeAI(
                model=settings.gemini_model_name,
                temperature=settings.model_temperature,
                google_api_key=settings.gemini_api_key.get_secret_value(),
            )
            return _model_cache
        except Exception as exc:
            agent_logger.warning("Gemini init failed (%s) — trying Groq.", exc)

    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        _model_cache = ChatGroq(
            model=settings.groq_model_name,
            temperature=settings.model_temperature,
            api_key=settings.groq_api_key,
        )
        return _model_cache

    raise RuntimeError("No LLM API key configured. Set GEMINI_API_KEY or GROQ_API_KEY.")


@tool
def semantic_log_search(
    query: str,
    k: int = 10,
    namespace_prefix: Optional[str] = None,
    interaction_type: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Perform a semantic vector similarity search over the HLP log store.

    Use this tool to:
    - Find log entries related to specific errors, tool calls, or agent reasoning steps.
    - Discover anomalies in past execution traces.
    - Retrieve all logs from a specific MCP interaction type.

    Parameters
    ----------
    query            : Natural language search query (e.g. "CRAG pipeline failures")
    k                : Number of top results to return (default 10)
    namespace_prefix : Filter by namespace prefix (e.g. "logs.mcp.server")
    interaction_type : Filter by type: tool_invocation, sampling_request, error, etc.
    session_id       : Filter to a specific session UUID

    Returns
    -------
    JSON string with top-k matching log entries including scores.
    """
    agent_logger.info("semantic_log_search: query='%s' k=%d", query[:80], k)
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available. Check LOG_DB_PATH."})

    results = _run_async(store.search(
        query=query,
        k=k,
        namespace_prefix=namespace_prefix,
        session_id=session_id,
        interaction_type=interaction_type,
    ))
    clean = []
    for r in results:
        r.pop("embedding_json", None)
        r.pop("metadata_json", None)
        clean.append(r)

    agent_logger.info("semantic_log_search returned %d results.", len(clean))
    return json.dumps(clean, indent=2, default=str)


@tool
def get_log_statistics() -> str:
    """
    Retrieve aggregate statistics from the HLP log store.

    Returns counts by interaction type, namespace distribution,
    average latency, error count, per-tool latency stats,
    and list of distinct session IDs.

    Use this tool first to understand the overall health and volume
    of the system before drilling into specific traces.
    """
    agent_logger.info("get_log_statistics called.")
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    stats    = _run_async(store.get_stats())
    sessions = _run_async(store.get_sessions())
    stats["session_ids"] = sessions
    agent_logger.info(
        "Stats: total=%d sessions=%d errors=%d",
        stats.get("total_entries", 0),
        len(sessions),
        stats.get("error_count", 0),
    )
    return json.dumps(stats, indent=2, default=str)


@tool
def sync_graph_to_neo4j(session_id: Optional[str] = None) -> str:
    """
    Project HLP interaction logs into the Neo4j Aura DB knowledge graph.

    Extracts relationships from log metadata and writes:
      • (:Session) nodes for each unique session
      • (:AgentAction) nodes for reasoning and final-answer steps
      • (:MCPServerCall) nodes for tool invocations, resource reads, and sampling
      • [:TRIGGERED] edges from Session → AgentAction
      • [:ROUTED_TO] edges from AgentAction → MCPServerCall
      • [:DEPENDS_ON] edges between chained MCP calls (sampling chains)

    Parameters
    ----------
    session_id : If provided, sync only that session. Otherwise syncs all.

    Returns
    -------
    JSON commit summary showing nodes and edges written.
    """
    agent_logger.info("sync_graph_to_neo4j called. session_id=%s", session_id)
    graph = _graph()
    if graph is None:
        return json.dumps({
            "error": "Neo4j not configured. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        })

    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    if session_id:
        entries = _run_async(store.get_by_session(session_id))
    else:
        entries = _run_async(store.get_all(limit=5000))

    from analysis_dashboard.graph_client import project_logs_to_graph
    result = project_logs_to_graph(entries, graph)

    agent_logger.info(
        "Graph sync complete: %d commits, %d errors, %d sessions processed.",
        result["total_commits"],
        len(result["errors"]),
        result["sessions_processed"],
    )
    return json.dumps(result, indent=2, default=str)


@tool
def latency_trend_chart(
    window: int = 5,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a latency trend chart with per-tool moving averages.

    Creates two panels:
      1. Scatter plot + moving average line per tool
      2. Box plot of latency distribution per tool

    Parameters
    ----------
    window    : Rolling window size for the moving average (default 5)
    save_path : If provided, saves the chart PNG to this file path.

    Returns
    -------
    JSON with: summary (text stats), chart_b64 (base64 PNG), saved_path.
    """
    agent_logger.info("latency_trend_chart: window=%d save_path=%s", window, save_path)
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    entries = _run_async(store.get_all(limit=2000))
    from analysis_dashboard.analytics import compute_latency_trend
    result = compute_latency_trend(entries, window=window, save_path=save_path)
    agent_logger.info("latency_trend_chart: %s", result.get("summary", "")[:120])
    return json.dumps({k: v for k, v in result.items() if k != "chart_b64"}, indent=2)


@tool
def token_metrics_chart(save_path: Optional[str] = None) -> str:
    """
    Generate a token consumption analysis chart.

    Creates two panels:
      1. Bar chart of total tokens by interaction type
      2. Cumulative token consumption over time

    Parameters
    ----------
    save_path : Optional path to save the PNG chart to disk.

    Returns
    -------
    JSON with: summary, total_tokens, by_type breakdown, saved_path.
    """
    agent_logger.info("token_metrics_chart called. save_path=%s", save_path)
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    entries = _run_async(store.get_all(limit=2000))
    from analysis_dashboard.analytics import compute_token_metrics
    result = compute_token_metrics(entries, save_path=save_path)
    agent_logger.info("token_metrics_chart: %s", result.get("summary", "")[:120])
    return json.dumps({k: v for k, v in result.items() if k != "chart_b64"}, indent=2)


@tool
def error_frequency_chart(
    window_minutes: int = 5,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate an error frequency analysis chart.

    Creates two panels:
      1. Error vs normal events timeline
      2. Event count distribution by interaction type

    Parameters
    ----------
    window_minutes : Time window for rolling error count (default 5)
    save_path      : Optional path to save the chart PNG.

    Returns
    -------
    JSON with: summary, total_errors, total_events, error_rate, saved_path.
    """
    agent_logger.info("error_frequency_chart called. window=%d", window_minutes)
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    entries = _run_async(store.get_all(limit=2000))
    from analysis_dashboard.analytics import compute_error_frequency
    result = compute_error_frequency(entries, window_minutes=window_minutes, save_path=save_path)
    agent_logger.info("error_frequency_chart: %s", result.get("summary", "")[:120])
    return json.dumps({k: v for k, v in result.items() if k != "chart_b64"}, indent=2)


@tool
def full_dashboard_chart(save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive 4-panel system health dashboard chart.

    Panels:
      1. Event distribution pie chart
      2. Latency over time with moving average
      3. Cumulative token consumption
      4. Error rate by namespace (horizontal bar)

    Returns
    -------
    JSON with: summary, saved_path.
    """
    agent_logger.info("full_dashboard_chart called. save_path=%s", save_path)
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    entries = _run_async(store.get_all(limit=2000))
    from analysis_dashboard.analytics import generate_dashboard_chart
    result = generate_dashboard_chart(entries, save_path=save_path)
    agent_logger.info("full_dashboard_chart: %s", result.get("summary", "")[:120])
    return json.dumps({k: v for k, v in result.items() if k != "chart_b64"}, indent=2)


@tool
def fetch_neo4j_subgraph(
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    hop_depth: int = 2,
) -> str:
    """
    Extract a causal relationship subgraph from Neo4j Aura DB.

    Runs Cypher path queries to retrieve the interaction graph surrounding
    a target session or specific trace (log entry) ID. Returns structured
    relationship data used as context injection for the XAI explainability
    engine.

    The query traverses:
      (:Session) -[:TRIGGERED]-> (:AgentAction) -[:ROUTED_TO]-> (:MCPServerCall)

    Parameters
    ----------
    session_id : UUID of the session to extract subgraph for.
    trace_id   : Specific log entry ID to anchor the subgraph on.
                 If both are provided, session_id takes priority.
    hop_depth  : Maximum relationship hops from the anchor node (default 2).

    Returns
    -------
    JSON containing:
      - nodes: list of graph nodes with their properties
      - relationships: list of edges with type and endpoint IDs
      - cypher_used: the exact Cypher query that was executed
      - context_summary: human-readable subgraph description for prompt injection
    """
    agent_logger.info(
        "fetch_neo4j_subgraph: session_id=%s trace_id=%s hop_depth=%d",
        session_id, trace_id, hop_depth,
    )

    graph = _graph()
    if graph is None:
        return json.dumps({
            "error": "Neo4j not configured. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        })

    if not graph._connected:
        return json.dumps({"error": "Neo4j driver not connected."})

    safe_hop_depth = max(1, min(int(hop_depth), 5))
    cache_key = f"subgraph:{session_id or ''}:{trace_id or ''}:{safe_hop_depth}"
    def _compute() -> str:
        if session_id:
            cypher = (
                "MATCH (s:Session {session_id: $anchor_id}) "
                "OPTIONAL MATCH path1 = (s)-[:TRIGGERED]->(a:AgentAction) "
                "OPTIONAL MATCH path2 = (a)-[:ROUTED_TO]->(m:MCPServerCall) "
                f"OPTIONAL MATCH path3 = (m)-[:DEPENDS_ON*1..{safe_hop_depth}]->(m2:MCPServerCall) "
                "RETURN s, collect(DISTINCT a) AS agent_actions, "
                "collect(DISTINCT m) AS mcp_calls, collect(DISTINCT m2) AS chained_calls"
            )
            anchor_id = session_id
        elif trace_id:
            cypher = (
                "MATCH (anchor) WHERE anchor.action_id = $anchor_id OR anchor.call_id = $anchor_id "
                "OPTIONAL MATCH path1 = (s:Session)-[:TRIGGERED]->(anchor) "
                "OPTIONAL MATCH path2 = (anchor)-[:ROUTED_TO]->(m:MCPServerCall) "
                f"OPTIONAL MATCH path3 = (anchor)-[:DEPENDS_ON*1..{safe_hop_depth}]->(m2:MCPServerCall) "
                "RETURN s, anchor, collect(DISTINCT m) AS mcp_calls, "
                "collect(DISTINCT m2) AS chained_calls"
            )
            anchor_id = trace_id
        else:
            cypher = (
                "MATCH (s:Session) "
                "WITH s ORDER BY s.created_at DESC LIMIT 5 "
                "OPTIONAL MATCH (s)-[:TRIGGERED]->(a:AgentAction) "
                "OPTIONAL MATCH (a)-[:ROUTED_TO]->(m:MCPServerCall) "
                "RETURN s, collect(DISTINCT a) AS agent_actions, collect(DISTINCT m) AS mcp_calls"
            )
            anchor_id = None

        try:
            with graph._driver.session() as neo_session:
                result = neo_session.run(cypher, anchor_id=anchor_id)
                records = [dict(r) for r in result]
        except Exception as exc:
            agent_logger.error("Cypher query failed: %s", exc)
            return json.dumps({"error": f"Cypher execution failed: {exc}", "cypher_used": cypher})

        def _node_to_dict(node) -> dict:
            if node is None:
                return {}
            try:
                return dict(node.items())
            except Exception:
                return {"_raw": str(node)}

        nodes: list[dict] = []
        relationships: list[dict] = []

        for record in records:
            for key, value in record.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    for item in value:
                        if item is not None:
                            nodes.append({"label": key, **_node_to_dict(item)})
                else:
                    node_dict = _node_to_dict(value)
                    if node_dict:
                        nodes.append({"label": key, **node_dict})

        seen = set()
        unique_nodes = []
        for n in nodes:
            uid = n.get("session_id") or n.get("action_id") or n.get("call_id") or str(n)
            if uid not in seen:
                seen.add(uid)
                unique_nodes.append(n)

        session_nodes   = [n for n in unique_nodes if "session_id" in n and "action_type" not in n]
        action_nodes    = [n for n in unique_nodes if "action_type" in n]
        call_nodes      = [n for n in unique_nodes if "call_id" in n]

        context_lines = [
            f"Neo4j Subgraph Context (anchor={anchor_id or 'recent sessions'}, hop_depth={safe_hop_depth}):",
            f"  Sessions found    : {len(session_nodes)}",
            f"  AgentAction nodes : {len(action_nodes)}",
            f"  MCPServerCall nodes: {len(call_nodes)}",
        ]
        for s in session_nodes[:3]:
            context_lines.append(
                f"  Session {s.get('session_id','?')[:8]}... "
                f"model={s.get('primary_model','?')} "
                f"entries={s.get('total_entries','?')}"
            )
        for a in action_nodes[:5]:
            context_lines.append(
                f"  AgentAction type={a.get('action_type','?')} "
                f"ns={a.get('namespace','?')[:40]} "
                f"latency={a.get('latency_ms','?')}ms"
            )
        for m in call_nodes[:5]:
            context_lines.append(
                f"  MCPCall tool={m.get('tool_name','?')} "
                f"type={m.get('interaction_type','?')} "
                f"latency={m.get('latency_ms','?')}ms"
            )

        context_summary = "\n".join(context_lines)
        agent_logger.info(
            "fetch_neo4j_subgraph: %d nodes extracted.", len(unique_nodes)
        )

        return json.dumps({
            "nodes": unique_nodes,
            "relationships": relationships,
            "cypher_used": cypher.strip(),
            "context_summary": context_summary,
            "node_counts": {
                "sessions": len(session_nodes),
                "agent_actions": len(action_nodes),
                "mcp_calls": len(call_nodes),
            },
        }, indent=2, default=str)

    return _tier2_get_or_compute(cache_key, _compute)

@tool
def run_explainability_audit(
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    top_k_logs: int = 20,
) -> str:
    """
    Run a full explainability audit on a session or trace execution path.

    This tool orchestrates:
      1. Graph Context Hydration  — Cypher subgraph from Neo4j as context
      2. Proxy LIME               — token-importance scores for text logs
      3. Proxy SHAP               — feature-contribution scores for numeric signatures
      4. Report Export            — saves explainability_audit_report.json

    Parameters
    ----------
    session_id  : UUID of the session to audit. If None, uses most recent session.
    trace_id    : Specific log entry ID to anchor the audit on.
    top_k_logs  : Number of log entries to include in the analysis (default 20).

    Returns
    -------
    JSON audit summary with:
      - lime_results: token importance arrays per analysed log entry
      - shap_results: feature contribution scores per numeric signature
      - graph_context: subgraph summary used as context
      - report_path: path to the saved explainability_audit_report.json
    """
    agent_logger.info(
        "run_explainability_audit: session_id=%s trace_id=%s top_k=%d",
        session_id, trace_id, top_k_logs,
    )

    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    resolved_session = session_id
    if not resolved_session and not trace_id:
        sessions = _run_async(store.get_sessions())
        if not sessions:
            return json.dumps({"error": "No sessions found in log store."})
        resolved_session = sessions[-1]
        agent_logger.info("No session_id provided — using most recent: %s", resolved_session[:8])

    if resolved_session:
        raw_entries = _run_async(store.get_by_session(resolved_session))
    else:
        raw_entries = _run_async(store.get_all(limit=top_k_logs))

    if not raw_entries:
        return json.dumps({"error": f"No log entries found for session={resolved_session}."})

    target_entries = [
        e for e in raw_entries
        if e.get("content") and len(str(e.get("content", ""))) > 20
    ][:top_k_logs]

    agent_logger.info("Audit targeting %d log entries.", len(target_entries))

    graph_context_raw = fetch_neo4j_subgraph.invoke({
        "session_id": resolved_session,
        "trace_id": trace_id,
        "hop_depth": 2,
    })
    try:
        graph_context = json.loads(graph_context_raw)
        context_summary = graph_context.get("context_summary", "No graph context available.")
    except Exception:
        graph_context = {}
        context_summary = "Graph context unavailable."

    text_entries: list = []
    structured_entries: list = []

    entries_fingerprint = hashlib.sha256(
        json.dumps(target_entries, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    xai_cache_key = f"xai:{entries_fingerprint}"

    def _compute_xai() -> str:
        text_entries_local: list = []
        structured_entries_local: list = []
        lime_results_local: dict = {}
        shap_results_local: dict = {}

        try:
            from analysis_dashboard.xai_engine import (
                run_proxy_lime,
                run_proxy_shap,
            )

            text_entries_local = [
                e for e in target_entries
                if e.get("mcp_interaction_type") in (
                    "error", "agent_reasoning", "tool_invocation", "sampling_request"
                )
            ]
            lime_results_local = run_proxy_lime(
                log_entries=text_entries_local,
                graph_context=context_summary,
            )

            structured_entries_local = [
                e for e in target_entries
                if e.get("latency_ms") is not None or e.get("token_count") is not None
            ]
            shap_results_local = run_proxy_shap(
                log_entries=structured_entries_local,
                graph_context=context_summary,
            )

        except ImportError as imp_err:
            agent_logger.error("xai_engine import failed: %s", imp_err)
            lime_results_local = {"error": f"xai_engine not available: {imp_err}"}
            shap_results_local = {"error": f"xai_engine not available: {imp_err}"}
        except Exception as xai_err:
            agent_logger.error("XAI computation failed: %s", xai_err)
            lime_results_local = {"error": str(xai_err)}
            shap_results_local  = {"error": str(xai_err)}

        return json.dumps({
            "text_entries_count": len(text_entries_local),
            "structured_entries_count": len(structured_entries_local),
            "lime_results": lime_results_local,
            "shap_results": shap_results_local,
            }, default=str)
    xai_payload = json.loads(_tier2_get_or_compute(xai_cache_key, _compute_xai))
    lime_results = xai_payload["lime_results"]
    shap_results = xai_payload["shap_results"]
    text_entries = [None] * xai_payload["text_entries_count"]
    structured_entries = [None] * xai_payload["structured_entries_count"]
    
    report = {
        "audit_metadata": {
            "session_id": resolved_session or trace_id,
            "entries_analysed": len(target_entries),
            "text_entries_for_lime": len(text_entries),
            "structured_entries_for_shap": len(structured_entries),
        },
        "graph_context": graph_context,
        "lime_results": lime_results,
        "shap_results": shap_results,
    }

    report_path = _REPO_ROOT / "explainability_audit_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        agent_logger.info("Audit report saved: %s", report_path)
    except Exception as save_err:
        agent_logger.error("Could not save audit report: %s", save_err)
        report_path = None

    summary = {
        "status": "success",
        "session_audited": resolved_session or trace_id,
        "entries_analysed": len(target_entries),
        "lime_tokens_scored": (
            sum(len(r.get("token_importances", [])) for r in lime_results.get("entries", []))
            if isinstance(lime_results, dict) and "entries" in lime_results else 0
        ),
        "shap_features_scored": (
            len(shap_results.get("feature_contributions", {}))
            if isinstance(shap_results, dict) else 0
        ),
        "graph_context_nodes": graph_context.get("node_counts", {}),
        "report_path": str(report_path) if report_path else None,
    }
    agent_logger.info("Explainability audit complete: %s", json.dumps(summary))
    return json.dumps(summary, indent=2, default=str)


ANALYSIS_TOOLS = [
    semantic_log_search,
    get_log_statistics,
    sync_graph_to_neo4j,
    latency_trend_chart,
    token_metrics_chart,
    error_frequency_chart,
    full_dashboard_chart,
    fetch_neo4j_subgraph,
    run_explainability_audit,
]


class AnalysisState(BaseModel):
    messages:       Annotated[list, operator.add] = []
    user_query:     str                           = ""
    planned_route:  list[str]                     = []
    stats_result:   str                           = ""
    search_results: str                           = ""
    xai_result:     str                           = ""
    chart_result:   str                           = ""
    final_answer:   str                           = ""
    routing_log:    Annotated[list, operator.add] = []


def _is_xai_query(query: str) -> bool:
    keywords = [
        "explain", "xai", "lime", "shap", "audit", "explainability",
        "feature importance", "token importance", "why did", "fault",
        "fallback", "error trace", "resilience",
    ]
    q = query.lower()
    return any(kw in q for kw in keywords)


def _is_chart_query(query: str) -> bool:
    keywords = [
        "chart", "plot", "graph", "visualise", "visualize",
        "latency", "token", "dashboard", "trend",
    ]
    q = query.lower()
    return any(kw in q for kw in keywords)


def _is_search_query(query: str) -> bool:
    keywords = ["search", "find", "trace", "session", "error", "anomaly"]
    q = query.lower()
    return any(kw in q for kw in keywords)


def _plan_route(query: str) -> list[str]:
    route = ["stats_node"]
    if _is_search_query(query):
        route.append("search_node")
    if _is_xai_query(query):
        route.append("xai_node")
    if _is_chart_query(query):
        route.append("chart_node")
    route.append("synthesize_node")
    return route


def initial_ingest_node(state: AnalysisState) -> Command:
    user_query = state.user_query
    if not user_query:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

    planned_route = _plan_route(user_query)
    next_node = planned_route[0]

    agent_logger.info("[initial_ingest_node] query='%s' planned_route=%s", user_query[:80], planned_route)

    log_entry = f"initial_ingest_node → {next_node}"
    agent_logger.debug("[ROUTE] %s", log_entry)

    return Command(
        goto=next_node,
        update={
            "user_query":    user_query,
            "planned_route": planned_route,
            "routing_log":   [log_entry],
        },
    )


def _next_from_route(state: AnalysisState, current: str) -> str:
    try:
        idx = state.planned_route.index(current)
        return state.planned_route[idx + 1]
    except (ValueError, IndexError):
        return "synthesize_node"


def stats_node(state: AnalysisState) -> Command:
    agent_logger.info("[stats_node] Fetching log statistics.")

    stats_raw = get_log_statistics.invoke({})
    agent_logger.info("[stats_node] Stats fetched (%d chars).", len(stats_raw))

    next_node = _next_from_route(state, "stats_node")
    log_entry = f"stats_node → {next_node}"
    agent_logger.debug("[ROUTE] %s", log_entry)

    return Command(
        goto=next_node,
        update={
            "stats_result": stats_raw,
            "routing_log":  [log_entry],
        },
    )


def search_node(state: AnalysisState) -> Command:
    agent_logger.info("[search_node] Running semantic log search.")

    results_raw = semantic_log_search.invoke({
        "query": state.user_query,
        "k": 15,
    })
    agent_logger.info("[search_node] Search returned (%d chars).", len(results_raw))

    next_node = _next_from_route(state, "search_node")
    log_entry = f"search_node → {next_node}"
    agent_logger.debug("[ROUTE] %s", log_entry)

    return Command(
        goto=next_node,
        update={
            "search_results": results_raw,
            "routing_log":    [log_entry],
        },
    )


def xai_node(state: AnalysisState) -> Command:
    agent_logger.info("[xai_node] Running explainability audit.")

    session_id = None
    if state.stats_result:
        try:
            stats_data = json.loads(state.stats_result)
            sessions = stats_data.get("session_ids", [])
            if sessions:
                session_id = sessions[-1]
        except Exception:
            pass

    xai_raw = run_explainability_audit.invoke({
        "session_id": session_id,
        "top_k_logs": 20,
    })
    agent_logger.info("[xai_node] XAI audit returned (%d chars).", len(xai_raw))

    next_node = _next_from_route(state, "xai_node")
    log_entry = f"xai_node → {next_node}"
    agent_logger.debug("[ROUTE] %s", log_entry)

    return Command(
        goto=next_node,
        update={
            "xai_result":  xai_raw,
            "routing_log": [log_entry],
        },
    )


def chart_node(state: AnalysisState) -> Command:
    agent_logger.info("[chart_node] Generating chart.")

    query = state.user_query.lower()
    if "latency" in query:
        chart_raw = latency_trend_chart.invoke({"window": 5})
    elif "token" in query:
        chart_raw = token_metrics_chart.invoke({})
    elif "error" in query or "frequency" in query:
        chart_raw = error_frequency_chart.invoke({"window_minutes": 5})
    else:
        chart_raw = full_dashboard_chart.invoke({})

    agent_logger.info("[chart_node] Chart generated (%d chars).", len(chart_raw))

    next_node = _next_from_route(state, "chart_node")
    log_entry = f"chart_node → {next_node}"
    agent_logger.debug("[ROUTE] %s", log_entry)

    return Command(
        goto=next_node,
        update={
            "chart_result": chart_raw,
            "routing_log":  [log_entry],
        },
    )


def synthesize_node(state: AnalysisState) -> Command:
    agent_logger.info("[synthesize_node] Synthesising final answer.")

    llm = _model()

    _STATS_LIMIT  = 1500
    _SEARCH_LIMIT = 2000
    _XAI_LIMIT    = 2000
    _CHART_LIMIT  = 800

    context_parts = [f"User query: {state.user_query}\n"]

    if state.stats_result:
        truncated = len(state.stats_result) > _STATS_LIMIT
        context_parts.append(
            f"Log Statistics:\n{state.stats_result[:_STATS_LIMIT]}"
            + (" [truncated]\n" if truncated else "\n")
        )

    if state.search_results:
        truncated = len(state.search_results) > _SEARCH_LIMIT
        context_parts.append(
            f"Semantic Search Results:\n{state.search_results[:_SEARCH_LIMIT]}"
            + (" [truncated]\n" if truncated else "\n")
        )

    if state.xai_result:
        truncated = len(state.xai_result) > _XAI_LIMIT
        context_parts.append(
            f"Explainability Audit Results:\n{state.xai_result[:_XAI_LIMIT]}"
            + (" [truncated]\n" if truncated else "\n")
        )

    if state.chart_result:
        truncated = len(state.chart_result) > _CHART_LIMIT
        context_parts.append(
            f"Chart Generation Results:\n{state.chart_result[:_CHART_LIMIT]}"
            + (" [truncated]\n" if truncated else "\n")
        )

    route_summary = " → ".join(
        entry.split(" → ")[0] for entry in state.routing_log
    )
    context_parts.append(f"\nExecution path: {route_summary} → synthesize_node → END\n")

    synthesis_prompt = (
        "You are the HLP Log Analysis Agent. Based on the diagnostic data collected "
        "below, produce a clear, structured analysis report answering the user's query.\n\n"
        "Include:\n"
        "1. A direct answer to the user's question\n"
        "2. Key findings from the data (errors, latencies, anomalies)\n"
        "3. If XAI data is present: explain what LIME/SHAP results indicate\n"
        "4. Recommendations or next steps\n\n"
        + "\n".join(context_parts)
    )

    try:
        response = llm.invoke([HumanMessage(content=synthesis_prompt)])
        final_answer = (
            response.content if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception as exc:
        agent_logger.error("[synthesize_node] LLM synthesis failed: %s", exc)
        final_answer = (
            f"Synthesis failed: {exc}\n\n"
            f"Raw data collected:\n"
            f"Stats: {state.stats_result[:500]}\n"
            f"XAI: {state.xai_result[:500]}"
        )

    agent_logger.info("[synthesize_node] Final answer (%d chars).", len(final_answer))

    log_entry = "synthesize_node → END"
    agent_logger.debug("[ROUTE] %s", log_entry)

    return Command(
        goto=END,
        update={
            "final_answer": final_answer,
            "routing_log":  [log_entry],
            "messages":     state.messages + [AIMessage(content=final_answer)],
        },
    )

def _compile_graph():
    builder = StateGraph(AnalysisState)
    builder.add_node("initial_ingest_node", initial_ingest_node)
    builder.add_node("stats_node",          stats_node)
    builder.add_node("search_node",         search_node)
    builder.add_node("xai_node",            xai_node)
    builder.add_node("chart_node",          chart_node)
    builder.add_node("synthesize_node",     synthesize_node)

    builder.add_edge(START, "initial_ingest_node")
    graph = builder.compile()
    agent_logger.info(
        "Edgeless StateGraph compiled. Nodes: %s. "
        "Only hardcoded edge: START → initial_ingest_node.",
        list(builder.nodes.keys()),
    )
    return graph


def build_analysis_agent():
    graph = _compile_graph()
    agent_logger.info("Analysis agent (edgeless LangGraph) ready.")
    return graph


async def run_analysis_query(
    agent,
    user_query: str,
    step_callback=None,
) -> tuple[str, list[dict]]:
    agent_logger.info("Analysis query: %s", user_query[:120])

    initial_state = AnalysisState(
        messages=[HumanMessage(content=user_query)],
        user_query=user_query,
    )

    steps: list[dict] = []
    final_answer = ""

    async for event in agent.astream(
        initial_state.model_dump(),
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            if node_name == "__end__":
                continue

            step = {
                "type":    "node_execution",
                "node":    node_name,
                "content": str(node_output)[:400],
                "routing": node_output.get("routing_log", []),
            }
            steps.append(step)

            if step_callback:
                await step_callback(step)

            if node_name == "synthesize_node" and node_output.get("final_answer"):
                final_answer = node_output["final_answer"]

            agent_logger.debug(
                "[STREAM] node=%s routing=%s",
                node_name,
                node_output.get("routing_log", []),
            )

    if not final_answer:
        agent_logger.warning("Streaming did not yield final_answer; extracting from steps.")
        for step in reversed(steps):
            if step["node"] == "synthesize_node" and "final_answer" in step.get("content", ""):
                break
        agent_logger.error("final_answer not captured from stream and fallback ainvoke skipped.")

    agent_logger.info(
        "Analysis query complete. steps=%d final_answer_len=%d",
        len(steps), len(final_answer),
    )
    return final_answer, steps


def main() -> None:
    import asyncio

    async def _run():
        agent = build_analysis_agent()
        queries = [
            "Give me an overview of the system health: error rate, latency stats, and session count.",
            "Search for any sampling_request logs and report what you find.",
            "Sync all sessions to the Neo4j graph and report what was committed.",
            "Generate a full dashboard chart and save it to disk.",
            "Run an explainability audit on the most recent session and show LIME and SHAP results.",
            "Find any fault injection or self-healing events in the logs and explain what happened.",
        ]
        for q in queries:
            agent_logger.info("=" * 60)
            agent_logger.info("Query: %s", q)
            answer, steps = await run_analysis_query(agent, q)
            agent_logger.info("Steps taken: %d", len(steps))
            agent_logger.info("Answer: %s", answer[:300])
            print(f"\n{'='*60}\nQuery: {q}\n\nAnswer:\n{answer}\n")

    asyncio.run(_run())


if __name__ == "__main__":
    main()