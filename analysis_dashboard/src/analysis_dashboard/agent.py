# analysis_dashboard/src/analysis_dashboard/agent.py
#
# HLP Graph Knowledge Agent — Stage 3 — Independent Log Analysis Agent
#
# Completely isolated from the MCP Client/Server runtime.
# Runs as a separate process via:  uv run --package analysis_dashboard start-analysis-agent
#
# Tools exposed to the agent:
#   1. semantic_log_search    — vector similarity search over HLPLogStore
#   2. get_log_statistics     — aggregate stats from the SQLite store
#   3. sync_graph_to_neo4j    — project all logs into Neo4j Aura DB
#   4. latency_trend_chart    — moving-average latency chart (matplotlib)
#   5. token_metrics_chart    — token consumption chart
#   6. error_frequency_chart  — rolling error frequency chart
#   7. full_dashboard_chart   — comprehensive 4-panel system health chart

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env"))

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─────────────────────────────────────────────
# 1. Logging — dedicated analysis_agent.log
#    Module-level setup so the log file is
#    created whether the agent is invoked via
#    the Streamlit dashboard OR the CLI.
# ─────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_LOG_FILE = _REPO_ROOT / "analysis_agent.log"
LOG_FORMAT  = "[%(asctime)s] [ANALYSIS_AGENT] [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_handler_console = logging.StreamHandler(sys.stdout)
_handler_file    = logging.FileHandler(ANALYSIS_LOG_FILE, encoding="utf-8")
for _h in (_handler_console, _handler_file):
    _h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

agent_logger = logging.getLogger("analysis_dashboard.agent")
agent_logger.setLevel(logging.DEBUG)
agent_logger.propagate = False
# Guard against duplicate handlers if module is reloaded (e.g. Streamlit hot-reload)
if not agent_logger.handlers:
    agent_logger.addHandler(_handler_console)
    agent_logger.addHandler(_handler_file)

agent_logger.info("Analysis Agent log: %s", ANALYSIS_LOG_FILE)


# ─────────────────────────────────────────────
# 2. Settings
# ─────────────────────────────────────────────

class AnalysisSettings(BaseSettings):
    groq_api_key:      SecretStr | None = None
    groq_model_name:   str              = "llama-3.3-70b-versatile"
    gemini_api_key:    SecretStr | None = None
    gemini_model_name: str              = "gemini-2.5-flash"
    model_temperature: float            = 0.0

    neo4j_uri:         str              = ""
    neo4j_username:    str              = "neo4j"
    neo4j_password:    SecretStr | None = None

    log_db_path:       str              = ""
    chart_output_dir:  str              = "./charts"

    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = AnalysisSettings()


# ─────────────────────────────────────────────
# 3. Shared resources (lazy-loaded singletons)
# ─────────────────────────────────────────────

def _get_log_store():
    """Lazy-load the HLPLogStore pointing at the shared DB."""
    from analysis_dashboard.store_reader import get_shared_log_store
    return get_shared_log_store(settings.log_db_path or None)


def _get_graph_client():
    """Lazy-load and connect to the Neo4j graph client."""
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


def _graph() :
    global _graph_client_cache
    if _graph_client_cache is None:
        _graph_client_cache = _get_graph_client()
    return _graph_client_cache


# ─────────────────────────────────────────────
# 4. LangChain @tool definitions
# ─────────────────────────────────────────────

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

    results = store.search(
        query=query,
        k=k,
        namespace_prefix=namespace_prefix,
        session_id=session_id,
        interaction_type=interaction_type,
    )

    # Strip large embedding vectors from output to keep response concise
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

    stats    = store.get_stats()
    sessions = store.get_sessions()
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
        entries = store.get_by_session(session_id)
    else:
        entries = store.get_all(limit=5000)

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
                Otherwise returns base64 only (displayed in Streamlit).

    Returns
    -------
    JSON with: summary (text stats), chart_b64 (base64 PNG), saved_path.
    """
    agent_logger.info("latency_trend_chart: window=%d save_path=%s", window, save_path)
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    entries = store.get_all(limit=2000)
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

    entries = store.get_all(limit=2000)
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

    entries = store.get_all(limit=2000)
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

    The chart is always saved to the CHART_OUTPUT_DIR directory.
    Optionally also saved to save_path if provided.

    Returns
    -------
    JSON with: summary, saved_path.
    """
    agent_logger.info("full_dashboard_chart called. save_path=%s", save_path)
    store = _get_log_store()
    if store is None:
        return json.dumps({"error": "Log store not available."})

    entries = store.get_all(limit=2000)
    from analysis_dashboard.analytics import generate_dashboard_chart
    result = generate_dashboard_chart(entries, save_path=save_path)
    agent_logger.info("full_dashboard_chart: %s", result.get("summary", "")[:120])
    return json.dumps({k: v for k, v in result.items() if k != "chart_b64"}, indent=2)


# ─────────────────────────────────────────────
# 5. Model builder
# ─────────────────────────────────────────────

def _build_analysis_model() -> BaseChatModel:
    if settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.gemini_model_name,
                temperature=settings.model_temperature,
                google_api_key=settings.gemini_api_key.get_secret_value(),
            )
        except Exception as exc:
            agent_logger.warning("Gemini init failed (%s) — trying Groq.", exc)

    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model_name,
            temperature=settings.model_temperature,
            api_key=settings.groq_api_key,
        )

    raise RuntimeError("No LLM API key configured. Set GEMINI_API_KEY or GROQ_API_KEY.")


# ─────────────────────────────────────────────
# 6. Agent construction
# ─────────────────────────────────────────────

ANALYSIS_TOOLS = [
    semantic_log_search,
    get_log_statistics,
    sync_graph_to_neo4j,
    latency_trend_chart,
    token_metrics_chart,
    error_frequency_chart,
    full_dashboard_chart,
]

ANALYSIS_SYSTEM_PROMPT = """You are the HLP Log Analysis Agent — a specialised \
diagnostic AI operating over the HLP distributed MCP system's execution traces.

You have exclusive access to the HLPLogStore (a SQLite vector database containing \
all interaction logs from the agent client and MCP server), the Neo4j Aura DB \
knowledge graph, and a suite of analytics and visualisation tools.

## Your Mission
Diagnose system health, find anomalies, trace execution paths, project causal \
interaction graphs into Neo4j, and produce clear trend charts.

## Available Tools
1. semantic_log_search    — Vector search over logs to find relevant traces
2. get_log_statistics     — Aggregate stats: counts, latency, error rate
3. sync_graph_to_neo4j    — Project session traces into Neo4j (Session/AgentAction/MCPServerCall nodes)
4. latency_trend_chart    — Per-tool latency moving averages and distributions
5. token_metrics_chart    — Token consumption by type and over time
6. error_frequency_chart  — Error rate and distribution across namespaces
7. full_dashboard_chart   — Comprehensive 4-panel system health chart

## Workflow
1. Always start with get_log_statistics to understand current data volume.
2. Use semantic_log_search to investigate specific anomalies or traces.
3. Use sync_graph_to_neo4j to populate the knowledge graph before graph questions.
4. Use chart tools to visualise trends — report the summary and saved_path.
5. Provide clear, structured diagnostic reports in your Final Answer.

## ReAct Format (STRICT)
Thought: your reasoning
Action: tool_name
Action Input: {{"param": "value"}}
Observation: tool result
... (repeat)
Thought: I have enough information.
Final Answer: structured diagnostic report.
"""


def build_analysis_agent():
    """Build the Log Analysis Agent with all diagnostic tools."""
    from langchain.agents import create_agent
    model = _build_analysis_model()
    agent = create_agent(
        model=model,
        tools=ANALYSIS_TOOLS,
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
    )
    agent_logger.info(
        "Analysis agent built. model=%s tools=%d",
        type(model).__name__,
        len(ANALYSIS_TOOLS),
    )
    return agent


# ─────────────────────────────────────────────
# 7. Streaming query runner (used by Streamlit)
# ─────────────────────────────────────────────

async def run_analysis_query(
    agent,
    user_query: str,
    step_callback=None,
) -> tuple[str, list[dict]]:
    """
    Run a query against the analysis agent and return (final_answer, steps).

    Parameters
    ----------
    agent         : Built analysis agent (from build_analysis_agent())
    user_query    : Natural language diagnostic question
    step_callback : Optional async callable(step_dict) called on each reasoning step

    Returns
    -------
    (final_answer: str, steps: list[dict])
    """
    from langchain_core.messages import AIMessage, ToolMessage

    agent_logger.info("Analysis query: %s", user_query[:120])
    steps: list[dict] = []
    final_answer = ""

    async for event in agent.astream({"messages": [HumanMessage(content=user_query)]}):
        if "model" in event:
            for msg in event["model"].get("messages", []):
                if isinstance(msg, AIMessage):
                    # Normalise content — Gemini returns a list of blocks, not a plain string
                    if isinstance(msg.content, list):
                        content_str = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in msg.content
                        )
                    else:
                        content_str = msg.content or ""

                    step = {
                        "type":    "reasoning",
                        "content": content_str[:500],
                        "tools":   [tc["name"] for tc in (msg.tool_calls or [])],
                    }
                    steps.append(step)
                    if step_callback:
                        await step_callback(step)
                    if content_str.strip() and not msg.tool_calls:
                        final_answer = content_str

        elif "tools" in event:
            for msg in event["tools"].get("messages", []):
                if isinstance(msg, ToolMessage):
                    # ToolMessage content may also be a list
                    if isinstance(msg.content, list):
                        tool_content_str = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in msg.content
                        )
                    else:
                        tool_content_str = str(msg.content) if msg.content else ""

                    step = {
                        "type":      "tool_result",
                        "tool_name": msg.name if hasattr(msg, "name") else "tool",
                        "content":   tool_content_str[:800],
                    }
                    steps.append(step)
                    if step_callback:
                        await step_callback(step)

    agent_logger.info("Analysis query complete. steps=%d", len(steps))
    return final_answer, steps


# ─────────────────────────────────────────────
# 8. CLI entry point (standalone run)
# ─────────────────────────────────────────────

def main() -> None:
    """
    Standalone CLI runner for the analysis agent.
    Runs a set of diagnostic queries and exits.
    """
    import asyncio

    async def _run():
        agent = build_analysis_agent()
        queries = [
            "Give me an overview of the system health: error rate, latency stats, and session count.",
            "Search for any sampling_request logs and report what you find.",
            "Sync all sessions to the Neo4j graph and report what was committed.",
            "Generate a full dashboard chart and save it to disk.",
        ]
        for q in queries:
            agent_logger.info("=" * 60)
            agent_logger.info("Query: %s", q)
            answer, steps = await run_analysis_query(agent, q)
            agent_logger.info("Answer: %s", answer[:300])
            print(f"\n{'='*60}\nQuery: {q}\n\nAnswer:\n{answer}\n")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
