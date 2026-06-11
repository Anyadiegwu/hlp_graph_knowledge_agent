# analysis_dashboard/src/analysis_dashboard/app.py
#
# HLP Graph Knowledge Agent — Stage 3 — Streamlit Diagnostic Dashboard
#
# Human-in-the-loop control plane for the HLP distributed MCP system.
#
# Features:
#   • Natural language chat interface → routes to Log Analysis Agent
#   • Step-by-step agent reasoning display
#   • Inline trend charts (latency, tokens, errors, full dashboard)
#   • Neo4j commit notifications
#   • Log store stats overview
#   • Session explorer sidebar

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env"))

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HLP Diagnostic Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; color: #00d4ff; }
    .main-header p  { margin: 0.3rem 0 0; color: #aaa; font-size: 0.9rem; }

    .stat-card {
        background: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        color: white;
    }
    .stat-card .value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .stat-card .label { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }

    .step-box {
        background: #111827;
        border-left: 3px solid #00d4ff;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
        font-family: monospace;
        font-size: 0.82rem;
        color: #e2e8f0;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .step-box.tool-result {
        border-left-color: #f59e0b;
        background: #1c1400;
    }
    .step-box.error {
        border-left-color: #ef4444;
        background: #1c0000;
    }

    .commit-badge {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 0.1rem;
    }
    .commit-badge.error {
        background: #7f1d1d;
        color: #fca5a5;
    }

    .answer-box {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    .chat-msg-user {
        background: #1e3a5f;
        border-radius: 10px 10px 2px 10px;
        padding: 0.7rem 1rem;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .chat-msg-agent {
        background: #0f2027;
        border-radius: 10px 10px 10px 2px;
        padding: 0.7rem 1rem;
        color: #a5f3fc;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _b64_to_image(b64: str):
    """Decode a base64 PNG string to bytes for st.image."""
    return base64.b64decode(b64)


def _run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
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


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────

if "agent"            not in st.session_state: st.session_state.agent           = None
if "chat_history"     not in st.session_state: st.session_state.chat_history    = []
if "last_steps"       not in st.session_state: st.session_state.last_steps      = []
if "last_charts"      not in st.session_state: st.session_state.last_charts     = {}
if "neo4j_commits"    not in st.session_state: st.session_state.neo4j_commits   = []
if "store_stats"      not in st.session_state: st.session_state.store_stats     = None
if "agent_ready"      not in st.session_state: st.session_state.agent_ready     = False
if "init_error"       not in st.session_state: st.session_state.init_error      = ""


# ─────────────────────────────────────────────
# Agent initialisation
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising Log Analysis Agent...")
def _init_agent():
    """Build the analysis agent once per session (cached by Streamlit)."""
    try:
        from analysis_dashboard.agent import build_analysis_agent
        agent = build_analysis_agent()
        return agent, None
    except Exception as exc:
        return None, str(exc)


def _load_store_stats():
    try:
        from analysis_dashboard.store_reader import get_shared_log_store
        store = get_shared_log_store()
        return store.get_stats(), store.get_sessions()
    except Exception as exc:
        return None, []


def _load_graph_summary():
    try:
        neo4j_uri = os.getenv("NEO4J_URI", "")
        if not neo4j_uri:
            return None
        from analysis_dashboard.graph_client import Neo4jGraphClient
        client = Neo4jGraphClient(
            uri=neo4j_uri,
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
        )
        if client.connect():
            summary = client.get_graph_summary()
            client.close()
            return summary
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🛡️ HLP Diagnostic Dashboard</h1>
    <p>Stage 3 — Distributed MCP Observability Control Plane</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — Stats & Session Explorer
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 System Overview")

    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.session_state.store_stats = None

    if st.session_state.store_stats is None:
        stats, sessions = _load_store_stats()
        st.session_state.store_stats = (stats, sessions)
    else:
        stats, sessions = st.session_state.store_stats

    if stats:
        st.metric("Total Log Entries",  stats.get("total_entries",  0))
        st.metric("Sessions Tracked",   stats.get("total_sessions", 0))
        st.metric("Error Count",        stats.get("error_count",    0))
        avg_lat = stats.get("avg_latency_ms")
        st.metric("Avg Latency",
                  f"{avg_lat:.0f} ms" if avg_lat else "—")

        st.markdown("### Interaction Breakdown")
        by_type = stats.get("by_interaction_type", {})
        if by_type:
            for itype, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
                st.progress(
                    cnt / max(by_type.values()),
                    text=f"`{itype}`: {cnt}",
                )
        else:
            st.info("No log entries yet. Run the agent client first.")
    else:
        st.warning("Log store not found.\nRun `uv run --package agent_client start-agent` first.")

    st.markdown("---")
    st.markdown("## 🔗 Neo4j Graph")

    graph_summary = _load_graph_summary()
    if graph_summary and graph_summary.get("connected"):
        st.metric("Sessions in Graph",  graph_summary.get("sessions",      0))
        st.metric("Agent Actions",       graph_summary.get("agent_actions", 0))
        st.metric("MCP Server Calls",    graph_summary.get("mcp_calls",     0))
        st.metric("Graph Edges",         graph_summary.get("edges",         0))
    else:
        if os.getenv("NEO4J_URI"):
            st.error("Neo4j connection failed.\nCheck NEO4J_URI / credentials.")
        else:
            st.info("Set NEO4J_URI in .env to enable graph features.")

    st.markdown("---")
    st.markdown("## 🗂️ Sessions")
    if sessions:
        selected_session = st.selectbox(
            "Filter by session",
            options=["All sessions"] + [s[:16] + "…" for s in sessions],
            index=0,
        )
    else:
        st.caption("No sessions yet.")
        selected_session = "All sessions"

    st.markdown("---")
    st.markdown("## ⚙️ Quick Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Latency\nChart", use_container_width=True):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Generate a latency trend chart for all sessions.",
            })
            st.session_state._trigger_query = "Generate a latency trend chart for all sessions."
    with col2:
        if st.button("🗺️ Sync\nNeo4j", use_container_width=True):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Sync all sessions to the Neo4j graph and report what was committed.",
            })
            st.session_state._trigger_query = "Sync all sessions to the Neo4j graph and report what was committed."


# ─────────────────────────────────────────────
# Main content tabs
# ─────────────────────────────────────────────

tab_chat, tab_reasoning, tab_charts, tab_graph = st.tabs([
    "💬 Analysis Chat",
    "🧠 Agent Reasoning",
    "📊 Trend Charts",
    "🗺️ Graph Updates",
])

# ── Tab 1: Chat Interface ─────────────────────────────────────
with tab_chat:
    st.markdown("### Natural Language Diagnostic Interface")
    st.caption(
        "Ask the Log Analysis Agent anything about your HLP system. "
        "Examples: *Find all error logs*, *What is the average latency for reflect_answer?*, "
        "*Sync sessions to Neo4j*, *Show me a dashboard chart*"
    )

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-msg-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-msg-agent">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask the analysis agent…",
                placeholder="e.g. Find sessions where reflect_answer latency exceeded 5000ms",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    # Process query
    query_to_run = None
    if submitted and user_input.strip():
        query_to_run = user_input.strip()
    elif hasattr(st.session_state, "_trigger_query"):
        query_to_run = st.session_state._trigger_query
        del st.session_state._trigger_query

    if query_to_run:
        st.session_state.chat_history.append({"role": "user", "content": query_to_run})

        # Initialise agent if needed
        agent, init_err = _init_agent()
        if init_err or agent is None:
            error_msg = f"Agent initialisation failed: {init_err}"
            st.session_state.chat_history.append({"role": "agent", "content": error_msg})
            st.error(error_msg)
        else:
            with st.spinner("🤔 Agent is reasoning…"):
                steps        = []
                charts_found = {}
                commits_found = []

                async def _collect_step(step: dict):
                    steps.append(step)
                    # Extract charts from tool results
                    if step.get("type") == "tool_result":
                        content = step.get("content", "")
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict) and "chart_b64" in parsed and parsed["chart_b64"]:
                                tool = step.get("tool_name", "chart")
                                charts_found[tool] = parsed["chart_b64"]
                            if isinstance(parsed, dict) and "commits" in parsed:
                                commits_found.extend(parsed.get("commits", []))
                        except Exception:
                            pass

                from analysis_dashboard.agent import run_analysis_query
                final_answer, all_steps = _run_async(
                    run_analysis_query(agent, query_to_run, step_callback=_collect_step)
                )

                # Also scan all_steps for charts
                for step in all_steps:
                    if step.get("type") == "tool_result":
                        try:
                            parsed = json.loads(step.get("content", "{}"))
                            if isinstance(parsed, dict) and parsed.get("chart_b64"):
                                charts_found[step.get("tool_name", "chart")] = parsed["chart_b64"]
                            if isinstance(parsed, dict) and "commits" in parsed:
                                commits_found.extend(parsed.get("commits", []))
                        except Exception:
                            pass

                st.session_state.last_steps    = all_steps
                st.session_state.last_charts   = charts_found
                st.session_state.neo4j_commits = commits_found

                answer = final_answer or "Analysis complete. Check the Reasoning tab for details."
                st.session_state.chat_history.append({"role": "agent", "content": answer})

        st.rerun()


# ── Tab 2: Agent Reasoning ────────────────────────────────────
with tab_reasoning:
    st.markdown("### Step-by-Step Agent Reasoning")

    if not st.session_state.last_steps:
        st.info("No reasoning trace yet. Ask a question in the Chat tab.")
    else:
        st.caption(f"Last query produced {len(st.session_state.last_steps)} reasoning steps.")
        for i, step in enumerate(st.session_state.last_steps, 1):
            step_type    = step.get("type", "reasoning")
            content      = step.get("content", "")
            tools_called = step.get("tools", [])

            if step_type == "reasoning":
                icon = "🧠"
                css_class = "step-box"
                header = f"Step {i} — Reasoning"
                if tools_called:
                    header += f" → calling: {', '.join(tools_called)}"
            else:
                icon = "🔧"
                css_class = "step-box tool-result"
                header = f"Step {i} — Tool Result: {step.get('tool_name', 'tool')}"

            with st.expander(f"{icon} {header}", expanded=(i == len(st.session_state.last_steps))):
                st.markdown(
                    f'<div class="{css_class}">{content}</div>',
                    unsafe_allow_html=True,
                )


# ── Tab 3: Trend Charts ───────────────────────────────────────
with tab_charts:
    st.markdown("### System Performance Trend Charts")

    if st.session_state.last_charts:
        for tool_name, b64 in st.session_state.last_charts.items():
            st.markdown(f"#### 📈 {tool_name.replace('_', ' ').title()}")
            try:
                img_bytes = _b64_to_image(b64)
                st.image(img_bytes, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render chart: {exc}")
    else:
        st.info(
            "No charts generated yet. Try asking:\n"
            "- *Generate a latency trend chart*\n"
            "- *Show me a token metrics chart*\n"
            "- *Generate a full dashboard chart*\n"
            "- *Show the error frequency chart and save it to disk*"
        )

    st.markdown("---")
    st.markdown("#### Quick Chart Generation")
    ccol1, ccol2, ccol3, ccol4 = st.columns(4)

    def _quick_chart(query: str):
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state._trigger_query = query

    with ccol1:
        if st.button("📉 Latency Trend", use_container_width=True):
            _quick_chart("Generate a latency trend chart with moving average window of 5.")
            st.rerun()
    with ccol2:
        if st.button("🔤 Token Metrics", use_container_width=True):
            _quick_chart("Generate a token metrics chart showing consumption by type.")
            st.rerun()
    with ccol3:
        if st.button("🚨 Error Freq", use_container_width=True):
            _quick_chart("Generate an error frequency chart.")
            st.rerun()
    with ccol4:
        if st.button("🎛️ Full Dashboard", use_container_width=True):
            _quick_chart("Generate a full 4-panel system health dashboard chart and save it to disk.")
            st.rerun()


# ── Tab 4: Graph Updates ─────────────────────────────────────
with tab_graph:
    st.markdown("### Neo4j Aura DB Graph Commit Log")

    if not st.session_state.neo4j_commits:
        if os.getenv("NEO4J_URI"):
            st.info(
                "No graph commits yet in this session.\n\n"
                "Ask the agent: *Sync all sessions to the Neo4j graph*"
            )
        else:
            st.warning(
                "Neo4j is not configured.\n\n"
                "Add `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` to your `.env` file."
            )
    else:
        st.success(f"✅ {len(st.session_state.neo4j_commits)} graph operations committed this session.")

        # Group by operation type
        nodes_committed   = [c for c in st.session_state.neo4j_commits if "node" in c and "error" not in c]
        edges_committed   = [c for c in st.session_state.neo4j_commits if "edge" in c and "error" not in c]
        errors_committed  = [c for c in st.session_state.neo4j_commits if "error" in c]

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            st.metric("Nodes Written",  len(nodes_committed))
        with gcol2:
            st.metric("Edges Written",  len(edges_committed))
        with gcol3:
            st.metric("Commit Errors",  len(errors_committed))

        st.markdown("#### Node Commits")
        for commit in nodes_committed[:50]:
            node_type  = commit.get("node", "?")
            operation  = commit.get("operation", "MERGE")
            identifier = commit.get("session_id") or commit.get("action_id") or commit.get("call_id") or "?"
            st.markdown(
                f'<span class="commit-badge">({node_type}) {operation} — {str(identifier)[:24]}</span>',
                unsafe_allow_html=True,
            )

        if edges_committed:
            st.markdown("#### Edge Commits")
            for commit in edges_committed[:50]:
                edge_type = commit.get("edge", "?")
                frm       = str(commit.get("from", "?"))[:16]
                to        = str(commit.get("to",   "?"))[:16]
                st.markdown(
                    f'<span class="commit-badge">-[:{edge_type}]-> {frm}…→{to}…</span>',
                    unsafe_allow_html=True,
                )

        if errors_committed:
            st.markdown("#### Commit Errors")
            for err in errors_committed:
                st.markdown(
                    f'<span class="commit-badge error">⚠ {str(err.get("error","?"))[:80]}</span>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    # Live graph summary
    if st.button("🔄 Refresh Graph Summary"):
        summary = _load_graph_summary()
        if summary and summary.get("connected"):
            gcol1, gcol2, gcol3, gcol4 = st.columns(4)
            gcol1.metric("Sessions",     summary["sessions"])
            gcol2.metric("AgentActions", summary["agent_actions"])
            gcol3.metric("MCPCalls",     summary["mcp_calls"])
            gcol4.metric("Edges",        summary["edges"])
        elif os.getenv("NEO4J_URI"):
            st.error("Could not connect to Neo4j. Check credentials.")
        else:
            st.info("NEO4J_URI not set.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "HLP Diagnostic Dashboard — Stage 3 | "
    "Kodecamp Agentic AI Bootcamp | "
    "Built with LangChain · LangGraph · Neo4j · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)


def main():
    """Entry point stub — Streamlit is launched via `uv run streamlit run`."""
    pass
