from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv
import redis as redis_client_lib

load_dotenv(find_dotenv(".env"))

st.set_page_config(
    page_title="HLP Diagnostic Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .step-box.node-exec {
        border-left-color: #a855f7;
        background: #1a0a2e;
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

    .token-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 12px;
        font-size: 0.78rem;
        font-family: monospace;
        margin: 0.15rem 0.1rem;
        font-weight: 600;
    }
    .token-high   { background: #7f1d1d; color: #fca5a5; }
    .token-medium { background: #78350f; color: #fde68a; }
    .token-low    { background: #1e3a5f; color: #93c5fd; }

    .resilience-card {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
        margin: 0.3rem;
    }
    .resilience-card .r-value { font-size: 2.2rem; font-weight: bold; }
    .resilience-card .r-label { font-size: 0.75rem; color: #94a3b8; }
    .r-green  { color: #4ade80; }
    .r-yellow { color: #facc15; }
    .r-red    { color: #f87171; }

    .audit-box {
        background: #0a0f1e;
        border: 1px solid #7c3aed;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #e2e8f0;
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def _b64_to_image(b64: str):
    return base64.b64decode(b64)


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

_tier3_redis: "redis_client_lib.Redis | None" = None
def _tier3_cache_client():
    global _tier3_redis
    if _tier3_redis is None:
        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            return None
        _tier3_redis = redis_client_lib.from_url(redis_url, decode_responses=True)
    return _tier3_redis

def _tier3_get_or_compute(cache_key: str, compute_fn, ttl_seconds: int = 60) -> dict | list | None:
    client = _tier3_cache_client()
    full_key = f"hlp:cache:tier3:{cache_key}"

    if client is not None:
        start = time.perf_counter()
        cached = client.get(full_key)
        if cached is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            client.incr("hlp:cache:tier3:hits")
            client.incrbyfloat("hlp:cache:tier3:latency_sum_hit", elapsed_ms)
            client.incr("hlp:cache:tier3:latency_count_hit")
            return json.loads(cached)
        client.incr("hlp:cache:tier3:misses")
    
    start = time.perf_counter()
    result = compute_fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if client is not None and result is not None:
        client.setex(full_key, ttl_seconds, json.dumps(result, default=str))
        client.incrbyfloat("hlp:cache:tier3:latency_sum_miss", elapsed_ms)
        client.incr("hlp:cache:tier3:latency_count_miss")

    return result
    
def _load_resilience_state() -> dict:
    def _compute():
        try:
            from analysis_dashboard.store_reader import get_shared_log_store
            store = get_shared_log_store()
            entries = _run_async(store.get_all(limit=5000))
        except Exception:
            return {
                "current": {
                    "fallback_activations": 0,
                    "self_heal_iterations": 0,
                    "absolute_fallback_hits": 0,
                },
                "history": [],
            }

        history = []
        for e in entries:
            raw_meta = e.get("metadata") or e.get("metadata_json") or "{}"
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except Exception:
                meta = {}
            if meta.get("event") == "resilience_state_flush":
                history.append({
                    "timestamp":              e.get("timestamp", ""),
                    "fallback_activations":   meta.get("fallback_activations", 0),
                    "self_heal_iterations":   meta.get("self_heal_iterations", 0),
                    "absolute_fallback_hits": meta.get("absolute_fallback_hits", 0),
                })

        current = (
            history[-1]
            if history
            else {
                "fallback_activations": 0,
                "self_heal_iterations": 0,
                "absolute_fallback_hits": 0,
            }
        )
        return {"current": current, "history": history}
    return _tier3_get_or_compute("resilience_state", _compute, ttl_seconds=30)

def _render_shap_chart(shap_data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_contributions = shap_data.get("feature_contributions", {})
    if not feature_contributions:
        st.info("No SHAP feature contributions to display.")
        return

    features = list(feature_contributions.keys())
    values   = [feature_contributions[f] for f in features]
    colors   = ["#ef4444" if v > 0 else "#3b82f6" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(features) * 1.1)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    bars = ax.barh(features, values, color=colors, edgecolor="none", height=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=4, color="#e2e8f0", fontsize=9)
    ax.axvline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (contribution to anomaly score)", color="#94a3b8", fontsize=9)
    ax.set_title(
        f"Proxy SHAP Feature Contributions\n"
        f"Dominant feature: {shap_data.get('dominant_feature', '—')}",
        color="#e2e8f0", fontsize=10, pad=10,
    )
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.caption(
        f"Features used: {', '.join(shap_data.get('features_used', []))}  |  "
        f"Entries analysed: {shap_data.get('total_entries', 0)}  |  "
        f"Anomalies detected: {shap_data.get('anomaly_count', 0)}"
    )


def _render_lime_annotations(lime_data: dict) -> None:
    entries = lime_data.get("entries", [])
    if not entries:
        st.info("No LIME token importance data to display.")
        return

    for entry in entries:
        entry_id   = entry.get("entry_id", "?")
        itype      = entry.get("interaction_type", "?")
        preview    = entry.get("content_preview", "")
        top_tokens = entry.get("token_importances", [])

        with st.expander(f"📝 {entry_id[:24]}  —  `{itype}`", expanded=False):
            st.caption(f"Content preview: *{preview[:100]}...*")

            if not top_tokens:
                st.caption("Content too short for perturbation analysis.")
                continue

            badge_html = ""
            for item in top_tokens:
                token = item.get("token", "")
                score = abs(item.get("importance", 0.0))
                css   = "token-high" if score > 0.6 else "token-medium" if score > 0.3 else "token-low"
                badge_html += (
                    f'<span class="token-badge {css}">'
                    f'{token} <small>({item["importance"]:+.3f})</small>'
                    f'</span> '
                )
            st.markdown(badge_html, unsafe_allow_html=True)

            import pandas as pd
            df_lime = pd.DataFrame(top_tokens)
            if not df_lime.empty:
                df_lime.columns = ["Token", "LIME Importance"]
                df_lime["LIME Importance"] = df_lime["LIME Importance"].round(4)
                df_lime = df_lime.sort_values("LIME Importance", ascending=False, key=abs)
                st.dataframe(df_lime, width="stretch", hide_index=True)

            st.caption(
                f"n_tokens={entry.get('n_tokens', '?')}  |  "
                f"n_perturbations={entry.get('n_perturbations', '?')}"
            )


defaults = {
    "agent":             None,
    "chat_history":      [],
    "last_steps":        [],
    "last_charts":       {},
    "neo4j_commits":     [],
    "store_stats":       None,
    "agent_ready":       False,
    "init_error":        "",
    "last_audit_result": None,
    "resilience_state":  None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


@st.cache_resource(show_spinner="Initialising Log Analysis Agent...")
def _init_agent():
    try:
        from analysis_dashboard.agent import build_analysis_agent
        return build_analysis_agent(), None
    except Exception as exc:
        return None, str(exc)

def _load_store_stats():
    def _compute():
        try:
            from analysis_dashboard.store_reader import get_shared_log_store

            async def _fetch():
                store = get_shared_log_store()
                stats = await store.get_stats()
                sessions = await store.get_sessions()
                return {"stats": stats, "sessions": sessions}

            return _run_async(_fetch())
        except Exception:
            return {"stats": None, "sessions": []}

    payload = _tier3_get_or_compute("store_stats", _compute, ttl_seconds=30)
    return payload["stats"], payload["sessions"]

def _load_graph_summary():
    def _compute():
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
    
    return _tier3_get_or_compute("graph_summary", _compute, ttl_seconds=60)

def _get_cache_telemetry() -> dict:
    client = _tier3_cache_client()
    if client is None:
        return {
            "tier1": {"hits": 0, "misses": 0, "hit_rate": 0.0, "tokens_saved": 0, "cost_saved_usd": 0.0},
            "tier2": {"hits": 0, "misses": 0, "hit_rate": 0.0},
            "tier3": {"hits": 0, "misses": 0, "hit_rate": 0.0},
        }
    
    def _int(key: str) -> int:
        val = client.get(key)
        return int(val) if val is not None else 0
    
    def _float(key: str) -> float:
        val = client.get(key)
        return float(val) if val is not None else 0.0
    
    def _hit_rate(hits: int, misses: int) -> float:
        total = hits + misses
        return round((hits / total) * 100,1) if total > 0 else 0.0
    
    def _avg_latency(tier: str, kind: str) -> float | None:
        count = _int(f"hlp:cache:{tier}:latency_count_{kind}")
        if count == 0:
            return None
        return round(_float(f"hlp:cache:{tier}:latency_sum_{kind}") / count, 2)
    
    result = {}
    for tier in ("tier1", "tier2", "tier3"):
        hits, misses = _int(f"hlp:cache:{tier}:hits"), _int(f"hlp:cache:{tier}:misses")
        result[tier] = {
            "hits": hits, "misses": misses,
            "hit_rate": _hit_rate(hits, misses),
            "avg_latency_hit_ms": _avg_latency(tier, "hit"),
            "avg_latency_miss_ms": _avg_latency(tier, "miss"),
        }
    result["tier1"]["tokens_saved"] = _int("hlp:cache:tier1:tokens_saved")
    result["tier1"]["cost_saved_usd"] = round(_float("hlp:cache:tier1:cost_saved_usd"), 4)

    return result

def _flush_cache_namespace(prefix: str) -> int:
    """Delete all Redis keys under a given prefix, Returns count of keys deleted."""
    client = _tier3_cache_client()
    if client is None:
        return 0
    keys = list(client.scan_iter(match=f"{prefix}*"))
    if keys:
        client.delete(*keys)
    return len(keys)

st.markdown("""
<div class="main-header">
    <h1>🛡️ HLP Diagnostic Dashboard</h1>
    <p>Fault-Tolerant · Edgeless LangGraph · XAI Explainability · Resilience Tracking</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 📊 System Overview")

    if st.button("🔄 Refresh Stats", width="stretch"):
        st.session_state.store_stats      = None
        st.session_state.resilience_state = None

    if st.session_state.store_stats is None:
        stats, sessions = _load_store_stats()
        st.session_state.store_stats = (stats, sessions)
    else:
        stats, sessions = st.session_state.store_stats

    if stats:
        st.metric("Total Log Entries", stats.get("total_entries",  0))
        st.metric("Sessions Tracked",  stats.get("total_sessions", 0))
        st.metric("Error Count",       stats.get("error_count",    0))
        avg_lat = stats.get("avg_latency_ms")
        st.metric("Avg Latency", f"{avg_lat:.0f} ms" if avg_lat else "—")

        st.markdown("### Interaction Breakdown")
        by_type = stats.get("by_interaction_type", {})
        if by_type:
            for itype, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
                st.progress(cnt / max(by_type.values()), text=f"`{itype}`: {cnt}")
        else:
            st.info("No log entries yet. Run the agent client first.")
    else:
        st.warning("Log store not found.\nRun `uv run --package agent_client start-agent` first.")

    st.markdown("---")
    st.markdown("## 🔗 Neo4j Graph")
    graph_summary = _load_graph_summary()
    if graph_summary and graph_summary.get("connected"):
        st.metric("Sessions in Graph", graph_summary.get("sessions",      0))
        st.metric("Agent Actions",     graph_summary.get("agent_actions", 0))
        st.metric("MCP Server Calls",  graph_summary.get("mcp_calls",     0))
        st.metric("Graph Edges",       graph_summary.get("edges",         0))
    else:
        if os.getenv("NEO4J_URI"):
            st.error("Neo4j connection failed.\nCheck NEO4J_URI / credentials.")
        else:
            st.info("Set NEO4J_URI in .env to enable graph features.")

    st.markdown("---")
    st.markdown("## 🧮 Cache Performance")
    telemetry = _get_cache_telemetry()

    t1 = telemetry["tier1"]
    st.markdown("**Tier 1 - LLM Semantic Cache**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Hit Rate", f"{t1['hit_rate']}%")
    col2.metric("Tokens Saved", t1['tokens_saved'])
    col3.metric("Cost Saved", f"${t1['cost_saved_usd']:.4f}")
    st.progress(t1['hit_rate'] / 100, text=f"{t1['hits']} hits / {t1['misses']} misses")

    t2 = telemetry["tier2"]
    st.markdown("**Tier 2 - Graph & XAI Cache**")
    st.progress(t2['hit_rate'] / 100, text=f"{t2['hit_rate']}% hit rate - {t2['hits']} hits / {t2['misses']} misses",)

    t3 = telemetry["tier3"]
    st.markdown("**Tier 3 - UI Query Cache**")
    st.progress(
        t3["hit_rate"] / 100,
        text=f"{t3['hit_rate']}% hit rate - {t3['hits']} hits / {t3['misses']} misses",
    )

    st.markdown("**Latency: Cached vs. Uncached**")
    for tier_key, tier_label in [("tier1", "LLM Calls"), ("tier2", "Graph/XAI"), ("tier3", "UI Queries")]:
        t = telemetry[tier_key]
        hit_ms = t.get("avg_latency_hit_ms")
        miss_ms = t.get("avg_latency_miss_ms")
        colA, colB = st.columns(2)
        colA.metric(f"{tier_label} — Cache Hit", f"{hit_ms:.1f} ms" if hit_ms is not None else "—")
        colB.metric(f"{tier_label} — Uncached", f"{miss_ms:.1f} ms" if miss_ms is not None else "—")
    st.markdown("**Cache Invalidation**")
    colf1, colf2, colf3, colf4 = st.columns(4)
    if colf1.button("Flush Tier 1"):
        n = _flush_cache_namespace("hlp:cache:tier1:")
        st.success(f"Flushed {n} Tier 1 key(s).")
    if colf2.button("Flush Tier 2"):
        n = _flush_cache_namespace("hlp:cache:tier2:")
        st.success(f"Flushed {n} Tier 2 key(s).")
    if colf3.button("Flush Tier 3"):
        n = _flush_cache_namespace("hlp:cache:tier3:")
        st.success(f"Flushed {n} Tier 3 key(s).")
    if colf4.button("Flush ALL"):
        n = _flush_cache_namespace("hlp:cache:")
        st.success(f"Flushed {n} key(s) across all tiers.")
        
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
        if st.button("📈 Latency\nChart", width="stretch"):
            q = "Generate a latency trend chart for all sessions."
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state._trigger_query = q
    with col2:
        if st.button("🗺️ Sync\nNeo4j", width="stretch"):
            q = "Sync all sessions to the Neo4j graph and report what was committed."
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state._trigger_query = q

    st.markdown("---")
    st.markdown("## 🔬 XAI Shortcuts")
    if st.button("🧪 Run Audit (latest session)", width="stretch"):
        st.session_state._trigger_xai_audit = True
    if st.button("⚡ Refresh Resilience", width="stretch"):
        st.session_state.resilience_state = None


tab_chat, tab_reasoning, tab_charts, tab_graph, tab_xai, tab_resilience = st.tabs([
    "💬 Analysis Chat",
    "🧠 Agent Reasoning",
    "📊 Trend Charts",
    "🗺️ Graph Updates",
    "🔬 XAI Audit",
    "⚡ Resilience",
])


with tab_chat:
    st.markdown("### Natural Language Diagnostic Interface")
    st.caption(
        "Ask the Log Analysis Agent anything about your HLP system. "
        "Examples: *Find all error logs*, *What is the average latency for reflect_answer?*, "
        "*Sync sessions to Neo4j*, *Show me a dashboard chart*, "
        "*Run an explainability audit on the most recent session*"
    )

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

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask the analysis agent…",
                placeholder="e.g. Run an explainability audit on the most recent session",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send 🚀", width="stretch")

    query_to_run = None
    if submitted and user_input.strip():
        query_to_run = user_input.strip()
    elif hasattr(st.session_state, "_trigger_query"):
        query_to_run = st.session_state._trigger_query
        del st.session_state._trigger_query

    if query_to_run:
        st.session_state.chat_history.append({"role": "user", "content": query_to_run})
        agent, init_err = _init_agent()
        if init_err or agent is None:
            error_msg = f"Agent initialisation failed: {init_err}"
            st.session_state.chat_history.append({"role": "agent", "content": error_msg})
            st.error(error_msg)
        else:
            with st.spinner("🤔 Agent is reasoning…"):
                charts_found  = {}
                commits_found = []

                async def _collect_step(step: dict):
                    content = step.get("content", "")
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and parsed.get("chart_b64"):
                            charts_found[step.get("tool_name", "chart")] = parsed["chart_b64"]
                        if isinstance(parsed, dict) and "commits" in parsed:
                            commits_found.extend(parsed.get("commits", []))
                    except Exception:
                        pass

                from analysis_dashboard.agent import run_analysis_query
                final_answer, all_steps = _run_async(
                    run_analysis_query(agent, query_to_run, step_callback=_collect_step)
                )

                for step in all_steps:
                    content = step.get("content", "")
                    try:
                        parsed = json.loads(content)
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


with tab_reasoning:
    st.markdown("### Step-by-Step Agent Reasoning")
    st.caption("Steps show edgeless LangGraph node executions with dynamic Command routing paths.")

    if not st.session_state.last_steps:
        st.info("No reasoning trace yet. Ask a question in the Chat tab.")
    else:
        st.caption(f"Last query produced {len(st.session_state.last_steps)} steps.")
        for i, step in enumerate(st.session_state.last_steps, 1):
            step_type = step.get("type", "reasoning")
            content   = step.get("content", "")
            routing   = step.get("routing", [])
            node_name = step.get("node", "")

            if step_type == "node_execution":
                icon      = "🟣"
                css_class = "step-box node-exec"
                route_str = " → ".join(routing) if routing else ""
                header    = f"Step {i} — Node: {node_name}"
                if route_str:
                    header += f"  ({route_str})"
            elif step_type == "reasoning":
                icon      = "🧠"
                css_class = "step-box"
                header    = f"Step {i} — Reasoning"
                tools_called = step.get("tools", [])
                if tools_called:
                    header += f" → calling: {', '.join(tools_called)}"
            else:
                icon      = "🔧"
                css_class = "step-box tool-result"
                header    = f"Step {i} — Tool Result: {step.get('tool_name', 'tool')}"

            with st.expander(f"{icon} {header}", expanded=(i == len(st.session_state.last_steps))):
                st.markdown(
                    f'<div class="{css_class}">{content[:800]}</div>',
                    unsafe_allow_html=True,
                )


with tab_charts:
    st.markdown("### System Performance Trend Charts")

    if st.session_state.last_charts:
        for tool_name, b64 in st.session_state.last_charts.items():
            st.markdown(f"#### 📈 {tool_name.replace('_', ' ').title()}")
            try:
                st.image(_b64_to_image(b64), width="stretch")
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
        if st.button("📉 Latency Trend", width="stretch"):
            _quick_chart("Generate a latency trend chart with moving average window of 5.")
            st.rerun()
    with ccol2:
        if st.button("🔤 Token Metrics", width="stretch"):
            _quick_chart("Generate a token metrics chart showing consumption by type.")
            st.rerun()
    with ccol3:
        if st.button("🚨 Error Freq", width="stretch"):
            _quick_chart("Generate an error frequency chart.")
            st.rerun()
    with ccol4:
        if st.button("🎛️ Full Dashboard", width="stretch"):
            _quick_chart("Generate a full 4-panel system health dashboard chart and save it to disk.")
            st.rerun()


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
        nodes_committed  = [c for c in st.session_state.neo4j_commits if "node"  in c and "error" not in c]
        edges_committed  = [c for c in st.session_state.neo4j_commits if "edge"  in c and "error" not in c]
        errors_committed = [c for c in st.session_state.neo4j_commits if "error" in c]

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1: st.metric("Nodes Written", len(nodes_committed))
        with gcol2: st.metric("Edges Written", len(edges_committed))
        with gcol3: st.metric("Commit Errors", len(errors_committed))

        st.markdown("#### Node Commits")
        for commit in nodes_committed[:50]:
            identifier = (
                commit.get("session_id") or
                commit.get("action_id") or
                commit.get("call_id") or "?"
            )
            st.markdown(
                f'<span class="commit-badge">({commit.get("node","?")}) '
                f'{commit.get("operation","MERGE")} — {str(identifier)[:24]}</span>',
                unsafe_allow_html=True,
            )
        if edges_committed:
            st.markdown("#### Edge Commits")
            for commit in edges_committed[:50]:
                frm = str(commit.get("from", "?"))[:16]
                to  = str(commit.get("to",   "?"))[:16]
                st.markdown(
                    f'<span class="commit-badge">-[:{commit.get("edge","?")}]-> {frm}…→{to}…</span>',
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


with tab_xai:
    st.markdown("### 🔬 Explainability Audit Report")
    st.caption(
        "Select a session or enter a trace ID to run a localised explainability audit. "
        "The engine extracts a Neo4j subgraph, runs proxy LIME on text logs, "
        "and computes proxy SHAP on numeric execution signatures."
    )

    st.markdown("#### Audit Activation Hub")
    _, sessions_list = _load_store_stats()

    xai_col1, xai_col2, xai_col3 = st.columns([3, 2, 1])
    with xai_col1:
        if sessions_list:
            audit_session = st.selectbox(
                "Session to audit",
                options=["Most recent"] + sessions_list,
                index=0,
                key="audit_session_select",
            )
            audit_session_id = sessions_list[-1] if audit_session == "Most recent" else audit_session
        else:
            st.warning("No sessions in log store yet.")
            audit_session_id = None

    with xai_col2:
        audit_trace_id = st.text_input(
            "Or enter a trace ID (overrides session)",
            placeholder="e.g. entry_uuid",
            key="audit_trace_input",
        )

    with xai_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_audit_btn = st.button("▶ Run Audit", width="stretch", type="primary")

    if getattr(st.session_state, "_trigger_xai_audit", False):
        run_audit_btn = True
        del st.session_state._trigger_xai_audit

    if run_audit_btn:
        if not audit_session_id and not audit_trace_id:
            st.error("No session or trace ID available. Run the agent client first.")
        else:
            with st.spinner("🔍 Running explainability audit (LIME + SHAP + Neo4j)…"):
                try:
                    from analysis_dashboard.agent import run_explainability_audit
                    audit_raw     = run_explainability_audit.invoke({
                        "session_id": audit_trace_id or audit_session_id,
                        "top_k_logs": 20,
                    })
                    audit_summary = json.loads(audit_raw)

                    _REPO_ROOT   = Path(__file__).resolve().parents[3]
                    report_path  = _REPO_ROOT / "explainability_audit_report.json"
                    full_report  = {}
                    if report_path.exists():
                        with open(report_path, "r", encoding="utf-8") as f:
                            full_report = json.load(f)

                    st.session_state.last_audit_result = {
                        "summary":     audit_summary,
                        "full_report": full_report,
                        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    }
                except Exception as exc:
                    st.error(f"Audit failed: {exc}")

    if st.session_state.last_audit_result:
        result    = st.session_state.last_audit_result
        summary   = result.get("summary", {})
        full      = result.get("full_report", {})
        timestamp = result.get("timestamp", "")

        st.markdown(
            f"---\n#### Audit Results  <small style='color:#64748b'>— {timestamp}</small>",
            unsafe_allow_html=True,
        )

        meta = summary.get("audit_metadata") or {
            "session_id":                  summary.get("session_audited", "?"),
            "entries_analysed":            summary.get("entries_analysed", 0),
            "text_entries_for_lime":       summary.get("lime_tokens_scored", 0),
            "structured_entries_for_shap": summary.get("shap_features_scored", 0),
        }
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Session",          str(meta.get("session_id", "?"))[:16] + "…")
        m2.metric("Entries Analysed", meta.get("entries_analysed", 0))
        m3.metric("LIME Entries",     meta.get("text_entries_for_lime", 0))
        m4.metric("SHAP Entries",     meta.get("structured_entries_for_shap", 0))

        graph_ctx = full.get("graph_context", {})
        if graph_ctx and "node_counts" in graph_ctx:
            with st.expander("🗺️ Neo4j Subgraph Context", expanded=False):
                nc = graph_ctx.get("node_counts", {})
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric("Session Nodes",      nc.get("sessions", 0))
                gc2.metric("AgentAction Nodes",  nc.get("agent_actions", 0))
                gc3.metric("MCPCall Nodes",       nc.get("mcp_calls", 0))
                ctx_summary = graph_ctx.get("context_summary", "")
                if ctx_summary:
                    st.code(ctx_summary, language=None)

        st.markdown("#### Proxy SHAP — Feature Contribution Chart")
        shap_data = full.get("shap_results", {})
        if shap_data and "error" not in shap_data:
            _render_shap_chart(shap_data)
            dominant = shap_data.get("dominant_feature")
            if dominant:
                st.info(
                    f"📌 **Dominant feature**: `{dominant}` — this metric contributed most "
                    f"to anomalous execution signatures in the selected session."
                )
        else:
            err = shap_data.get("error", "No SHAP data available.") if shap_data else "No SHAP data."
            st.warning(f"SHAP: {err}")

        st.markdown("---")

        st.markdown("#### Proxy LIME — Token Importance Annotations")
        lime_data = full.get("lime_results", {})
        if lime_data and "error" not in lime_data:
            _render_lime_annotations(lime_data)
        else:
            err = lime_data.get("error", "No LIME data available.") if lime_data else "No LIME data."
            st.warning(f"LIME: {err}")

        st.markdown("---")
        _REPO_ROOT  = Path(__file__).resolve().parents[3]
        report_path = _REPO_ROOT / "explainability_audit_report.json"
        if report_path.exists():
            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download explainability_audit_report.json",
                    data=f,
                    file_name="explainability_audit_report.json",
                    mime="application/json",
                )
    else:
        st.info("Run an audit above to see LIME token importance and SHAP feature contribution results.")


with tab_resilience:
    st.markdown("### ⚡ Resilience Tracking Panel")
    st.caption(
        "Live and historical counts of how many times the MCP client fell back "
        "using RunnableWithFallbacks or initiated an LLM self-healing iteration. "
        "These counters are written to the log store after every agent query."
    )

    if st.session_state.resilience_state is None:
        st.session_state.resilience_state = _load_resilience_state()

    res     = st.session_state.resilience_state
    current = res.get("current", {})
    history = res.get("history", [])

    if st.button("🔄 Refresh Resilience Data"):
        st.session_state.resilience_state = None
        st.rerun()

    st.markdown("#### Current Session Resilience State")

    fa  = current.get("fallback_activations",   0)
    shi = current.get("self_heal_iterations",    0)
    afh = current.get("absolute_fallback_hits",  0)

    r1, r2, r3 = st.columns(3)
    with r1:
        colour = "r-yellow" if fa > 0 else "r-green"
        st.markdown(
            f'<div class="resilience-card">'
            f'<div class="r-value {colour}">{fa}</div>'
            f'<div class="r-label">Fallback Activations</div>'
            f'<div style="font-size:0.7rem;color:#64748b;margin-top:0.4rem">'
            f'RunnableWithFallbacks caught an error and rerouted</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with r2:
        colour = "r-green" if shi > 0 else "r-yellow" if fa > 0 else "r-green"
        st.markdown(
            f'<div class="resilience-card">'
            f'<div class="r-value {colour}">{shi}</div>'
            f'<div class="r-label">Self-Heal Successes</div>'
            f'<div style="font-size:0.7rem;color:#64748b;margin-top:0.4rem">'
            f'LLM self-correction recovered successfully</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with r3:
        colour = "r-red" if afh > 0 else "r-green"
        st.markdown(
            f'<div class="resilience-card">'
            f'<div class="r-value {colour}">{afh}</div>'
            f'<div class="r-label">Absolute Fallback Hits</div>'
            f'<div style="font-size:0.7rem;color:#64748b;margin-top:0.4rem">'
            f'All retries + self-heals exhausted — hardcoded fallback fired</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if fa > 0:
        recovery_rate = (shi / fa) * 100
        st.markdown(
            f"**Self-heal recovery rate:** `{recovery_rate:.0f}%`  "
            f"({shi} recovered / {fa} total fallbacks)"
        )
        st.progress(min(recovery_rate / 100, 1.0))
        if afh > 0:
            st.warning(
                f"⚠️ {afh} absolute fallback(s) fired this session — "
                f"all retry and self-heal paths were exhausted. "
                f"Check the log store for entries with `event=absolute_fallback`."
            )
    else:
        st.success("✅ No fallback activations recorded — primary execution path succeeded for all queries.")

    if history:
        st.markdown("---")
        st.markdown("#### Historical Resilience Log")
        st.caption(f"{len(history)} resilience state snapshots recorded across all sessions.")

        import pandas as pd
        df_res = pd.DataFrame(history)
        if not df_res.empty:
            df_res = df_res.rename(columns={
                "timestamp":              "Timestamp",
                "fallback_activations":   "Fallbacks",
                "self_heal_iterations":   "Self-Heals",
                "absolute_fallback_hits": "Absolute Fallbacks",
            })
            st.dataframe(
                df_res.sort_values("Timestamp", ascending=False).head(50),
                width="stretch",
                hide_index=True,
            )

        if len(history) >= 2:
            st.markdown("#### Fallback Trend Over Time")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")

            xs = list(range(len(history)))
            ax.plot(xs, [h["fallback_activations"]   for h in history],
                    label="Fallback Activations", color="#facc15", linewidth=2)
            ax.plot(xs, [h["self_heal_iterations"]   for h in history],
                    label="Self-Heal Successes",  color="#4ade80", linewidth=2)
            ax.plot(xs, [h["absolute_fallback_hits"] for h in history],
                    label="Absolute Fallbacks",   color="#f87171", linewidth=2)

            ax.set_xlabel("Snapshot Index", color="#94a3b8", fontsize=9)
            ax.set_ylabel("Count",          color="#94a3b8", fontsize=9)
            ax.set_title("Resilience Counters Over Time", color="#e2e8f0", fontsize=10)
            ax.tick_params(colors="#94a3b8")
            ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e3a5f")

            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
    else:
        st.info(
            "No historical resilience data yet.\n\n"
            "Run the agent client with a resilience test query:\n"
            "`Run a resilience test: trigger a runtime_error fault on the query_knowledge tool.`"
        )

    st.markdown("---")
    with st.expander("ℹ️ Resilience Configuration Reference", expanded=False):
        st.markdown("""
| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_RETRY_ATTEMPTS` | 3 | Total invocation attempts (1 original + 2 retries) |
| `wait_exponential_jitter` | True | Exponential backoff + random jitter between retries |
| `exception_key` | `"error_trace"` | Key used to inject caught exception into fallback input |
| Fallback 1 | `_SelfHealRunnable` | LLM self-correction using corrective prompt |
| Fallback 2 | `_AbsoluteFallbackRunnable` | Hardcoded safe exit — never raises |
        """)


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "HLP Diagnostic Dashboard | "
    "Kodecamp Agentic AI Bootcamp | "
    "Built with LangChain · LangGraph · Neo4j · Streamlit · Proxy LIME/SHAP"
    "</p>",
    unsafe_allow_html=True,
)
