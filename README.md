# HLP Graph Knowledge Agent — Stage 3: Observability & Diagnostics

**KodeCamp Agentic AI Bootcamp | Stage 3**

Extends the Stage 2 distributed MCP architecture with a full observability stack:
structured vector log persistence, a semantic log analysis agent, a Neo4j Aura DB
causal knowledge graph, and an interactive Streamlit diagnostic dashboard.

**GitHub:** https://github.com/Anyadiegwu/hlp_graph_knowledge_agent

---

## Repository Structure

```
hlp_graph_knowledge_agent/
├── pyproject.toml                  ← uv workspace root (3 members)
├── .env.example                    ← all required environment keys
├── REFLECTION_STAGE3.md
├── README.md
│
├── mcp_server/                     ← FastMCP server (Stage 2 + Stage 3)
│   ├── pyproject.toml
│   └── src/mcp_server/server.py
│
├── agent_client/                   ← MCP client + SQLite vector log store
│   ├── pyproject.toml
│   └── src/agent_client/
│       ├── client.py               ← ReAct agent, MCP sampling, _persist() calls
│       └── log_store.py            ← HLPLogStore, LogEntry schema, NS namespaces
│
└── analysis_dashboard/             ← Log Analysis Agent + Streamlit UI
    ├── pyproject.toml
    └── src/analysis_dashboard/
        ├── agent.py                ← Log Analysis Agent (7 tools)
        ├── app.py                  ← Streamlit dashboard
        ├── analytics.py            ← matplotlib/seaborn chart tools
        ├── graph_client.py         ← Neo4j Aura DB client + projection logic
        └── store_reader.py         ← read-only SQLite adapter for analysis process
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        agent_client                             │
│  LangChain ReAct Agent                                          │
│    ├── reflect_answer_tool  ─────────────────────┐              │
│    └── query_knowledge_tool ─────────────────┐   │              │
│                                              │   │              │
│  _persist() ──► HLPLogStore (SQLite)         │   │              │
│    Namespaces:                               │   │              │
│      logs.agent.planning.reflexive_loop      │   │              │
│      logs.mcp.client.tool_invocation         │   │              │
│      logs.mcp.server.tools.crag_pipeline     │   │              │
│      logs.mcp.server.sampling_request        │   │              │
│      ... (12 namespace paths total)          │   │              │
└──────────────────────────────────────────────┼───┼──────────────┘
                         streamable-http        │   │
                         localhost:8000         │   │
┌──────────────────────────────────────────────▼───▼──────────────┐
│                        mcp_server                               │
│  @tool reflect_answer   ← ctx.sample() ← client                 │
│  @tool query_knowledge  ← Hierarchical CRAG + ToT + Tavily      │
└─────────────────────────────────────────────────────────────────┘

                mcp_agent_log.db  (shared SQLite file)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   analysis_dashboard                            │
│  (separate process — reads DB, never writes to it)             │
│                                                                 │
│  Log Analysis Agent (LangChain ReAct)                          │
│    ├── semantic_log_search    ← vector cosine similarity        │
│    ├── get_log_statistics     ← aggregate counts + latency      │
│    ├── sync_graph_to_neo4j    ← project traces into Neo4j       │
│    ├── latency_trend_chart    ← matplotlib moving avg chart     │
│    ├── token_metrics_chart    ← token consumption chart         │
│    ├── error_frequency_chart  ← rolling error rate chart        │
│    └── full_dashboard_chart   ← 4-panel health dashboard        │
│                                                                 │
│  Streamlit Dashboard                                            │
│    ├── Chat tab      ← natural language → agent                 │
│    ├── Reasoning tab ← step-by-step agent trace                 │
│    ├── Charts tab    ← inline rendered trend charts             │
│    └── Graph tab     ← Neo4j commit log + live summary          │
│                                     │                           │
│                                     ▼                           │
│                           Neo4j Aura DB                         │
│                  (:Session)-[:TRIGGERED]->(:AgentAction)        │
│                  (:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)  │
│                  (:MCPServerCall)-[:DEPENDS_ON]->(:MCPServerCall)│
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A free [Neo4j Aura DB](https://neo4j.com/cloud/aura/) instance (for graph features)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anyadiegwu/hlp_graph_knowledge_agent.git
cd hlp_graph_knowledge_agent
```

### 2. Create your `.env` file

```bash
# Bash / macOS / Linux
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```cmd
:: Windows CMD
copy .env.example .env
```

Then edit `.env` with your actual keys. Required:
- `GROQ_API_KEY` and/or `GEMINI_API_KEY` (at least one)
- `TAVILY_API_KEY` (for CRAG web fallback)
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` (for graph features)

### 3. Sync all workspace dependencies

```bash
uv sync
```

---

## Running the System

You need **three terminal windows**, all opened at the repo root.

---

### Terminal 1 — MCP Server

```bash
# Bash / macOS / Linux
uv run --package mcp_server start-server
```

```powershell
# Windows PowerShell
uv run --package mcp_server start-server
```

```cmd
:: Windows CMD
uv run --package mcp_server start-server
```

Wait for:
```
[SERVER] [INFO] Starting ThinkingAgentServer (streamable-http) on http://0.0.0.0:8000 ...
INFO:     Application startup complete.
```

Verify the server is healthy:
```bash
curl http://localhost:8000/health
# Expected: OK
```

---

### Terminal 2 — Agent Client

```bash
# Bash / macOS / Linux
uv run --package agent_client start-agent
```

```powershell
# Windows PowerShell
uv run --package agent_client start-agent
```

```cmd
:: Windows CMD
uv run --package agent_client start-agent
```

This will:
1. Connect to the MCP server and register its tools
2. Pre-fetch knowledge context and run queries through the full ReAct + CRAG + Reflection pipeline
3. Write all interaction logs to `mcp_agent_system.log` (flat) and `mcp_agent_log.db` (SQLite vector store)

**The agent client must complete at least one full run before launching the dashboard.**

---

### Terminal 3 — Streamlit Dashboard

```bash
# Bash / macOS / Linux
uv run --package analysis_dashboard python -m streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

```powershell
# Windows PowerShell
uv run --package analysis_dashboard python -m streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

```cmd
:: Windows CMD
uv run --package analysis_dashboard python -m streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

### Optional — Standalone Analysis Agent CLI

Runs a set of hardcoded diagnostic queries and produces `analysis_agent.log`:

```bash
# Bash / macOS / Linux
uv run --package analysis_dashboard start-analysis-agent
```

```powershell
# Windows PowerShell
uv run --package analysis_dashboard start-analysis-agent
```

```cmd
:: Windows CMD
uv run --package analysis_dashboard start-analysis-agent
```

---

## Log & Output Files

| File | Description |
|---|---|
| `mcp_agent_system.log` | Flat dual-stream log (CLIENT + SERVER tags) |
| `mcp_agent_log.db` | SQLite vector store — all structured log entries with embeddings |
| `analysis_agent.log` | Analysis agent execution log (created on first dashboard or CLI use) |
| `charts/` | PNG charts saved by the analysis agent |

---

## Key Design Decisions

**MCP Sampling** — The `reflect_answer` tool on the server holds no LLM API keys. All
Critic/Corrector LLM calls are delegated to the client via `ctx.sample()`, which
executes them using the client's locally configured Gemini or Groq model.

**Relevance-based Tavily fallback** — The CRAG pipeline triggers Tavily web search when
the top KB chunk scores below 0.15, meaning the internal knowledge base has no genuinely
relevant content for the query. This ensures off-topic queries always get web-sourced answers.

**Vector log store** — Every significant agent/MCP event is persisted to `mcp_agent_log.db`
as a validated `LogEntry` with a Gemini embedding for semantic search. The store uses
hierarchical dot-separated namespaces (e.g. `logs.mcp.server.tools.crag_pipeline`) to
co-locate related traces.

**Process isolation** — The analysis dashboard is a completely separate `uv` package that
reads the SQLite DB via a read-only adapter (`store_reader.py`) without importing any code
from `agent_client`. The Neo4j sync, semantic search, and chart tools all run in the
dashboard process independently.

---

## Dependency Management

```bash
# Add a dependency to a specific package
uv add --package analysis_dashboard plotly

# Sync after any pyproject.toml change
uv sync

# Run a one-off command in package context
uv run --package analysis_dashboard python -c "import neo4j; print(neo4j.__version__)"
```

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `ConnectError` on client start | MCP server not running | Start Terminal 1 first, wait for startup |
| `no such table: log_entries` in dashboard | Client hasn't run yet | Run Terminal 2 first |
| `Neo4j connection failed` | Wrong URI or credentials | Check `.env` NEO4J_* values |
| `ModuleNotFoundError` | Dependencies not synced | Run `uv sync` from repo root |
| `No model available` | No API keys in `.env` | Add `GROQ_API_KEY` or `GEMINI_API_KEY` |
| Port 8000 already in use | Old server process still running | `taskkill /F /IM python.exe` (Windows) or `kill $(lsof -ti:8000)` (Linux/Mac) |
| `uv trampoline failed` (Windows paths with spaces) | Space in directory path | Use `python -m streamlit run` instead of `streamlit run` |
#   h l p _ g r a p h _ k n o w l e d g e _ a g e n t  
 