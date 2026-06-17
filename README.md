# HLP Graph Knowledge Agent — Stage 3: Observability & Diagnostics

Extends the Stage 2 distributed MCP architecture with a full observability stack: structured vector log persistence, a semantic log analysis agent, a Neo4j Aura DB causal knowledge graph, and an interactive Streamlit diagnostic dashboard.

🌐 **GitHub Repository:** [hlp_graph_knowledge_agent](https://github.com/Anyadiegwu/hlp_graph_knowledge_agent)

---

## 📂 Repository Structure

```directory
hlp_graph_knowledge_agent/
├── 📄 pyproject.toml               # uv workspace root (3 members)
├── 📄 .env.example                 # Template for required environment keys
├── 📄 REFLECTION_STAGE3.md
├── 📄 README.md
│
├── 📁 mcp_server/                  # FastMCP server (Stage 2 + Stage 3)
│   ├── pyproject.toml
│   └── 📁 src/mcp_server/
│       └── server.py               
│
├── 📁 agent_client/                # MCP client + SQLite vector log store
│   ├── pyproject.toml
│   └── 📁 src/agent_client/
│       ├── client.py               # ReAct agent, MCP sampling, _persist() calls
│       └── log_store.py            # HLPLogStore, LogEntry schema, namespaces
│
└── 📁 analysis_dashboard/          # Log Analysis Agent + Streamlit UI
    ├── pyproject.toml
    └── 📁 src/analysis_dashboard/
        ├── agent.py                # Log Analysis Agent (7 tools)
        ├── app.py                  # Streamlit dashboard UI
        ├── analytics.py            # matplotlib / seaborn chart tools
        ├── graph_client.py         # Neo4j Aura DB client + projection logic
        └── store_reader.py         # Read-only SQLite adapter for analysis

```

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                          agent_client                           │
│  LangChain ReAct Agent                                          │
│     ├── reflect_answer_tool  ─────────────────────┐             │
│     └── query_knowledge_tool ─────────────────┐   │             │
│                                               │   │             │
│  _persist() ──► HLPLogStore (SQLite)          │   │             │
│     Namespaces:                               │   │             │
│        logs.agent.planning.reflexive_loop     │   │             │
│        logs.mcp.client.tool_invocation        │   │             │
│        logs.mcp.server.tools.crag_pipeline    │   │             │
│        logs.mcp.server.sampling_request       │   │             │
│        ... (12 namespace paths total)         │   │             │
└──────────────────────────────────────────────┼───┼──────────────┘
                                 streamable-http│   │
                                 localhost:8000 │   │
┌──────────────────────────────────────────────▼───▼──────────────┐
│                          mcp_server                             │
│  @tool reflect_answer   ← ctx.sample() ← client                 │
│  @tool query_knowledge  ← Hierarchical CRAG + ToT + Tavily      │
└─────────────────────────────────────────────────────────────────┘

                 mcp_agent_log.db  (shared SQLite file)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                       analysis_dashboard                        │
│  (separate process — reads DB, never writes to it)              │
│                                                                 │
│  Log Analysis Agent (LangChain ReAct)                           │
│     ├── semantic_log_search    ← vector cosine similarity       │
│     ├── get_log_statistics     ← aggregate counts + latency     │
│     ├── sync_graph_to_neo4j    ← project traces into Neo4j      │
│     ├── latency_trend_chart    ← matplotlib moving avg chart    │
│     ├── token_metrics_chart    ← token consumption chart        │
│     ├── error_frequency_chart  ← rolling error rate chart       │
│     └── full_dashboard_chart   ← 4-panel health dashboard       │
│                                                                 │
│  Streamlit Dashboard                                            │
│     ├── Chat tab       ← natural language → agent               │
│     ├── Reasoning tab  ← step-by-step agent trace               │
│     ├── Charts tab     ← inline rendered trend charts           │
│     └── Graph tab      ← Neo4j commit log + live summary        │
│                                     │                           │
│                                     ▼                           │
│                           Neo4j Aura DB                         │
│                  (:Session)-[:TRIGGERED]->(:AgentAction)        │
│                  (:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)  │
│                  (:MCPServerCall)-[:DEPENDS_ON]->(:MCPServerCall)│
└─────────────────────────────────────────────────────────────────┘

```

---

## 🛠️ Prerequisites

* **Python:** Version 3.11 or higher
* **Package Manager:** [uv](https://docs.astral.sh/uv/getting-started/installation/) installed globally
* **Database:** A free [Neo4j Aura DB](https://neo4j.com/cloud/aura/) instance (required for graph/network features)

---

## 🚀 Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Anyadiegwu/hlp_graph_knowledge_agent.git
cd hlp_graph_knowledge_agent

```

### 2. Configure Environment Variables

Duplicate `.env.example` into a local `.env` file depending on your operating system:

* **Bash / macOS / Linux:**
```bash
cp .env.example .env

```


* **Windows PowerShell:**
```powershell
Copy-Item .env.example .env

```


* **Windows CMD:**
```cmd
copy .env.example .env

```



> ⚠️ **Important:** Open your new `.env` file and insert your production keys. You will need at least one primary LLM key (`GROQ_API_KEY` or `GEMINI_API_KEY`), a `TAVILY_API_KEY` for web fallbacks, and your `NEO4J_*` credentials.

### 3. Sync Workspace Dependencies

Leverage `uv` to automatically install and resolve all internal package dependencies:

```bash
uv sync

```

---

## 💻 Running the System

You will need **three terminal windows** open simultaneously, all pointed at the repository root. Follow the steps below in chronological order.

### 📌 Step 1: Initialize MCP Server (Terminal 1)

* **Bash / macOS / Linux:** `uv run --package mcp_server start-server`
* **Windows PowerShell:** `uv run --package mcp_server start-server`
* **Windows CMD:** `uv run --package mcp_server start-server`

Ensure the startup completes successfully by tracking the logs:

```text
[SERVER] [INFO] Starting ThinkingAgentServer (streamable-http) on http://0.0.0.0:8000 ...
INFO:     Application startup complete.

```

To verify the health of your runtime, open another window or tab and query the health route:

```bash
curl http://localhost:8000/health
# Expected Output: OK

```

### 📌 Step 2: Spin Up Agent Client (Terminal 2)

* **Bash / macOS / Linux:** `uv run --package agent_client start-agent`
* **Windows PowerShell:** `uv run --package agent_client start-agent`
* **Windows CMD:** `uv run --package agent_client start-agent`

This client script initializes tool mappings, queries the baseline knowledge base via the ReAct + CRAG pipelines, and seeds your local database outputs (`mcp_agent_system.log` and `mcp_agent_log.db`).

> 🛑 **Note:** The client must complete at least **one full initialization pass** before starting the dashboard.

### 📌 Step 3: Launch Streamlit Dashboard (Terminal 3)

* **Bash / macOS / Linux:** `uv run --package analysis_dashboard python -m streamlit run analysis_dashboard/src/analysis_dashboard/app.py`
* **Windows PowerShell:** `uv run --package analysis_dashboard python -m streamlit run analysis_dashboard/src/analysis_dashboard/app.py`
* **Windows CMD:** `uv run --package analysis_dashboard python -m streamlit run analysis_dashboard/src/analysis_dashboard/app.py`

Once active, open your browser and navigate to: **`http://localhost:8501`**

---

### 💡 Optional — Standalone Analysis Agent CLI

If you prefer testing inside your terminal rather than via a web browser, use the built-in diagnostic loop to generate a localized `analysis_agent.log`:

* **Bash / macOS / Linux:** `uv run --package analysis_dashboard start-analysis-agent`
* **Windows PowerShell:** `uv run --package analysis_dashboard start-analysis-agent`
* **Windows CMD:** `uv run --package analysis_dashboard start-analysis-agent`

---

## 📁 Log & Output Files Reference

| File / Folder Path | Type | Description |
| --- | --- | --- |
| `mcp_agent_system.log` | Text Log | Flat dual-stream trace capturing client and server exchanges via text tags. |
| `mcp_agent_log.db` | SQLite DB | Vector repository containing all structured entries coupled with vector embeddings. |
| `analysis_agent.log` | Text Log | Isolated diagnostics run record generated during dashboard or CLI interaction loops. |

---

## 📐 Key Design Decisions

* **MCP Sampling Architecture:** The `@tool reflect_answer` module deployed on the server is intentionally kept decoupled from LLM API authorization keys. Instead, complex Critic/Corrector verification requests are piped back down to the calling client through `ctx.sample()`.
* **Thresholded Tavily Fallback:** To maintain rigorous query scoping, the Corrective RAG (CRAG) loop triggers a Tavily web search exclusively when the highest-ranking local Vector KB chunk drops below a semantic confidence score of `0.15`.
* **Hierarchical Vector Log Namespaces:** Agent operational steps pass through runtime type validation as a `LogEntry` before writing to `mcp_agent_log.db`. Records are organized through structural namespaces (e.g., `logs.mcp.server.tools.crag_pipeline`) for efficient query lookups.
* **Process Isolation:** The diagnostic engine dashboard is split into an independent workspace package. It accesses database updates safely through a read-only bridge module (`store_reader.py`) ensuring zero transactional conflicts with write loops inside the active client application.

---

## 📦 Dependency Management Cheatsheet

```bash
# Add a third-party dependency to a designated package scope
uv add --package analysis_dashboard plotly

# Re-align workspace environments after changing a local pyproject.toml configuration file
uv sync

# Run a localized, isolated command directly within a package environment context
uv run --package analysis_dashboard python -c "import neo4j; print(neo4j.__version__)"

```

---

## 🛠️ Common Errors & Troubleshooting

| Exception / Error | Root Cause Analysis | Corrective Action |
| --- | --- | --- |
| ❌ `ConnectError` on client startup | The underlying MCP server engine has not finished initializing. | Start **Terminal 1** first, verify it states *Application startup complete*, then retry. |
| ❌ `no such table: log_entries` | The Streamlit dashboard is attempting to read an unitialized database. | Run **Terminal 2** (the agent client) at least once to build the table schemas. |
| ❌ `Neo4j connection failed` | Missing or invalid authentication configurations inside the `.env` configuration file. | double-check your `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` environment strings. |
| ❌ `ModuleNotFoundError` | Virtual environments or package dependencies are out of sync. | Run a clean `uv sync` execution from your workspace root directory. |
| ❌ `No model available` | The runtime environment cannot locate usable LLM access configurations. | Populate your local `.env` target file with a valid `GROQ_API_KEY` or `GEMINI_API_KEY`. |
| ❌ Port `8000` already in use | A dangling, orphaned server process from a previous runtime session is locking the port. | **Windows:** `taskkill /F /IM python.exe`<br>

<br>**macOS / Linux:** `kill $(lsof -ti:8000)` |
| ❌ `uv trampoline failed` | Windows path parser constraints encountered spaces within directory paths. | Avoid global system binaries inside paths with spaces; use `python -m streamlit run` instead. |