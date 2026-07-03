# HLP Graph Knowledge Agent

A fault-tolerant, self-healing, and fully auditable multi-agent network built on LangChain, LangGraph, FastMCP, Neo4j Aura DB, and Streamlit.

The system is structured as a `uv` workspace monorepo containing three isolated packages:

- **`mcp_server`** — FastMCP server exposing a Hierarchical CRAG knowledge tool with Tree-of-Thought relevance grading, an MCP Sampling reflection tool, and a fault injection tool for resilience testing.
- **`agent_client`** — LangChain ReAct agent with a three-tier LLM provider fallback (Gemini → Groq → local Ollama) and a three-layer runtime resilience stack: `RunnableWithRetry`, `RunnableWithFallbacks`, and a hardcoded absolute fallback. Writes structured interaction logs to a SQLite vector store.
- **`analysis_dashboard`** — Edgeless LangGraph `StateGraph` analysis agent with proxy LIME/SHAP explainability engine and a Streamlit diagnostic interface.

---

## Repository Structure

```
hlp_graph_knowledge_agent/
├── mcp_server/
│   ├── pyproject.toml
│   └── src/mcp_server/
│       └── server.py
├── agent_client/
│   ├── pyproject.toml
│   └── src/agent_client/
│       ├── client.py
│       └── log_store.py
├── analysis_dashboard/
│   ├── pyproject.toml
│   └── src/analysis_dashboard/
│       ├── agent.py
│       ├── app.py
│       ├── store_reader.py
│       ├── graph_client.py
│       ├── analytics.py
│       └── xai_engine.py
├── pyproject.toml
├── .env
├── mcp_agent_system.log
├── mcp_agent_log.db
├── explainability_audit_report.json
└── REFLECTION_STAGE4.md
```

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed globally
- A [Google Gemini API key](https://aistudio.google.com/) (primary LLM)
- A [Groq API key](https://console.groq.com/) (secondary LLM fallback + self-healing)
- [Ollama](https://ollama.com/) running locally with a pulled model (optional — only used as the final LLM fallback if neither `GEMINI_API_KEY` nor `GROQ_API_KEY` is configured; defaults to `llama3.2:3b`)
- A [Neo4j Aura DB](https://console.neo4j.io/) free instance (graph features)
- A [Tavily API key](https://tavily.com/) (optional — CRAG web fallback)

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
TAVILY_API_KEY=your_tavily_api_key_here
```

`GEMINI_API_KEY` and `GROQ_API_KEY` are both optional individually — if neither is set, the client falls back to a local Ollama model — but at least one of Gemini or Groq is strongly recommended for reliable output quality.

---

## Installation

Install all workspace packages from the repo root:

```bash
uv sync
```

---

## Running the System

The system requires three terminals running in order.

### Terminal 1 — MCP Server

```bash
uv run --package mcp_server start-server
```

```cmd
:: CMD
uv run --package mcp_server start-server
```

```powershell
# PowerShell
uv run --package mcp_server start-server
```

The server starts on `http://localhost:8000`. Verify it is running by visiting `http://localhost:8000/health` in a browser — it should return `OK`.

---

### Terminal 2 — Agent Client

```bash
uv run --package agent_client start-agent
```

```cmd
:: CMD
uv run --package agent_client start-agent
```

```powershell
# PowerShell
uv run --package agent_client start-agent
```

The client runs four queries and a dedicated resilience test. Expected output in `mcp_agent_system.log`:

```
RunnableWithRetry configured: max_attempts=3 jitter=True
RunnableWithFallbacks configured: fallbacks=[_SelfHealRunnable, _AbsoluteFallbackRunnable] exception_key='error_trace'
--- [RESILIENCE TEST] Sending fault to primary chain... ---
CRITICAL: Primary Chain failed!
--- [STEP 2] Routing to Self-Heal Chain with Error Trace... ---
HEALED: Self-Heal Chain recovered successfully via ChatGroq!
--- [RESILIENCE TEST COMPLETE] self_heal_iterations=1 ---
Resilience summary: fallback_activations=1 self_heal_iterations=1 absolute_fallback_hits=0
```

---

### Terminal 3 — Analysis Dashboard

```bash
uv run streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

```cmd
:: CMD
uv run streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

```powershell
# PowerShell
uv run streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

Open `http://localhost:8501` in a browser.

---

## Dashboard Features

### 💬 Analysis Chat
Natural language interface to the edgeless LangGraph analysis agent. Example queries:

- `Give me an overview of the system health: error rate, latency stats, and session count.`
- `Search for any sampling_request logs and report what you find.`
- `Sync all sessions to the Neo4j graph and report what was committed.`
- `Generate a full dashboard chart and save it to disk.`
- `Run an explainability audit on the most recent session and show LIME and SHAP results.`

### 🧠 Agent Reasoning
Step-by-step trace of the edgeless LangGraph node executions with dynamic `Command(goto=...)` routing paths.

### 📊 Trend Charts
Latency trend, token metrics, error frequency, and full 4-panel dashboard charts with quick-generate buttons.

### 🗺️ Graph Updates
Neo4j Aura DB graph commit log showing nodes and edges written per session.

### 🔬 XAI Audit
Explainability audit interface:
1. Select a session from the dropdown or enter a trace ID
2. Click **▶ Run Audit**
3. The engine extracts a Neo4j subgraph, runs proxy LIME on text logs, and computes proxy SHAP on numeric execution signatures
4. Results render as LIME token importance badges and a SHAP horizontal bar chart
5. Download `explainability_audit_report.json` directly from the UI

### ⚡ Resilience
Live and historical counts of fallback activations, self-heal successes, and absolute fallback hits with a trend chart over time.

---

## CRAG Knowledge Retrieval & Tree-of-Thought Grading

`query_knowledge` (in `mcp_server`) runs a Corrective RAG pipeline: expand the query into variants, retrieve candidate chunks hierarchically (domain → section → leaf), then grade each candidate for relevance before falling back to Tavily web search if internal knowledge is insufficient.

Relevance grading is done via genuine Tree-of-Thought reasoning over MCP Sampling, not a static heuristic:

1. **Branch** — for each candidate chunk, sample several independent one-sentence reasoning attempts ("thoughts") about whether the chunk helps answer the query.
2. **Evaluate** — a separate sampling call reviews all generated thoughts together, picks the strongest one, and issues a final relevance verdict.
3. **Decide** — the chunk is kept only if the evaluator's verdict says relevant.

Chunks that survive grading, or web results if the internal KB scores below threshold, are assembled into the final CRAG response.

---

## Resilience Architecture

Two independent resilience mechanisms exist in `agent_client`:

**LLM provider fallback** (which model backs the agent at all):
```
Gemini (primary)
      │ init/call fails
      ▼
Groq (secondary)
      │ init/call fails
      ▼
Ollama (local, e.g. llama3.2:3b)
```

**Runtime resilience stack** (how a single query is protected once a model is selected):
```
Primary Agent
      │
      ▼
RunnableWithRetry (max_attempts=3, exponential backoff + jitter)
      │ all retries exhausted
      ▼
_SelfHealRunnable (Groq LLM generates corrective response from error trace)
      │ self-heal fails
      ▼
_AbsoluteFallbackRunnable (never raises — logs to SQLite, returns safe error payload)
```

Fault injection is available via the `simulate_fault` MCP tool which supports: `runtime_error`, `bad_schema`, `timeout_sim`, `partial_payload`, and `random`.

---

## XAI Engine

Located at `analysis_dashboard/src/analysis_dashboard/xai_engine.py`.

**Proxy LIME** — for unstructured text logs:
- Tokenises log content and generates 30 masked variants per entry
- Measures cosine similarity drop between original and masked TF-IDF vectors
- Tokens causing the largest similarity drop are scored as high-importance

**Proxy SHAP** — for structured numeric signatures (`latency_ms`, `token_count`, `content_length`):
- Uses `IsolationForest` (scikit-learn) to flag anomalous entries
- Computes exact Shapley values via exhaustive subset enumeration (8 subsets for 3 features)
- Returns the dominant feature driving anomalous execution behaviour

---

## Key Output Files

| File | Description |
|---|---|
| `mcp_agent_system.log` | Dual-stream flat log (CLIENT + SERVER relay) |
| `mcp_agent_log.db` | SQLite vector log store with embedded interaction traces |
| `explainability_audit_report.json` | LIME token importance arrays and SHAP feature contributions for the last audit run |
| `REFLECTION_STAGE4.md` | Conceptual reflection on edgeless orchestration, XAI limitations, and fallback strategy |

---

## Dependencies

Key packages per workspace package:

**mcp_server**
- `fastmcp`, `langchain-tavily`, `python-dotenv`, `pydantic`

**agent_client**
- `langchain`, `langchain-core`, `langchain-groq`, `langchain-google-genai`, `langchain-ollama`
- `langchain-mcp-adapters`, `langgraph`, `langgraph-checkpoint-sqlite`, `mcp`
- `pydantic`, `pydantic-settings`, `aiosqlite`

**analysis_dashboard**
- `langchain`, `langgraph`, `neo4j`
- `streamlit`, `matplotlib`, `seaborn`, `pandas`
- `numpy`, `scikit-learn`

Install all dependencies with `uv sync` from the repo root.