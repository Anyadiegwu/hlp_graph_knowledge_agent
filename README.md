# HLP Graph Knowledge Agent

A fault-tolerant, self-healing, and fully auditable multi-agent network built on LangChain, LangGraph, FastMCP, Neo4j Aura DB, Supabase (Postgres + pgvector), Redis, and Streamlit — with an x402 stablecoin billing layer and Algorithmic FinOps governance on top.

The system is structured as a `uv` workspace monorepo containing four isolated packages:

- **`mcp_server`** — FastMCP server exposing a Hierarchical CRAG knowledge tool with Tree-of-Thought relevance grading, an MCP Sampling reflection tool, a fault injection tool for resilience testing, and an x402 paywall in front of `query_knowledge`.
- **`agent_client`** — LangChain ReAct agent with a three-tier LLM provider fallback (Gemini → Groq → local Ollama), a three-layer runtime resilience stack (`RunnableWithRetry`, `RunnableWithFallbacks`, a hardcoded absolute fallback), a Supabase/pgvector-backed hierarchical log store, a Redis-backed semantic cache in front of the MCP sampling handler, an x402 payment-retry client for calling paywalled server tools, and a reciprocal x402 sampling wall that charges untrusted MCP servers for this client's own LLM compute.
- **`analysis_dashboard`** — Edgeless LangGraph `StateGraph` analysis agent with proxy LIME/SHAP explainability engine, Redis-cached Neo4j subgraph and XAI compute paths, and a Streamlit diagnostic interface with live cache telemetry and a FinOps/x402 billing tab.
- **`x402_core`** — Shared, framework-agnostic x402 stablecoin billing and Algorithmic FinOps governance primitives (wallet signing, facilitator client, payment schemas, budget governor, shared audit trail) used by both `mcp_server` and `agent_client`.

---

## Repository Structure

```
hlp_graph_knowledge_agent/
├── mcp_server/
│   ├── pyproject.toml
│   └── src/mcp_server/
│       ├── server.py           # FastMCP tools/resources, query_knowledge (paywalled)
│       ├── crag_index.py       # RedisVL vector index for hierarchical retrieval
│       └── paywall.py          # @require_payment decorator — MCP-transport x402 paywall
├── agent_client/
│   ├── pyproject.toml
│   └── src/agent_client/
│       ├── client.py           # ReAct agent, Tier 1 semantic cache, sampling handler + reciprocal 402 wall
│       ├── log_store.py        # Supabase/pgvector-backed hierarchical log store
│       └── x402_client.py      # Client-side payment-retry helper for calling paywalled tools
├── analysis_dashboard/
│   ├── pyproject.toml
│   └── src/analysis_dashboard/
│       ├── agent.py            # Edgeless LangGraph agent, Tier 2 cache
│       ├── app.py              # Streamlit UI, Tier 3 cache, cache + FinOps telemetry dashboard
│       ├── store_reader.py     # Shared read-only Supabase log reader
│       ├── graph_client.py     # Neo4j graph client + log-to-graph projection
│       ├── analytics.py        # Matplotlib/Seaborn chart generation
│       ├── xai_engine.py       # Proxy LIME / proxy SHAP explainability engine
│       └── finops_view.py      # Wallet balance, ledger summary, audit-trail readers for the FinOps tab
├── x402_core/
│   ├── pyproject.toml
│   ├── generate_wallet.py          # Prints a fresh Base Sepolia wallet (address + private key)
│   ├── generate_finops_audit.py    # Produces finops_compliance_audit.json from a real (or --offline) run
│   ├── src/x402_core/
│   │   ├── schemas.py           # PaymentRequirements, X402Challenge, X402PaymentPayload, receipts
│   │   ├── constants.py         # Base Sepolia / USDC / facilitator constants, usd<->atomic helpers
│   │   ├── wallet.py            # AgentWallet — signs EIP-3009 transferWithAuthorization payloads
│   │   ├── facilitator.py       # Async client for the Xpay Staging Facilitator (/verify, /settle)
│   │   ├── finops.py            # FinOpsGovernor — pre-flight per-session USD budget ceiling
│   │   ├── audit.py             # Shared Redis-backed 402 challenge/verify/settle audit trail
│   │   └── exceptions.py        # X402PaymentRequiredError, X402VerificationError, X402SettlementError, FinOpsBudgetExceededException
│   └── tests/                   # Unit tests for wallet signing, schemas, FinOps governor
├── charts/                    # Auto-generated PNG charts (latency trend, tokens, errors, dashboard)
├── pyproject.toml
├── .env.example
├── mcp_agent_system.log
├── cache_performance_audit.json
├── explainability_audit_report.json
├── finops_compliance_audit.json
├── REFLECTION_STAGE5.md
└── REFLECTION_STAGE6.md
```

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed globally
- A [Google Gemini API key](https://aistudio.google.com/) (primary LLM)
- A [Groq API key](https://console.groq.com/) (secondary LLM fallback + self-healing)
- [Ollama](https://ollama.com/) running locally with a pulled model (optional — only used as the final LLM fallback if neither `GEMINI_API_KEY` nor `GROQ_API_KEY` is configured; defaults to `llama3.2:3b`)
- A [Neo4j Aura DB](https://console.neo4j.io/) free instance (graph features)
- A [Supabase](https://supabase.com/) project with the `pgvector` extension enabled (log store)
- A [Redis](https://redis.io/) instance — e.g. [Redis Cloud](https://redis.io/cloud/) free tier — reachable over the network (multi-tier caching, and shared by the x402 FinOps ledger/audit trail)
- A [Tavily API key](https://tavily.com/) (optional — CRAG web fallback)
- A **Base Sepolia testnet wallet** funded via the [Circle faucet](https://faucet.circle.com/) (optional — only required to exercise real, non-`--offline` x402 payments; the system runs without it, issuing 402 challenges that just can't be paid)

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
GEMINI_MODEL_NAME=gemini-3.5-flash-lite
USE_GROQ=true
MODEL_TEMPERATURE=0.0
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=your_neo4j_username_here
NEO4J_PASSWORD=your_neo4j_aura_password_here
SUPABASE_DB_URL=your_pooled_connection_string_here
REDIS_URL=redis://default:your_actual_password@your_actual_host:your_port
AGENT_WALLET_PRIVATE_KEY=0xyour_testnet_wallet_private_key_here
SERVER_WALLET_ADDRESS=0xyour_server_payout_wallet_address_here
AGENT_SAMPLING_PAY_TO_ADDRESS=0xyour_agent_payout_wallet_address_here

```

`GEMINI_API_KEY` and `GROQ_API_KEY` are both optional individually — if neither is set, the client falls back to a local Ollama model — but at least one of Gemini or Groq is strongly recommended for reliable output quality.

`SUPABASE_DB_URL` should be the **pooled** (transaction-mode / pgbouncer) connection string from your Supabase project settings — the client connects with `statement_cache_size=0` to stay compatible with pgbouncer's prepared-statement handling.

`REDIS_URL` is shared across all three cache tiers, and also backs the x402 FinOps ledger and audit trail; no separate credentials are needed per tier.

The x402/FinOps keys are optional in the sense that the system still runs without them — `query_knowledge` will simply return an unpayable 402 challenge, and the FinOps tab will show empty telemetry — but a funded `AGENT_WALLET_PRIVATE_KEY` is required to actually exercise a real payment end to end.

Generate a fresh testnet wallet with:

```bash
uv run --package x402_core python x402_core/generate_wallet.py
```

This prints an address and private key straight to the terminal (never written to disk or logged) — fund the address at the [Circle faucet](https://faucet.circle.com/), then paste the private key into `AGENT_WALLET_PRIVATE_KEY`.

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

Watch for `Tier 1 semantic cache HIT` / `MISS` log lines on repeated or paraphrased queries — this confirms the Redis-backed semantic cache is intercepting the sampling handler correctly.

If `AGENT_WALLET_PRIVATE_KEY` is configured, watch also for the client transparently absorbing a 402 challenge from `query_knowledge`: an initial call returns an `X402Challenge`, `agent_client.x402_client` signs an EIP-3009 authorization and retries with `x_payment` populated, and the retried call returns the real CRAG result.

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
Latency trend, token metrics, error frequency, and full 4-panel dashboard charts with quick-generate buttons. Chart-backing queries are cached in Tier 3 (Streamlit UI cache) to avoid recomputation on every re-render.

Every generated chart is saved as a timestamped PNG to `charts/` at the repo root (e.g. `charts/latency_trend_20260717_115737.png`), so each run is preserved rather than overwritten.

### 🗺️ Graph Updates
Neo4j Aura DB graph commit log showing nodes and edges written per session. Subgraph fetches are cached in Tier 2.

### 🔬 XAI Audit
Explainability audit interface:
1. Select a session from the dropdown or enter a trace ID
2. Click **▶ Run Audit**
3. The engine extracts a Neo4j subgraph (Tier 2 cached), runs proxy LIME on text logs, and computes proxy SHAP on numeric execution signatures (Tier 2 cached)
4. Results render as LIME token importance badges and a SHAP horizontal bar chart
5. Download `explainability_audit_report.json` directly from the UI

### ⚡ Resilience
Live and historical counts of fallback activations, self-heal successes, and absolute fallback hits with a trend chart over time.

### 🧮 Cache Telemetry
Live panel showing, per tier:
- Cache hit rate (%)
- Total tokens saved and estimated dollar cost savings (Tier 1 only)
- Average latency for cache hits vs. cache misses
- A manual **flush** control per cache namespace (Tier 1 / Tier 2 / Tier 3 / all), for clearing stale entries during testing

### 💸 x402 / FinOps
Live wallet telemetry, spend/earn totals, spend velocity, and the chronological 402 challenge → verify → settle audit stream, shared across the MCP server's paywall and the agent client's reciprocal sampling wall via one Redis-backed audit trail:
- Wallet USDC balance (read-only `eth_call` against Base Sepolia — never signs or sends)
- Cumulative USDC spent and earned, and net position
- Current session spend velocity (USD/hour)
- A settlement bar chart (red = spent, green = earned)
- A scrollable, chronological stream of every `challenge_issued`, `verify_attempted`, `settlement_recorded` (and their `reciprocal_*` counterparts) event
- A **🔄 Refresh FinOps Telemetry** button to re-fetch the wallet balance and audit trail on demand

---

## CRAG Knowledge Retrieval & Tree-of-Thought Grading

`query_knowledge` (in `mcp_server`) runs a Corrective RAG pipeline: expand the query into variants, retrieve candidate chunks hierarchically (domain → section → leaf), then grade each candidate for relevance before falling back to Tavily web search if internal knowledge is insufficient.

Relevance grading is done via genuine Tree-of-Thought reasoning over MCP Sampling, not a static heuristic:

1. **Branch** — for each candidate chunk, sample several independent one-sentence reasoning attempts ("thoughts") about whether the chunk helps answer the query.
2. **Evaluate** — a separate sampling call reviews all generated thoughts together, picks the strongest one, and issues a final relevance verdict.
3. **Decide** — the chunk is kept only if the evaluator's verdict says relevant.

Chunks that survive grading, or web results if the internal KB scores below threshold, are assembled into the final CRAG response. Every sampling call in this pipeline passes through the Tier 1 Redis semantic cache before reaching the LLM provider.

`query_knowledge` is also the system's paywalled resource — see **x402 Stablecoin Billing & Algorithmic FinOps Governance** below — and runs a FinOps pre-flight budget check before retrieval even starts, independent of the x402 payment itself.

---

## x402 Stablecoin Billing & Algorithmic FinOps Governance

The `x402_core` package implements the [x402](https://x402.org) HTTP-402 payment pattern — adapted to travel over MCP tool arguments and MCP sampling metadata rather than raw HTTP headers, since MCP transports don't expose header injection the way a REST framework would — plus a shared, pre-flight USD budget governor.

### Server-side paywall (`mcp_server/paywall.py`)

`@require_payment(resource, price_usd)` wraps an `@mcp.tool()` function and adds an `x_payment: str = ""` keyword argument to it. Per call:

1. **No `x_payment` presented** → return a structured `X402Challenge` JSON in place of the tool's real result. This *is* the 402.
2. **`x_payment` presented** → verify with the Xpay facilitator's `/verify` endpoint first (cheap, no chain write). Invalid → raise `X402VerificationError`.
3. **Verified** → run the wrapped tool for real.
4. **Tool succeeded** → settle via `/settle` (relayed on-chain, gas-sponsored by Xpay) and append a structured entry to the shared FinOps audit trail.

`query_knowledge` is the paywalled resource, priced at `QUERY_KNOWLEDGE_PRICE_USD` (default `0.01` USDC/call).

### Client-side payment retry (`agent_client/x402_client.py`)

When a remote MCP tool call comes back as a parsed `X402Challenge` instead of its normal result, `invoke_with_payment_retry` signs an EIP-3009 authorization with the agent's own wallet (`AGENT_WALLET_PRIVATE_KEY`) and retries the call with `x_payment` populated — this is what lets the agent transparently pay for `query_knowledge` without any special-casing in the ReAct loop itself.

### Reciprocal sampling wall (`agent_client/client.py`)

The agent client's own `sampling_callback` — which backs any MCP server's sampling requests against this client's local LLM — charges third-party callers for that compute. Any caller whose `caller_server` isn't in `TRUSTED_SAMPLING_SERVERS` (our own first-party server, by default) must present a valid x402 payment in the sampling request's `metadata` field before the client will spend its local LLM budget on their behalf; unpaid/untrusted requests are refused with a raised `X402PaymentRequiredError` carrying the challenge, priced at `SAMPLING_PRICE_USD` (default `0.002` USDC/request).

### Wallet signing (`x402_core/wallet.py`)

`AgentWallet` signs `transferWithAuthorization` (EIP-3009) payloads off-chain via EIP-712 typed-data signing — no native Base Sepolia ETH ever leaves the wallet and no gas is spent directly by it; the signed authorization is handed to the Xpay facilitator, which relays it on-chain as the gas sponsor. Every authorization gets a fresh random 32-byte nonce (`secrets.token_hex(32)`), and every signature is self-verified via `Account.recover_message` immediately after signing, before it's ever sent anywhere.

### Facilitator client (`x402_core/facilitator.py`)

`XpayFacilitatorClient` splits payment handling into `/verify` (checks the signature and authorization window, no chain write) and `/settle` (relays on-chain, gas-sponsored, returns a transaction hash once the transfer lands). Splitting the two lets the server reject an obviously-bad payload cheaply before paying for the slower settlement relay. `/settle` is the only result the paywall trusts as the basis for granting access — `/verify` is advisory only, since a payload can pass verification and still fail (or race) at settlement. See `REFLECTION_STAGE6.md` for a fuller discussion of this verify/settle race window, signature replay protections, chain-reorg assumptions, and session/ledger boundary trust.

### Algorithmic FinOps Governance (`x402_core/finops.py`)

`FinOpsGovernor` is a **pre-flight** budget interceptor, independent of the x402 payment layer itself — it governs LLM *spend* (token cost), not on-chain payment for tool access. Every LLM-touching step calls `governor.preflight(session_id, projected_usd)` *before* firing the LLM call; if the session's running spend plus the projected cost would exceed `FINOPS_MAX_USD_PER_CHAIN`, it raises `FinOpsBudgetExceededException` and the call site is expected to fall back to a cached or structurally-safe response rather than let the exception surface as a generic failure. Spend is tracked in Redis (shared ledger across the server and client processes) when `REDIS_URL` is set, or an in-memory dict otherwise. The governor also tracks spend velocity (USD/hour) over a rolling window, surfaced live in the dashboard's FinOps tab.

### Shared audit trail (`x402_core/audit.py`)

Both the MCP server's paywall and the agent client's reciprocal sampling wall append to the **same** Redis list (`hlp:finops:audit`), so the dashboard can render one chronological stream of challenges issued, payments verified, and settlements confirmed on both sides of the protocol, regardless of which process wrote them.

### Generating a compliance trace

`x402_core/generate_finops_audit.py` produces `finops_compliance_audit.json` — a structural trace of one paywalled operation end to end (challenge → signed payment metadata → verify result → settlement result):

```bash
# From the repo root, against a real funded wallet + live facilitator:
uv run --package x402_core python x402_core/generate_finops_audit.py \
    --resource query_knowledge --price-usd 0.01

# Without network access to the facilitator, produce a schema-accurate
# structural example instead (NOT a substitute for a real run):
uv run --package x402_core python x402_core/generate_finops_audit.py --offline
```

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
_AbsoluteFallbackRunnable (never raises — logs to Supabase, returns safe error payload)
```

Fault injection is available via the `simulate_fault` MCP tool which supports: `runtime_error`, `bad_schema`, `timeout_sim`, `partial_payload`, and `random`.

---

## Data Tier: Supabase + pgvector

`agent_client/log_store.py` and `analysis_dashboard/store_reader.py` implement the log store on top of Postgres via an async connection pool:

- **Relational schema**: `sessions` and `log_entries` tables, linked by foreign key. Tool invocations, sampling requests, and errors are all rows in `log_entries`, distinguished by an `interaction_type` column.
- **Vector search**: log content embeddings are stored in a `pgvector` column (`output_dimensionality=768`) and queried via cosine-distance similarity search directly in Postgres.
- **Connection pooling**: both packages connect through `asyncpg.create_pool(..., statement_cache_size=0)` for pgbouncer-compatible pooled access, supporting concurrent multi-agent executions without exhausting connections.

---

## Multi-Tiered Redis Caching Topology

```
[MCP Client Sampling Handler] ---> Tier 1: Vector-Based Semantic Cache (RedisSemanticCache)
                                             │
[Edgeless Log Analysis Agent] ---> Tier 2: Subgraph & Compute Cache (Neo4j subgraphs, SHAP/LIME)
                                             │
[Streamlit Presentation UI]   ---> Tier 3: Slow Analytics Query Cache
```

- **Tier 1** uses a strict `distance_threshold` (0.15) to guard against false-positive hits returning a stale or unrelated cached generation.
- **Tier 2** and **Tier 3** use standard Redis key-value caching with TTLs, keyed by namespace (`hlp:cache:tier2:...`, `hlp:cache:tier3:...`) so each tier can be flushed independently from the dashboard's Cache Invalidation Control Hub.

The same `REDIS_URL` instance also backs the x402 FinOps ledger (`hlp:finops:ledger:<session_id>`) and shared audit trail (`hlp:finops:audit`) — separate namespaces, same Redis connection.

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

Both computations are wrapped in the Tier 2 Redis cache, so repeated audits of the same trace return instantly instead of re-running SHAP/LIME from scratch.

---

## Key Output Files

| File | Description |
|---|---|
| `mcp_agent_system.log` | Dual-stream flat log (CLIENT + SERVER relay) |
| `cache_performance_audit.json` | Trace of a Tier 1 cache miss followed by a semantic cache hit, with latency delta |
| `explainability_audit_report.json` | LIME token importance arrays and SHAP feature contributions for the last audit run |
| `finops_compliance_audit.json` | End-to-end trace of one x402 paywalled operation: 402 challenge, signed payment metadata, verify result, settlement result |
| `charts/*.png` | Timestamped latency/token/error/dashboard charts generated from the Trend Charts tab |
| `REFLECTION_STAGE5.md` | Reflection on semantic-cache threshold tuning, cloud state synchronization, and local vs. centralized caching trade-offs |
| `REFLECTION_STAGE6.md` | Reflection on cosine vs. Euclidean vector indexing, and cross-layer attack surfaces (signature replay, verify/settle race, chain reorgs, session/ledger trust) in the x402 flow |

---

## Dependencies

Key packages per workspace package:

**x402_core**
- `pydantic`, `httpx`, `eth-account` (EIP-712 signing), `redis`

**mcp_server**
- `fastmcp`, `langchain-tavily`, `python-dotenv`, `pydantic`, `numpy`, `redisvl`
- `asyncpg`, `pgvector`, `httpx`
- `x402_core` (workspace dependency)

**agent_client**
- `langchain`, `langchain-core`, `langchain-groq`, `langchain-google-genai`, `langchain-ollama`
- `langchain-mcp-adapters`, `langgraph`, `mcp`
- `pydantic`, `pydantic-settings`
- `asyncpg`, `pgvector`, `psycopg2-binary` (Supabase/pgvector access)
- `redis`, `langchain-redis` (Tier 1 semantic cache)
- `httpx`, `x402_core` (workspace dependency)

**analysis_dashboard**
- `langchain`, `langgraph`, `neo4j`
- `streamlit`, `matplotlib`, `seaborn`, `pandas`
- `numpy`, `scikit-learn`
- `asyncpg`, `pgvector`, `psycopg2-binary` (shared Supabase log reader)
- `redis` (Tier 2 + Tier 3 caching)
- `httpx`, `x402_core` (workspace dependency, via `finops_view.py`)

Install all dependencies with `uv sync` from the repo root.