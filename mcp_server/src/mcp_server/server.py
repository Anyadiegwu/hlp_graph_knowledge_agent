from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dotenv import find_dotenv, load_dotenv
from typing import Any

from fastmcp import Context, FastMCP
from langchain_tavily import TavilySearch
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from pydantic import BaseModel, Field

from mcp_server.crag_index import CragMultiIndex
from mcp_server.paywall import require_payment
from x402_core.exceptions import FinOpsBudgetExceededException
from x402_core.finops import get_governor


load_dotenv(find_dotenv(".env"))

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [SERVER] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ThinkingAgentServer",
    instructions=(
        "Production MCP server exposing a Sampling-based Reflection tool, "
        "A Hierarchical CRAG knowledge tool, and a fault-injection tool for "
        "Resilience Testing."
    ),
)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(_: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

_tavily_key = os.getenv("TAVILY_API_KEY", "")
_tavily = TavilySearch(max_results=3, include_images=False) if _tavily_key else None

class _EmbeddingProvider:
    """Lazy-loaded embedding provider with Gemini → zero-vector stub fallback."""
    def __init__(self) -> None:
        self._model = None
        self._tried_init = False

    def _ensure_model(self) -> None:
        if self._tried_init:
            return
        self._tried_init = True
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set — CRAG will use zero-vector stub embeddings "
                "(cosine scores will be meaningless until a key is configured)."
            )
            return
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self._model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
            output_dimensionality=768,
        )
        logger.info("CRAG embedding model loaded: models/gemini-embedding-001")

    def embed(self, text: str) -> list[float]:
        self._ensure_model()
        if self._model is None:
            return [0.0] * 768
        return self._model.embed_query(text)

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> tuple[list[list[float]], list[bool]]:
        """Returns (vectors, ok_flags) — ok_flags[i] is False if that item's
        sub-batch fell back to a zero-vector stub."""
        self._ensure_model()
        if self._model is None:
            return [[0.0] * 768 for _ in texts], [False] * len(texts)

        vectors: list[list[float]] = []
        ok_flags: list[bool] = []
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i:i + batch_size]
            try:
                vectors.extend(self._model.embed_documents(sub_batch))
                ok_flags.extend([True] * len(sub_batch))
            except Exception as exc:
                logger.error(
                    "Batch embedding failed for items %d-%d: %s — using zero-vector stubs.",
                    i, i + len(sub_batch), exc,
                )
                vectors.extend([[0.0] * 768 for _ in sub_batch])
                ok_flags.extend([False] * len(sub_batch))
        return vectors, ok_flags

_embedder = _EmbeddingProvider()

KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "langchain": {
        "summary": "LangChain is a framework for building LLM-powered applications.",
        "keywords": ["langchain", "framework", "llm", "chain", "agent"],
        "sections": {
            "agents": {
                "summary": "LangChain agent patterns and factories.",
                "keywords": ["agent", "react", "create_agent", "tool", "loop"],
                "chunks": [
                    {
                        "id": "lc-a-001",
                        "title": "ReAct Agent Loop",
                        "content": (
                            "LangChain agents use a ReAct loop: Thought → Action → Observation. "
                            "The create_agent factory accepts a model, a list of tools, and a "
                            "system prompt. It returns a LangGraph compiled graph that streams "
                            "events including model outputs and tool calls."
                        ),
                        "keywords": ["react", "thought", "action", "observation", "create_agent", "langgraph"],
                    },
                    {
                        "id": "lc-a-002",
                        "title": "Tool Binding",
                        "content": (
                            "Tools are bound to the model via model.bind_tools(tools). "
                            "Each tool must have a name, description, and args_schema. "
                            "The @tool decorator from langchain.tools creates StructuredTool instances."
                        ),
                        "keywords": ["bind_tools", "tool", "decorator", "structured_tool", "args_schema"],
                    },
                ],
            },
            "resilience": {
                "summary": "LangChain resilience primitives: RunnableWithRetry and RunnableWithFallbacks.",
                "keywords": ["retry", "fallback", "resilience", "runnable", "fault", "recovery"],
                "chunks": [
                    {
                        "id": "lc-r-001",
                        "title": "RunnableWithRetry",
                        "content": (
                            "RunnableWithRetry wraps any Runnable with automatic retry logic. "
                            "Configure max_attempt_number, wait_exponential_jitter=True for "
                            "exponential backoff with jitter. Handles transient network faults, "
                            "HTTP 429 rate limits, and socket drops before bubbling the exception."
                        ),
                        "keywords": ["retry", "runnable_with_retry", "exponential", "jitter", "backoff", "429"],
                    },
                    {
                        "id": "lc-r-002",
                        "title": "RunnableWithFallbacks",
                        "content": (
                            "RunnableWithFallbacks chains a primary runnable with one or more "
                            "fallback runnables. The exception_key argument injects the caught "
                            "exception string into the next runnable's input. A final hardcoded "
                            "fallback ensures the system never crashes unhandled."
                        ),
                        "keywords": ["fallback", "runnable_with_fallbacks", "exception_key", "chain", "self_heal"],
                    },
                ],
            },
            "chains": {
                "summary": "LangChain LCEL chains and composition patterns.",
                "keywords": ["chain", "lcel", "pipe", "prompt", "runnable"],
                "chunks": [
                    {
                        "id": "lc-c-001",
                        "title": "LCEL Pipe Operator",
                        "content": (
                            "LCEL uses the | operator to compose Runnables: "
                            "prompt | model | parser. Each step is a Runnable with "
                            "invoke(), stream(), and batch() methods."
                        ),
                        "keywords": ["lcel", "pipe", "runnable", "invoke", "stream", "batch"],
                    },
                ],
            },
        },
    },
    "mcp": {
        "summary": "Model Context Protocol — open standard for agent-tool communication.",
        "keywords": ["mcp", "protocol", "transport", "tool", "resource"],
        "sections": {
            "sampling": {
                "summary": "MCP Sampling — server-initiated LLM calls delegated to the client.",
                "keywords": ["sampling", "llm", "create_message", "client", "server", "delegate"],
                "chunks": [
                    {
                        "id": "mcp-s-001",
                        "title": "MCP Sampling Overview",
                        "content": (
                            "MCP Sampling inverts the normal request direction: the server "
                            "sends a CreateMessageRequest to the client, which executes the "
                            "LLM call using its locally configured model and returns the result. "
                            "This keeps all API keys and billing on the client side."
                        ),
                        "keywords": ["sampling", "create_message", "api_key", "billing", "invert"],
                    },
                    {
                        "id": "mcp-s-002",
                        "title": "Sampling Security",
                        "content": (
                            "Because the server crafts the prompts sent to the client's LLM, "
                            "clients should validate incoming sampling requests before forwarding "
                            "them to the model. A malicious server could craft prompts designed "
                            "to exfiltrate system prompt content or inject instructions."
                        ),
                        "keywords": ["security", "validate", "malicious", "prompt_injection", "exfiltrate"],
                    },
                ],
            },
            "transport": {
                "summary": "MCP transport layers: stdio, SSE, streamable-http.",
                "keywords": ["transport", "stdio", "sse", "http", "streamable"],
                "chunks": [
                    {
                        "id": "mcp-t-001",
                        "title": "Streamable HTTP Transport",
                        "content": (
                            "The streamable-http transport multiplexes requests and SSE streams "
                            "over a single HTTP endpoint. In FastMCP, enable it with "
                            "mcp.run(transport='streamable-http', port=8000). "
                            "The client connects using transport='streamable-http' in its config."
                        ),
                        "keywords": ["streamable-http", "sse", "multiplex", "fastmcp", "port", "http"],
                    },
                ],
            },
            "resources": {
                "summary": "MCP Resources — read-only data endpoints addressable by URI.",
                "keywords": ["resource", "uri", "read", "template", "data"],
                "chunks": [
                    {
                        "id": "mcp-r-001",
                        "title": "Resource URI Templates",
                        "content": (
                            "Resources use URI templates like knowledge://domain/{query}. "
                            "FastMCP maps path parameters automatically to function arguments. "
                            "Resources are read via resources/read in the MCP protocol."
                        ),
                        "keywords": ["uri", "template", "path_param", "resources/read", "fastmcp"],
                    },
                ],
            },
        },
    },
    "fastmcp": {
        "summary": "FastMCP — high-level Python framework wrapping the MCP SDK.",
        "keywords": ["fastmcp", "framework", "python", "decorator", "server"],
        "sections": {
            "context": {
                "summary": "FastMCP Context object for server-side tool utilities.",
                "keywords": ["context", "ctx", "log", "progress", "sample", "elicit"],
                "chunks": [
                    {
                        "id": "fm-c-001",
                        "title": "Context Injection",
                        "content": (
                            "The Context object is injected into tools that declare "
                            "'ctx: Context' as a parameter. It provides: ctx.info(), "
                            "ctx.debug(), ctx.error() for forwarding logs to the client; "
                            "ctx.report_progress() for progress updates; "
                            "ctx.sample() for MCP Sampling requests."
                        ),
                        "keywords": ["ctx", "info", "debug", "error", "report_progress", "sample", "inject"],
                    },
                    {
                        "id": "fm-c-002",
                        "title": "ctx.sample() Usage",
                        "content": (
                            "ctx.sample(messages, max_tokens=512) sends a CreateMessageRequest "
                            "to the client. The client's sampling_callback executes the LLM call "
                            "and returns a SamplingResult. The server accesses the text via "
                            "result.text or result.content.text."
                        ),
                        "keywords": ["ctx.sample", "create_message", "sampling_callback", "result", "text"],
                    },
                ],
            },
        },
    },
    "crag": {
        "summary": "Corrective RAG — retrieval pipeline with grading and web fallback.",
        "keywords": ["crag", "retrieval", "grading", "fallback", "rag"],
        "sections": {
            "pipeline": {
                "summary": "CRAG pipeline stages: expand, retrieve, grade, fallback.",
                "keywords": ["expand", "retrieve", "grade", "fallback", "pipeline"],
                "chunks": [
                    {
                        "id": "crag-p-001",
                        "title": "Multi-Query Expansion",
                        "content": (
                            "Multi-query expansion generates N semantically distinct rephrasings "
                            "of the user's original query to improve retrieval recall. Each variant "
                            "targets a different aspect or phrasing of the same information need."
                        ),
                        "keywords": ["multi-query", "expansion", "rephrase", "semantic", "recall", "variant"],
                    },
                    {
                        "id": "crag-p-002",
                        "title": "Hierarchical Indexing",
                        "content": (
                            "Hierarchical indexing searches top-level summaries first to identify "
                            "relevant domains, then narrows to section-level summaries, and finally "
                            "retrieves leaf chunks. This prunes irrelevant branches early and "
                            "reduces noise in the final result set."
                        ),
                        "keywords": ["hierarchical", "index", "domain", "section", "leaf", "prune"],
                    },
                    {
                        "id": "crag-p-003",
                        "title": "Tree-of-Thought Evaluation",
                        "content": (
                            "ToT evaluation runs 3 independent reasoning chains to score each "
                            "retrieved chunk for relevance to the query. Each chain reasons from "
                            "a different angle: semantic overlap, query intent coverage, and "
                            "factual utility. The majority verdict across chains determines "
                            "whether the chunk is accepted or rejected."
                        ),
                        "keywords": ["tot", "tree-of-thought", "reasoning", "chain", "majority", "relevance"],
                    },
                ],
            },
        },
    },
}

CRAG_MULTI_INDEX = CragMultiIndex()

_CHUNK_LOOKUP: dict[str, dict[str, Any]] = {}
for _domain_name, _domain in KNOWLEDGE_BASE.items():
    for _section_name, _section in _domain.get("sections", {}).items():
        for _chunk in _section.get("chunks", []):
            _CHUNK_LOOKUP[_chunk["id"]] = {**_chunk, "_domain": _domain_name, "_section": _section_name}


async def _ingest_knowledge_base(kb: dict[str, dict[str, Any]]) -> None:
    """One-time (idempotent) ingestion: embeds every domain/section/chunk
    and upserts into BOTH the Redis hot tier and the Supabase durable
    tier, each configured for standard cosine distance at the index
    level (see crag_index.py) rather than any hand-rolled metric."""
    keys: list[tuple[str, str, str, str]] = []  # (bucket, item_id, domain, section)
    texts: list[str] = []

    for domain_name, domain in kb.items():
        keys.append(("domains", domain_name, domain_name, ""))
        texts.append(domain["summary"] + " " + " ".join(domain.get("keywords", [])))
        for section_name, section in domain.get("sections", {}).items():
            sec_key = f"{domain_name}::{section_name}"
            keys.append(("sections", sec_key, domain_name, section_name))
            texts.append(section["summary"] + " " + " ".join(section.get("keywords", [])))
            for chunk in section.get("chunks", []):
                keys.append(("chunks", chunk["id"], domain_name, section_name))
                texts.append(
                    chunk["title"] + " " + chunk["content"] + " " + " ".join(chunk.get("keywords", []))
                )

    vectors, ok_flags = _embedder.embed_batch(texts)
    failed = 0
    for (bucket, item_id, domain_name, section_name), vector, ok in zip(keys, vectors, ok_flags):
        if not ok:
            failed += 1
            continue
        await CRAG_MULTI_INDEX.upsert_everywhere(bucket, item_id, vector, domain_name, section_name)

    logger.info(
        "CRAG multi-index ingestion complete: %d items embedded, %d failed "
        "(redis_tier_available=%s, supabase_tier_configured=%s).",
        len(keys) - failed, failed,
        CRAG_MULTI_INDEX.redis_tier.available, CRAG_MULTI_INDEX.supabase_tier.configured,
    )


try:
    asyncio.run(_ingest_knowledge_base(KNOWLEDGE_BASE))
except RuntimeError:
    # Defensive: if an event loop is already running at import time (e.g.
    # certain test runners), schedule ingestion on it instead of crashing.
    logger.warning("Event loop already running at import — scheduling CRAG ingestion as a task.")
    asyncio.get_event_loop().create_task(_ingest_knowledge_base(KNOWLEDGE_BASE))

class CriticResponse(BaseModel):
    has_issues: bool = Field(
        description="True if the draft answer fails any constraints or fails to fully answer the query."
    )
    issues: list[str] = Field(
        default=[],
        description="List of specific, actionable issues identified. Empty if has_issues is False."
    )

class CorrectorResponse(BaseModel):
    corrected_answer: str = Field(
        description="The full, revised answer incorporating all requested fixes without introducing unsupported claims."
    )
    changes_made: list[str] = Field(
        description="A brief log of the specific changes made during this iteration."
    )

def _handle_bad_schema(target: str) -> str:
    return json.dumps({
        "tool_target": target,
        "fault_injected": True,
        "error_hint": "missing required fields: content, session_id",
        "latency_ms": "not-a-number",
    })

def _handle_partial_payload(target: str) -> str:
    return json.dumps({
        "session_id": 12345,             
        "mcp_interaction_type": None,   
        "content": ["not", "a", "string"],
        "latency_ms": "fast",              
        "tool_target": target,
        "fault_injected": True,
    })

def _handle_runtime_error(target: str) -> None:
    raise RuntimeError(
        f"Simulated execution failure in '{target}': "
        f"downstream dependency returned HTTP 503. "
        f"Retry budget may be available."
    )

def _handle_timeout(target: str) -> None:
    raise TimeoutError(
        f"Simulated timeout in '{target}'. "
        f"Network latency exceeded acceptable threshold."
    )

FAULT_HANDLERS = {
    "bad_schema": {"type": "payload", "func": _handle_bad_schema},
    "partial_payload": {"type": "payload", "func": _handle_partial_payload},
    "runtime_error": {"type": "exception", "func": _handle_runtime_error},
    "timeout_sim": {"type": "exception", "func": _handle_timeout},
}

@mcp.tool()
async def simulate_fault(
    fault_type: str = "random",
    tool_target: str = "query_knowledge",
    severity: str = "medium",
    ctx: Context = None,
) -> str:
    resolved = fault_type if fault_type != "random" else random.choice(list(FAULT_HANDLERS.keys()))
    
    if resolved not in FAULT_HANDLERS:
        warn_msg = f"[FAULT SIMULATION ANOMALY] Received unmapped fault type '{resolved}'. Defaulting to 'runtime_error'."
        logger.warning(warn_msg)
        if ctx: await ctx.log(level="warning", message=warn_msg)
        resolved = "runtime_error"

    severity_map = {"low": 2, "medium": 6, "high": 16}
    sleep_secs = severity_map.get(severity, 6)

    msg = f"[FAULT INJECTION] tool_target={tool_target} fault_type={resolved} severity={severity}"
    logger.warning(msg)
    if ctx: await ctx.log(level="warning", message=msg)

    handler_cfg = FAULT_HANDLERS[resolved]
    handler_func = handler_cfg["func"]

    if resolved == "timeout_sim":
        await asyncio.sleep(sleep_secs)

    if handler_cfg["type"] == "payload":
        return handler_func(tool_target)
    else:
        handler_func(tool_target)

def _expand_queries(query: str) -> list[str]:
    q = query.strip()
    terms = [w for w in q.lower().split() if len(w) > 3]
    variants = [q]
    if terms:
        core = " ".join(terms[:3])
        variants.append(f"What is {core}?")
    variants.append(f"How does {q.lower().rstrip('?')} work in practice?")
    return variants[:3]

DOMAIN_THRESHOLD  = 0.35
SECTION_THRESHOLD = 0.35
LEAF_THRESHOLD    = 0.40
TOP_K_PER_LEVEL   = 5


async def _hierarchical_retrieve(queries: list[str]) -> list[dict[str, Any]]:
    """True cosine multi-index retrieval: each hierarchy level is a real
    ANN cosine query against `CRAG_MULTI_INDEX` (Redis hot tier, Supabase
    durable tier) — no per-item Python cosine loop, no keyword weighting."""
    try:
        query_vectors = [_embedder.embed(q) for q in queries]
    except Exception as exc:
        logger.error("Query embedding failed: %s — returning no candidates (will trigger Tavily fallback).", exc)
        return []

    # Merge scores across the expanded query variants by taking the max
    # similarity seen for a given item_id — a chunk is relevant if it
    # matches ANY good phrasing of the question, not the average of all of them.
    domain_scores: dict[str, float] = {}
    for qv in query_vectors:
        for item in await CRAG_MULTI_INDEX.cosine_query("domains", qv, top_k=TOP_K_PER_LEVEL):
            domain_scores[item.item_id] = max(domain_scores.get(item.item_id, 0.0), item.similarity)

    relevant_domains = {d for d, s in domain_scores.items() if s >= DOMAIN_THRESHOLD}

    section_scores: dict[str, float] = {}
    for qv in query_vectors:
        for item in await CRAG_MULTI_INDEX.cosine_query("sections", qv, top_k=TOP_K_PER_LEVEL * 2):
            if item.domain and item.domain not in relevant_domains:
                continue
            section_scores[item.item_id] = max(section_scores.get(item.item_id, 0.0), item.similarity)

    relevant_sections = {s for s, sc in section_scores.items() if sc >= SECTION_THRESHOLD}

    chunk_scores: dict[str, float] = {}
    for qv in query_vectors:
        for item in await CRAG_MULTI_INDEX.cosine_query("chunks", qv, top_k=TOP_K_PER_LEVEL * 3):
            sec_key = f"{item.domain}::{item.section}"
            if relevant_sections and sec_key not in relevant_sections:
                continue
            chunk_scores[item.item_id] = max(chunk_scores.get(item.item_id, 0.0), item.similarity)

    leaf_candidates: list[dict[str, Any]] = []
    for chunk_id, chunk_score in chunk_scores.items():
        chunk = _CHUNK_LOOKUP.get(chunk_id)
        if chunk is None:
            continue
        domain_name, section_name = chunk["_domain"], chunk["_section"]
        d_score = domain_scores.get(domain_name, 0.0)
        s_score = section_scores.get(f"{domain_name}::{section_name}", 0.0)
        leaf_candidates.append({
            **chunk,
            "_score": d_score * 0.3 + s_score * 0.3 + chunk_score * 0.4,
        })

    leaf_candidates.sort(key=lambda c: c["_score"], reverse=True)
    logger.info(
        "True cosine multi-index retrieval yielded %d leaf candidates (top score: %.3f).",
        len(leaf_candidates),
        leaf_candidates[0]["_score"] if leaf_candidates else 0.0,
    )
    return leaf_candidates

TOT_BRANCHES = 3  

class ToTThought(BaseModel):
    reasoning: str = Field(description="A short line of reasoning about this chunk's relevance.")


class ToTVerdict(BaseModel):
    best_thought_index: int = Field(description="Index (0-based) of the strongest reasoning line.")
    is_relevant: bool = Field(description="Final verdict based on the strongest reasoning line.")


async def _tot_evaluate(
    chunks: list[dict[str, Any]],
    queries: list[str],
    ctx: Context = None,
    session_id: str = "anonymous",
) -> list[dict[str, Any]]:
    """
    Tree-of-Thought chunk evaluation: generate several independent lines of
    reasoning (thoughts) per chunk, then use a separate evaluator call to
    pick the strongest one and decide relevance from it.

    Each branch is a real LLM call (via MCP sampling), so each one is
    gated by a FinOps preflight check first — a runaway ToT expansion
    (many chunks x many branches) is exactly the kind of multi-step trace
    the governor exists to cap before it burns through the session's budget.
    """
    if not chunks or ctx is None:
        return [c for c in chunks if c["_score"] >= LEAF_THRESHOLD]

    governor = get_governor()
    query_block = "\n".join(f"- {q}" for q in queries)
    accepted: list[dict[str, Any]] = []

    for chunk in chunks:
        thoughts = []
        for _ in range(TOT_BRANCHES):
            try:
                governor.preflight(session_id, projected_usd=0.0004, step_label="tot_thought_branch")
            except FinOpsBudgetExceededException as exc:
                logger.warning("FinOps ceiling reached mid-ToT (%s) — truncating remaining branches/chunks.", exc)
                if ctx: await ctx.log(level="warning", message=f"FinOps budget exceeded during ToT: {exc}")
                return accepted or [c for c in chunks if c["_score"] >= LEAF_THRESHOLD][:1]
            prompt = f"""[TASK:TOT_THOUGHT]
            Queries:
            {query_block}

            Chunk: {chunk.get('title', '')}
            {chunk.get('content', '')}

            In one sentence, reason about whether this chunk helps answer
            the queries. Respond ONLY with JSON matching:
            {json.dumps(ToTThought.model_json_schema(), indent=2)}
            """
            try:
                result = await ctx.sample(prompt, max_tokens=100)
                text = result.text.strip().strip("```json").strip("```").strip()
                thoughts.append(ToTThought.model_validate_json(text).reasoning)
                governor.record_spend(session_id, 0.0004)
            except Exception as exc:
                logger.warning("ToT thought generation failed: %s", exc)

        if not thoughts:
            if chunk["_score"] >= LEAF_THRESHOLD:
                accepted.append(chunk)
            continue

        thoughts_block = "\n".join(f"{i}: {t}" for i, t in enumerate(thoughts))
        eval_prompt = f"""[TASK:TOT_VERDICT]
        Here are independent reasoning attempts about whether a chunk is
        relevant to these queries:
        {query_block}

        Reasoning attempts:
        {thoughts_block}

        Pick the strongest reasoning attempt and give a final verdict.
        Respond ONLY with JSON matching:
        {json.dumps(ToTVerdict.model_json_schema(), indent=2)}
        """
        try:
            governor.preflight(session_id, projected_usd=0.0004, step_label="tot_verdict_eval")
        except FinOpsBudgetExceededException as exc:
            logger.warning("FinOps ceiling reached before ToT verdict (%s) — using cosine score as fallback verdict.", exc)
            if chunk["_score"] >= LEAF_THRESHOLD:
                accepted.append(chunk)
            continue
        try:
            result = await ctx.sample(eval_prompt, max_tokens=100)
            text = result.text.strip().strip("```json").strip("```").strip()
            verdict = ToTVerdict.model_validate_json(text)
            governor.record_spend(session_id, 0.0004)
            await ctx.log(
                level="debug",
                message=f"ToT chunk '{chunk['id']}': best_thought=\"{thoughts[verdict.best_thought_index]}\" → relevant={verdict.is_relevant}",
            )
            if verdict.is_relevant:
                accepted.append(chunk)
        except Exception as exc:
            logger.warning("ToT evaluation failed for chunk '%s': %s", chunk["id"], exc)
            if chunk["_score"] >= LEAF_THRESHOLD:
                accepted.append(chunk)

    logger.info("ToT accepted %d/%d chunks.", len(accepted), len(chunks))
    return accepted

async def _tavily_fallback(query: str) -> list[dict[str, Any]]:
    if not _tavily_key or _tavily is None:
        logger.warning("Tavily not configured — skipping web fallback.")
        return []
    try:
        results = await _tavily.ainvoke(query)
        if isinstance(results, list):
            return [
                {
                    "document_id": f"web-{i}",
                    "title": r.get("title", "Web Result"),
                    "content": r.get("content", r.get("snippet", "")),
                    "url": r.get("url", ""),
                    "_domain": "web",
                    "_section": "web",
                    "_score": r.get("score", 0.5),
                    "keywords": [],
                }
                for i, r in enumerate(results)
            ]
        return []
    except Exception as exc:
        logger.error("Tavily fallback error: %s", exc)
        return []

@mcp.resource("knowledge://domain/docs")
async def domain_knowledge_resource() -> str:
    lines = ["Available knowledge domains:\n"]
    for domain, data in KNOWLEDGE_BASE.items():
        lines.append(f"• {domain}: {data['summary']}")
        for sec_name, sec in data.get("sections", {}).items():
            lines.append(f"    └─ {sec_name}: {sec['summary']}")
    return "\n".join(lines)

QUERY_KNOWLEDGE_PRICE_USD = float(os.getenv("QUERY_KNOWLEDGE_PRICE_USD", "0.01"))


@mcp.tool()
@require_payment("query_knowledge", QUERY_KNOWLEDGE_PRICE_USD)
async def query_knowledge(query: str, session_id: str = "", ctx: Context = None) -> str:
    msg = f"CRAG pipeline started for query: '{query}'"
    logger.info(msg)
    if ctx: await ctx.info(msg)

    governor = get_governor()
    fin_session = session_id or "anonymous"
    try:
        governor.preflight(fin_session, projected_usd=0.0015, step_label="query_knowledge.retrieval")
    except FinOpsBudgetExceededException as exc:
        msg = f"FinOps ceiling hit before retrieval even started: {exc}"
        logger.warning(msg)
        if ctx: await ctx.log(level="warning", message=msg)
        return f"[FinOps budget exceeded] Falling back to a safe static summary for: {query}"

    queries = _expand_queries(query)
    msg = f"Expanded queries: {queries}"
    logger.debug(msg)
    if ctx: await ctx.log(level="debug", message=msg)

    candidates = await _hierarchical_retrieve(queries)
    governor.record_spend(fin_session, 0.0015)
    top_score = candidates[0]["_score"] if candidates else 0.0
    msg = f"Cosine hierarchical retrieval yielded {len(candidates)} leaf candidates (top score: {top_score:.3f})."
    logger.info(msg)
    if ctx: await ctx.info(msg)

    accepted = await _tot_evaluate(candidates, queries, ctx, session_id=fin_session)

    msg = f"ToT accepted {len(accepted)}/{len(candidates)} chunks."
    logger.info(msg)
    if ctx: await ctx.info(msg)

    top_score = accepted[0]["_score"] if accepted else 0.0
    kb_is_relevant = top_score >= LEAF_THRESHOLD and len(accepted) >= 1

    used_fallback = False
    if not kb_is_relevant:
        msg = f"KB relevance too low (top_score={top_score:.3f}, threshold={LEAF_THRESHOLD}) — triggering Tavily fallback."
        logger.info(msg)
        if ctx: await ctx.info(msg)
        web = await _tavily_fallback(query)
        if web:
            accepted = web
            used_fallback = True
        else:
            msg = "Tavily fallback returned nothing — using low-relevance KB results."
            logger.warning(msg)
            if ctx: await ctx.log(level="warning", message=msg)

    if not accepted:
        return f"No relevant documentation found for: {query}"

    top_chunks  = accepted[:5]
    domains_hit = {c["_domain"] for c in top_chunks if c["_domain"] in KNOWLEDGE_BASE}
    summaries   = "\n".join(f"[{d}] {KNOWLEDGE_BASE[d]['summary']}" for d in domains_hit)
    chunks_text = "\n\n".join(
        f"### {c['title']} ({c['id']})\n{c['content']}"
        + (f"\nSource: {c['url']}" if c.get("url") else "")
        for c in top_chunks
    )
    fallback_note = "\n\n> Note: Internal KB was insufficient; web results included." if used_fallback else ""
    result = f"""## Query
        {query}

        ## Expanded Queries
        {"\n".join(f"- {q}" for q in queries)}

        ## Domain Summaries
        {summaries}

        ## Retrieved Chunks ({len(top_chunks)}):

        {chunks_text}{fallback_note}
    """
    if ctx: await ctx.info(f"CRAG returning {len(top_chunks)} chunks for '{query}'.")
    return result

MAX_REFLECTION_ITERATIONS = 3

@mcp.tool()
async def reflect_answer(
    draft_answer: str,
    original_query: str,
    constraints: str = "accuracy, completeness, no hallucinations",
    session_id: str = "",
    ctx: Context = None,
) -> str:
    if ctx is None:
        return "Error: Context not available."

    msg = f"Reflection started for query: {original_query[:80]}"
    logger.info(msg)
    await ctx.info(msg)

    governor = get_governor()
    fin_session = session_id or "anonymous"
    current_draft = draft_answer
    critic_schema = json.dumps(CriticResponse.model_json_schema(), indent=2)
    corrector_schema = json.dumps(CorrectorResponse.model_json_schema(), indent=2)

    for iteration in range(1, MAX_REFLECTION_ITERATIONS + 1):
        # Pre-flight: a Critic + Corrector pair is two LLM calls, so budget
        # for both before running either — this is the "intercept planning
        # cycles" requirement for the Reflection/Self-correction loop.
        try:
            governor.preflight(fin_session, projected_usd=0.002, step_label=f"reflection_iter_{iteration}")
        except FinOpsBudgetExceededException as exc:
            msg = f"FinOps ceiling reached at reflection iteration {iteration}: {exc} — returning current draft."
            logger.warning(msg)
            await ctx.log(level="warning", message=msg)
            break

        msg = f"Reflection iteration {iteration}/{MAX_REFLECTION_ITERATIONS}"
        logger.info(msg)
        await ctx.info(msg)
        await ctx.report_progress(
            progress=iteration - 1,
            total=MAX_REFLECTION_ITERATIONS,
            message=f"Iteration {iteration}",
        )

        critic_prompt = f"""[TASK:CRITIC]
        You are a rigorous Critic AI. Audit the draft answer against the original query and constraints.

        Constraints:
        {constraints}

        Original Query:
        {original_query}

        Draft Answer:
        {current_draft}

        You must respond ONLY with a raw JSON object matching this schema:
        {critic_schema}
        """

        try:
            critic_result = await ctx.sample(critic_prompt, max_tokens=512)
            critic_text = critic_result.text if hasattr(critic_result, "text") else str(critic_result)
            clean_critic = critic_text.strip().strip("```json").strip("```").strip()
            critic_data = CriticResponse.model_validate_json(clean_critic)
            governor.record_spend(fin_session, 0.001)
        except Exception as exc:
            logger.error(
                "Critic step failed parsing or validation: %s — this often indicates "
                "a semantic-cache cross-contamination (wrong cached response served "
                "for this prompt) rather than a genuine LLM formatting error. "
                "Raw response was: %s", exc, clean_critic[:200] if 'clean_critic' in dir() else "N/A",
            )
            await ctx.log(level="error", message=f"Critic step parsing failed: {exc}")
            break

        logger.info("Critic: has_issues=%s, count=%d", critic_data.has_issues, len(critic_data.issues))
        await ctx.info(f"Critic result: has_issues={critic_data.has_issues}, issues_count={len(critic_data.issues)}")

        if not critic_data.has_issues:
            await ctx.info(f"No issues at iteration {iteration}. Finalising.")
            break

        issues_text = "\n".join(f"- {issue}" for issue in critic_data.issues)
        corrector_prompt = f"""[TASK:CORRECTOR]
        You are a precise Corrector AI. Fix every issue listed below in the draft.

        Original Query: {original_query}

        Draft Answer:
        {current_draft}

        Issues to Fix:
        {issues_text}

        You must respond ONLY with a raw JSON object matching this schema:
        {corrector_schema}
        """

        try:
            corrector_result = await ctx.sample(corrector_prompt, max_tokens=1024)
            corrector_text = corrector_result.text if hasattr(corrector_result, "text") else str(corrector_result)
            clean_corrector = corrector_text.strip().strip("```json").strip("```").strip()
            corrector_data = CorrectorResponse.model_validate_json(clean_corrector)
            current_draft = corrector_data.corrected_answer
            governor.record_spend(fin_session, 0.001)
            await ctx.info(f"Corrector applied {len(corrector_data.changes_made)} changes.")
        except Exception as exc:
            logger.error("Corrector step failed parsing or validation: %s", exc)
            await ctx.log(level="warning", message=f"Corrector step skipped due to structural error: {exc}")
            break

    await ctx.report_progress(
        progress=MAX_REFLECTION_ITERATIONS,
        total=MAX_REFLECTION_ITERATIONS,
        message="Reflection complete",
    )
    return current_draft

def main() -> None:
    logger.info("Starting ThinkingAgentServer (streamable-http) on http://0.0.0.0:8000 ...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()