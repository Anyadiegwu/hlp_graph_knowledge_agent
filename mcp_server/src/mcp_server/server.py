from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any

from fastmcp import Context, FastMCP
from langchain_tavily import TavilySearch
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from pydantic import BaseModel, Field

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

def _score_against_queries(text: str, keywords: list[str], queries: list[str]) -> float:
    combined = (text + " " + " ".join(keywords)).lower()
    query_terms = set()
    for q in queries:
        query_terms.update(w for w in q.lower().split() if len(w) > 2)
    hits = sum(1 for term in query_terms if term in combined)
    return hits / max(len(query_terms), 1)

def _hierarchical_retrieve(queries: list[str]) -> list[dict[str, Any]]:
    DOMAIN_THRESHOLD  = 0.05
    SECTION_THRESHOLD = 0.05
    leaf_candidates: list[dict[str, Any]] = []

    for domain_name, domain in KNOWLEDGE_BASE.items():
        domain_score = _score_against_queries(
            domain["summary"], domain.get("keywords", []), queries
        )
        if domain_score < DOMAIN_THRESHOLD:
            continue
        for section_name, section in domain.get("sections", {}).items():
            section_score = _score_against_queries(
                section["summary"], section.get("keywords", []), queries
            )
            if section_score < SECTION_THRESHOLD:
                continue
            for chunk in section.get("chunks", []):
                chunk_score = _score_against_queries(
                    chunk["content"] + " " + chunk["title"],
                    chunk.get("keywords", []),
                    queries,
                )
                leaf_candidates.append({
                    **chunk,
                    "_domain":  domain_name,
                    "_section": section_name,
                    "_score":   domain_score * 0.3 + section_score * 0.3 + chunk_score * 0.4,
                })

    leaf_candidates.sort(key=lambda c: c["_score"], reverse=True)
    logger.info("Hierarchical retrieval yielded %d leaf candidates.", len(leaf_candidates))
    return leaf_candidates

def _tot_evaluate(chunks: list[dict[str, Any]], queries: list[str]) -> list[dict[str, Any]]:
    accepted = []
    query_terms = set(w for q in queries for w in q.lower().split() if len(w) > 3)

    for chunk in chunks:
        content_lower = (chunk.get("content", "") + " " + chunk.get("title", "")).lower()

        overlap = sum(1 for t in query_terms if t in content_lower)
        vote_a  = overlap >= max(1, len(query_terms) * 0.15)
        title_lower = chunk.get("title", "").lower()
        vote_b  = any(t in title_lower for t in query_terms) or chunk["_score"] >= 0.12
        vote_c  = len(chunk.get("content", "")) >= 60 and len(chunk.get("keywords", [])) >= 2

        votes   = sum([vote_a, vote_b, vote_c])
        verdict = "ACCEPT" if votes >= 2 else "REJECT"
        
        # CHANGED: Log directly to the standard server logger right where it happens!
        logger.debug(
            f"ToT chunk '{chunk['id']}' [{chunk['_domain']}/{chunk['_section']}]: "
            f"A={vote_a} B={vote_b} C={vote_c} → {votes}/3 → {verdict}"
        )
        
        if votes >= 2:
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
                    document_id: f"web-{i}",
                    "title":  r.get("title", "Web Result"),
                    "content": r.get("content", r.get("snippet", "")),
                    "url":     r.get("url", ""),
                    "_domain":  "web",
                    "_section": "web",
                    "_score":   r.get("score", 0.5),
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

@mcp.tool()
async def query_knowledge(query: str, ctx: Context = None) -> str:
    msg = f"CRAG pipeline started for query: '{query}'"
    logger.info(msg)
    if ctx: await ctx.info(msg)

    queries = _expand_queries(query)
    msg = f"Expanded queries: {queries}"
    logger.debug(msg)
    if ctx: await ctx.log(level="debug", message=msg)

    candidates = _hierarchical_retrieve(queries)
    msg = f"Hierarchical retrieval yielded {len(candidates)} leaf candidates."
    logger.info(msg)
    if ctx: await ctx.info(msg)

    tot_log: list[str] = []
    accepted = _tot_evaluate(candidates, queries, tot_log)
    for log_msg in tot_log:
        if ctx: await ctx.log(level="debug", message=log_msg)

    msg = f"ToT accepted {len(accepted)}/{len(candidates)} chunks."
    logger.info(msg)
    if ctx: await ctx.info(msg)

    RELEVANCE_THRESHOLD = 0.15
    top_score = accepted[0]["_score"] if accepted else 0.0
    kb_is_relevant = top_score >= RELEVANCE_THRESHOLD and len(accepted) >= 1

    used_fallback = False
    if not kb_is_relevant:
        msg = f"KB relevance too low (top_score={top_score:.3f}, threshold={RELEVANCE_THRESHOLD}) — triggering Tavily fallback."
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
    ctx: Context = None,
) -> str:
    if ctx is None:
        return "Error: Context not available."

    msg = f"Reflection started for query: {original_query[:80]}"
    logger.info(msg)
    await ctx.info(msg)

    current_draft = draft_answer
    critic_schema = json.dumps(CriticResponse.model_json_schema(), indent=2)
    corrector_schema = json.dumps(CorrectorResponse.model_json_schema(), indent=2)

    for iteration in range(1, MAX_REFLECTION_ITERATIONS + 1):
        msg = f"Reflection iteration {iteration}/{MAX_REFLECTION_ITERATIONS}"
        logger.info(msg)
        await ctx.info(msg)
        await ctx.report_progress(
            progress=iteration - 1,
            total=MAX_REFLECTION_ITERATIONS,
            message=f"Iteration {iteration}",
        )

        critic_prompt = f"""
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
        except Exception as exc:
            logger.error("Critic step failed parsing or validation: %s", exc)
            await ctx.log(level="error", message=f"Critic step parsing failed: {exc}")
            break

        logger.info("Critic: has_issues=%s, count=%d", critic_data.has_issues, len(critic_data.issues))
        await ctx.info(f"Critic result: has_issues={critic_data.has_issues}, issues_count={len(critic_data.issues)}")

        if not critic_data.has_issues:
            await ctx.info(f"No issues at iteration {iteration}. Finalising.")
            break

        issues_text = "\n".join(f"- {issue}" for issue in critic_data.issues)
        corrector_prompt = f"""
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