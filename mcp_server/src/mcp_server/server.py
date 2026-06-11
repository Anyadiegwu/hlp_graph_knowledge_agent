# MCP Server — Stage 3 (HLP Graph Knowledge Agent)
# Identical to Stage 2 — server holds no LLM, all sampling delegated to client.
# Exposes:
#   @tool  reflect_answer  — Critic/Corrector loop via MCP Sampling
#   @tool  query_knowledge — Hierarchical CRAG with ToT + Tavily fallback

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastmcp import Context, FastMCP
from langchain_tavily import TavilySearch
from starlette.requests import Request
from starlette.responses import PlainTextResponse

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [SERVER] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ThinkingAgentServer",
    instructions=(
        "Production MCP server exposing a Sampling-based Reflection tool "
        "and a Hierarchical CRAG knowledge tool."
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
    combined = text.lower()
    all_terms = set()
    for q in queries:
        all_terms.update(w for w in q.lower().split() if len(w) > 2)
    all_terms.update(kw.lower() for kw in keywords)
    hits = sum(1 for term in all_terms if term in combined)
    return hits / max(len(all_terms), 1)


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


def _tot_evaluate(
    chunks: list[dict[str, Any]], queries: list[str], ctx_log: list[str]
) -> list[dict[str, Any]]:
    accepted = []
    for chunk in chunks:
        content_lower = (chunk.get("content", "") + " " + chunk.get("title", "")).lower()
        query_terms = set(w for q in queries for w in q.lower().split() if len(w) > 3)

        overlap = sum(1 for t in query_terms if t in content_lower)
        vote_a  = overlap >= max(1, len(query_terms) * 0.15)
        title_lower = chunk.get("title", "").lower()
        vote_b  = any(t in title_lower for t in query_terms) or chunk["_score"] >= 0.12
        vote_c  = len(chunk.get("content", "")) >= 60 and len(chunk.get("keywords", [])) >= 2

        votes   = sum([vote_a, vote_b, vote_c])
        verdict = "ACCEPT" if votes >= 2 else "REJECT"
        log_msg = (
            f"ToT chunk '{chunk['id']}' [{chunk['_domain']}/{chunk['_section']}]: "
            f"A={vote_a} B={vote_b} C={vote_c} → {votes}/3 → {verdict}"
        )
        logger.debug(log_msg)
        ctx_log.append(log_msg)
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
                    "id":       f"web-{i}",
                    "title":    r.get("title", "Web Result"),
                    "content":  r.get("content", r.get("snippet", "")),
                    "url":      r.get("url", ""),
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
    """
    Hierarchical CRAG retrieval pipeline:
      1. Multi-query expansion   — 3 semantic variants
      2. Hierarchical retrieval  — domain → section → chunk (3 levels)
      3. ToT evaluation          — 3-chain majority vote per chunk
      4. Tavily fallback         — if ToT accepts < 2 chunks
    """
    msg = f"CRAG pipeline started for query: '{query}'"
    logger.info(msg)
    if ctx: await ctx.info(msg)

    queries  = _expand_queries(query)
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

    # Relevance-based fallback trigger:
    # Fire Tavily if the KB has no genuinely relevant content for this query,
    # regardless of chunk count. A top score below 0.15 means the KB is
    # returning marginally-matching chunks, not real answers.
    RELEVANCE_THRESHOLD = 0.15
    top_score = accepted[0]["_score"] if accepted else 0.0
    kb_is_relevant = top_score >= RELEVANCE_THRESHOLD and len(accepted) >= 2

    used_fallback = False
    if not kb_is_relevant:
        msg = (
            f"KB relevance too low (top_score={top_score:.3f}, threshold={RELEVANCE_THRESHOLD}) "
            f"— triggering Tavily fallback."
        )
        logger.info(msg)
        if ctx: await ctx.info(msg)
        web = await _tavily_fallback(query)
        if web:
            # Replace low-relevance KB chunks with web results entirely
            accepted = web
            used_fallback = True
        else:
            # Tavily unavailable — keep whatever KB returned
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
    fallback_note = (
        "\n\n> Note: Internal KB was insufficient; web results included." if used_fallback else ""
    )
    result = (
        f"## Query\n{query}\n\n"
        f"## Expanded Queries\n" + "\n".join(f"- {q}" for q in queries) + "\n\n"
        f"## Domain Summaries\n{summaries}\n\n"
        f"## Retrieved Chunks ({len(top_chunks)})\n\n{chunks_text}"
        f"{fallback_note}"
    )
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
    """
    Iteratively critiques and corrects a draft answer via MCP Sampling.
    The server sends Critic and Corrector prompts to the client via ctx.sample().
    The client executes them with its local LLM. Server holds NO API keys.
    """
    if ctx is None:
        return "Error: Context not available."

    msg = f"Reflection started for query: {original_query[:80]}"
    logger.info(msg)
    await ctx.info(msg)

    current_draft = draft_answer

    for iteration in range(1, MAX_REFLECTION_ITERATIONS + 1):
        msg = f"Reflection iteration {iteration}/{MAX_REFLECTION_ITERATIONS}"
        logger.info(msg)
        await ctx.info(msg)
        await ctx.report_progress(
            progress=iteration - 1,
            total=MAX_REFLECTION_ITERATIONS,
            message=f"Iteration {iteration}",
        )

        critic_prompt = (
            f"You are a rigorous Critic AI. Audit the draft answer against the query "
            f"and constraints.\n\n"
            f"IMPORTANT: The draft answer may be grounded in a retrieved knowledge base. "
            f"Do NOT flag content simply because you are personally unfamiliar with it "
            f"or because the topic seems niche or technical. Only flag genuine logical "
            f"contradictions, internal inconsistencies, unsupported speculative claims, "
            f"or direct violations of the listed constraints. "
            f"If the answer is internally consistent and addresses the query, report no issues.\n\n"
            f"Constraints: {constraints}\n\n"
            f"Original Query: {original_query}\n\n"
            f"Draft Answer:\n{current_draft}\n\n"
            f"Respond ONLY in valid JSON with no extra text or markdown fences:\n"
            f'{{"has_issues": true, "issues": ["issue1", "issue2"]}}\n'
            f"If no issues: "
            f'{{"has_issues": false, "issues": []}}'
        )

        logger.info("Sending Critic sampling request (iteration %d).", iteration)
        await ctx.info(f"Sending Critic sampling request to client (iteration {iteration}).")

        try:
            critic_result = await ctx.sample(critic_prompt, max_tokens=512)
            critic_text = critic_result.text if hasattr(critic_result, "text") else str(critic_result)
        except Exception as exc:
            logger.error("Critic sampling failed: %s", exc)
            await ctx.error(f"Critic sampling failed: {exc}")
            break

        try:
            clean = critic_text.strip().strip("```json").strip("```").strip()
            critic_data = json.loads(clean)
            has_issues: bool = critic_data.get("has_issues", False)
            issues: list[str] = critic_data.get("issues", [])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Critic response not valid JSON — treating as no issues.")
            await ctx.log(level="warning", message="Critic response not valid JSON — treating as no issues.")
            has_issues = False
            issues = []

        logger.info("Critic: has_issues=%s, count=%d", has_issues, len(issues))
        await ctx.info(f"Critic result: has_issues={has_issues}, issues_count={len(issues)}")

        if not has_issues:
            await ctx.info(f"No issues at iteration {iteration}. Finalising.")
            break

        issues_text = "\n".join(f"- {issue}" for issue in issues)
        corrector_prompt = (
            f"You are a precise Corrector AI. Fix every issue listed below in the draft. "
            f"Do not introduce new unsupported claims.\n\n"
            f"Original Query: {original_query}\n\n"
            f"Draft Answer:\n{current_draft}\n\n"
            f"Issues to Fix:\n{issues_text}\n\n"
            f"Respond ONLY in valid JSON with no extra text or markdown fences:\n"
            f'{{"corrected_answer": "full corrected answer here", '
            f'"changes_made": ["change1", "change2"]}}'
        )

        await ctx.info(f"Sending Corrector sampling request to client (iteration {iteration}).")

        try:
            corrector_result = await ctx.sample(corrector_prompt, max_tokens=1024)
            corrector_text = corrector_result.text if hasattr(corrector_result, "text") else str(corrector_result)
        except Exception as exc:
            logger.error("Corrector sampling failed: %s", exc)
            await ctx.error(f"Corrector sampling failed: {exc}")
            break

        try:
            clean = corrector_text.strip().strip("```json").strip("```").strip()
            corrector_data = json.loads(clean)
            current_draft = corrector_data.get("corrected_answer", current_draft)
            changes = corrector_data.get("changes_made", [])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrector response not valid JSON — keeping current draft.")
            changes = []

        await ctx.info(f"Corrector applied {len(changes)} changes.")

    await ctx.report_progress(
        progress=MAX_REFLECTION_ITERATIONS,
        total=MAX_REFLECTION_ITERATIONS,
        message="Reflection complete",
    )
    await ctx.info(f"Reflection complete for query: '{original_query[:80]}'")
    return current_draft


def main() -> None:
    logger.info("Starting ThinkingAgentServer (streamable-http) on http://0.0.0.0:8000 ...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
