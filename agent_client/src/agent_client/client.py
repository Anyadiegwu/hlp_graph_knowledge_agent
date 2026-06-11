# agent_client/src/agent_client/client.py
#
# HLP Graph Knowledge Agent — Stage 3
#
# Extends Stage 2 with:
#   • HLPLogStore: LangGraph-style SQLite vector store with hierarchical namespaces
#   • Every significant agent/MCP event is persisted as a validated LogEntry
#   • session_id propagated across all log entries in a single run
#   • Dual-stream logging retained (flat file + structured SQLite)

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env"))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    LoggingMessageNotificationParams,
    TextContent,
)
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_client.log_store import (
    HLPLogStore,
    LogEntry,
    MCPInteractionType,
    NS,
    get_log_store,
)

# ─────────────────────────────────────────────
# 1. Dual-stream logging (flat file — Stage 2)
#    Stage 3 ADDS the SQLite vector store on
#    top of this; the flat log is retained.
# ─────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_FILE   = _REPO_ROOT / "mcp_agent_system.log"
LOG_FORMAT = "[%(asctime)s] [%(log_source)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _SourceFilter(logging.Filter):
    def __init__(self, source: str) -> None:
        super().__init__()
        self._source = source

    def filter(self, record: logging.LogRecord) -> bool:
        record.log_source = self._source
        return True


def _make_logger(name: str, source: str) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.addFilter(_SourceFilter(source))
    lg.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(_SourceFilter(source))
    lg.addHandler(fh)
    return lg


client_logger = _make_logger("agent.client",       "CLIENT")
server_logger = _make_logger("agent.server_relay",  "SERVER")
client_logger.info("Flat log file : %s", LOG_FILE)


# ─────────────────────────────────────────────
# 2. Settings
# ─────────────────────────────────────────────

class Settings(BaseSettings):
    groq_api_key:    SecretStr | None = None
    groq_model_name: str              = "llama-3.3-70b-versatile"
    gemini_api_key:  SecretStr | None = None
    gemini_model_name: str            = "gemini-2.5-flash"
    model_temperature: float          = 0.0
    mcp_server_url:  str              = "http://localhost:8000/mcp"
    use_groq:        bool             = True
    log_db_path:     str              = ""       # empty → auto-detected at repo root

    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# ─────────────────────────────────────────────
# 3. Stage 3 — SQLite Vector Log Store
# ─────────────────────────────────────────────

_db_path = settings.log_db_path or (_REPO_ROOT / "mcp_agent_log.db")
log_store: HLPLogStore = get_log_store(_db_path)
client_logger.info("Vector log store: %s", _db_path)

# Each process run = one session
SESSION_ID = str(uuid.uuid4())
client_logger.info("Session ID: %s", SESSION_ID)


def _persist(
    interaction_type: MCPInteractionType,
    content: str,
    namespace_path: str,
    component: str = "agent_client",
    tool_name: str | None = None,
    latency_ms: float | None = None,
    token_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    embed: bool = True,
) -> None:
    """
    Write a validated LogEntry to the SQLite vector store.
    Non-blocking: failures are logged to flat file but never raise.
    """
    try:
        entry = LogEntry(
            session_id=SESSION_ID,
            mcp_interaction_type=interaction_type,
            content=content,
            namespace_path=namespace_path,
            component=component,
            tool_name=tool_name,
            latency_ms=latency_ms,
            token_count=token_count,
            metadata=metadata or {},
        )
        log_store.put(entry, embed=embed)
    except Exception as exc:
        client_logger.warning("Log store write failed: %s", exc)


# ─────────────────────────────────────────────
# 4. Model builders
# ─────────────────────────────────────────────

def _build_primary_model() -> BaseChatModel:
    # Prefer Gemini for the primary agent loop — Groq has a known issue where
    # it emits malformed tool-call XML (<function=name{...}>) instead of JSON
    # when the conversation history grows long, causing a 400 BadRequestError.
    if settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.gemini_model_name,
                temperature=settings.model_temperature,
                google_api_key=settings.gemini_api_key.get_secret_value(),
            )
        except Exception as exc:
            client_logger.warning("Gemini primary model failed (%s) — trying Groq.", exc)

    if settings.use_groq and settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq
            # disable_streaming avoids the parallel tool-call path that triggers
            # the malformed function-call generation on some Groq models.
            return ChatGroq(
                model=settings.groq_model_name,
                temperature=settings.model_temperature,
                api_key=settings.groq_api_key,
                model_kwargs={"parallel_tool_calls": False},
            )
        except Exception as exc:
            client_logger.warning("Groq init failed (%s) — trying Ollama.", exc)

    from langchain_ollama import ChatOllama
    client_logger.warning("No cloud API keys — falling back to Ollama.")
    return ChatOllama(model="llama3.2:3b", temperature=settings.model_temperature)


def _build_sampling_model() -> BaseChatModel:
    if settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.gemini_model_name,
                temperature=settings.model_temperature,
                google_api_key=settings.gemini_api_key.get_secret_value(),
            )
        except Exception as exc:
            client_logger.warning("Gemini sampling model failed (%s) — trying Groq.", exc)

    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model_name,
            temperature=settings.model_temperature,
            api_key=settings.groq_api_key,
        )

    raise RuntimeError("No model available for MCP Sampling. Set GEMINI_API_KEY or GROQ_API_KEY.")


primary_model  = _build_primary_model()
sampling_model = _build_sampling_model()
client_logger.info("Primary model : %s", type(primary_model).__name__)
client_logger.info("Sampling model: %s", type(sampling_model).__name__)

# ─────────────────────────────────────────────────────────────
# Log startup event to vector store
# ─────────────────────────────────────────────────────────────
_persist(
    interaction_type=MCPInteractionType.SYSTEM_EVENT,
    content=f"Agent session started. primary_model={type(primary_model).__name__} sampling_model={type(sampling_model).__name__}",
    namespace_path=NS.SYSTEM_STARTUP,
    component="agent_client",
    embed=False,
)

# ─────────────────────────────────────────────
# 5. MCP Sampling handler
# ─────────────────────────────────────────────

async def sampling_callback(
    context: RequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    client_logger.info("MCP Sampling request | max_tokens=%s", params.maxTokens)
    _t0 = time.perf_counter()

    # Persist the incoming sampling request
    prompt_texts = []
    lc_messages  = []
    for msg in params.messages:
        if hasattr(msg.content, "text"):
            text = msg.content.text
        elif isinstance(msg.content, list):
            text = " ".join(
                block.get("text", "") for block in msg.content if isinstance(block, dict)
            )
        else:
            text = str(msg.content)
        prompt_texts.append(text)
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=text))
        else:
            lc_messages.append(AIMessage(content=text))

    _persist(
        interaction_type=MCPInteractionType.SAMPLING_REQUEST,
        content="\n---\n".join(prompt_texts),
        namespace_path=NS.MCP_SERVER_SAMPLING,
        component="mcp_server",
        metadata={"max_tokens": params.maxTokens},
    )

    # Execute sampling
    response_text = None
    model_used = settings.gemini_model_name if settings.gemini_api_key else settings.groq_model_name

    try:
        response = await sampling_model.ainvoke(lc_messages)
        response_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        client_logger.info("Sampling response generated (%d chars).", len(response_text))
    except Exception as exc:
        client_logger.error("Sampling model invocation failed: %s", exc)
        if settings.groq_api_key and type(sampling_model).__name__ != "ChatGroq":
            client_logger.warning("Falling back to Groq for sampling.")
            try:
                from langchain_groq import ChatGroq
                groq_fallback = ChatGroq(
                    model=settings.groq_model_name,
                    temperature=settings.model_temperature,
                    api_key=settings.groq_api_key,
                )
                response = await groq_fallback.ainvoke(lc_messages)
                response_text = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                model_used = settings.groq_model_name
            except Exception as exc2:
                client_logger.error("Groq fallback also failed: %s", exc2)
                response_text = f"Error during sampling: {exc2}"
        else:
            response_text = f"Error during sampling: {exc}"

    latency = (time.perf_counter() - _t0) * 1000

    # Persist the sampling response
    _persist(
        interaction_type=MCPInteractionType.SAMPLING_RESPONSE,
        content=response_text or "",
        namespace_path=NS.MCP_CLIENT_SAMPLING,
        component="agent_client",
        latency_ms=latency,
        token_count=len((response_text or "").split()),
        metadata={"model_used": model_used},
    )

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model=model_used,
    )


# ─────────────────────────────────────────────
# 6. MCP callback handlers
# ─────────────────────────────────────────────

async def log_handler(
    params: LoggingMessageNotificationParams,
    context: CallbackContext,
) -> None:
    level_map = {
        "debug":    logging.DEBUG,
        "info":     logging.INFO,
        "warning":  logging.WARNING,
        "error":    logging.ERROR,
        "critical": logging.CRITICAL,
    }
    level = level_map.get(str(params.level).lower(), logging.INFO)
    if isinstance(params.data, dict):
        message = params.data.get("msg", str(params.data))
    else:
        message = str(params.data)
    server_logger.log(level, message)

    # Also persist server logs to the vector store
    # Determine namespace based on message content
    ns = NS.MCP_SERVER_TOOL
    if "crag" in message.lower() or "retrieval" in message.lower() or "tot" in message.lower():
        ns = NS.MCP_SERVER_CRAG
    elif "reflection" in message.lower() or "critic" in message.lower() or "corrector" in message.lower():
        ns = NS.MCP_SERVER_REFLECTION
    elif "sampling" in message.lower():
        ns = NS.MCP_SERVER_SAMPLING

    _persist(
        interaction_type=(
            MCPInteractionType.ERROR if level >= logging.ERROR
            else MCPInteractionType.TOOL_INVOCATION
        ),
        content=message,
        namespace_path=ns,
        component="mcp_server",
        embed=(level >= logging.INFO),   # only embed INFO+ to save API calls
    )


async def progress_handler(
    progress: float,
    total: float | None,
    message: str | None,
    context: CallbackContext,
) -> None:
    tool_info = f" ({context.tool_name})" if context.tool_name else ""
    if total and total > 0:
        pct = (progress / total) * 100
        client_logger.info("[PROGRESS%s] %.1f%% — %s", tool_info, pct, message or "")
    else:
        client_logger.info("[PROGRESS%s] step=%.0f — %s", tool_info, progress, message or "")


# ─────────────────────────────────────────────
# 7. MCP Client
# ─────────────────────────────────────────────

MCP_SERVER_NAME = "ThinkingAgentServer"

mcp_client = MultiServerMCPClient(
    {
        MCP_SERVER_NAME: {
            "transport": "streamable-http",
            "url": settings.mcp_server_url,
            "session_kwargs": {
                "sampling_callback": sampling_callback,
            },
        }
    },
    callbacks=Callbacks(
        on_logging_message=log_handler,
        on_progress=progress_handler,
    ),
)

_remote_tools: dict[str, Any] = {}


async def connect_to_server() -> None:
    client_logger.info("Connecting to MCP server at %s ...", settings.mcp_server_url)
    _t0 = time.perf_counter()

    tools_list = await mcp_client.get_tools()
    for t in tools_list:
        _remote_tools[t.name] = t
        client_logger.info("  Registered remote tool: %s", t.name)

    latency = (time.perf_counter() - _t0) * 1000
    client_logger.info(
        "MCP connection established — %d tools registered.", len(_remote_tools)
    )

    _persist(
        interaction_type=MCPInteractionType.SYSTEM_EVENT,
        content=(
            f"MCP connection established. tools={[t for t in _remote_tools.keys()]} "
            f"server_url={settings.mcp_server_url}"
        ),
        namespace_path=NS.MCP_CLIENT_CONNECT,
        component="agent_client",
        latency_ms=latency,
        embed=False,
    )

    try:
        async with mcp_client.session(MCP_SERVER_NAME) as session:
            await session.set_logging_level("debug")
        client_logger.info("Server log level set to DEBUG.")
    except Exception as exc:
        client_logger.warning("Could not set server log level: %s", exc)


# ─────────────────────────────────────────────
# 8. LangChain @tool wrappers
# ─────────────────────────────────────────────

@tool
async def reflect_answer_tool(
    draft_answer: str,
    original_query: str,
    constraints: str = "accuracy, completeness, no hallucinations",
) -> str:
    """
    Critiques and iteratively corrects a draft answer via the remote MCP
    Reflection tool. The server runs a Critic/Corrector loop using MCP
    Sampling — all LLM calls are executed by this client's sampling_callback.

    Use this tool when:
    - You have a draft answer and want to verify it before finalising.
    - The query is technical, high-stakes, or requires factual precision.
    - You suspect your answer may contain unverified claims.
    """
    client_logger.info("Invoking remote reflect_answer tool.")
    _t0 = time.perf_counter()

    _persist(
        interaction_type=MCPInteractionType.TOOL_INVOCATION,
        content=f"reflect_answer invoked. query='{original_query[:120]}' draft_len={len(draft_answer)}",
        namespace_path=NS.MCP_CLIENT_TOOL_CALL,
        component="agent_client",
        tool_name="reflect_answer",
        metadata={"constraints": constraints, "draft_length": len(draft_answer)},
    )

    if "reflect_answer" not in _remote_tools:
        return "Error: reflect_answer tool not available on server."

    result = await _remote_tools["reflect_answer"].ainvoke({
        "draft_answer": draft_answer,
        "original_query": original_query,
        "constraints": constraints,
    })

    latency = (time.perf_counter() - _t0) * 1000
    client_logger.info("reflect_answer tool returned (%.0f ms).", latency)

    _persist(
        interaction_type=MCPInteractionType.TOOL_INVOCATION,
        content=f"reflect_answer completed. result_len={len(result)} latency_ms={latency:.0f}",
        namespace_path=NS.MCP_SERVER_REFLECTION,
        component="mcp_server",
        tool_name="reflect_answer",
        latency_ms=latency,
        metadata={"result_length": len(result)},
    )
    return result


@tool
async def query_knowledge_tool(query: str) -> str:
    """
    Queries the remote Hierarchical CRAG knowledge base on the MCP server.

    The server performs:
      1. Multi-query expansion  — 3 semantic rephrasings
      2. Hierarchical retrieval — domain → section → chunk (3 levels)
      3. ToT evaluation         — 3-chain majority vote per chunk
      4. Tavily web fallback    — if internal KB is insufficient

    Use this tool when:
    - You need technical documentation about LangChain, MCP, FastMCP, or CRAG.
    - You want structured knowledge before drafting a technical answer.
    - The query involves framework usage, architecture, or design patterns.
    """
    client_logger.info("Invoking remote query_knowledge tool: '%s'", query[:80])
    _t0 = time.perf_counter()

    _persist(
        interaction_type=MCPInteractionType.TOOL_INVOCATION,
        content=f"query_knowledge invoked. query='{query}'",
        namespace_path=NS.MCP_CLIENT_TOOL_CALL,
        component="agent_client",
        tool_name="query_knowledge",
        metadata={"query": query},
    )

    if "query_knowledge" not in _remote_tools:
        return "Error: query_knowledge tool not available on server."

    result = await _remote_tools["query_knowledge"].ainvoke({"query": query})
    latency = (time.perf_counter() - _t0) * 1000

    # Fix: result may be a list of content blocks rather than a plain string.
    # Extract text before passing to _persist() which requires a str.
    if isinstance(result, str):
        result_str = result
    elif isinstance(result, list):
        result_str = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in result
        )
    else:
        result_str = str(result)

    client_logger.info("query_knowledge tool returned (%d chars, %.0f ms).", len(result_str), latency)

    _persist(
        interaction_type=MCPInteractionType.RESOURCE_READ,
        content=result_str[:2000],
        namespace_path=NS.MCP_SERVER_CRAG,
        component="mcp_server",
        tool_name="query_knowledge",
        latency_ms=latency,
        metadata={"query": query, "result_length": len(result_str)},
    )
    return result_str


# ─────────────────────────────────────────────
# 9. Agent construction
# ─────────────────────────────────────────────

def _build_agent():
    tools = [reflect_answer_tool, query_knowledge_tool]
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description[:120]}" for t in tools
    )
    tool_names = ", ".join(t.name for t in tools)

    system_prompt = f"""You are a meticulous Thinking Agent operating in a distributed \
MCP architecture. You reason carefully, never fabricate facts, and always verify your \
answers before finalising them.

## Available Tools
{tool_descriptions}

## Tool Usage Rules

1. **query_knowledge_tool** — Call this FIRST for any technical question about \
LangChain, MCP, FastMCP, CRAG, or system design.

2. **reflect_answer_tool** — ALWAYS call this on your draft answer before giving \
a Final Answer.

## Workflow

**Technical queries:**
query_knowledge_tool → draft answer → reflect_answer_tool → Final Answer

**General queries:**
draft answer → reflect_answer_tool → Final Answer

## ReAct Format (STRICT)

Thought: your reasoning step
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: I now have a verified answer.
Final Answer: complete, verified response.
"""

    agent = create_agent(
        model=primary_model,
        tools=tools,
        system_prompt=system_prompt,
    )
    client_logger.info("Agent built with tools: %s", [t.name for t in tools])
    return agent, tools


# ─────────────────────────────────────────────
# 10. Query runner
# ─────────────────────────────────────────────

async def run_query(agent: Any, user_query: str) -> str:
    client_logger.info("=" * 70)
    client_logger.info("Query: %s", user_query)

    _persist(
        interaction_type=MCPInteractionType.AGENT_REASONING,
        content=f"Agent received query: {user_query}",
        namespace_path=NS.AGENT_PLANNING,
        component="agent_client",
        metadata={"query": user_query},
    )

    # Hardcoded pre-fetch: always call query_knowledge_tool before the agent
    # starts reasoning. This guarantees Tavily fires for off-KB queries
    # regardless of what the agent decides to do on its own.
    client_logger.info("Pre-fetching knowledge context for query...")
    try:
        kb_context = await query_knowledge_tool.ainvoke({"query": user_query})
    except Exception as exc:
        client_logger.warning("Pre-fetch failed: %s", exc)
        kb_context = ""

    # Inject KB context into the initial message so the agent has it from the start
    augmented_query = user_query
    if kb_context and kb_context.strip():
        augmented_query = (
            f"{user_query}\n\n"
            f"[Pre-fetched Knowledge Context]:\n{kb_context}"
        )

    final_response = ""
    _t0 = time.perf_counter()

    async for event in agent.astream({"messages": [HumanMessage(content=augmented_query)]}):
        if "model" in event:
            for msg in event["model"].get("messages", []):
                if isinstance(msg, AIMessage):
                    # msg.content may be a list of content blocks (Gemini) or a plain string
                    if isinstance(msg.content, list):
                        content_str = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in msg.content
                        )
                    else:
                        content_str = msg.content or ""

                    if content_str.strip() and msg.tool_calls:
                        _persist(
                            interaction_type=MCPInteractionType.AGENT_REASONING,
                            content=content_str[:1000],
                            namespace_path=NS.AGENT_REASONING,
                            component="agent_client",
                            embed=False,
                        )
                    elif content_str.strip() and not msg.tool_calls:
                        final_response = content_str

    latency = (time.perf_counter() - _t0) * 1000

    _persist(
        interaction_type=MCPInteractionType.AGENT_FINAL_ANSWER,
        content=final_response,
        namespace_path=NS.AGENT_FINAL_ANSWER,
        component="agent_client",
        latency_ms=latency,
        metadata={"query": user_query, "response_length": len(final_response)},
    )

    client_logger.info("Answer:\n%s", final_response)
    client_logger.info("=" * 70)
    print(f"\nQuery: {user_query}\n\nAnswer:\n{final_response}\n")
    return final_response


# ─────────────────────────────────────────────
# 11. Entry point
# ─────────────────────────────────────────────

async def _async_main() -> None:
    await connect_to_server()
    agent, _ = _build_agent()

    queries = [
        "What is MCP Sampling and how does it work?",
        "How should I design a LangChain agent that uses MCP tools?",
    ]

    for query in queries:
        await run_query(agent, query)

    # Log shutdown
    stats = log_store.get_stats()
    client_logger.info(
        "Session complete. Log store stats: total_entries=%d sessions=%d",
        stats["total_entries"], stats["total_sessions"],
    )
    _persist(
        interaction_type=MCPInteractionType.SYSTEM_EVENT,
        content=f"Agent session ended. total_entries={stats['total_entries']}",
        namespace_path=NS.SYSTEM_SHUTDOWN,
        component="agent_client",
        embed=False,
    )
    log_store.close()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
