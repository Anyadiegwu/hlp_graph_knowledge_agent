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

from dotenv import find_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
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
from tenacity import wait_exponential

from agent_client.log_store import (
    HLPLogStore,
    LogEntry,
    MCPInteractionType,
    NS,
    get_log_store,
)

_REPO_ROOT  = Path(__file__).resolve().parents[3]
LOG_FILE    = _REPO_ROOT / "mcp_agent_system.log"
LOG_FORMAT  = "[%(asctime)s] [%(log_source)s] [%(levelname)s] %(message)s"
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


client_logger = _make_logger("agent.client",      "CLIENT")
server_logger = _make_logger("agent.server_relay", "SERVER")
client_logger.info("Flat log file : %s", LOG_FILE)

MAX_RETRY_ATTEMPTS = 3


class Settings(BaseSettings):
    groq_api_key:      SecretStr | None = None
    groq_model_name:   str              = "llama-3.3-70b-versatile"
    gemini_api_key:    SecretStr | None = None
    gemini_model_name: str              = "gemini-2.5-flash"
    model_temperature: float            = 0.0
    mcp_server_url:    str              = "http://localhost:8000/mcp"
    use_groq:          bool             = True
    log_db_path:       str              = ""
    ollama_model_name: str              = "llama3.2:3b"

    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

_db_path  = settings.log_db_path or (_REPO_ROOT / "mcp_agent_log.db")
log_store: HLPLogStore = get_log_store(_db_path)
client_logger.info("Vector log store: %s", _db_path)

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


RESILIENCE_STATE: dict[str, int] = {
    "fallback_activations":   0,
    "self_heal_iterations":   0,
    "absolute_fallback_hits": 0,
}


def _flush_resilience_state() -> None:
    _persist(
        interaction_type=MCPInteractionType.SYSTEM_EVENT,
        content=json.dumps(RESILIENCE_STATE),
        namespace_path=NS.SYSTEM_STARTUP,
        component="agent_client",
        metadata={"event": "resilience_state_flush", **RESILIENCE_STATE},
        embed=False,
    )


def _build_groq_fallback() -> BaseChatModel | None:
    if not settings.groq_api_key:
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model_name,
            temperature=settings.model_temperature,
            api_key=settings.groq_api_key,
        )
    except Exception as exc:
        client_logger.warning("Could not build Groq fallback model: %s", exc)
        return None


class _SelfHealRunnable:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def _extract_query(self, input_data: Any) -> str:
        if isinstance(input_data, dict):
            messages = input_data.get("messages", [])
            if messages:
                last = messages[-1]
                return last.content if isinstance(last.content, str) else str(last.content)
            return str(input_data.get("input", input_data))
        if isinstance(input_data, str):
            return input_data
        return str(input_data)

    async def ainvoke(self, input_data: Any, config: RunnableConfig | None = None) -> dict:
        RESILIENCE_STATE["fallback_activations"] += 1

        full_error_trace = ""
        if isinstance(input_data, dict):
            full_error_trace = str(input_data.get("error_trace", "unknown error"))

        original_query = self._extract_query(input_data)

        client_logger.warning(
            "\nCRITICAL: Primary Chain failed!\n"
            "Error Captured: %s\n"
            "\n--- [STEP 2] Routing to Self-Heal Chain with Error Trace... ---\n",
            full_error_trace[:300],
        )

        _persist(
            interaction_type=MCPInteractionType.ERROR,
            content=(
                f"[SELF-HEAL] Primary chain failed. Attempting LLM self-correction.\n"
                f"error_trace: {full_error_trace}\n"
                f"original_query: {original_query}"
            ),
            namespace_path=NS.AGENT_REASONING,
            component="agent_client",
            metadata={
                "event": "self_heal_activated",
                "error_trace": full_error_trace,
                "fallback_activations": RESILIENCE_STATE["fallback_activations"],
            },
            embed=False,
        )

        corrective_prompt = (
            f"You are a resilient AI assistant. One of your tool calls just failed "
            f"with the following error traceback:\n\n"
            f"ERROR:\n{full_error_trace}\n\n"
            f"Despite this failure, you must still answer the user's original query "
            f"as helpfully as possible using only your internal knowledge — do NOT "
            f"attempt to call any tools.\n\n"
            f"Original query: {original_query}\n\n"
            f"Provide a complete, honest answer. If you genuinely cannot answer "
            f"without the failed tool, say so clearly and explain what information "
            f"would be needed."
        )

        response_text: str | None = None
        model_used = type(self._model).__name__

        try:
            response = await self._model.ainvoke([HumanMessage(content=corrective_prompt)])
            response_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
        except Exception as primary_exc:
            client_logger.warning(
                "[SELF-HEAL] Primary model failed (%s) — trying Groq fallback.", primary_exc
            )
            groq = _build_groq_fallback()
            if groq is not None and type(self._model).__name__ != "ChatGroq":
                try:
                    response = await groq.ainvoke([HumanMessage(content=corrective_prompt)])
                    response_text = (
                        response.content
                        if isinstance(response.content, str)
                        else str(response.content)
                    )
                    model_used = settings.groq_model_name
                except Exception as groq_exc:
                    client_logger.error("[SELF-HEAL] Groq fallback also failed: %s", groq_exc)
                    raise groq_exc
            else:
                raise primary_exc

        RESILIENCE_STATE["self_heal_iterations"] += 1
        client_logger.info(
            "\nHEALED: Self-Heal Chain recovered successfully via %s!\n"
            "Recovery answer: %s\n",
            model_used,
            response_text[:300],
        )
        _persist(
            interaction_type=MCPInteractionType.AGENT_FINAL_ANSWER,
            content=f"[SELF-HEAL RECOVERY] {response_text}",
            namespace_path=NS.AGENT_FINAL_ANSWER,
            component="agent_client",
            metadata={
                "event": "self_heal_success",
                "model_used": model_used,
                "self_heal_iterations": RESILIENCE_STATE["self_heal_iterations"],
            },
        )
        return {"messages": [AIMessage(content=response_text)], "self_healed": True}


class _AbsoluteFallbackRunnable:
    async def ainvoke(self, input_data: Any, config: RunnableConfig | None = None) -> dict:
        RESILIENCE_STATE["absolute_fallback_hits"] += 1

        full_error_trace = ""
        if isinstance(input_data, dict):
            full_error_trace = str(input_data.get("error_trace", "unknown error"))

        client_logger.critical(
            "[ABSOLUTE FALLBACK] All retries and self-healing exhausted. "
            "error_trace=%s absolute_fallback_hits=%d",
            full_error_trace[:300],
            RESILIENCE_STATE["absolute_fallback_hits"],
        )

        try:
            _persist(
                interaction_type=MCPInteractionType.ERROR,
                content=(
                    f"[ABSOLUTE FALLBACK] Catastrophic execution failure.\n"
                    f"All retries ({MAX_RETRY_ATTEMPTS}) and self-healing chains exhausted.\n"
                    f"error_trace: {full_error_trace}\n"
                    f"session_id: {SESSION_ID}\n"
                    f"resilience_state: {json.dumps(RESILIENCE_STATE)}"
                ),
                namespace_path=NS.AGENT_FINAL_ANSWER,
                component="agent_client",
                metadata={
                    "event": "absolute_fallback",
                    "error_trace": full_error_trace,
                    "session_id": SESSION_ID,
                    **RESILIENCE_STATE,
                },
                embed=False,
            )
        except Exception:
            pass

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I'm sorry — I encountered a critical system error and was unable "
                        "to process your request. All automatic recovery attempts were "
                        "exhausted. Please try again or contact support if this persists."
                    )
                )
            ],
            "absolute_fallback": True,
            "error_trace": full_error_trace,
            "session_id": SESSION_ID,
            "resilience_state": dict(RESILIENCE_STATE),
        }


def _build_resilient_chain(runnable: Any, model: BaseChatModel) -> Any:
    retried = runnable.with_retry(
        stop_after_attempt=MAX_RETRY_ATTEMPTS,
        wait_exponential_jitter=True,
    )
    client_logger.info(
        "RunnableWithRetry configured: max_attempts=%d jitter=True",
        MAX_RETRY_ATTEMPTS,
    )

    from langchain_groq import ChatGroq
    healing_model = (
        ChatGroq(
            model=settings.groq_model_name,
            temperature=settings.model_temperature,
            api_key=settings.groq_api_key,
        )
        if settings.groq_api_key
        else model
    )

    resilient = retried.with_fallbacks(
        fallbacks=[
            RunnableLambda(_SelfHealRunnable(healing_model).ainvoke),
            RunnableLambda(_AbsoluteFallbackRunnable().ainvoke),
        ],
        exception_key="error_trace",
    )
    client_logger.info(
        "RunnableWithFallbacks configured: "
        "fallbacks=[_SelfHealRunnable, _AbsoluteFallbackRunnable] "
        "exception_key='error_trace'"
    )
    return resilient


def _build_primary_model() -> BaseChatModel:
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
    return ChatOllama(model=settings.ollama_model_name, temperature=settings.model_temperature)


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

_persist(
    interaction_type=MCPInteractionType.SYSTEM_EVENT,
    content=(
        f"Agent session started. "
        f"primary_model={type(primary_model).__name__} "
        f"sampling_model={type(sampling_model).__name__} "
        f"max_retry_attempts={MAX_RETRY_ATTEMPTS}"
    ),
    namespace_path=NS.SYSTEM_STARTUP,
    component="agent_client",
    embed=False,
)

async def sampling_callback(
    context: RequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    client_logger.info("MCP Sampling request | max_tokens=%s", params.maxTokens)
    _t0 = time.perf_counter()

    prompt_texts = []
    lc_messages  = []

    if getattr(params, "systemPrompt", None):
        lc_messages.append({"role": "system", "content": params.systemPrompt})
        prompt_texts.append(f"[system] {params.systemPrompt}")

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
        lc_messages.append({"role": getattr(msg, "role", "user"), "content": text})

    _persist(
        interaction_type=MCPInteractionType.SAMPLING_REQUEST,
        content="\n---\n".join(prompt_texts),
        namespace_path=NS.MCP_SERVER_SAMPLING,
        component="mcp_server",
        metadata={"max_tokens": params.maxTokens},
    )

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
        embed=(level >= logging.INFO),
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
            f"MCP connection established. tools={list(_remote_tools.keys())} "
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

    if isinstance(result, str):
        result_str = result
    elif isinstance(result, list):
        result_str = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in result
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


@tool
async def simulate_fault_tool(
    fault_type: str = "random",
    tool_target: str = "query_knowledge",
    severity: str = "medium",
) -> str:
    """
    Triggers a controlled fault injection on the MCP server for resilience testing.

    This tool deliberately causes the server to raise a specific error type so
    that the client's RunnableWithRetry and RunnableWithFallbacks self-healing
    chains are exercised with realistic failure traces.

    Parameters
    ----------
    fault_type  : bad_schema | runtime_error | timeout_sim | partial_payload | random
    tool_target : The logical tool name to annotate the fault against (for logs).
    severity    : low | medium | high — controls error intensity and sleep duration.

    Use this tool when:
    - You want to test that the resilience stack is working end-to-end.
    - You need to generate fault trace data for the explainability audit report.
    - You are running a resilience demo.
    """
    client_logger.info(
        "Invoking simulate_fault: fault_type=%s tool_target=%s severity=%s",
        fault_type, tool_target, severity,
    )
    _t0 = time.perf_counter()

    _persist(
        interaction_type=MCPInteractionType.TOOL_INVOCATION,
        content=(
            f"simulate_fault invoked. "
            f"fault_type={fault_type} tool_target={tool_target} severity={severity}"
        ),
        namespace_path=NS.MCP_CLIENT_TOOL_CALL,
        component="agent_client",
        tool_name="simulate_fault",
        metadata={"fault_type": fault_type, "tool_target": tool_target, "severity": severity},
    )

    if "simulate_fault" not in _remote_tools:
        return "Error: simulate_fault tool not available on server."

    result = await _remote_tools["simulate_fault"].ainvoke({
        "fault_type": fault_type,
        "tool_target": tool_target,
        "severity": severity,
    })

    latency = (time.perf_counter() - _t0) * 1000
    _persist(
        interaction_type=MCPInteractionType.TOOL_INVOCATION,
        content=f"simulate_fault returned (unexpectedly): {str(result)[:200]}",
        namespace_path=NS.MCP_SERVER_TOOL,
        component="mcp_server",
        tool_name="simulate_fault",
        latency_ms=latency,
    )
    return str(result)


def _build_agent():
    tools = [reflect_answer_tool, query_knowledge_tool, simulate_fault_tool]
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

3. **simulate_fault_tool** — Use ONLY when explicitly asked to run a resilience test \
or fault injection demo. Never use this during normal queries.

## Workflow

**Technical queries:**
query_knowledge_tool → draft answer → reflect_answer_tool → Final Answer

**General queries:**
draft answer → reflect_answer_tool → Final Answer

**Resilience tests:**
simulate_fault_tool (with appropriate fault_type) → observe recovery → Final Answer

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

    resilient_agent = _build_resilient_chain(agent, primary_model)

    final_response = ""
    _t0 = time.perf_counter()

    try:
        result = await resilient_agent.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}
        )

        if isinstance(result, dict) and "absolute_fallback" in result:
            final_response = result["messages"][0].content
            client_logger.critical("[ABSOLUTE FALLBACK RESPONSE] %s", final_response)
        elif isinstance(result, dict) and result.get("self_healed"):
            final_response = result["messages"][-1].content
            client_logger.warning("[SELF-HEAL RESPONSE] %s", final_response[:200])
        else:
            messages = result.get("messages", []) if isinstance(result, dict) else []
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    if isinstance(msg.content, list):
                        content = " ".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in msg.content
                        )
                    elif isinstance(msg.content, str):
                        content = msg.content
                    else:
                        content = str(msg.content)

                    if content.strip() and not msg.tool_calls:
                        final_response = content
                        break

    except Exception as exc:
        client_logger.critical("Unhandled exception escaped resilient chain: %s", exc)
        final_response = f"Critical error: {exc}"

    latency = (time.perf_counter() - _t0) * 1000

    _persist(
        interaction_type=MCPInteractionType.AGENT_FINAL_ANSWER,
        content=final_response,
        namespace_path=NS.AGENT_FINAL_ANSWER,
        component="agent_client",
        latency_ms=latency,
        metadata={
            "query": user_query,
            "response_length": len(final_response),
            **RESILIENCE_STATE,
        },
    )

    _flush_resilience_state()

    client_logger.info("Answer:\n%s", final_response)
    client_logger.info("=" * 70)
    print(f"\nQuery: {user_query}\n\nAnswer:\n{final_response}\n")
    return final_response


async def run_resilience_test() -> None:
    client_logger.info("=" * 70)
    client_logger.info("\n--- [RESILIENCE TEST] Sending fault to primary chain... ---\n")

    async def _fault_call(_: dict) -> dict:
        if "simulate_fault" not in _remote_tools:
            raise RuntimeError("simulate_fault tool not registered.")
        try:
            await _remote_tools["simulate_fault"].ainvoke({
                "fault_type": "runtime_error",
                "tool_target": "query_knowledge",
                "severity":    "medium",
            })
        except Exception as exc:
            raise RuntimeError(
                f"Simulated fault propagated to resilient chain: {exc}"
            ) from exc
        raise RuntimeError(
            "Simulated execution failure in 'query_knowledge': "
            "downstream dependency returned HTTP 503. Retry budget may be available."
        )

    fault_runnable  = RunnableLambda(_fault_call)
    resilient_fault = _build_resilient_chain(fault_runnable, primary_model)

    _persist(
        interaction_type=MCPInteractionType.TOOL_INVOCATION,
        content="Resilience test started: direct fault injection bypassing agent loop.",
        namespace_path=NS.MCP_CLIENT_TOOL_CALL,
        component="agent_client",
        tool_name="simulate_fault",
        metadata={"test": "resilience_demo", "fault_type": "runtime_error"},
        embed=False,
    )

    try:
        result = await resilient_fault.ainvoke(
            {"messages": [HumanMessage(content="resilience test input")]}
        )

        if result.get("absolute_fallback"):
            client_logger.critical(
                "\nCRITICAL: Primary chain failed AND self-heal failed!\n"
                "Error Captured: %s\n"
                "\n--- [ABSOLUTE FALLBACK] Hardcoded safe response returned. ---\n"
                "absolute_fallback_hits=%d",
                result.get("error_trace", "")[:300],
                RESILIENCE_STATE["absolute_fallback_hits"],
            )
        elif result.get("self_healed"):
            client_logger.info(
                "\nHEALED: Self-Heal Chain recovered successfully!\n"
                "Recovery answer: %s\n"
                "\n--- [RESILIENCE TEST COMPLETE] self_heal_iterations=%d ---\n",
                result["messages"][-1].content[:300],
                RESILIENCE_STATE["self_heal_iterations"],
            )

    except Exception as exc:
        client_logger.error("[RESILIENCE TEST] Unhandled exception: %s", exc)

    _flush_resilience_state()
    client_logger.info(
        "Resilience summary: fallback_activations=%d "
        "self_heal_iterations=%d absolute_fallback_hits=%d",
        RESILIENCE_STATE["fallback_activations"],
        RESILIENCE_STATE["self_heal_iterations"],
        RESILIENCE_STATE["absolute_fallback_hits"],
    )
    client_logger.info("=" * 70)


async def _async_main() -> None:
    await connect_to_server()
    agent, _ = _build_agent()

    queries = [
        "What is MCP Sampling and how does it work?",
        "How should I design a LangChain agent that uses MCP tools?",
        "What are the LangChain resilience primitives for building fault-tolerant chains?",
    ]

    for query in queries:
        await run_query(agent, query)

    await run_resilience_test()

    stats = log_store.get_stats()
    client_logger.info(
        "Session complete. Log store stats: total_entries=%d sessions=%d",
        stats["total_entries"], stats["total_sessions"],
    )
    client_logger.info(
        "Resilience summary: fallback_activations=%d self_heal_iterations=%d "
        "absolute_fallback_hits=%d",
        RESILIENCE_STATE["fallback_activations"],
        RESILIENCE_STATE["self_heal_iterations"],
        RESILIENCE_STATE["absolute_fallback_hits"],
    )
    _persist(
        interaction_type=MCPInteractionType.SYSTEM_EVENT,
        content=(
            f"Agent session ended. total_entries={stats['total_entries']} "
            f"resilience_state={json.dumps(RESILIENCE_STATE)}"
        ),
        namespace_path=NS.SYSTEM_SHUTDOWN,
        component="agent_client",
        embed=False,
    )
    log_store.close()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()