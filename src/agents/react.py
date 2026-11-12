"""Fábrica de especialistas: un mini-agente ReAct reutilizable (patrón de la pre-entrega 5).

    START -> model --(¿pidió herramientas?)--> tools
               ^                                  |
               +----------------------------------+   (el ciclo: razonar -> actuar -> observar)
             si no pidió herramientas -> END

Cada especialista es un SUBGRAFO con sus propias herramientas acotadas. El nodo que lo
envuelve (`make_specialist_node`) le entrega SOLO una instrucción de entrada (no el
historial completo) y devuelve al grafo principal una `Contribution` ya validada.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src import config
from src.schemas import AgentName, Contribution, TraceEvent

logger = logging.getLogger("orchestrator.agents")

# Herramientas cuyo resultado trae artículos del Código (para registrar qué se consultó).
_ARTICLE_TOOLS = {"search_articles", "read_article"}


def build_react_agent(llm: BaseChatModel, tools: list, system_prompt: str) -> CompiledStateGraph:
    """Arma el subgrafo ReAct de un especialista con sus herramientas y su prompt de rol."""
    llm_with_tools = llm.bind_tools(tools).with_retry(
        stop_after_attempt=config.RETRY_ATTEMPTS, wait_exponential_jitter=config.RETRY_BACKOFF
    )

    async def model_node(state: MessagesState) -> dict[str, Any]:
        response = await llm_with_tools.ainvoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", model_node)
    # handle_tool_errors=True: si una herramienta falla (p. ej. argumentos inválidos), el error
    # vuelve al LLM como mensaje en vez de romper el programa.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


def extract_text(message: BaseMessage) -> str:
    """Normaliza el `content` (Gemini a veces devuelve una lista de bloques en vez de un str)."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


@dataclass
class SpecialistRun:
    text: str = ""
    articles: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


async def run_specialist(agent: CompiledStateGraph, user_input: str, recursion_limit: int) -> SpecialistRun:
    """Ejecuta el subgrafo con UN solo mensaje de entrada y extrae texto final, herramientas y artículos."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": recursion_limit},
    )
    messages: list[BaseMessage] = result["messages"]

    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    run = SpecialistRun(text=extract_text(final).strip() if final else "")

    articles: dict[str, None] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        run.tools_used.append(message.name or "?")
        if message.name in _ARTICLE_TOOLS:
            try:
                for item in json.loads(extract_text(message)).get("results", []):
                    articles.setdefault(str(item["article"]), None)
            except (ValueError, AttributeError, KeyError, TypeError):
                continue  # la herramienta devolvió un error en texto plano: no hay artículos que registrar
    run.articles = list(articles)
    return run


def make_specialist_node(
    name: AgentName,
    agent: CompiledStateGraph,
    build_input: Callable[[dict[str, Any]], str],
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Envuelve un especialista como nodo del grafo principal.

    Garantiza que una falla del especialista NO rompe el grafo: se registra como
    `Contribution(status="error")` y en `errors`, y el supervisor decide qué hacer.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        step = state.get("step_count", 0)
        instruction = state.get("instruction") or state["question"]
        errors: list[str] = []
        try:
            run = await run_specialist(agent, build_input(state), config.SPECIALIST_RECURSION_LIMIT)
            status, output = "ok", run.text or "(el agente no devolvió texto)"
            articles, tools_used = run.articles, run.tools_used
        except Exception as exc:  # frontera del especialista: cualquier falla se aísla y se informa
            logger.exception("Falló el especialista %s", name)
            status, output, articles, tools_used = "error", f"{type(exc).__name__}: {exc}", [], []
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

        ms = round((time.perf_counter() - t0) * 1000, 1)
        contribution = Contribution(
            agent=name, step=step, instruction=instruction, output=output,
            articles=articles, tools_used=tools_used, status=status, ms=ms,
        )
        detail = f"{status} · herramientas: {', '.join(tools_used) or '-'} · artículos: {', '.join(articles) or '-'}"
        trace = TraceEvent(node=name, step=step, detail=detail, ms=ms)
        return {
            "contributions": [contribution.model_dump()],
            "traces": [trace.model_dump()],
            "errors": errors,
        }

    node.__name__ = f"{name}_node"
    return node
