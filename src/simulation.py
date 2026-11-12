"""Modo SIMULACIÓN: corre el grafo completo sin API keys, ni Pinecone, ni red.

Sirve para (1) ver el flujo de delegación de punta a punta, (2) probar el cableado en tests y
(3) generar el diagrama del grafo. Está rotulado como simulación a propósito: NO prueba la
calidad del LLM real ni del RAG real; prueba que la orquestación (supervisor, guardas,
reducers, especialistas, síntesis) funciona.

Qué es real y qué es simulado:
  * REAL: el grafo, el supervisor y sus guardas, las herramientas del analista y el texto de
    los artículos del Código Tributario (data/chunks.jsonl).
  * SIMULADO: el LLM (responde un guion cargado de antemano) y el retriever (búsqueda léxica
    local simple en lugar de Pinecone + BM25 + embeddings).
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.graph import build_graph
from src.rag.chunks_io import load_chunks
from src.rag.system import RAGSystem, tokenize_es
from src.schemas import SupervisorDecision

SIM_QUESTION = (
    "Pagué con 100 días de atraso un tributo de $50.000. "
    "¿Qué multa por mora corresponde y de cuánto sería?"
)


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """Un AIMessage que 'pide' ejecutar una herramienta."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """LLM falso: entrega, en orden, las respuestas cargadas. Guarda lo que recibió en cada llamada."""

    responses: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        response = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class ScriptedDecider:
    """Reemplaza al `llm.with_structured_output(SupervisorDecision)` con decisiones guionadas."""

    def __init__(self, decisions: list[SupervisorDecision | Exception]):
        self.decisions = decisions
        self.seen: list[list[BaseMessage]] = []
        self.index = 0

    async def ainvoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> SupervisorDecision:
        self.seen.append(list(messages))
        decision = self.decisions[self.index]
        self.index += 1
        if isinstance(decision, Exception):
            raise decision
        return decision


class LexicalRetriever:
    """Retriever offline: ordena los chunks por palabras en común con la consulta."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.calls: list[str] = []

    async def ainvoke(self, query: str, *args: Any, **kwargs: Any) -> list[Document]:
        self.calls.append(query)
        wanted = set(tokenize_es(query))
        scored = [(len(wanted & set(tokenize_es(c.page_content))), c) for c in self.chunks]
        return [c for score, c in sorted(scored, key=lambda x: -x[0]) if score > 0][:10]


def _guion_llm() -> list[AIMessage]:
    return [
        # --- Investigador: busca, lee el artículo completo y entrega un informe ---
        tool_call("search_articles", {"query": "multa por mora en el pago de tributos"}, "c1"),
        tool_call("read_article", {"number": "94"}, "c2"),
        AIMessage(
            content=(
                "[SIMULACIÓN] Informe del investigador. Art. 94 (Mora): la mora se sanciona con una multa sobre el "
                "tributo no pagado en término y un recargo mensual. Multa: A) 5% si se paga dentro de los 5 días "
                "hábiles siguientes al vencimiento; B) 10% entre ese plazo y 90 días corridos; C) 20% pasados los "
                "90 días corridos. El recargo mensual lo fija el Poder Ejecutivo (no figura la tasa en el artículo)."
            )
        ),
        # --- Analista: aplica el 20% (100 días > 90) con la herramienta ---
        tool_call("calculate_percentage", {"base": 50000, "percentage": 20}, "c3"),
        AIMessage(
            content=(
                "[SIMULACIÓN] Con 100 días de atraso corresponde el tramo C del Art. 94 (más de 90 días): 20%. "
                "Cálculo: 20% de 50000 = 10000. Dato faltante: la tasa del recargo mensual (la fija el Poder "
                "Ejecutivo y no figura en el artículo), por lo que no se calculó."
            )
        ),
        # --- Síntesis final ---
        AIMessage(
            content=(
                "[SIMULACIÓN] Al pagar con 100 días de atraso corresponde la multa del 20% sobre el tributo no "
                "pagado en término (Art. 94, literal C). Sobre $50.000, la multa es de $10.000. Además se aplica "
                "un recargo mensual, cuya tasa fija el Poder Ejecutivo y no figura en el artículo, por lo que no "
                "puede calcularse con este texto.\n\n"
                "Información orientativa basada en el texto del Código Tributario; no constituye asesoramiento legal."
            )
        ),
    ]


def _guion_decisiones() -> list[SupervisorDecision]:
    return [
        SupervisorDecision(
            next="research",
            instruction="Buscá en el Código Tributario el artículo sobre mora y multas por pago fuera de plazo, "
            "con los porcentajes y plazos de cada tramo.",
            reason="Todavía no hay evidencia: hay que investigar primero.",
        ),
        SupervisorDecision(
            next="analyst",
            instruction="Con los tramos de la evidencia, determiná cuál corresponde a 100 días de atraso y "
            "calculá la multa sobre $50.000.",
            reason="La pregunta pide un monto y ya hay artículo citado.",
        ),
        SupervisorDecision(
            next="FINISH",
            instruction="",
            reason="Rúbrica cumplida: artículo citado, cálculo con fórmula y dato faltante declarado.",
        ),
    ]


def build_simulation_graph(checkpointer=None):
    """Grafo real con LLM guionado y retriever offline. Devuelve (grafo, llm, decisor, retriever)."""
    chunks = load_chunks()
    retriever = LexicalRetriever(chunks)
    llm = ScriptedChatModel(responses=_guion_llm())
    decider = ScriptedDecider(_guion_decisiones())
    graph = build_graph(llm, RAGSystem(retriever, k=5, mode="simulacion"), chunks, checkpointer, decider=decider)
    return graph, llm, decider, retriever
