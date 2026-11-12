"""Piezas compartidas por los tests: construyen el grafo REAL con un LLM guionado.

Estos tests verifican el CABLEADO (enrutamiento, guardas, reducers, aislamiento de errores).
Lo que no pueden verificar es la decisión del LLM real: eso se ve ejecutando `python main.py demo`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.graph import build_graph
from src.rag.chunks_io import load_chunks
from src.rag.system import RAGSystem
from src.schemas import SupervisorDecision
from src.simulation import LexicalRetriever, ScriptedChatModel, ScriptedDecider, tool_call

__all__ = [
    "AIMessage", "SupervisorDecision", "ScriptedChatModel", "ScriptedDecider", "tool_call",
    "make_graph", "decision", "research_script", "analyst_script",
]


def decision(next_: str, instruction: str = "", reason: str = "motivo de prueba") -> SupervisorDecision:
    return SupervisorDecision(next=next_, instruction=instruction, reason=reason)


def research_script(tag: str = "") -> list[AIMessage]:
    """Un ciclo ReAct completo del investigador: busca, lee el artículo 94 y entrega el informe."""
    return [
        tool_call("search_articles", {"query": "multa por mora en el pago"}, f"s{tag}"),
        tool_call("read_article", {"number": "94"}, f"r{tag}"),
        AIMessage(content="Informe: Art. 94 (Mora): multa del 20% pasados 90 días corridos."),
    ]


def analyst_script(tag: str = "") -> list[AIMessage]:
    return [
        tool_call("calculate_percentage", {"base": 50000, "percentage": 20}, f"p{tag}"),
        AIMessage(content="20% de 50000 = 10000 (Art. 94)."),
    ]


def make_graph(llm_responses: list[AIMessage], decisions: list, checkpointer=None):
    """Devuelve (grafo, llm, decisor). El retriever es léxico y offline, sobre los chunks reales."""
    chunks = load_chunks()
    llm = ScriptedChatModel(responses=llm_responses)
    decider = ScriptedDecider(decisions)
    rag = RAGSystem(LexicalRetriever(chunks), k=5, mode="test")
    return build_graph(llm, rag, chunks, checkpointer, decider=decider), llm, decider
