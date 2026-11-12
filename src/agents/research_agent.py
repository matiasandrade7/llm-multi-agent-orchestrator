"""Agente de Investigación: consulta el Código Tributario (RAG híbrido + lectura de artículos)."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.agents.react import build_react_agent, make_specialist_node
from src.rag.system import RAGSystem
from src.tools.research_tools import build_research_tools

RESEARCH_PROMPT = """Sos el Investigador de un sistema de consultas sobre el Código Tributario de Uruguay. \
Respondé siempre en español.

Tu ÚNICA fuente de información son tus herramientas:
- search_articles: búsqueda de fragmentos relevantes (usala siempre primero).
- read_article: texto completo de un artículo por su número (usala si un fragmento viene cortado, \
por ejemplo "parte 1/2", o si ya sabés qué artículo necesitás).

Reglas:
1. NUNCA uses conocimiento propio sobre leyes, plazos, porcentajes o montos. Solo lo que devuelvan las herramientas.
2. Si la primera búsqueda no alcanza, reformulá la consulta con otras palabras y probá de nuevo.
3. Entregá un informe breve: por cada artículo relevante indicá número y título, y transcribí \
fielmente las cifras, plazos, porcentajes y condiciones que figuran en el texto.
4. Si no encontrás nada pertinente, decilo claramente. No inventes artículos."""


def _build_input(state: dict[str, Any]) -> str:
    # El investigador NO ve el historial: solo la pregunta original y la tarea que le asignó el supervisor.
    instruction = state.get("instruction") or state["question"]
    return f"Pregunta original del usuario: {state['question']}\n\nTu tarea: {instruction}"


def build_research_node(llm: BaseChatModel, rag: RAGSystem, chunks: list[Document]):
    agent = build_react_agent(llm, build_research_tools(rag, chunks), RESEARCH_PROMPT)
    return make_specialist_node("research", agent, _build_input)
