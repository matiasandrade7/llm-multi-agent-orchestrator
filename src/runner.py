"""Ejecución de una consulta y presentación de la traza.

`run_query` es el único punto de entrada para correr el grafo (lo usan el CLI, el notebook
y, en la pre-entrega 7, el worker de la API). Siempre fija un `recursion_limit` y captura
el caso de que se agote, para que un bug de enrutamiento no cuelgue ni gaste tokens sin límite.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src import config

logger = logging.getLogger("orchestrator.runner")


async def run_query(graph: CompiledStateGraph, question: str, thread_id: str | None = None) -> dict[str, Any]:
    """Corre el grafo completo y devuelve el estado final."""
    initial = {"question": question, "messages": [HumanMessage(content=question)]}
    run_config = {
        "recursion_limit": config.RECURSION_LIMIT,
        # Un thread_id por consulta: los acumuladores (contributions, traces...) no se mezclan entre tareas.
        "configurable": {"thread_id": thread_id or uuid4().hex},
    }
    try:
        return await graph.ainvoke(initial, config=run_config)
    except GraphRecursionError:
        logger.error("Se agotó el recursion_limit (%s) para: %s", config.RECURSION_LIMIT, question)
        return {
            "question": question,
            "final_answer": "No se pudo completar la consulta: el sistema alcanzó el límite de pasos.",
            "contributions": [],
            "traces": [],
            "errors": [f"GraphRecursionError: límite de {config.RECURSION_LIMIT} pasos"],
        }


def format_trace(state: dict[str, Any]) -> str:
    """Tabla de texto con la secuencia de nodos: quién actuó, qué decidió y cuánto tardó."""
    rows = state.get("traces", [])
    if not rows:
        return "(sin trazas)"
    lines = [f"{'#':>2}  {'nodo':<11} {'paso':>4} {'ms':>9}  detalle", "-" * 100]
    for i, t in enumerate(rows, 1):
        lines.append(f"{i:>2}  {t['node']:<11} {t['step']:>4} {t['ms']:>9.1f}  {t['detail']}")
    return "\n".join(lines)


def save_trace(state: dict[str, Any], name: str, path: Path | None = None, origin: str = "real") -> Path:
    """Guarda pregunta, trazas, contribuciones y respuesta en traces/<name>.json (evidencia reproducible)."""
    path = path or config.TRACES_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"origen": origin, "fecha": datetime.now().isoformat(timespec="seconds")},
        "pregunta": state.get("question"),
        "pasos_supervisor": state.get("step_count"),
        "trazas": state.get("traces", []),
        "contribuciones": state.get("contributions", []),
        "errores": state.get("errors", []),
        "respuesta_final": state.get("final_answer"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
