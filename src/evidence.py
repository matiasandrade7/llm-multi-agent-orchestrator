"""Utilidades para leer las contribuciones acumuladas en el estado.

El analista y la síntesis final no reciben "todo el historial": reciben la
EVIDENCIA, un resumen acotado de lo que aportó el Investigador. Esto evita la
contaminación de contexto (pasar metadata y mensajes irrelevantes a cada agente).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src import config


def contributions_of(state: dict[str, Any], agent: str, only_ok: bool = True) -> list[dict]:
    return [
        c
        for c in state.get("contributions", [])
        if c["agent"] == agent and (c["status"] == "ok" or not only_ok)
    ]


def calls_per_agent(state: dict[str, Any]) -> Counter:
    """Cuántas veces se ejecutó cada especialista (incluye las ejecuciones con error)."""
    return Counter(c["agent"] for c in state.get("contributions", []))


def has_valid_research(state: dict[str, Any]) -> bool:
    """¿El investigador aportó al menos un artículo consultado? Es la condición mínima de evidencia."""
    return any(c["articles"] for c in contributions_of(state, "research"))


def cited_articles(state: dict[str, Any]) -> list[str]:
    """Artículos consultados por el investigador, sin repetir y en orden de aparición."""
    seen: dict[str, None] = {}
    for c in contributions_of(state, "research"):
        for article in c["articles"]:
            seen.setdefault(article, None)
    return list(seen)


def collect_evidence(state: dict[str, Any], agents: tuple[str, ...] = ("research",)) -> str:
    """Texto de las contribuciones de los agentes pedidos, acotado a MAX_EVIDENCE_CHARS en total.

    El presupuesto se reparte entre los bloques: así un informe largo del investigador
    no deja afuera lo que calculó el analista.
    """
    blocks: list[str] = []
    for c in state.get("contributions", []):
        if c["agent"] in agents and c["status"] == "ok":
            arts = ", ".join(c["articles"]) or "ninguno"
            blocks.append(f"[{c['agent']} · artículos consultados: {arts}]\n{c['output']}")
    if not blocks:
        return ""
    per_block = max(500, config.MAX_EVIDENCE_CHARS // len(blocks))
    return "\n\n".join(block[:per_block] for block in blocks)
