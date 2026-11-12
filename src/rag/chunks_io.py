"""Lectura de los chunks del Código Tributario (data/chunks.jsonl).

Pinecone guarda los vectores; BM25 es una búsqueda léxica que necesita los textos
EN MEMORIA. Este archivo (generado por la ingesta de la pre-entrega 4 con
`python -m src.ingest --dry-run`) está versionado en este repo para que BM25 y la
herramienta `read_article` funcionen en un clon limpio, sin reingestar.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from src import config


def load_chunks(path: Path | None = None) -> list[Document]:
    path = path or config.CHUNKS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Generalo con la ingesta de la pre-entrega 4 "
            "(`python -m src.ingest --dry-run`) y copialo a data/chunks.jsonl."
        )
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            docs.append(Document(page_content=record["page_content"], metadata=record["metadata"], id=record["id"]))
    return docs


def group_by_article(chunks: list[Document]) -> dict[str, list[Document]]:
    """Agrupa los chunks por número de artículo (en mayúsculas: '20-BIS'), en orden de chunk."""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.metadata.get("articulo", "?")).upper()].append(chunk)
    for parts in grouped.values():
        parts.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return dict(grouped)
