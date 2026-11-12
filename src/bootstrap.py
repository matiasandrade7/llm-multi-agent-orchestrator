"""Arma el sistema REAL: Gemini + RAG híbrido sobre Pinecone + grafo.

Se hace una sola vez al arrancar (cargar BM25 y el modelo de embeddings es lo más lento).
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from src.graph import build_graph
from src.llm import build_llm
from src.rag.chunks_io import load_chunks
from src.rag.system import RAGSystem


def build_real_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    llm = build_llm()
    chunks = load_chunks()
    rag = RAGSystem.from_pinecone()
    return build_graph(llm, rag, chunks, checkpointer)
