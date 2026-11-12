"""Configuración central del proyecto.

Todo lo que puede cambiar entre máquinas o ejecuciones (claves, nombre del índice,
modelo, límites del grafo) se lee de variables de entorno / archivo .env. Ningún
otro módulo tiene IDs de índices, nombres de modelos ni números "mágicos".

Los imports pesados (pinecone, sentence-transformers) se hacen dentro de las
funciones: así los tests y `python main.py simular` corren sin descargar modelos.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT: Path = Path(__file__).resolve().parent.parent

# --- LLM --------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
LLM_MODEL: str | None = os.getenv("LLM_MODEL") or None

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}

# --- Límites del grafo (defensa contra el "Supervisor Infinito") -------------
# Máximo de decisiones del supervisor por consulta.
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "6"))
# Cuántas veces puede invocarse el mismo especialista (1 ejecución + 1 refinamiento).
MAX_CALLS_PER_AGENT: int = int(os.getenv("MAX_CALLS_PER_AGENT", "2"))
# Techo duro de LangGraph para todo el grafo (red de seguridad adicional).
RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", "30"))
# Techo de pasos del ciclo ReAct interno de cada especialista.
SPECIALIST_RECURSION_LIMIT: int = int(os.getenv("SPECIALIST_RECURSION_LIMIT", "12"))
# Cuánta evidencia (en caracteres) se le pasa al analista y a la síntesis final.
MAX_EVIDENCE_CHARS: int = int(os.getenv("MAX_EVIDENCE_CHARS", "6000"))

# --- Resiliencia ante fallas del proveedor (429, 503, timeouts) ----------------
RETRY_ATTEMPTS: int = int(os.getenv("LLM_RETRIES", "3"))
# Espera exponencial con jitter entre reintentos. Los tests lo desactivan para no esperar.
RETRY_BACKOFF: bool = os.getenv("LLM_RETRY_BACKOFF", "true").lower() != "false"

# --- RAG (índice Pinecone de la pre-entrega 4) -------------------------------
INDEX_NAME: str = os.getenv("INDEX_NAME", "codigo-tributario-rag")
NAMESPACE: str = os.getenv("NAMESPACE", "codigo-tributario")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
CHUNKS_PATH: Path = Path(os.getenv("CHUNKS_PATH", str(ROOT / "data" / "chunks.jsonl")))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

# --- Trazas -----------------------------------------------------------------
TRACES_DIR: Path = ROOT / "traces"


def require_env(name: str) -> str:
    """Devuelve una variable obligatoria o falla con un mensaje claro."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}. Copiá .env.example a .env y completala.")
    return value


@lru_cache(maxsize=1)
def get_embeddings():
    """Carga el modelo de embeddings una sola vez por proceso (es lo más pesado del sistema)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    class _E5Embeddings(HuggingFaceEmbeddings):
        """Los modelos E5 esperan los prefijos 'query: ' y 'passage: '."""

        def embed_documents(self, texts):
            return super().embed_documents([f"passage: {t}" for t in texts])

        def embed_query(self, text):
            return super().embed_query(f"query: {text}")

    cls = _E5Embeddings if "e5" in EMBEDDING_MODEL.lower() else HuggingFaceEmbeddings
    return cls(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def get_pinecone_client():
    from pinecone import Pinecone

    return Pinecone(api_key=require_env("PINECONE_API_KEY"))
