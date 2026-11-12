"""RAGSystem: recuperador híbrido (BM25 + vectorial en Pinecone), versión asíncrona.

Es el RAG de la pre-entrega 4, con dos cambios para poder usarlo dentro de un
sistema async (y, más adelante, detrás de FastAPI):
  * `abuscar()` usa `retriever.ainvoke`, que no bloquea el event loop.
  * Los imports pesados son perezosos y el retriever se inyecta por el constructor,
    así los tests usan un retriever falso sin Pinecone ni modelos.

Cómo se combinan los dos mundos:
  - BM25Retriever (léxico): vive en memoria, se arma con data/chunks.jsonl.
  - Retriever de PineconeVectorStore (semántico): consulta el índice en la nube.
  - EnsembleRetriever fusiona ambos RANKINGS con Reciprocal Rank Fusion ponderada.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal, Protocol

from langchain_core.documents import Document

from src import config
from src.rag.chunks_io import load_chunks

Mode = Literal["hybrid", "vector", "bm25"]

# Palabras vacías muy frecuentes (ya sin tildes): evitan ruido en BM25 con preguntas largas.
STOPWORDS_ES = frozenset(
    "de la el en y a los las del que se por con un una para al lo como su sus es son o u e ni si no "
    "mas pero sin sobre entre cuando cual cuales quien quienes le les este esta estos estas ese esa "
    "hay ser fue sera seran donde".split()
)


def tokenize_es(text: str) -> list[str]:
    """Tokenizador para BM25 en español: minúsculas, sin tildes, sin stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", text) if t not in STOPWORDS_ES]


class AsyncRetriever(Protocol):
    """Lo único que RAGSystem le pide a un retriever (los de LangChain lo cumplen)."""

    async def ainvoke(self, input: str, *args: Any, **kwargs: Any) -> list[Document]: ...


class RAGSystem:
    """Encapsula un retriever y devuelve siempre los top-k documentos (chunks)."""

    def __init__(self, retriever: AsyncRetriever, k: int = 5, mode: str = "hybrid"):
        self.retriever = retriever
        self.k = k
        self.mode = mode

    async def abuscar(self, query: str) -> list[Document]:
        """Top-k chunks para la consulta, ya fusionados y ordenados (no bloquea el event loop)."""
        docs = await self.retriever.ainvoke(query)
        return docs[: self.k]

    @classmethod
    def from_pinecone(
        cls,
        mode: Mode = "hybrid",
        k: int | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        candidates: int = 10,
    ) -> "RAGSystem":
        """Fábrica para uso real. Cargar BM25 y el modelo de embeddings es costoso:
        hacelo UNA vez al arrancar, no en cada consulta."""
        k = k or config.RAG_TOP_K
        if mode == "bm25":
            return cls(_build_bm25(load_chunks(), k), k, mode)
        if mode == "vector":
            return cls(_build_vector_retriever(k), k, mode)
        if mode != "hybrid":
            raise ValueError(f"modo inválido: {mode!r}")
        try:  # langchain >= 1.0 movió los retrievers "clásicos" a langchain-classic
            from langchain_classic.retrievers import EnsembleRetriever
        except ImportError:  # pragma: no cover
            from langchain.retrievers import EnsembleRetriever
        ensemble = EnsembleRetriever(
            retrievers=[_build_bm25(load_chunks(), candidates), _build_vector_retriever(candidates)],
            weights=list(weights),
        )
        return cls(ensemble, k, mode)


def _build_bm25(chunks: list[Document], k: int):
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize_es)


def _build_vector_retriever(k: int):
    from langchain_pinecone import PineconeVectorStore

    index = config.get_pinecone_client().Index(config.INDEX_NAME)
    store = PineconeVectorStore(index=index, embedding=config.get_embeddings(), namespace=config.NAMESPACE)
    return store.as_retriever(search_kwargs={"k": k})
