"""Herramientas del Investigador (lo ÚNICO que este agente puede hacer).

Ideas de la clase 5 aplicadas acá:
  * El docstring de cada @tool es lo único que el LLM lee para decidir cuándo usarla.
  * Los argumentos se validan con Pydantic (`args_schema`) ANTES de ejecutar.
  * Los resultados se arman con un modelo Pydantic y se devuelven como JSON.
  * Los errores no rompen el programa: vuelven como JSON con la clave "error" para
    que el LLM decida si reintenta con otra consulta.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from src.rag.system import RAGSystem
from src.rag.chunks_io import group_by_article
from src.schemas import ArticleFound, ReadArticleInput, SearchArticlesInput, SearchResult

logger = logging.getLogger("orchestrator.tools")

# Un artículo largo no debe llenar el contexto del LLM: se recorta (el texto completo sigue en los chunks).
MAX_CHARS_PER_ARTICLE = 1800


def _to_found(doc: Document) -> ArticleFound:
    meta = doc.metadata
    return ArticleFound(
        article=str(meta.get("articulo", "?")),
        title=str(meta.get("titulo_articulo", "")),
        page=int(meta["page"]) if meta.get("page") is not None else None,
        category=str(meta.get("categoria", "")),
        content=doc.page_content[:MAX_CHARS_PER_ARTICLE],
    )


def build_research_tools(rag: RAGSystem, chunks: list[Document]) -> list[BaseTool]:
    """Crea las herramientas del Investigador sobre un RAGSystem y los chunks ya cargados."""
    by_article = group_by_article(chunks)

    @tool("search_articles", args_schema=SearchArticlesInput)
    async def search_articles(query: str) -> str:
        """Busca en el Código Tributario de Uruguay los fragmentos más relevantes para una consulta
        (búsqueda híbrida léxica + semántica). Usar SIEMPRE primero, con una consulta en lenguaje
        natural. Devuelve un JSON con "results": una lista de {article, title, page, content}.
        Un artículo largo puede venir partido en partes ("parte 1/2"): si el fragmento parece
        cortado, usá read_article para traer el texto completo. Si falla, devuelve "error"."""
        try:
            docs = await rag.abuscar(query)
        except Exception as exc:  # frontera de E/S (Pinecone, red): se informa al LLM, no se rompe el grafo
            logger.warning("search_articles falló: %s: %s", type(exc).__name__, exc)
            return SearchResult(ok=False, query=query, error=f"{type(exc).__name__}: {exc}").model_dump_json()
        return SearchResult(query=query, results=[_to_found(d) for d in docs]).model_dump_json()

    @tool("read_article", args_schema=ReadArticleInput)
    async def read_article(number: str) -> str:
        """Devuelve el texto COMPLETO de un artículo del Código Tributario a partir de su número
        (por ejemplo '94' o '20-BIS'), uniendo todas sus partes. Usar cuando search_articles
        devolvió un artículo relevante pero cortado, o cuando ya sabés qué artículo necesitás.
        Devuelve un JSON con "results" (un solo artículo) o con "error" si el número no existe."""
        key = number.upper()
        parts = by_article.get(key)
        if not parts:
            return SearchResult(ok=False, query=number, error=f"No existe el artículo {number}.").model_dump_json()
        full = " ".join(p.page_content for p in parts)
        first = parts[0].metadata
        found = ArticleFound(
            article=str(first.get("articulo", key)),
            title=str(first.get("titulo_articulo", "")),
            page=int(first["page"]) if first.get("page") is not None else None,
            category=str(first.get("categoria", "")),
            content=full[: MAX_CHARS_PER_ARTICLE * 2],
        )
        return SearchResult(query=number, results=[found]).model_dump_json()

    return [search_articles, read_article]
