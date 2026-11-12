"""Pruebas de las herramientas y de los esquemas Pydantic (sin red ni API keys)."""

import json

import pytest
from pydantic import ValidationError

from src.rag.chunks_io import group_by_article, load_chunks
from src.rag.system import RAGSystem, tokenize_es
from src.schemas import ReadArticleInput, SearchArticlesInput, SupervisorDecision
from src.simulation import LexicalRetriever
from src.tools.analyst_tools import add_years, calculate_expression, calculate_percentage, days_between, safe_eval
from src.tools.research_tools import build_research_tools


# --------------------------- calculadora segura ---------------------------
def test_safe_eval_calcula_expresiones_validas() -> None:
    assert safe_eval("50000 + 50000 * 0.2") == 60000.0
    assert safe_eval("(2 + 3) * 4") == 20.0
    assert safe_eval("-5 + 10") == 5.0


@pytest.mark.parametrize(
    "malicioso",
    ["__import__('os').system('ls')", "open('/etc/passwd')", "[1, 2]", "'a' * 3", "2 ** 1000", "1 / 0", "5 +"],
)
def test_safe_eval_rechaza_todo_lo_que_no_sea_aritmetica(malicioso: str) -> None:
    with pytest.raises(ValueError):
        safe_eval(malicioso)


async def test_calculate_percentage_devuelve_resultado_y_formula() -> None:
    out = json.loads(await calculate_percentage.ainvoke({"base": 50000, "percentage": 20}))
    assert out["ok"] is True
    assert out["result"] == 10000.0
    assert "20% de 50000" in out["detail"]


async def test_calculate_expression_reporta_error_sin_romper() -> None:
    out = json.loads(await calculate_expression.ainvoke({"expression": "__import__('os')"}))
    assert out["ok"] is False and out["error"]


async def test_days_between_y_add_years() -> None:
    d = json.loads(await days_between.ainvoke({"start": "2026-01-01", "end": "2026-04-11"}))
    assert d["result"] == 100
    y = json.loads(await add_years.ainvoke({"start": "2024-02-29", "years": 5}))
    assert y["result"] == "2029-02-28"  # 29/02 hacia un año no bisiesto


@pytest.mark.parametrize(
    "args",
    [{"start": "2026-13-45", "end": "2026-01-01"}, {"start": "ayer", "end": "2026-01-01"}],
)
async def test_fechas_invalidas_fallan_en_el_esquema_antes_de_ejecutar(args: dict) -> None:
    with pytest.raises(ValidationError):
        await days_between.ainvoke(args)


async def test_porcentaje_negativo_es_rechazado_por_pydantic() -> None:
    with pytest.raises(ValidationError):
        await calculate_percentage.ainvoke({"base": 100, "percentage": -5})


# --------------------------- esquemas ---------------------------
def test_supervisor_decision_solo_acepta_destinos_validos() -> None:
    SupervisorDecision(next="FINISH", instruction="", reason="listo")
    with pytest.raises(ValidationError):
        SupervisorDecision(next="inventor", instruction="", reason="no existe ese agente")


def test_esquemas_de_entrada_de_las_herramientas() -> None:
    ReadArticleInput(number="94")
    ReadArticleInput(number="20-BIS")
    for invalido in ["94; DROP", "", "abc", "1234"]:
        with pytest.raises(ValidationError):
            ReadArticleInput(number=invalido)
    with pytest.raises(ValidationError):
        SearchArticlesInput(query="ab")


# --------------------------- herramientas del investigador ---------------------------
@pytest.fixture
def research_tools():
    chunks = load_chunks()
    rag = RAGSystem(LexicalRetriever(chunks), k=5)
    tools = {t.name: t for t in build_research_tools(rag, chunks)}
    return tools["search_articles"], tools["read_article"]


async def test_read_article_une_las_partes_de_un_articulo_largo(research_tools) -> None:
    _, read_article = research_tools
    out = json.loads(await read_article.ainvoke({"number": "94"}))
    assert out["ok"] is True
    texto = out["results"][0]["content"]
    assert "5%" in texto and "20%" in texto  # el artículo 94 está partido en 2 chunks: llegan los dos


async def test_read_article_inexistente_devuelve_error_como_dato(research_tools) -> None:
    _, read_article = research_tools
    out = json.loads(await read_article.ainvoke({"number": "999"}))
    assert out["ok"] is False and "999" in out["error"]


async def test_search_articles_devuelve_resultados_validados(research_tools) -> None:
    search, _ = research_tools
    out = json.loads(await search.ainvoke({"query": "multa por mora en el pago de tributos"}))
    assert out["ok"] is True
    assert "94" in [r["article"] for r in out["results"]]


async def test_search_articles_con_retriever_caido_informa_el_error_sin_romper() -> None:
    class RetrieverCaido:
        async def ainvoke(self, query, *a, **k):
            raise ConnectionError("Pinecone no responde")

    chunks = load_chunks()
    tools = {t.name: t for t in build_research_tools(RAGSystem(RetrieverCaido()), chunks)}
    out = json.loads(await tools["search_articles"].ainvoke({"query": "prescripción de tributos"}))
    assert out["ok"] is False
    assert "ConnectionError" in out["error"]


# --------------------------- RAG: utilidades ---------------------------
def test_tokenizador_en_espanol_normaliza_tildes_y_stopwords() -> None:
    assert tokenize_es("Artículo 20-BIS: la Prescripción") == ["articulo", "20", "bis", "prescripcion"]


def test_chunks_del_codigo_estan_agrupados_por_articulo() -> None:
    grouped = group_by_article(load_chunks())
    assert len(grouped) == 115 or len(grouped) > 100
    assert [d.metadata["chunk_index"] for d in grouped["94"]] == [0, 1]
