"""Pruebas del grafo completo con un LLM guionado: flujo feliz, bucles, fallas y aislamiento de contexto."""

from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.runner import format_trace, run_query, save_trace
from src.simulation import SIM_QUESTION, build_simulation_graph
from tests.helpers import AIMessage, analyst_script, decision, make_graph, research_script


def nodes(state: dict) -> list[str]:
    return [t["node"] for t in state["traces"]]


# --------------------------- flujo feliz ---------------------------
async def test_flujo_completo_investigar_calcular_y_sintetizar() -> None:
    graph, *_ = build_simulation_graph()
    state = await run_query(graph, SIM_QUESTION)

    # El supervisor vuelve a decidir después de CADA especialista (topología jerárquica).
    assert nodes(state) == ["supervisor", "research", "supervisor", "analyst", "supervisor", "finalize"]
    assert [c["agent"] for c in state["contributions"]] == ["research", "analyst"]
    assert state["task_completed"] is True
    assert state["step_count"] == 3  # el reducer operator.add acumuló las 3 decisiones
    assert "94" in state["contributions"][0]["articles"]
    assert state["contributions"][1]["tools_used"] == ["calculate_percentage"]
    assert "Artículos consultados" in state["final_answer"]
    assert state["errors"] == []


async def test_cada_contribucion_registra_quien_aporto_que() -> None:
    graph, *_ = build_simulation_graph()
    state = await run_query(graph, SIM_QUESTION)
    research, analyst = state["contributions"]
    assert research["status"] == "ok" and research["tools_used"] == ["search_articles", "read_article"]
    assert research["step"] == 1 and analyst["step"] == 2
    assert analyst["articles"] == []  # el analista no consulta artículos: calcula


async def test_si_no_hay_nada_que_calcular_el_analista_no_interviene() -> None:
    graph, *_ = make_graph(
        research_script() + [AIMessage(content="Respuesta final.")],
        [decision("research", "buscá"), decision("FINISH")],
    )
    state = await run_query(graph, "¿Qué es el conjunto económico?")
    assert nodes(state) == ["supervisor", "research", "supervisor", "finalize"]
    assert [c["agent"] for c in state["contributions"]] == ["research"]


# --------------------------- aislamiento de contexto ---------------------------
async def test_el_investigador_recibe_solo_pregunta_e_instruccion_no_el_historial() -> None:
    graph, llm, _ = make_graph(
        research_script() + analyst_script() + [AIMessage(content="ok")],
        [decision("research", "Buscá el artículo de mora"), decision("analyst", "Calculá la multa"), decision("FINISH")],
    )
    await run_query(graph, "Pregunta original del usuario")

    primera_llamada_investigador = llm.seen[0]
    humanos = [m for m in primera_llamada_investigador if isinstance(m, HumanMessage)]
    assert len(humanos) == 1  # un único mensaje de entrada, no el historial
    assert "Pregunta original del usuario" in humanos[0].content
    assert "Buscá el artículo de mora" in humanos[0].content
    sistema = next(m for m in primera_llamada_investigador if isinstance(m, SystemMessage))
    assert "Investigador" in sistema.content


async def test_el_analista_recibe_la_evidencia_del_investigador() -> None:
    graph, llm, _ = make_graph(
        research_script() + analyst_script() + [AIMessage(content="ok")],
        [decision("research", "buscá"), decision("analyst", "calculá"), decision("FINISH")],
    )
    await run_query(graph, "pregunta")
    # llm.seen[0..2] son las 3 llamadas del investigador; la 4ª (índice 3) es la primera del analista.
    entrada_analista = next(m for m in llm.seen[3] if isinstance(m, HumanMessage)).content
    assert "EVIDENCIA" in entrada_analista and "Art. 94" in entrada_analista


# --------------------------- el "Supervisor Infinito" ---------------------------
async def test_un_supervisor_que_siempre_pide_refinar_no_entra_en_bucle() -> None:
    # El decisor insiste en 'research' sin parar; las guardas lo cortan al llegar al tope de llamadas.
    repeticiones = config.MAX_CALLS_PER_AGENT
    graph, *_ = make_graph(
        research_script("a") + research_script("b") + [AIMessage(content="final")],
        [decision("research", "otra vez")] * 10,
    )
    state = await run_query(graph, "pregunta")

    assert nodes(state).count("research") == repeticiones
    assert state["step_count"] <= config.MAX_STEPS
    assert state["task_completed"] is True
    assert "max_calls" in format_trace(state)


async def test_el_recursion_limit_es_una_red_de_seguridad_final(monkeypatch) -> None:
    monkeypatch.setattr(config, "RECURSION_LIMIT", 3)  # absurdamente bajo, para forzar el corte
    graph, *_ = build_simulation_graph()
    state = await run_query(graph, SIM_QUESTION)
    assert "límite de pasos" in state["final_answer"]
    assert state["errors"] and "GraphRecursionError" in state["errors"][0]


# --------------------------- fallas aisladas ---------------------------
async def test_si_el_llm_del_supervisor_falla_se_usa_la_politica_de_respaldo() -> None:
    graph, *_ = make_graph(
        research_script() + analyst_script() + [AIMessage(content="final")],
        [RuntimeError("503 sobrecargado"), RuntimeError("503"), RuntimeError("503")],
    )
    state = await run_query(graph, "Pagué $50.000 con atraso, ¿cuánto es la multa?")

    assert nodes(state) == ["supervisor", "research", "supervisor", "analyst", "supervisor", "finalize"]
    assert len([e for e in state["errors"] if e.startswith("supervisor")]) == 3
    assert "origen=respaldo" in format_trace(state)


async def test_si_un_especialista_falla_el_grafo_no_se_rompe_y_no_se_inventa_la_respuesta() -> None:
    from src.simulation import ScriptedChatModel

    class LlmCaido(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise ConnectionError("sin conexión")

    from src.graph import build_graph
    from src.rag.chunks_io import load_chunks
    from src.rag.system import RAGSystem
    from src.simulation import LexicalRetriever, ScriptedDecider

    chunks = load_chunks()
    graph = build_graph(
        LlmCaido(responses=[]),
        RAGSystem(LexicalRetriever(chunks)),
        chunks,
        decider=ScriptedDecider([decision("research", "buscá")] * 5 + [decision("FINISH")] * 5),
    )
    state = await run_query(graph, "¿Qué multa corresponde?")

    assert all(c["status"] == "error" for c in state["contributions"])
    assert len(state["contributions"]) == config.MAX_CALLS_PER_AGENT  # se reintentó y se cortó
    assert any("ConnectionError" in e for e in state["errors"])
    assert "No encontré en el Código Tributario evidencia suficiente" in state["final_answer"]


async def test_si_el_llm_de_sintesis_falla_hay_respuesta_de_respaldo() -> None:
    # Se agotan las respuestas del guion justo antes de la síntesis => la llamada final falla.
    graph, *_ = make_graph(
        research_script() + analyst_script(),
        [decision("research", "buscá"), decision("analyst", "calculá"), decision("FINISH")],
    )
    state = await run_query(graph, "pregunta")
    assert "No pude redactar la respuesta final" in state["final_answer"]
    assert any(e.startswith("finalize") for e in state["errors"])


# --------------------------- salida y diagrama ---------------------------
async def test_la_traza_se_guarda_como_json_reproducible(tmp_path) -> None:
    import json

    graph, *_ = build_simulation_graph()
    state = await run_query(graph, SIM_QUESTION)
    path = save_trace(state, "t", path=tmp_path / "t.json", origin="SIMULACION")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["origen"] == "SIMULACION"
    assert [t["node"] for t in data["trazas"]][-1] == "finalize"
    assert data["pasos_supervisor"] == 3


def test_el_diagrama_mermaid_incluye_todos_los_nodos() -> None:
    graph, *_ = build_simulation_graph()
    mermaid = graph.get_graph().draw_mermaid()
    for nodo in ["supervisor", "research", "analyst", "finalize"]:
        assert nodo in mermaid
