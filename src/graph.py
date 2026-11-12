"""Grafo principal: topología JERÁRQUICA con un Supervisor al centro.

                        ┌──────────────► research ──┐
    START ──► supervisor ├──────────────► analyst  ──┤   (cada especialista VUELVE al supervisor)
                 ▲       └──► finalize ──► END       │
                 └──────────────────────────────────┘

* El supervisor decide a quién delegar (arista condicional con `Literal` en el retorno).
* Los especialistas nunca hablan entre sí: todo pasa por el supervisor, que valida con la
  rúbrica antes de permitir el cierre.
* `finalize` redacta la respuesta solo con la evidencia reunida.

`build_graph` recibe sus dependencias por parámetro (LLM, RAG, checkpointer): así los tests
inyectan componentes falsos y, en la pre-entrega 7, se le pasa el checkpointer de Redis
sin tocar el resto.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src import config
from src.agents.analyst_agent import build_analyst_node
from src.agents.research_agent import build_research_node
from src.finalizer import build_finalize_node
from src.rag.system import RAGSystem
from src.schemas import SupervisorDecision
from src.state import OrchestratorState
from src.supervisor import build_supervisor_node, route

SUPERVISOR = "supervisor"
RESEARCH = "research"
ANALYST = "analyst"
FINALIZE = "finalize"


def build_graph(
    llm: BaseChatModel,
    rag: RAGSystem,
    chunks: list[Document],
    checkpointer: BaseCheckpointSaver | None = None,
    decider: Runnable | None = None,
) -> CompiledStateGraph:
    """Arma y compila el grafo.

    `decider` es opcional: por defecto el supervisor usa el mismo LLM con salida estructurada.
    Los tests lo reemplazan por un decisor guionado para probar el enrutamiento sin LLM real.
    """
    decider = decider or llm.with_structured_output(SupervisorDecision).with_retry(
        stop_after_attempt=2, wait_exponential_jitter=config.RETRY_BACKOFF
    )

    graph = StateGraph(OrchestratorState)
    graph.add_node(SUPERVISOR, build_supervisor_node(decider))
    graph.add_node(RESEARCH, build_research_node(llm, rag, chunks))
    graph.add_node(ANALYST, build_analyst_node(llm))
    graph.add_node(FINALIZE, build_finalize_node(llm))

    graph.add_edge(START, SUPERVISOR)
    graph.add_conditional_edges(
        SUPERVISOR,
        route,  # devuelve Literal["research", "analyst", "finalize"]
        {"research": RESEARCH, "analyst": ANALYST, "finalize": FINALIZE},
    )
    graph.add_edge(RESEARCH, SUPERVISOR)  # los especialistas vuelven al supervisor
    graph.add_edge(ANALYST, SUPERVISOR)
    graph.add_edge(FINALIZE, END)

    return graph.compile(checkpointer=checkpointer)
