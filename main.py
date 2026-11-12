"""Punto de entrada (CLI) del orquestador multi-agente.

    python main.py simular                 # flujo completo SIN API keys (LLM y retriever simulados)
    python main.py preguntar "texto"       # consulta real: Gemini + Pinecone (requiere .env)
    python main.py demo                    # 3 consultas reales y guarda las trazas en traces/
    python main.py mermaid                 # imprime el diagrama Mermaid del grafo
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.runner import format_trace, run_query, save_trace
from src.simulation import SIM_QUESTION, build_simulation_graph

DEMO_QUESTIONS = [
    # Investigar + calcular + sintetizar (el camino completo).
    "Pagué con 100 días de atraso un tributo de $50.000. ¿Qué multa por mora corresponde y de cuánto sería?",
    # Solo investigar: no hay nada que calcular, el analista no debería intervenir.
    "¿En cuántos años prescribe el derecho al cobro de los tributos y cuándo se amplía ese plazo?",
    # Fuera de tema: el sistema debe reconocer que no hay evidencia en vez de inventar.
    "¿Cuál es la receta tradicional del chivito uruguayo?",
]


def _print_result(state: dict, title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"PREGUNTA: {state.get('question')}\n")
    print("TRAZA DE EJECUCIÓN")
    print(format_trace(state))
    print(f"\nDecisiones del supervisor: {state.get('step_count')} | errores: {state.get('errors') or 'ninguno'}")
    print(f"\nRESPUESTA FINAL\n{state.get('final_answer')}\n")


async def cmd_simular() -> None:
    graph, *_ = build_simulation_graph()
    state = await run_query(graph, SIM_QUESTION)
    _print_result(state, "SIMULACIÓN (LLM y retriever simulados; grafo, supervisor y herramientas reales)")
    print("Traza guardada en:", save_trace(state, "traza_simulacion", origin="SIMULACION"))


async def cmd_preguntar(question: str) -> None:
    from src.bootstrap import build_real_graph

    graph = build_real_graph()
    state = await run_query(graph, question)
    _print_result(state, "EJECUCIÓN REAL")


async def cmd_demo() -> None:
    from src.bootstrap import build_real_graph

    graph = build_real_graph()  # se arma UNA vez y se reutiliza
    for i, question in enumerate(DEMO_QUESTIONS, 1):
        state = await run_query(graph, question)
        _print_result(state, f"DEMO {i}/{len(DEMO_QUESTIONS)} (ejecución real)")
        print("Traza guardada en:", save_trace(state, f"traza_demo_{i}", origin="real"))


def cmd_mermaid() -> None:
    graph, *_ = build_simulation_graph()
    print(graph.get_graph().draw_mermaid())


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("simular", help="flujo completo sin API keys")
    ask = sub.add_parser("preguntar", help="una consulta real")
    ask.add_argument("question")
    sub.add_parser("demo", help="3 consultas reales con trazas")
    sub.add_parser("mermaid", help="diagrama Mermaid del grafo")
    args = parser.parse_args()

    if args.command == "mermaid":
        cmd_mermaid()
    elif args.command == "simular":
        asyncio.run(cmd_simular())
    elif args.command == "preguntar":
        asyncio.run(cmd_preguntar(args.question))
    else:
        asyncio.run(cmd_demo())


if __name__ == "__main__":
    sys.exit(main())
