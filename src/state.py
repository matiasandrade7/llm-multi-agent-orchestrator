"""Estado compartido del grafo.

Es el "pizarrón" que todos los nodos leen y escriben. Hereda de `MessagesState`
(que ya trae `messages`) y suma campos propios para que el Supervisor pueda decidir
y para saber QUÉ agente aportó QUÉ información.

Reglas de diseño:
  * Solo tipos simples y serializables (str, int, bool, list[dict]). Cuando se
    persista el estado (checkpointer / Redis), todo tiene que poder pasar a JSON.
  * Los campos que acumulan información de varios nodos usan un REDUCER
    (`Annotated[..., operator.add]`): cada nodo devuelve solo lo nuevo y LangGraph
    lo SUMA a lo anterior. Nadie pisa el trabajo de otro.
  * Los demás campos se sobrescriben: guardan "el valor actual".
"""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.graph import MessagesState


class OrchestratorState(MessagesState):
    # --- Entrada -----------------------------------------------------------
    question: str                      # pregunta original del usuario

    # --- Decisión del supervisor (se sobrescriben en cada vuelta) ------------
    next_agent: str                    # "research" | "analyst" | "FINISH"
    instruction: str                   # instrucción puntual para el especialista elegido
    reason: str                        # por qué decidió eso (queda auditable)
    task_completed: bool               # True cuando el supervisor decide FINISH

    # --- Acumuladores (con reducer: suman, no pisan) --------------------------
    step_count: Annotated[int, operator.add]            # decisiones del supervisor hasta ahora
    contributions: Annotated[list[dict], operator.add]  # qué aportó cada agente (ver schemas.Contribution)
    traces: Annotated[list[dict], operator.add]         # traza por nodo con milisegundos
    errors: Annotated[list[str], operator.add]          # fallas aisladas (no rompen el grafo)

    # --- Salida ---------------------------------------------------------------
    final_answer: str
