# CLAUDE.md · Orquestador multi-agente (CoderHouse AI Engineering)

Contexto para trabajar este repo con Claude Code. Leelo entero antes de tocar código.

## Quién y para qué

- Matías (Uruguay) cursa **AI Engineering en CoderHouse**. Escribile en **español rioplatense**.
- Este repo es la **Pre-entrega 6: Orquestador multi-agente especializado**.
  Entrega: **jueves 2026-11-12 21:00 (America/Montevideo)**, repo público en GitHub.
- Matías está **aprendiendo**: cuando cambies o agregues algo, explicá brevemente el *por qué*
  (concepto + decisión de diseño), no solo el diff. No metas complejidad que no pueda defender
  en una devolución del corrector.
- Dominio: asistente de consultas sobre el **Código Tributario de Uruguay** (Decreto-Ley 14.306),
  indexado en su Pinecone (pre-entrega 4). La **entrega final (2026-12-05)** integra todas las
  pre-entregas, así que esto debe poder crecer sin reescribirse.

## Consigna de la Pre-entrega 6 (checklist de evaluación)

- `state.py` con estado compartido y campos tipo `next_agent` / `task_completed`.
- `agents/` con agente investigador y agente analista, cada uno con **tools funcionales**.
- Grafo con nodo **Supervisor** y **aristas condicionales con `Literal`**.
- **Rúbrica de validación** en el prompt del Supervisor (o nodo de validación).
- Evitar el **"Supervisor Infinito"** (contador/techo de pasos) y la **contaminación de contexto**.
- README con diagrama **Mermaid** + explicación de topología y manejo de conflictos.
- Video **o** notebook que muestre el flujo de delegación.

Pre-entrega 7 (vence 2026-11-19) suma: FastAPI async, checkpointer Redis, observabilidad
(Phoenix/LangSmith), human-in-the-loop y prueba de carga con 5 requests concurrentes.
Verificar en la doc vigente si el checkpointer de Redis de LangGraph requiere módulos
JSON/Search de Redis antes de elegir imagen/servicio.

## Arquitectura

Topología jerárquica: el **Supervisor** está en el centro; los especialistas siempre devuelven
el control a él. `START → supervisor → (research | analyst | finalize)`, y `research/analyst →
supervisor`, `finalize → END`.

| Archivo | Rol |
|---|---|
| `src/state.py` | `OrchestratorState(MessagesState)`. Acumuladores con reducer `operator.add`: `step_count`, `contributions`, `traces`, `errors`. Todo serializable a JSON. |
| `src/supervisor.py` | Prompt + rúbrica (a–d), `apply_guards` (determinista, manda sobre el LLM), `fallback_decision`, `route()` con `Literal["research","analyst","finalize"]`. |
| `src/agents/react.py` | Ciclo ReAct propio (modelo + `ToolNode` + `tools_condition`) y `make_specialist_node`, que **aísla fallas** (Contribution `status="error"`). |
| `src/agents/research_agent.py`, `analyst_agent.py` | Prompts y nodos. Cada uno recibe solo pregunta + instrucción (el analista además la evidencia). |
| `src/tools/research_tools.py` | `search_articles` (RAG híbrido), `read_article`. Errores como JSON, no excepciones. |
| `src/tools/analyst_tools.py` | `calculate_percentage`, `calculate_expression` (AST seguro, **sin `eval`**), `days_between`, `add_years`. |
| `src/finalizer.py` | Síntesis solo con evidencia; sin evidencia responde que no la hay; respaldo determinista si falla el LLM. |
| `src/graph.py` | `build_graph(llm, rag, chunks, checkpointer=None, decider=None)`. |
| `src/runner.py` | `run_query` (fija `recursion_limit`, captura `GraphRecursionError`), `format_trace`, `save_trace`. |
| `src/simulation.py` | LLM guionado + retriever léxico para correr el grafo real **sin API keys**. |
| `src/config.py` | Todo sale de variables de entorno. Imports pesados son perezosos. |

## Comandos

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python 3.12+
pip install -r requirements.txt
pytest -q                          # 52 tests, sin API keys ni torch
python main.py simular             # flujo completo simulado
python main.py preguntar "..."     # real: Gemini + Pinecone (requiere .env)
python main.py demo                # 3 consultas reales -> traces/traza_demo_{1,2,3}.json
python main.py mermaid             # diagrama del grafo real
```

## Reglas del proyecto (no romper)

1. **Los tests no pueden depender de torch/sentence-transformers ni de API keys.** El sandbox del
   corrector tiene pocos recursos. Los imports pesados van dentro de funciones.
2. **Sin IDs ni modelos hardcodeados**: índice, namespace, modelo y límites salen de `.env`
   (`src/config.py`). Si agregás una variable, sumala a `.env.example` y al README.
3. **Todo `async`**, tipado, validación con Pydantic en herramientas y salidas del LLM.
4. **Estado serializable** (str/int/bool/list[dict]); nada de objetos en el estado.
5. **El Supervisor nunca confía solo en el LLM**: cualquier regla de seguridad va en
   `apply_guards`, con test.
6. **No inventar**: sin evidencia no se calcula ni se finaliza; si falta un dato, se declara.
7. Nunca commitear `.env` ni claves. `data/chunks.jsonl` **sí** se commitea (lo usan BM25 y
   `read_article`).
8. Todo cambio de comportamiento lleva test. Correr `pytest -q` antes de dar algo por terminado.

## Lecciones de las correcciones anteriores (el corrector lo valora)

- Tests automatizados que pasen en una instalación limpia.
- `.env.example` que coincida con las variables realmente usadas.
- Declarar la versión de Python (3.12+) y `requirements.txt` mínimo, escrito a mano (no `pip freeze`).
- Evidencia de ejecución real en el README (trazas/salidas), no solo la simulación.
- Manejo de errores específico (no `except Exception` mudo), reintentos y logs con tiempos por nodo.
- Tabla "Cumplimiento de la consigna" en el README.

## Preferencias de trabajo

- Los pendientes y todo lo que Matías deba completar o rellenar se le dicen en el chat, no se dejan
  como comentarios ni checklists en los archivos.
- Los comentarios en el código solo van si explican algo no obvio o ayudan a organizar.

## Pensando en la Pre-entrega 7 y la entrega final

- Envolver `run_query` en un endpoint FastAPI async; un `thread_id` por consulta.
- Pasar un checkpointer a `build_graph(checkpointer=...)` (ya soportado).
- Observabilidad: trazar los nodos existentes (ya hay `traces` con ms por nodo).
- HITL: candidato natural es una interrupción antes de `finalize` o ante `status="error"`.
- Cuidar la concurrencia: el grafo se construye **una vez** y se reutiliza; no compartir estado mutable.
