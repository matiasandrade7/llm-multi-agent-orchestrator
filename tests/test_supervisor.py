"""Pruebas del Supervisor: guardas deterministas, política de respaldo y arista condicional."""

import pytest

from src import config
from src.schemas import Contribution
from src.supervisor import apply_guards, fallback_decision, needs_calculation, route
from tests.helpers import decision


def contribution(agent: str, articles=None, status="ok") -> dict:
    return Contribution(
        agent=agent, step=0, instruction="x", output="salida", articles=articles or [], status=status
    ).model_dump()


def state(contribs=None, steps=0, question="¿Qué multa corresponde?") -> dict:
    return {"question": question, "step_count": steps, "contributions": contribs or []}


# --------------------------- guardas ---------------------------
def test_una_decision_valida_pasa_sin_cambios() -> None:
    d = decision("analyst", "calculá")
    out, override = apply_guards(d, state([contribution("research", ["94"])]))
    assert out == d and override is None


def test_techo_de_pasos_fuerza_finish_pase_lo_que_pase() -> None:
    out, override = apply_guards(decision("research", "otra vez"), state(steps=config.MAX_STEPS))
    assert out.next == "FINISH" and override == "max_steps"


def test_no_se_puede_calcular_sin_evidencia() -> None:
    out, override = apply_guards(decision("analyst", "calculá"), state())
    assert out.next == "research" and "sin evidencia" in override
    assert out.instruction == "¿Qué multa corresponde?"  # la instrucción del LLM ya no aplicaba


def test_no_se_puede_finalizar_sin_evidencia() -> None:
    out, _ = apply_guards(decision("FINISH"), state())
    assert out.next == "research"


def test_una_investigacion_sin_articulos_no_cuenta_como_evidencia() -> None:
    out, _ = apply_guards(decision("FINISH"), state([contribution("research", articles=[])]))
    assert out.next == "research"


def test_una_investigacion_con_error_no_cuenta_como_evidencia() -> None:
    out, _ = apply_guards(decision("FINISH"), state([contribution("research", ["94"], status="error")]))
    assert out.next == "research"


def test_tope_de_llamadas_por_agente_corta_el_bucle_de_refinamientos() -> None:
    contribs = [contribution("research", ["94"]) for _ in range(config.MAX_CALLS_PER_AGENT)]
    out, override = apply_guards(decision("research", "refiná"), state(contribs))
    assert out.next == "FINISH" and override == "max_calls"


PREGUNTA_CON_CALCULO = "Pagué con 100 días de atraso un tributo de $50.000. ¿Qué multa corresponde?"


def test_investigador_agotado_con_calculo_pendiente_deriva_al_analista() -> None:
    contribs = [contribution("research", ["94"]) for _ in range(config.MAX_CALLS_PER_AGENT)]
    out, override = apply_guards(decision("research", "otra vez"), state(contribs, question=PREGUNTA_CON_CALCULO))
    assert out.next == "analyst" and override == "max_calls_to_analyst"
    assert out.instruction  # el analista recibe una instrucción concreta


def test_investigador_agotado_no_deriva_si_el_analista_ya_intervino() -> None:
    contribs = [contribution("research", ["94"]) for _ in range(config.MAX_CALLS_PER_AGENT)] + [contribution("analyst")]
    out, override = apply_guards(decision("research", "otra vez"), state(contribs, question=PREGUNTA_CON_CALCULO))
    assert out.next == "FINISH" and override == "max_calls"


def test_investigador_agotado_no_deriva_si_la_pregunta_no_pide_calculo() -> None:
    contribs = [contribution("research", ["38"]) for _ in range(config.MAX_CALLS_PER_AGENT)]
    out, override = apply_guards(
        decision("research", "otra vez"), state(contribs, question="¿Qué es el conjunto económico?")
    )
    assert out.next == "FINISH" and override == "max_calls"


def test_sin_evidencia_y_con_investigador_agotado_se_finaliza_sin_inventar() -> None:
    contribs = [contribution("research", [], status="error") for _ in range(config.MAX_CALLS_PER_AGENT)]
    out, _ = apply_guards(decision("FINISH"), state(contribs))
    assert out.next == "FINISH"  # la síntesis responderá "no hay evidencia suficiente"


# --------------------------- respaldo ---------------------------
def test_respaldo_investiga_primero() -> None:
    assert fallback_decision(state()).next == "research"


def test_respaldo_calcula_solo_si_la_pregunta_lo_requiere() -> None:
    con_calculo = state([contribution("research", ["94"])], question="Pagué $50.000 con atraso, ¿cuánto es la multa?")
    sin_calculo = state([contribution("research", ["38"])], question="¿Qué es el conjunto económico?")
    assert fallback_decision(con_calculo).next == "analyst"
    assert fallback_decision(sin_calculo).next == "FINISH"


def test_respaldo_finaliza_si_el_analista_ya_intervino() -> None:
    s = state([contribution("research", ["94"]), contribution("analyst")], question="calculá $50.000")
    assert fallback_decision(s).next == "FINISH"


@pytest.mark.parametrize(
    "pregunta, esperado",
    [
        ("Pagué con 100 días de atraso un tributo de $50.000", True),
        ("¿Cuánto es la multa?", True),
        ("Calculame el recargo", True),
        ("¿Qué es la instigación pública a no pagar tributos?", False),
    ],
)
def test_deteccion_de_necesidad_de_calculo(pregunta: str, esperado: bool) -> None:
    assert needs_calculation(pregunta) is esperado


# --------------------------- arista condicional ---------------------------
@pytest.mark.parametrize(
    "next_agent, destino",
    [("research", "research"), ("analyst", "analyst"), ("FINISH", "finalize")],
)
def test_route_traduce_la_decision_al_nombre_del_nodo(next_agent: str, destino: str) -> None:
    assert route({"next_agent": next_agent}) == destino
