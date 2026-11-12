"""Contratos de datos (Pydantic) del sistema.

Todo lo que cruza una frontera se valida con Pydantic:
  * la decisión del Supervisor (qué agente sigue),
  * lo que cada especialista aporta (`Contribution`),
  * los argumentos y resultados de cada herramienta.

Así, si el LLM manda un argumento inválido o un resultado viene mal formado,
falla en el esquema y no se propaga como dato sucio.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Nombres de nodos/agentes. "FINISH" no es un nodo: es la señal de "ya está resuelto".
AgentName = Literal["research", "analyst"]
Destination = Literal["research", "analyst", "FINISH"]


# --------------------------------------------------------------------------- #
# Supervisor y estado compartido
# --------------------------------------------------------------------------- #
class SupervisorDecision(BaseModel):
    """Salida estructurada del Supervisor: a quién delega (o si termina) y por qué."""

    next: Destination = Field(
        description="Próximo paso: 'research' (buscar artículos), 'analyst' (calcular) o 'FINISH' (ya se puede responder)."
    )
    instruction: str = Field(
        default="",
        description=(
            "Instrucción puntual y AUTOSUFICIENTE para el especialista (no ve la conversación). "
            "Vacía si next es 'FINISH'."
        ),
    )
    reason: str = Field(min_length=3, description="Justificación breve de la decisión, según la rúbrica.")


class Contribution(BaseModel):
    """Qué aportó un agente: queda en el estado para saber quién dijo qué."""

    agent: AgentName
    step: int = Field(ge=0, description="Cuántas decisiones del supervisor llevaba el grafo cuando se ejecutó.")
    instruction: str
    output: str
    articles: list[str] = Field(default_factory=list, description="Artículos que el agente efectivamente consultó.")
    tools_used: list[str] = Field(default_factory=list)
    status: Literal["ok", "error"] = "ok"
    ms: float = Field(default=0.0, ge=0)


class TraceEvent(BaseModel):
    """Una fila de la traza de ejecución (para la tabla del notebook y los logs)."""

    node: str
    step: int
    detail: str
    ms: float = 0.0


# --------------------------------------------------------------------------- #
# Herramientas del Investigador
# --------------------------------------------------------------------------- #
class SearchArticlesInput(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Consulta en lenguaje natural sobre el Código Tributario, por ejemplo 'multa por mora en el pago'.",
    )


class ReadArticleInput(BaseModel):
    number: str = Field(
        ...,
        pattern=r"^\d{1,3}(-[A-Za-z]+)?$",
        description="Número de artículo, por ejemplo '94' o '20-BIS'.",
    )


class ArticleFound(BaseModel):
    article: str
    title: str = ""
    page: int | None = None
    category: str = ""
    content: str


class SearchResult(BaseModel):
    ok: bool = True
    query: str = ""
    results: list[ArticleFound] = Field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------- #
# Herramientas del Analista
# --------------------------------------------------------------------------- #
_ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"


class PercentageInput(BaseModel):
    base: float = Field(..., ge=0, description="Monto base, por ejemplo el tributo no pagado.")
    percentage: float = Field(..., ge=0, le=1000, description="Porcentaje a aplicar, por ejemplo 20 para 20%.")


class ExpressionInput(BaseModel):
    expression: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Expresión aritmética con números, + - * / ** y paréntesis. Ejemplo: '50000 * 0.2'.",
    )


class DaysBetweenInput(BaseModel):
    start: str = Field(..., pattern=_ISO_DATE, description="Fecha inicial en formato AAAA-MM-DD.")
    end: str = Field(..., pattern=_ISO_DATE, description="Fecha final en formato AAAA-MM-DD.")

    @field_validator("start", "end")
    @classmethod
    def _valid_date(cls, value: str) -> str:
        date.fromisoformat(value)  # lanza ValueError (=> falla de validación) si el día no existe
        return value


class AddYearsInput(BaseModel):
    start: str = Field(..., pattern=_ISO_DATE, description="Fecha inicial en formato AAAA-MM-DD.")
    years: int = Field(..., ge=0, le=100, description="Cantidad de años a sumar.")

    @field_validator("start")
    @classmethod
    def _valid_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


class CalculationResult(BaseModel):
    ok: bool = True
    result: float | str | None = None
    detail: str = ""
    error: str | None = None
