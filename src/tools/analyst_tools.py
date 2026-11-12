"""Herramientas del Analista: cómputo determinista.

Un LLM razona bien pero calcula mal. Por eso el analista NO hace cuentas "de cabeza":
usa estas herramientas, que son código normal, sin red y 100% reproducibles.

Importante: ninguna herramienta conoce tasas ni plazos legales. Los números salen
de los artículos que trajo el Investigador; acá solo se hacen las operaciones.
"""

from __future__ import annotations

import ast
import operator
from datetime import date

from langchain_core.tools import BaseTool, tool

from src.schemas import (
    AddYearsInput,
    CalculationResult,
    DaysBetweenInput,
    ExpressionInput,
    PercentageInput,
)

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_MAX_EXPONENT = 10


def safe_eval(expression: str) -> float:
    """Evalúa una expresión aritmética SIN usar eval(): solo números y operadores permitidos.

    `eval()` ejecutaría cualquier código que mande el LLM; recorrer el árbol sintáctico
    (ast) y aceptar únicamente nodos conocidos elimina ese riesgo.
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponente demasiado grande (máximo {_MAX_EXPONENT}).")
            return _BINARY_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Expresión no permitida: solo números, + - * / ** y paréntesis.")

    try:
        return float(_eval(ast.parse(expression.strip(), mode="eval")))
    except ZeroDivisionError as exc:
        raise ValueError("División por cero.") from exc
    except SyntaxError as exc:
        raise ValueError("Expresión mal formada.") from exc


def _ok(result: float | str, detail: str) -> str:
    return CalculationResult(result=result, detail=detail).model_dump_json()


def _error(message: str) -> str:
    return CalculationResult(ok=False, error=message).model_dump_json()


@tool("calculate_percentage", args_schema=PercentageInput)
async def calculate_percentage(base: float, percentage: float) -> str:
    """Calcula `percentage`% de `base`. Usar para multas, recargos o cualquier porcentaje sobre un monto
    (por ejemplo, el 20% de 50000). El porcentaje debe haber salido de un artículo citado por el
    investigador. Devuelve un JSON con "result" (el monto) y "detail" (la fórmula aplicada)."""
    result = round(base * percentage / 100, 2)
    return _ok(result, f"{percentage:g}% de {base:g} = {result:g}")


@tool("calculate_expression", args_schema=ExpressionInput)
async def calculate_expression(expression: str) -> str:
    """Evalúa una expresión aritmética (números, + - * / ** y paréntesis), por ejemplo '50000 + 50000 * 0.2'.
    Usar para sumas de varios conceptos (tributo + multa + recargo) o cualquier cuenta que no sea
    un porcentaje simple. Devuelve un JSON con "result" y "detail", o "error" si la expresión no es válida."""
    try:
        result = round(safe_eval(expression), 2)
    except ValueError as exc:
        return _error(str(exc))
    return _ok(result, f"{expression} = {result}")


@tool("days_between", args_schema=DaysBetweenInput)
async def days_between(start: str, end: str) -> str:
    """Cuenta los días corridos entre dos fechas AAAA-MM-DD (end - start). Usar para saber cuántos días
    de atraso hay entre un vencimiento y un pago, y así elegir el tramo que corresponde según el
    artículo. Devuelve un JSON con "result" (cantidad de días, negativa si end es anterior a start)."""
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return _ok(days, f"{end} - {start} = {days} días corridos")


@tool("add_years", args_schema=AddYearsInput)
async def add_years(start: str, years: int) -> str:
    """Suma años a una fecha AAAA-MM-DD. Usar para plazos de prescripción (por ejemplo, 5 años desde
    una fecha). Devuelve un JSON con "result" (la fecha resultante AAAA-MM-DD)."""
    base = date.fromisoformat(start)
    try:
        result = base.replace(year=base.year + years)
    except ValueError:  # 29 de febrero hacia un año no bisiesto
        result = base.replace(year=base.year + years, day=28)
    return _ok(result.isoformat(), f"{start} + {years} años = {result.isoformat()}")


ANALYST_TOOLS: list[BaseTool] = [calculate_percentage, calculate_expression, days_between, add_years]
