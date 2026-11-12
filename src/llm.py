"""Fábrica del LLM: elige el proveedor según la configuración (.env).

Los imports de cada proveedor son perezosos: solo necesitás instalar el paquete
del proveedor que vayas a usar.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from src import config

# Variables de entorno aceptadas como API key para cada proveedor.
_API_KEY_VARS: dict[str, tuple[str, ...]] = {
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def model_name(provider: str | None = None, model: str | None = None) -> str:
    """Resuelve el nombre del modelo: argumento > LLM_MODEL > valor por defecto del proveedor."""
    provider = (provider or config.LLM_PROVIDER).lower()
    return model or config.LLM_MODEL or config.DEFAULT_MODELS[provider]


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Construye el chat model. Temperatura 0 para que las decisiones sean lo más predecibles posible."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider not in _API_KEY_VARS:
        raise ValueError(f"Proveedor '{provider}' no soportado. Usá uno de: {', '.join(_API_KEY_VARS)}.")

    variables = _API_KEY_VARS[provider]
    if not any(os.getenv(v) for v in variables):
        raise RuntimeError(
            f"Falta la API key de {provider}: definí {' o '.join(variables)} en tu archivo .env "
            "(copiá .env.example como .env)."
        )

    name = model_name(provider, model)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=name, temperature=temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=temperature)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=name, temperature=temperature)
