import pytest

from src import config


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Sin espera exponencial entre reintentos: los tests de fallas no deben tardar segundos."""
    monkeypatch.setattr(config, "RETRY_BACKOFF", False)
