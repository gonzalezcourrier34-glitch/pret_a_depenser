# tests/test_security.py

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from app.core import security


def test_verify_api_key_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(security, "API_KEY", "secret_test_key")

    result = security.verify_api_key("secret_test_key")

    assert result is None


def test_verify_api_key_server_key_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "API_KEY", "")

    with pytest.raises(HTTPException) as exc_info:
        security.verify_api_key("any_key")

    exc = exc_info.value
    assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc.detail == "La clé API du serveur n'est pas configurée."


def test_verify_api_key_missing_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "API_KEY", "secret_test_key")

    with pytest.raises(HTTPException) as exc_info:
        security.verify_api_key(None)

    exc = exc_info.value
    assert exc.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc.detail == "Clé API manquante."


def test_verify_api_key_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "API_KEY", "secret_test_key")

    with pytest.raises(HTTPException) as exc_info:
        security.verify_api_key("wrong_key")

    exc = exc_info.value
    assert exc.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc.detail == "Clé API invalide."