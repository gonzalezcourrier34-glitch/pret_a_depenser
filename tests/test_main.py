# tests/test_main.py

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main_module


# =============================================================================
# Test route racine
# =============================================================================

def test_root_redirects_to_docs() -> None:
    client = TestClient(main_module.app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code in (307, 302)
    assert response.headers["location"] == "/docs"


# =============================================================================
# Test présence des routes
# =============================================================================

def test_routes_are_registered() -> None:
    paths = {route.path for route in main_module.app.routes}

    assert "/" in paths
    assert "/docs" in paths
    assert "/openapi.json" in paths


# =============================================================================
# Test middleware enregistré
# =============================================================================

def test_logging_middleware_is_registered() -> None:
    middleware_classes = [m.cls for m in main_module.app.user_middleware]

    if main_module.BENCHMARK_MODE:
        assert main_module.LoggingMiddleware not in middleware_classes
    else:
        assert main_module.LoggingMiddleware in middleware_classes

# =============================================================================
# Tests lifespan
# =============================================================================

class FakeDBSession:
    def __init__(self) -> None:
        self.executed = []
        self.closed = False

    def execute(self, query):
        self.executed.append(query)
        return 1

    def close(self):
        self.closed = True


def build_test_app_with_lifespan(lifespan_func):
    app = FastAPI(lifespan=lifespan_func)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


def test_lifespan_startup_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBSession()

    called = {
        "get_model": 0,
        "get_threshold": 0,
        "init_full_data_cache": 0,
    }

    def fake_get_model():
        called["get_model"] += 1
        return object()

    def fake_get_threshold():
        called["get_threshold"] += 1
        return 0.5

    def fake_init_full_data_cache(debug=False):
        called["init_full_data_cache"] += 1
        assert debug is False

    def fake_session_local():
        return fake_db

    monkeypatch.setattr(main_module, "get_model", fake_get_model)
    monkeypatch.setattr(main_module, "get_threshold", fake_get_threshold)
    monkeypatch.setattr(main_module, "init_full_data_cache", fake_init_full_data_cache)
    monkeypatch.setattr(main_module, "SessionLocal", fake_session_local)

    app = build_test_app_with_lifespan(main_module.lifespan)

    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    assert called["get_model"] == 1
    assert called["get_threshold"] == 1
    assert called["init_full_data_cache"] == 1
    assert fake_db.closed is True
    assert len(fake_db.executed) == 1


def test_lifespan_startup_failure_on_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_model():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(main_module, "get_model", fake_get_model)

    app = build_test_app_with_lifespan(main_module.lifespan)

    with pytest.raises(RuntimeError, match="model load failed"):
        with TestClient(app):
            pass


def test_lifespan_startup_failure_on_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_model():
        return object()

    def fake_get_threshold():
        raise RuntimeError("threshold load failed")

    monkeypatch.setattr(main_module, "get_model", fake_get_model)
    monkeypatch.setattr(main_module, "get_threshold", fake_get_threshold)

    app = build_test_app_with_lifespan(main_module.lifespan)

    with pytest.raises(RuntimeError, match="threshold load failed"):
        with TestClient(app):
            pass


def test_lifespan_startup_failure_on_db_check(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenDBSession:
        def execute(self, query):
            raise RuntimeError("db check failed")

        def close(self):
            self.closed = True

    def fake_get_model():
        return object()

    def fake_get_threshold():
        return 0.5

    def fake_session_local():
        return BrokenDBSession()

    monkeypatch.setattr(main_module, "get_model", fake_get_model)
    monkeypatch.setattr(main_module, "get_threshold", fake_get_threshold)
    monkeypatch.setattr(main_module, "SessionLocal", fake_session_local)

    app = build_test_app_with_lifespan(main_module.lifespan)

    with pytest.raises(RuntimeError, match="db check failed"):
        with TestClient(app):
            pass


def test_lifespan_startup_failure_on_cache_init(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDBSession()

    def fake_get_model():
        return object()

    def fake_get_threshold():
        return 0.5

    def fake_session_local():
        return fake_db

    def fake_init_full_data_cache(debug=False):
        raise RuntimeError("cache init failed")

    monkeypatch.setattr(main_module, "get_model", fake_get_model)
    monkeypatch.setattr(main_module, "get_threshold", fake_get_threshold)
    monkeypatch.setattr(main_module, "SessionLocal", fake_session_local)
    monkeypatch.setattr(main_module, "init_full_data_cache", fake_init_full_data_cache)

    app = build_test_app_with_lifespan(main_module.lifespan)

    with pytest.raises(RuntimeError, match="cache init failed"):
        with TestClient(app):
            pass

    assert fake_db.closed is True