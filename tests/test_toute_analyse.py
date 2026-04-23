# tests/test_route_analyse.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import route_analyse as ra


# =============================================================================
# Faux objets
# =============================================================================

class FakeDB:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class FakeEvidentlyService:
    def __init__(self, db) -> None:
        self.db = db

    def run_and_persist_data_drift_analysis(
        self,
        *,
        model_name,
        model_version,
        reference_kind,
        current_kind,
        monitoring_dir,
        max_rows,
    ):
        return {
            "success": True,
            "message": "ok",
            "model_name": model_name,
            "model_version": model_version,
            "reference_kind": reference_kind,
            "current_kind": current_kind,
            "monitoring_dir": monitoring_dir,
            "logged_metrics": 3,
            "reference_rows": 100,
            "current_rows": 80,
            "max_rows": max_rows,
        }


class FakeMonitoringEvaluationService:
    def __init__(self, db) -> None:
        self.db = db

    def run_and_persist_monitoring_evaluation(
        self,
        *,
        model_name,
        model_version,
        dataset_name,
        window_start,
        window_end,
        beta,
        cost_fn,
        cost_fp,
    ):
        return {
            "success": True,
            "message": "ok",
            "model_name": model_name,
            "model_version": model_version,
            "dataset_name": dataset_name,
            "logged_metrics": 1,
            "sample_size": 50,
            "matched_rows": 40,
            "threshold_used": 0.5,
            "window_start": window_start,
            "window_end": window_end,
            "beta": beta,
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
        }


# =============================================================================
# App de test
# =============================================================================

def create_test_client(fake_db: FakeDB) -> TestClient:
    app = FastAPI()
    app.include_router(ra.router)

    app.dependency_overrides[ra.get_db] = lambda: fake_db
    app.dependency_overrides[ra.verify_api_key] = lambda: None

    return TestClient(app)


# =============================================================================
# Tests helpers
# =============================================================================

def test_parse_optional_datetime_none() -> None:
    assert ra._parse_optional_datetime(None) is None


def test_parse_optional_datetime_success() -> None:
    result = ra._parse_optional_datetime("2026-04-23T10:30:00")

    assert isinstance(result, datetime)
    assert result.year == 2026
    assert result.month == 4
    assert result.day == 23


def test_parse_optional_datetime_invalid() -> None:
    with pytest.raises(Exception) as exc_info:
        ra._parse_optional_datetime("not-a-date")

    exc = exc_info.value
    assert getattr(exc, "status_code", None) == 400
    assert "Format de date invalide" in str(exc.detail)


def test_resolve_monitoring_dir_explicit() -> None:
    result = ra._resolve_monitoring_dir("custom/monitoring")
    assert result == str(Path("custom/monitoring"))


def test_resolve_monitoring_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ra, "MONITORING_DIR", Path("artifacts/monitoring"))
    result = ra._resolve_monitoring_dir(None)
    assert result == "artifacts/monitoring"


# =============================================================================
# Route Evidently
# =============================================================================

def test_run_evidently_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(ra, "EvidentlyService", FakeEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "xgb",
            "model_version": "v1",
            "reference_kind": "transformed",
            "current_kind": "transformed",
            "monitoring_dir": "tmp/monitoring",
            "max_rows": 500,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    assert body["model_name"] == "xgb"
    assert body["model_version"] == "v1"
    assert body["reference_kind"] == "transformed"
    assert body["current_kind"] == "transformed"
    assert body["logged_metrics"] == 3
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_run_evidently_rejects_mismatched_kinds() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "xgb",
            "reference_kind": "raw",
            "current_kind": "transformed",
        },
    )

    assert response.status_code == 400
    assert "doivent être identiques" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_evidently_failed_result_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class FailedEvidentlyService:
        def __init__(self, db) -> None:
            self.db = db

        def run_and_persist_data_drift_analysis(self, **kwargs):
            return {
                "success": False,
                "message": "no drift metrics logged",
                "model_name": kwargs["model_name"],
                "model_version": kwargs["model_version"],
                "reference_kind": kwargs["reference_kind"],
                "current_kind": kwargs["current_kind"],
                "logged_metrics": 0,
                "reference_rows": 10,
                "current_rows": 8,
                "max_rows": kwargs["max_rows"],
            }

    monkeypatch.setattr(ra, "EvidentlyService", FailedEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "xgb",
            "model_version": "v1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_evidently_value_error_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class ValueErrorEvidentlyService:
        def __init__(self, db) -> None:
            self.db = db

        def run_and_persist_data_drift_analysis(self, **kwargs):
            raise ValueError("monitoring directory invalid")

    monkeypatch.setattr(ra, "EvidentlyService", ValueErrorEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={"model_name": "xgb"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "monitoring directory invalid"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_evidently_unexpected_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenEvidentlyService:
        def __init__(self, db) -> None:
            self.db = db

        def run_and_persist_data_drift_analysis(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(ra, "EvidentlyService", BrokenEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={"model_name": "xgb"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de l'analyse Evidently."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# Route Monitoring Evaluation
# =============================================================================

def test_run_monitoring_evaluation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(ra, "MonitoringEvaluationService", FakeMonitoringEvaluationService)

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "xgb",
            "model_version": "v1",
            "dataset_name": "scoring_prod",
            "window_start": "2026-04-01T00:00:00",
            "window_end": "2026-04-23T00:00:00",
            "beta": 2.0,
            "cost_fn": 10.0,
            "cost_fp": 1.0,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    assert body["model_name"] == "xgb"
    assert body["model_version"] == "v1"
    assert body["dataset_name"] == "scoring_prod"
    assert body["logged_metrics"] == 1
    assert body["sample_size"] == 50
    assert body["matched_rows"] == 40
    assert body["threshold_used"] == 0.5
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_run_monitoring_evaluation_invalid_window_start() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "xgb",
            "window_start": "not-a-date",
        },
    )

    assert response.status_code == 400
    assert "Format de date invalide" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 0


def test_run_monitoring_evaluation_failed_result_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class FailedMonitoringEvaluationService:
        def __init__(self, db) -> None:
            self.db = db

        def run_and_persist_monitoring_evaluation(self, **kwargs):
            return {
                "success": False,
                "message": "no matched rows",
                "model_name": kwargs["model_name"],
                "model_version": kwargs["model_version"],
                "dataset_name": kwargs["dataset_name"],
                "logged_metrics": 0,
                "sample_size": 0,
                "matched_rows": 0,
                "threshold_used": None,
            }

    monkeypatch.setattr(ra, "MonitoringEvaluationService", FailedMonitoringEvaluationService)

    response = client.post(
        "/analyse/evaluation/run",
        params={"model_name": "xgb"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_monitoring_evaluation_value_error_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class ValueErrorMonitoringEvaluationService:
        def __init__(self, db) -> None:
            self.db = db

        def run_and_persist_monitoring_evaluation(self, **kwargs):
            raise ValueError("ground truth missing")

    monkeypatch.setattr(
        ra,
        "MonitoringEvaluationService",
        ValueErrorMonitoringEvaluationService,
    )

    response = client.post(
        "/analyse/evaluation/run",
        params={"model_name": "xgb"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "ground truth missing"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_monitoring_evaluation_unexpected_error_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenMonitoringEvaluationService:
        def __init__(self, db) -> None:
            self.db = db

        def run_and_persist_monitoring_evaluation(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        ra,
        "MonitoringEvaluationService",
        BrokenMonitoringEvaluationService,
    )

    response = client.post(
        "/analyse/evaluation/run",
        params={"model_name": "xgb"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de l'évaluation monitoring."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1