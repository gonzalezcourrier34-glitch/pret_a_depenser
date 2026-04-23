# tests/test_route_analyse.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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
        model_name: str,
        model_version: str | None = None,
        reference_kind: str,
        current_kind: str,
        monitoring_dir: str,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
        return {
            "success": True,
            "message": "Analyse Evidently exécutée avec succès.",
            "model_name": model_name,
            "model_version": model_version or "v1",
            "reference_kind": reference_kind,
            "current_kind": current_kind,
            "logged_metrics": 12,
            "html_report_path": str(Path(monitoring_dir) / "report.html"),
            "reference_rows": 1000,
            "current_rows": 900,
            "analyzed_columns": ["AMT_CREDIT", "AMT_ANNUITY"],
            "report": {"dataset_drift": False},
        }


class FakeMonitoringEvaluationService:
    def __init__(self, db) -> None:
        self.db = db

    def run_and_persist_monitoring_evaluation(
        self,
        *,
        model_name: str,
        model_version: str | None = None,
        dataset_name: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        beta: float = 2.0,
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
    ) -> dict[str, Any]:
        return {
            "success": True,
            "message": "Évaluation monitoring exécutée avec succès.",
            "model_name": model_name,
            "model_version": model_version or "v1",
            "dataset_name": dataset_name,
            "logged_metrics": 1,
            "sample_size": 120,
            "matched_rows": 120,
            "threshold_used": 0.0593,
            "window_start": window_start,
            "window_end": window_end,
            "metrics": {
                "roc_auc": 0.81,
                "precision_score": 0.32,
                "recall_score": 0.61,
                "fbeta_score": 0.50,
                "business_cost": 85.0,
                "beta": beta,
                "cost_fn": cost_fn,
                "cost_fp": cost_fp,
            },
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
    value = "2026-04-23T12:30:45"
    result = ra._parse_optional_datetime(value)

    assert isinstance(result, datetime)
    assert result == datetime(2026, 4, 23, 12, 30, 45)


def test_parse_optional_datetime_invalid() -> None:
    with pytest.raises(Exception) as exc_info:
        ra._parse_optional_datetime("pas-une-date")

    exc = exc_info.value
    assert getattr(exc, "status_code", None) == 400
    assert "Format de date invalide" in str(exc.detail)


def test_resolve_monitoring_dir_with_argument() -> None:
    result = ra._resolve_monitoring_dir("artifacts/monitoring_test")
    assert result == str(Path("artifacts/monitoring_test"))


def test_resolve_monitoring_dir_default() -> None:
    result = ra._resolve_monitoring_dir(None)
    assert result == str(ra.MONITORING_DIR)


def test_resolve_monitoring_dir_blank_uses_default() -> None:
    result = ra._resolve_monitoring_dir("   ")
    assert result == str(ra.MONITORING_DIR)


# =============================================================================
# /analyse/evidently/run
# =============================================================================

def test_run_evidently_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(ra, "EvidentlyService", FakeEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "credit_scoring_model",
            "model_version": "v1",
            "reference_kind": "transformed",
            "current_kind": "transformed",
            "monitoring_dir": "artifacts/monitoring",
            "max_rows": 5000,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["model_name"] == "credit_scoring_model"
    assert body["model_version"] == "v1"
    assert body["reference_kind"] == "transformed"
    assert body["current_kind"] == "transformed"
    assert body["logged_metrics"] == 12
    assert body["reference_rows"] == 1000
    assert body["current_rows"] == 900
    assert len(body["analyzed_columns"]) == 2
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_run_evidently_uses_default_monitoring_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class InspectEvidentlyService(FakeEvidentlyService):
        def run_and_persist_data_drift_analysis(self, **kwargs):
            assert kwargs["monitoring_dir"] == str(ra.MONITORING_DIR)
            return super().run_and_persist_data_drift_analysis(**kwargs)

    monkeypatch.setattr(ra, "EvidentlyService", InspectEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "credit_scoring_model",
            "reference_kind": "transformed",
            "current_kind": "transformed",
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_run_evidently_rejects_incompatible_kinds() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "credit_scoring_model",
            "reference_kind": "raw",
            "current_kind": "transformed",
        },
    )

    assert response.status_code == 400
    assert "doivent être identiques" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_evidently_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenEvidentlyService(FakeEvidentlyService):
        def run_and_persist_data_drift_analysis(self, **kwargs):
            raise ValueError("Dataset de référence introuvable.")

    monkeypatch.setattr(ra, "EvidentlyService", BrokenEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "credit_scoring_model",
            "reference_kind": "transformed",
            "current_kind": "transformed",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Dataset de référence introuvable."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_evidently_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenEvidentlyService(FakeEvidentlyService):
        def run_and_persist_data_drift_analysis(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(ra, "EvidentlyService", BrokenEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "credit_scoring_model",
            "reference_kind": "transformed",
            "current_kind": "transformed",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de l'analyse Evidently."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_evidently_failed_result_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class FailedResultEvidentlyService(FakeEvidentlyService):
        def run_and_persist_data_drift_analysis(self, **kwargs):
            return {
                "success": False,
                "message": "Analyse Evidently terminée sans succès.",
                "model_name": kwargs["model_name"],
                "model_version": kwargs.get("model_version") or "v1",
                "reference_kind": kwargs["reference_kind"],
                "current_kind": kwargs["current_kind"],
                "logged_metrics": 0,
                "html_report_path": None,
                "reference_rows": 0,
                "current_rows": 0,
                "analyzed_columns": [],
                "report": None,
            }

    monkeypatch.setattr(ra, "EvidentlyService", FailedResultEvidentlyService)

    response = client.post(
        "/analyse/evidently/run",
        params={
            "model_name": "credit_scoring_model",
            "reference_kind": "transformed",
            "current_kind": "transformed",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["logged_metrics"] == 0
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_evidently_requires_model_name() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.post("/analyse/evidently/run")

    assert response.status_code == 422


# =============================================================================
# /analyse/evaluation/run
# =============================================================================

def test_run_monitoring_evaluation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        ra,
        "MonitoringEvaluationService",
        FakeMonitoringEvaluationService,
    )

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "credit_scoring_model",
            "model_version": "v1",
            "dataset_name": "scoring_prod",
            "window_start": "2026-04-01T00:00:00",
            "window_end": "2026-04-20T00:00:00",
            "beta": 2.0,
            "cost_fn": 10.0,
            "cost_fp": 1.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["model_name"] == "credit_scoring_model"
    assert body["model_version"] == "v1"
    assert body["dataset_name"] == "scoring_prod"
    assert body["logged_metrics"] == 1
    assert body["sample_size"] == 120
    assert body["matched_rows"] == 120
    assert body["threshold_used"] == 0.0593
    assert body["metrics"]["roc_auc"] == 0.81
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_run_monitoring_evaluation_without_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        ra,
        "MonitoringEvaluationService",
        FakeMonitoringEvaluationService,
    )

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "credit_scoring_model",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["dataset_name"] == "scoring_prod"
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_run_monitoring_evaluation_invalid_window_start_format() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "credit_scoring_model",
            "window_start": "date-invalide",
        },
    )

    assert response.status_code == 400
    assert "Format de date invalide" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 0


def test_run_monitoring_evaluation_invalid_window_end_format() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "credit_scoring_model",
            "window_end": "date-invalide",
        },
    )

    assert response.status_code == 400
    assert "Format de date invalide" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 0


def test_run_monitoring_evaluation_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenMonitoringEvaluationService(FakeMonitoringEvaluationService):
        def run_and_persist_monitoring_evaluation(self, **kwargs):
            raise ValueError("Aucune vérité terrain disponible.")

    monkeypatch.setattr(
        ra,
        "MonitoringEvaluationService",
        BrokenMonitoringEvaluationService,
    )

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "credit_scoring_model",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Aucune vérité terrain disponible."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_monitoring_evaluation_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenMonitoringEvaluationService(FakeMonitoringEvaluationService):
        def run_and_persist_monitoring_evaluation(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        ra,
        "MonitoringEvaluationService",
        BrokenMonitoringEvaluationService,
    )

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "credit_scoring_model",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de l'évaluation monitoring."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_monitoring_evaluation_failed_result_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class FailedResultMonitoringEvaluationService(FakeMonitoringEvaluationService):
        def run_and_persist_monitoring_evaluation(self, **kwargs):
            return {
                "success": False,
                "message": "Évaluation monitoring terminée sans succès.",
                "model_name": kwargs["model_name"],
                "model_version": kwargs.get("model_version") or "v1",
                "dataset_name": kwargs["dataset_name"],
                "logged_metrics": 0,
                "sample_size": 0,
                "matched_rows": 0,
                "threshold_used": None,
                "window_start": kwargs.get("window_start"),
                "window_end": kwargs.get("window_end"),
                "metrics": {},
            }

    monkeypatch.setattr(
        ra,
        "MonitoringEvaluationService",
        FailedResultMonitoringEvaluationService,
    )

    response = client.post(
        "/analyse/evaluation/run",
        params={
            "model_name": "credit_scoring_model",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["logged_metrics"] == 0
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_run_monitoring_evaluation_requires_model_name() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.post("/analyse/evaluation/run")

    assert response.status_code == 422