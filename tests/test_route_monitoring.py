# tests/test_route_monitoring.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import route_monitoring as rm


# =============================================================================
# Helpers temporels
# =============================================================================

def _utc_now() -> datetime:
    """
    Retourne un datetime timezone-aware pour les tests.
    """
    return datetime.now(timezone.utc)


# =============================================================================
# Faux objets
# =============================================================================

class FakeDB:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.refreshed: list[Any] = []

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def refresh(self, obj: Any) -> None:
        self.refreshed.append(obj)


@dataclass
class FakeModel:
    model_name: str = "xgb"
    model_version: str = "v1"
    stage: str = "production"
    run_id: str | None = "run_1"
    source_path: str | None = "artifacts/model.joblib"
    training_data_version: str | None = "train_v1"
    feature_list: list[str] | None = None
    hyperparameters: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    deployed_at: datetime | None = None
    is_active: bool = True
    created_at: datetime = field(default_factory=_utc_now)


@dataclass
class FakeAlert:
    id: int = 1
    alert_type: str = "drift"
    severity: str = "high"
    model_name: str | None = "xgb"
    model_version: str | None = "v1"
    feature_name: str | None = "AMT_CREDIT"
    title: str = "Drift detected"
    message: str = "Feature drift detected"
    context: dict[str, Any] | None = None
    status: str = "open"
    created_at: datetime = field(default_factory=_utc_now)
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


class FakeMonitoringService:
    def __init__(self, db) -> None:
        self.db = db

    def get_active_model(self, model_name=None):
        return FakeModel(model_name=model_name or "xgb")

    def get_models(self, *, limit=200, model_name=None, is_active=None):
        return {
            "count": 2,
            "items": [
                {
                    "id": 1,
                    "model_name": "xgb",
                    "model_version": "v1",
                    "stage": "production",
                    "run_id": "run_1",
                    "source_path": "artifacts/model_v1.joblib",
                    "training_data_version": "train_v1",
                    "feature_list": ["f1", "f2"],
                    "hyperparameters": {"max_depth": 4},
                    "metrics": {"roc_auc": 0.81},
                    "deployed_at": None,
                    "is_active": True,
                    "created_at": _utc_now(),
                },
                {
                    "id": 2,
                    "model_name": "xgb",
                    "model_version": "v2",
                    "stage": "staging",
                    "run_id": "run_2",
                    "source_path": "artifacts/model_v2.joblib",
                    "training_data_version": "train_v2",
                    "feature_list": ["f1", "f2"],
                    "hyperparameters": {"max_depth": 5},
                    "metrics": {"roc_auc": 0.83},
                    "deployed_at": None,
                    "is_active": False,
                    "created_at": _utc_now(),
                },
            ],
        }

    def register_model_version(self, **kwargs):
        return {
            "message": "Version de modèle enregistrée avec succès.",
            "model_name": kwargs["model_name"],
            "model_version": kwargs["model_version"],
            "stage": kwargs["stage"],
            "is_active": kwargs["is_active"],
            "deployed_at": kwargs.get("deployed_at"),
        }

    def get_drift_metrics(self, **kwargs):
        return {
            "count": 1,
            "items": [
                {
                    "id": 1,
                    "model_name": "xgb",
                    "model_version": "v1",
                    "feature_name": "AMT_CREDIT",
                    "metric_name": "psi",
                    "reference_window_start": None,
                    "reference_window_end": None,
                    "current_window_start": None,
                    "current_window_end": None,
                    "metric_value": 0.22,
                    "threshold_value": 0.10,
                    "drift_detected": True,
                    "details": {"bucket_count": 10},
                    "computed_at": _utc_now(),
                }
            ],
        }

    def get_evaluation_metrics(self, **kwargs):
        return {
            "count": 1,
            "items": [
                {
                    "id": 1,
                    "model_name": "xgb",
                    "model_version": "v1",
                    "dataset_name": "prod",
                    "window_start": None,
                    "window_end": None,
                    "roc_auc": 0.81,
                    "pr_auc": 0.42,
                    "precision_score": 0.3,
                    "recall_score": 0.6,
                    "f1_score": 0.4,
                    "fbeta_score": 0.5,
                    "business_cost": 100.0,
                    "tn": 100,
                    "fp": 10,
                    "fn": 5,
                    "tp": 20,
                    "sample_size": 135,
                    "computed_at": _utc_now(),
                }
            ],
        }

    def get_feature_store(self, **kwargs):
        return {
            "count": 1,
            "items": [
                {
                    "id": 1,
                    "request_id": "req_1",
                    "client_id": 100001,
                    "model_name": "xgb",
                    "model_version": "v1",
                    "feature_name": "AMT_CREDIT",
                    "feature_value": "50000.0",
                    "feature_type": "float",
                    "source_table": "prediction_features_snapshot",
                    "snapshot_timestamp": _utc_now(),
                }
            ],
        }

    def get_recent_alerts(self, **kwargs):
        return [FakeAlert(id=1), FakeAlert(id=2, status="acknowledged")]

    def acknowledge_alert(self, alert_id: int):
        return FakeAlert(
            id=alert_id,
            status="acknowledged",
            acknowledged_at=_utc_now(),
        )

    def resolve_alert(self, alert_id: int):
        return FakeAlert(
            id=alert_id,
            status="resolved",
            acknowledged_at=_utc_now(),
            resolved_at=_utc_now(),
        )

    def get_monitoring_summary(self, **kwargs):
        return {
            "model_name": kwargs["model_name"],
            "model_version": kwargs.get("model_version"),
            "window_start": kwargs.get("window_start"),
            "window_end": kwargs.get("window_end"),
            "predictions": {
                "total_predictions": 100,
                "total_errors": 5,
                "error_rate": 0.05,
                "avg_latency_ms": 123.4,
                "last_prediction_at": None,
            },
            "drift": {
                "total_drift_metrics": 20,
                "detected_drifts": 4,
                "drift_rate": 0.2,
                "last_drift_at": None,
            },
            "latest_evaluation": None,
            "alerts": {
                "open_alerts": 3,
                "acknowledged_alerts": 2,
                "resolved_alerts": 1,
            },
        }

    def get_monitoring_health(self, **kwargs):
        return {
            "model_name": kwargs["model_name"],
            "model_version": kwargs.get("model_version"),
            "window_start": kwargs.get("window_start"),
            "window_end": kwargs.get("window_end"),
            "has_predictions": True,
            "has_drift_metrics": True,
            "has_latest_evaluation": False,
            "open_alerts": 2,
            "avg_latency_ms": 88.0,
            "last_prediction_at": None,
            "last_drift_at": None,
            "latest_evaluation_at": None,
        }


# =============================================================================
# App de test
# =============================================================================

def create_test_client(fake_db: FakeDB) -> TestClient:
    app = FastAPI()
    app.include_router(rm.router)

    app.dependency_overrides[rm.get_db] = lambda: fake_db
    app.dependency_overrides[rm.verify_api_key] = lambda: None

    return TestClient(app)


# =============================================================================
# Helpers
# =============================================================================

def test_validate_window_ok() -> None:
    rm._validate_window(None, None)
    rm._validate_window(
        datetime(2026, 4, 1, 10, 0, 0),
        datetime(2026, 4, 2, 10, 0, 0),
    )


def test_validate_window_missing_bound() -> None:
    with pytest.raises(Exception) as exc_info:
        rm._validate_window(datetime(2026, 4, 1, 10, 0, 0), None)

    exc = exc_info.value
    assert getattr(exc, "status_code", None) == 400
    assert "doivent être fournis ensemble" in str(exc.detail)


def test_validate_window_invalid_order() -> None:
    with pytest.raises(Exception) as exc_info:
        rm._validate_window(
            datetime(2026, 4, 2, 10, 0, 0),
            datetime(2026, 4, 1, 10, 0, 0),
        )

    exc = exc_info.value
    assert getattr(exc, "status_code", None) == 400
    assert "strictement supérieur" in str(exc.detail)


def test_serialize_active_model() -> None:
    entity = FakeModel(model_name="xgb", model_version="v2")
    result = rm._serialize_active_model(entity)

    assert result.model_name == "xgb"
    assert result.model_version == "v2"
    assert result.stage == "production"


def test_serialize_alert() -> None:
    alert = FakeAlert(id=9, status="open")
    result = rm._serialize_alert(alert)

    assert result.id == 9
    assert result.alert_type == "drift"
    assert result.status == "open"


# =============================================================================
# /monitoring/active-model
# =============================================================================

def test_get_active_model_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/active-model", params={"model_name": "xgb"})

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "xgb"
    assert body["model_version"] == "v1"
    assert body["is_active"] is True


def test_get_active_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class ServiceNotFound(FakeMonitoringService):
        def get_active_model(self, model_name=None):
            return None

    monkeypatch.setattr(rm, "MonitoringService", ServiceNotFound)

    response = client.get("/monitoring/active-model")

    assert response.status_code == 404
    assert response.json()["detail"] == "Aucun modèle actif trouvé."


def test_get_active_model_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_active_model(self, model_name=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/active-model")

    assert response.status_code == 500
    assert "Erreur lors de la récupération du modèle actif" in response.json()["detail"]


# =============================================================================
# /monitoring/models
# =============================================================================

def test_get_models_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/models", params={"limit": 10, "model_name": "xgb"})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["items"][0]["model_name"] == "xgb"
    assert body["items"][0]["id"] == 1


def test_get_models_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_models(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/models")

    assert response.status_code == 500
    assert "Erreur lors de la récupération du registre des modèles" in response.json()["detail"]


# =============================================================================
# /monitoring/models/register
# =============================================================================

def test_register_model_version_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.post(
        "/monitoring/models/register",
        json={
            "model_name": "xgb",
            "model_version": "v3",
            "stage": "production",
            "run_id": "run_3",
            "source_path": "artifacts/model_v3.joblib",
            "training_data_version": "train_v3",
            "feature_list": ["f1", "f2"],
            "hyperparameters": {"max_depth": 5},
            "metrics": {"roc_auc": 0.84},
            "deployed_at": None,
            "is_active": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "xgb"
    assert body["model_version"] == "v3"
    assert body["stage"] == "production"
    assert body["is_active"] is True
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_register_model_version_error_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def register_model_version(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.post(
        "/monitoring/models/register",
        json={
            "model_name": "xgb",
            "model_version": "v3",
            "stage": "production",
            "is_active": True,
        },
    )

    assert response.status_code == 500
    assert "Erreur lors de l'enregistrement du modèle" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /monitoring/drift
# =============================================================================

def test_get_drift_metrics_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/drift", params={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["feature_name"] == "AMT_CREDIT"


def test_get_drift_metrics_invalid_window() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.get(
        "/monitoring/drift",
        params={"window_start": "2026-04-10T00:00:00"},
    )

    assert response.status_code == 400
    assert "doivent être fournis ensemble" in response.json()["detail"]


def test_get_drift_metrics_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_drift_metrics(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/drift")

    assert response.status_code == 500
    assert "Erreur lors de la récupération des métriques de drift" in response.json()["detail"]


# =============================================================================
# /monitoring/evaluation
# =============================================================================

def test_get_evaluation_metrics_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/evaluation", params={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["dataset_name"] == "prod"


def test_get_evaluation_metrics_invalid_window() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.get(
        "/monitoring/evaluation",
        params={
            "window_start": "2026-04-10T00:00:00",
            "window_end": "2026-04-09T00:00:00",
        },
    )

    assert response.status_code == 400
    assert "strictement supérieur" in response.json()["detail"]


def test_get_evaluation_metrics_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_evaluation_metrics(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/evaluation")

    assert response.status_code == 500
    assert "Erreur lors de la récupération des métriques d'évaluation" in response.json()["detail"]


# =============================================================================
# /monitoring/feature-store
# =============================================================================

def test_get_feature_store_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/feature-store", params={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["request_id"] == "req_1"


def test_get_feature_store_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_feature_store(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/feature-store")

    assert response.status_code == 500
    assert "Erreur lors de la récupération du feature store" in response.json()["detail"]


# =============================================================================
# /monitoring/alerts
# =============================================================================

def test_get_recent_alerts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/alerts", params={"limit": 10, "status": "open"})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["items"][0]["alert_type"] == "drift"


def test_get_recent_alerts_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_recent_alerts(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/alerts")

    assert response.status_code == 500
    assert "Erreur lors de la récupération des alertes" in response.json()["detail"]


# =============================================================================
# /monitoring/alerts/{id}/acknowledge
# =============================================================================

def test_acknowledge_alert_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.post("/monitoring/alerts/12/acknowledge")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 12
    assert body["status"] == "acknowledged"
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0
    assert len(fake_db.refreshed) == 1


def test_acknowledge_alert_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class ServiceNotFound(FakeMonitoringService):
        def acknowledge_alert(self, alert_id: int):
            return None

    monkeypatch.setattr(rm, "MonitoringService", ServiceNotFound)

    response = client.post("/monitoring/alerts/12/acknowledge")

    assert response.status_code == 404
    assert response.json()["detail"] == "Alerte introuvable."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_acknowledge_alert_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def acknowledge_alert(self, alert_id: int):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.post("/monitoring/alerts/12/acknowledge")

    assert response.status_code == 500
    assert "Erreur lors de la reconnaissance de l'alerte" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /monitoring/alerts/{id}/resolve
# =============================================================================

def test_resolve_alert_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.post("/monitoring/alerts/15/resolve")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 15
    assert body["status"] == "resolved"
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0
    assert len(fake_db.refreshed) == 1


def test_resolve_alert_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class ServiceNotFound(FakeMonitoringService):
        def resolve_alert(self, alert_id: int):
            return None

    monkeypatch.setattr(rm, "MonitoringService", ServiceNotFound)

    response = client.post("/monitoring/alerts/15/resolve")

    assert response.status_code == 404
    assert response.json()["detail"] == "Alerte introuvable."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_resolve_alert_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def resolve_alert(self, alert_id: int):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.post("/monitoring/alerts/15/resolve")

    assert response.status_code == 500
    assert "Erreur lors de la résolution de l'alerte" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /monitoring/summary
# =============================================================================

def test_get_monitoring_summary_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/summary", params={"model_name": "xgb"})

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "xgb"
    assert body["predictions"]["total_predictions"] == 100
    assert body["drift"]["detected_drifts"] == 4


def test_get_monitoring_summary_invalid_window() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.get(
        "/monitoring/summary",
        params={
            "model_name": "xgb",
            "window_start": "2026-04-10T00:00:00",
        },
    )

    assert response.status_code == 400
    assert "doivent être fournis ensemble" in response.json()["detail"]


def test_get_monitoring_summary_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_monitoring_summary(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/summary", params={"model_name": "xgb"})

    assert response.status_code == 500
    assert "Erreur lors du calcul de la synthèse" in response.json()["detail"]


# =============================================================================
# /monitoring/health
# =============================================================================

def test_get_monitoring_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)
    monkeypatch.setattr(rm, "MonitoringService", FakeMonitoringService)

    response = client.get("/monitoring/health", params={"model_name": "xgb"})

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "xgb"
    assert body["has_predictions"] is True
    assert body["open_alerts"] == 2


def test_get_monitoring_health_invalid_window() -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    response = client.get(
        "/monitoring/health",
        params={
            "model_name": "xgb",
            "window_start": "2026-04-10T00:00:00",
            "window_end": "2026-04-09T00:00:00",
        },
    )

    assert response.status_code == 400
    assert "strictement supérieur" in response.json()["detail"]


def test_get_monitoring_health_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    class BrokenService(FakeMonitoringService):
        def get_monitoring_health(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm, "MonitoringService", BrokenService)

    response = client.get("/monitoring/health", params={"model_name": "xgb"})

    assert response.status_code == 500
    assert "Erreur lors du calcul de l'état du monitoring" in response.json()["detail"]