# tests/test_monitoring_service.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from app.services.monitoring_service import MonitoringService, _safe_divide


# =============================================================================
# Faux objets métier
# =============================================================================

@dataclass
class FakeModelRegistry:
    id: int = 1
    model_name: str = "credit_model"
    model_version: str = "v1"
    stage: str = "Production"
    run_id: str | None = "run_123"
    source_path: str | None = "artifacts/model.joblib"
    training_data_version: str | None = "train_v1"
    feature_list: list[str] | None = None
    hyperparameters: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    deployed_at: datetime | None = None
    is_active: bool = True
    created_at: datetime | None = None


@dataclass
class FakeDriftMetric:
    id: int = 10
    model_name: str = "credit_model"
    model_version: str = "v1"
    feature_name: str = "AMT_CREDIT"
    metric_name: str = "psi"
    reference_window_start: datetime | None = None
    reference_window_end: datetime | None = None
    current_window_start: datetime | None = None
    current_window_end: datetime | None = None
    metric_value: float = 0.21
    threshold_value: float | None = 0.10
    drift_detected: bool = True
    details: dict[str, Any] | None = None
    computed_at: datetime | None = None


@dataclass
class FakeEvaluationMetric:
    id: int = 20
    model_name: str = "credit_model"
    model_version: str = "v1"
    dataset_name: str = "production_window"
    window_start: datetime | None = None
    window_end: datetime | None = None
    roc_auc: float | None = 0.81
    pr_auc: float | None = 0.42
    precision_score: float | None = 0.30
    recall_score: float | None = 0.66
    f1_score: float | None = 0.41
    fbeta_score: float | None = 0.55
    business_cost: float | None = 120.0
    tn: int | None = 100
    fp: int | None = 10
    fn: int | None = 5
    tp: int | None = 20
    sample_size: int | None = 135
    computed_at: datetime | None = None


@dataclass
class FakeFeatureStoreRecord:
    id: int = 30
    request_id: str = "req_1"
    client_id: int = 100001
    model_name: str = "credit_model"
    model_version: str = "v1"
    feature_name: str = "AMT_CREDIT"
    feature_value: str = "50000.0"
    feature_type: str = "float"
    source_table: str | None = "prediction_features_snapshot"
    snapshot_timestamp: datetime | None = None


@dataclass
class FakeAlert:
    id: int = 40
    alert_type: str = "drift"
    severity: str = "high"
    title: str = "Drift detected"
    message: str = "AMT_CREDIT drift detected"
    model_name: str | None = "credit_model"
    model_version: str | None = "v1"
    feature_name: str | None = "AMT_CREDIT"
    context: dict[str, Any] | None = None
    status: str = "open"
    created_at: datetime | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


@dataclass
class FakePredictionLog:
    prediction_timestamp: datetime | None = None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def db_session() -> object:
    return object()


@pytest.fixture
def service(db_session: object) -> MonitoringService:
    return MonitoringService(db_session)


# =============================================================================
# Tests helpers
# =============================================================================

def test_safe_divide_normal_case() -> None:
    assert _safe_divide(10, 2) == 5.0


def test_safe_divide_zero_denominator() -> None:
    assert _safe_divide(10, 0) == 0.0


# =============================================================================
# Active model / registry
# =============================================================================

def test_get_active_model_found(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    model = FakeModelRegistry(model_name="xgb", model_version="v2")

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_active_model_record",
        lambda db, model_name=None: model,
    )

    result = service.get_active_model(model_name="xgb")

    assert result is model
    assert result.model_name == "xgb"
    assert result.model_version == "v2"


def test_get_active_model_not_found(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_active_model_record",
        lambda db, model_name=None: None,
    )

    result = service.get_active_model(model_name="missing")

    assert result is None


def test_get_models(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    rows = [
        FakeModelRegistry(id=1, model_name="xgb", model_version="v1"),
        FakeModelRegistry(id=2, model_name="xgb", model_version="v2", is_active=False),
    ]

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.list_model_records",
        lambda db, limit, model_name=None, is_active=None: rows,
    )

    result = service.get_models(limit=2, model_name="xgb", is_active=None)

    assert result["count"] == 2
    assert result["items"][0]["model_name"] == "xgb"
    assert result["items"][1]["model_version"] == "v2"


def test_register_model_version_create(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    created = FakeModelRegistry(
        id=7,
        model_name="xgb",
        model_version="v3",
        stage="Production",
        is_active=True,
    )

    called = {"deactivate": False}

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_model_record_by_name_version",
        lambda db, model_name, model_version: None,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.create_model_record",
        lambda db, **kwargs: created,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.update_model_record",
        lambda db, entity, **kwargs: entity,
    )

    def fake_deactivate(db, model_name, keep_model_id):
        called["deactivate"] = True
        assert model_name == "xgb"
        assert keep_model_id == 7

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.deactivate_other_model_versions",
        fake_deactivate,
    )

    result = service.register_model_version(
        model_name="xgb",
        model_version="v3",
        stage="Production",
        is_active=True,
    )

    assert result["model_name"] == "xgb"
    assert result["model_version"] == "v3"
    assert result["is_active"] is True
    assert called["deactivate"] is True


def test_register_model_version_update(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    existing = FakeModelRegistry(
        id=8,
        model_name="lgbm",
        model_version="v1",
        stage="Staging",
        is_active=False,
    )
    updated = FakeModelRegistry(
        id=8,
        model_name="lgbm",
        model_version="v1",
        stage="Production",
        is_active=False,
    )

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_model_record_by_name_version",
        lambda db, model_name, model_version: existing,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.create_model_record",
        lambda db, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.update_model_record",
        lambda db, entity, **kwargs: updated,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.deactivate_other_model_versions",
        lambda db, model_name, keep_model_id: None,
    )

    result = service.register_model_version(
        model_name="lgbm",
        model_version="v1",
        stage="Production",
        is_active=False,
    )

    assert result["model_name"] == "lgbm"
    assert result["stage"] == "Production"
    assert result["is_active"] is False


# =============================================================================
# Drift
# =============================================================================

def test_log_drift_metric(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    fixed_now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    created = FakeDriftMetric(
        id=11,
        model_name="xgb",
        model_version="v1",
        feature_name="EXT_SOURCE_2",
        metric_name="psi",
        drift_detected=True,
        computed_at=fixed_now,
    )

    monkeypatch.setattr("app.services.monitoring_service._utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.create_drift_metric_record",
        lambda db, **kwargs: created,
    )

    result = service.log_drift_metric(
        model_name="xgb",
        model_version="v1",
        feature_name="EXT_SOURCE_2",
        metric_name="psi",
        metric_value=0.3,
        drift_detected=True,
    )

    assert result["id"] == 11
    assert result["feature_name"] == "EXT_SOURCE_2"
    assert result["drift_detected"] is True
    assert result["computed_at"] == fixed_now


def test_get_drift_metrics(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    rows = [
        FakeDriftMetric(id=1, feature_name="A"),
        FakeDriftMetric(id=2, feature_name="B", drift_detected=False),
    ]

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.list_drift_metrics",
        lambda db, **kwargs: rows,
    )

    result = service.get_drift_metrics(limit=2)

    assert result["count"] == 2
    assert result["items"][0]["feature_name"] == "A"
    assert result["items"][1]["drift_detected"] is False


# =============================================================================
# Evaluation
# =============================================================================

def test_log_evaluation_metrics(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    fixed_now = datetime(2026, 4, 23, 13, 0, tzinfo=timezone.utc)
    created = FakeEvaluationMetric(
        id=21,
        model_name="xgb",
        model_version="v1",
        dataset_name="prod_window",
        computed_at=fixed_now,
    )

    monkeypatch.setattr("app.services.monitoring_service._utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.create_evaluation_metric_record",
        lambda db, **kwargs: created,
    )

    result = service.log_evaluation_metrics(
        model_name="xgb",
        model_version="v1",
        dataset_name="prod_window",
        sample_size=100,
    )

    assert result["id"] == 21
    assert result["dataset_name"] == "prod_window"
    assert result["computed_at"] == fixed_now


def test_get_evaluation_metrics(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    rows = [
        FakeEvaluationMetric(id=1, dataset_name="test"),
        FakeEvaluationMetric(id=2, dataset_name="prod"),
    ]

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.list_evaluation_metrics",
        lambda db, **kwargs: rows,
    )

    result = service.get_evaluation_metrics(limit=2)

    assert result["count"] == 2
    assert result["items"][0]["dataset_name"] == "test"
    assert result["items"][1]["dataset_name"] == "prod"


# =============================================================================
# Feature store
# =============================================================================

def test_get_feature_store(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    rows = [
        FakeFeatureStoreRecord(id=1, feature_name="AMT_CREDIT"),
        FakeFeatureStoreRecord(id=2, feature_name="EXT_SOURCE_2"),
    ]

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.list_feature_store_records",
        lambda db, **kwargs: rows,
    )

    result = service.get_feature_store(limit=2)

    assert result["count"] == 2
    assert result["items"][0]["feature_name"] == "AMT_CREDIT"
    assert result["items"][1]["feature_name"] == "EXT_SOURCE_2"


# =============================================================================
# Alerts
# =============================================================================

def test_create_alert(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    fixed_now = datetime(2026, 4, 23, 14, 0, tzinfo=timezone.utc)
    alert = FakeAlert(id=41, created_at=fixed_now)

    monkeypatch.setattr("app.services.monitoring_service._utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.create_alert_record",
        lambda db, **kwargs: alert,
    )

    result = service.create_alert(
        alert_type="drift",
        severity="high",
        title="Title",
        message="Message",
    )

    assert result.id == 41
    assert result.alert_type == "drift"
    assert result.created_at == fixed_now


def test_get_recent_alerts(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    alerts = [
        FakeAlert(id=1, status="open"),
        FakeAlert(id=2, status="acknowledged"),
    ]

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.list_alert_records",
        lambda db, **kwargs: alerts,
    )

    result = service.get_recent_alerts(limit=2)

    assert len(result) == 2
    assert result[0].id == 1
    assert result[1].status == "acknowledged"


def test_acknowledge_alert_not_found(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_alert_by_id",
        lambda db, alert_id: None,
    )

    result = service.acknowledge_alert(123)

    assert result is None


def test_acknowledge_alert_already_resolved(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    alert = FakeAlert(id=50, status="resolved")

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_alert_by_id",
        lambda db, alert_id: alert,
    )

    result = service.acknowledge_alert(50)

    assert result is alert
    assert result.status == "resolved"


def test_acknowledge_alert_success(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    fixed_now = datetime(2026, 4, 23, 15, 0, tzinfo=timezone.utc)
    alert = FakeAlert(id=51, status="open")
    updated = FakeAlert(id=51, status="acknowledged", acknowledged_at=fixed_now)

    monkeypatch.setattr("app.services.monitoring_service._utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_alert_by_id",
        lambda db, alert_id: alert,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.update_alert_status",
        lambda db, alert, status, acknowledged_at=None, resolved_at=None: updated,
    )

    result = service.acknowledge_alert(51)

    assert result is not None
    assert result.status == "acknowledged"
    assert result.acknowledged_at == fixed_now


def test_resolve_alert_not_found(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_alert_by_id",
        lambda db, alert_id: None,
    )

    result = service.resolve_alert(777)

    assert result is None


def test_resolve_alert_success_with_existing_ack_time(
    monkeypatch: pytest.MonkeyPatch,
    service: MonitoringService,
) -> None:
    ack_time = datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc)
    resolve_time = datetime(2026, 4, 23, 16, 0, tzinfo=timezone.utc)

    alert = FakeAlert(id=52, status="acknowledged", acknowledged_at=ack_time)
    updated = FakeAlert(
        id=52,
        status="resolved",
        acknowledged_at=ack_time,
        resolved_at=resolve_time,
    )

    monkeypatch.setattr("app.services.monitoring_service._utc_now", lambda: resolve_time)
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_alert_by_id",
        lambda db, alert_id: alert,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.update_alert_status",
        lambda db, alert, status, acknowledged_at=None, resolved_at=None: updated,
    )

    result = service.resolve_alert(52)

    assert result is not None
    assert result.status == "resolved"
    assert result.acknowledged_at == ack_time
    assert result.resolved_at == resolve_time


def test_resolve_alert_success_without_existing_ack_time(
    monkeypatch: pytest.MonkeyPatch,
    service: MonitoringService,
) -> None:
    fixed_now = datetime(2026, 4, 23, 17, 0, tzinfo=timezone.utc)

    alert = FakeAlert(id=53, status="open", acknowledged_at=None)
    updated = FakeAlert(
        id=53,
        status="resolved",
        acknowledged_at=fixed_now,
        resolved_at=fixed_now,
    )

    monkeypatch.setattr("app.services.monitoring_service._utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_alert_by_id",
        lambda db, alert_id: alert,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.update_alert_status",
        lambda db, alert, status, acknowledged_at=None, resolved_at=None: updated,
    )

    result = service.resolve_alert(53)

    assert result is not None
    assert result.status == "resolved"
    assert result.acknowledged_at == fixed_now


# =============================================================================
# Summary / health
# =============================================================================

def test_get_monitoring_summary(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    pred_time = datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc)
    drift_time = datetime(2026, 4, 23, 8, 0, tzinfo=timezone.utc)
    eval_time = datetime(2026, 4, 23, 7, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.count_prediction_logs",
        lambda db, **kwargs: 100,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.count_prediction_errors",
        lambda db, **kwargs: 5,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.get_latest_prediction_log",
        lambda db, **kwargs: FakePredictionLog(prediction_timestamp=pred_time),
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.get_average_latency_ms",
        lambda db, **kwargs: 123.4,
    )

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.count_drift_metrics",
        lambda db, drift_detected=None, **kwargs: 20 if drift_detected is None else 4,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_latest_drift_metric",
        lambda db, **kwargs: FakeDriftMetric(computed_at=drift_time),
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_latest_evaluation_metric",
        lambda db, **kwargs: FakeEvaluationMetric(computed_at=eval_time),
    )

    def fake_count_alerts(db, status=None, **kwargs):
        return {
            "open": 3,
            "acknowledged": 2,
            "resolved": 7,
        }[status]

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.count_alert_records",
        fake_count_alerts,
    )

    result = service.get_monitoring_summary(
        model_name="xgb",
        model_version="v1",
    )

    assert result["model_name"] == "xgb"
    assert result["model_version"] == "v1"

    assert result["predictions"]["total_predictions"] == 100
    assert result["predictions"]["total_errors"] == 5
    assert result["predictions"]["error_rate"] == pytest.approx(0.05)
    assert result["predictions"]["avg_latency_ms"] == 123.4
    assert result["predictions"]["last_prediction_at"] == pred_time

    assert result["drift"]["total_drift_metrics"] == 20
    assert result["drift"]["detected_drifts"] == 4
    assert result["drift"]["drift_rate"] == pytest.approx(0.2)
    assert result["drift"]["last_drift_at"] == drift_time

    assert result["latest_evaluation"]["computed_at"] == eval_time
    assert result["alerts"]["open_alerts"] == 3
    assert result["alerts"]["acknowledged_alerts"] == 2
    assert result["alerts"]["resolved_alerts"] == 7


def test_get_monitoring_summary_without_latest_entities(
    monkeypatch: pytest.MonkeyPatch,
    service: MonitoringService,
) -> None:
    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.count_prediction_logs",
        lambda db, **kwargs: 0,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.count_prediction_errors",
        lambda db, **kwargs: 0,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.get_latest_prediction_log",
        lambda db, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.prediction_crud.get_average_latency_ms",
        lambda db, **kwargs: None,
    )

    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.count_drift_metrics",
        lambda db, drift_detected=None, **kwargs: 0,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_latest_drift_metric",
        lambda db, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.get_latest_evaluation_metric",
        lambda db, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.services.monitoring_service.monitoring_crud.count_alert_records",
        lambda db, status=None, **kwargs: 0,
    )

    result = service.get_monitoring_summary(model_name="xgb")

    assert result["predictions"]["error_rate"] == 0.0
    assert result["predictions"]["last_prediction_at"] is None
    assert result["drift"]["drift_rate"] == 0.0
    assert result["drift"]["last_drift_at"] is None
    assert result["latest_evaluation"] is None
    assert result["alerts"]["open_alerts"] == 0


def test_get_monitoring_health(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    summary = {
        "model_name": "xgb",
        "model_version": "v1",
        "window_start": None,
        "window_end": None,
        "predictions": {
            "total_predictions": 12,
            "total_errors": 1,
            "error_rate": 1 / 12,
            "avg_latency_ms": 88.0,
            "last_prediction_at": "2026-04-23T10:00:00",
        },
        "drift": {
            "total_drift_metrics": 5,
            "detected_drifts": 1,
            "drift_rate": 0.2,
            "last_drift_at": "2026-04-23T09:00:00",
        },
        "latest_evaluation": {
            "computed_at": "2026-04-23T08:00:00",
        },
        "alerts": {
            "open_alerts": 2,
            "acknowledged_alerts": 1,
            "resolved_alerts": 4,
        },
    }

    monkeypatch.setattr(service, "get_monitoring_summary", lambda **kwargs: summary)

    result = service.get_monitoring_health(model_name="xgb", model_version="v1")

    assert result["model_name"] == "xgb"
    assert result["model_version"] == "v1"
    assert result["has_predictions"] is True
    assert result["has_drift_metrics"] is True
    assert result["has_latest_evaluation"] is True
    assert result["open_alerts"] == 2
    assert result["avg_latency_ms"] == 88.0
    assert result["latest_evaluation_at"] == "2026-04-23T08:00:00"


def test_get_monitoring_health_empty(monkeypatch: pytest.MonkeyPatch, service: MonitoringService) -> None:
    summary = {
        "model_name": "xgb",
        "model_version": None,
        "window_start": None,
        "window_end": None,
        "predictions": {
            "total_predictions": 0,
            "avg_latency_ms": None,
            "last_prediction_at": None,
        },
        "drift": {
            "total_drift_metrics": 0,
            "last_drift_at": None,
        },
        "latest_evaluation": None,
        "alerts": {
            "open_alerts": 0,
        },
    }

    monkeypatch.setattr(service, "get_monitoring_summary", lambda **kwargs: summary)

    result = service.get_monitoring_health(model_name="xgb")

    assert result["has_predictions"] is False
    assert result["has_drift_metrics"] is False
    assert result["has_latest_evaluation"] is False
    assert result["open_alerts"] == 0