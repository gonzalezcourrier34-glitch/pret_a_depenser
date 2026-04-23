# tests/test_crud_monitoring.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytest

from app.crud import monitoring as cm


# =============================================================================
# Faux outils ORM
# =============================================================================

class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __ne__(self, other):
        return ("ne", self.name, other)

    def is_(self, other):
        return ("is", self.name, other)

    def __ge__(self, other):
        return ("ge", self.name, other)

    def __lt__(self, other):
        return ("lt", self.name, other)

    def desc(self):
        return ("desc", self.name)


class FakeEntity:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeModelRegistry(FakeEntity):
    id = FakeColumn("id")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    stage = FakeColumn("stage")
    deployed_at = FakeColumn("deployed_at")
    created_at = FakeColumn("created_at")
    is_active = FakeColumn("is_active")


class FakeDriftMetric(FakeEntity):
    id = FakeColumn("id")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    feature_name = FakeColumn("feature_name")
    metric_name = FakeColumn("metric_name")
    drift_detected = FakeColumn("drift_detected")
    computed_at = FakeColumn("computed_at")


class FakeEvaluationMetric(FakeEntity):
    id = FakeColumn("id")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    dataset_name = FakeColumn("dataset_name")
    computed_at = FakeColumn("computed_at")


class FakeFeatureStoreMonitoring(FakeEntity):
    id = FakeColumn("id")
    request_id = FakeColumn("request_id")
    client_id = FakeColumn("client_id")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    feature_name = FakeColumn("feature_name")
    source_table = FakeColumn("source_table")
    snapshot_timestamp = FakeColumn("snapshot_timestamp")


class FakeAlert(FakeEntity):
    id = FakeColumn("id")
    status = FakeColumn("status")
    severity = FakeColumn("severity")
    alert_type = FakeColumn("alert_type")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    feature_name = FakeColumn("feature_name")
    created_at = FakeColumn("created_at")


class FakeQuery:
    def __init__(self, *, first_result=None, all_result=None, count_result=0):
        self.first_result = first_result
        self.all_result = all_result or []
        self.count_result = count_result
        self.filters = []
        self.order_by_args = []
        self.limit_value = None
        self.updated_with = None
        self.synchronize_session = None

    def filter(self, *conditions):
        self.filters.extend(conditions)
        return self

    def order_by(self, *args):
        self.order_by_args.extend(args)
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def all(self):
        return self.all_result

    def first(self):
        return self.first_result

    def count(self):
        return self.count_result

    def update(self, values, synchronize_session=False):
        self.updated_with = values
        self.synchronize_session = synchronize_session
        return 1


class FakeSession:
    def __init__(self):
        self.queries = {}
        self.added = []
        self.added_all = []
        self.flush_count = 0

    def set_query(self, model, query: FakeQuery):
        self.queries[model] = query

    def query(self, model):
        return self.queries[model]

    def add(self, entity):
        if not hasattr(entity, "id"):
            entity.id = len(self.added) + 1
        self.added.append(entity)

    def add_all(self, entities):
        for idx, entity in enumerate(entities, start=1):
            if not hasattr(entity, "id"):
                entity.id = idx
        self.added_all.extend(entities)

    def flush(self):
        self.flush_count += 1


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def patch_models(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cm, "ModelRegistry", FakeModelRegistry)
    monkeypatch.setattr(cm, "DriftMetric", FakeDriftMetric)
    monkeypatch.setattr(cm, "EvaluationMetric", FakeEvaluationMetric)
    monkeypatch.setattr(cm, "FeatureStoreMonitoring", FakeFeatureStoreMonitoring)
    monkeypatch.setattr(cm, "Alert", FakeAlert)


@pytest.fixture
def db() -> FakeSession:
    return FakeSession()


# =============================================================================
# Helpers query builders
# =============================================================================

def test_build_drift_metrics_query_applies_filters(db: FakeSession) -> None:
    query = FakeQuery()
    db.set_query(FakeDriftMetric, query)

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 2)

    result = cm._build_drift_metrics_query(
        db,
        model_name="xgb",
        model_version="v1",
        feature_name="AMT_CREDIT",
        metric_name="psi",
        drift_detected=True,
        window_start=start,
        window_end=end,
    )

    assert result is query
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("eq", "feature_name", "AMT_CREDIT") in query.filters
    assert ("eq", "metric_name", "psi") in query.filters
    assert ("is", "drift_detected", True) in query.filters
    assert ("ge", "computed_at", start) in query.filters
    assert ("lt", "computed_at", end) in query.filters


def test_build_evaluation_metrics_query_applies_filters(db: FakeSession) -> None:
    query = FakeQuery()
    db.set_query(FakeEvaluationMetric, query)

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 2)

    result = cm._build_evaluation_metrics_query(
        db,
        model_name="xgb",
        model_version="v1",
        dataset_name="prod",
        window_start=start,
        window_end=end,
    )

    assert result is query
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("eq", "dataset_name", "prod") in query.filters
    assert ("ge", "computed_at", start) in query.filters
    assert ("lt", "computed_at", end) in query.filters


def test_build_feature_store_query_applies_filters(db: FakeSession) -> None:
    query = FakeQuery()
    db.set_query(FakeFeatureStoreMonitoring, query)

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 2)

    result = cm._build_feature_store_query(
        db,
        request_id="req_1",
        client_id=100001,
        feature_name="AMT_CREDIT",
        model_name="xgb",
        model_version="v1",
        source_table="snapshot",
        window_start=start,
        window_end=end,
    )

    assert result is query
    assert ("eq", "request_id", "req_1") in query.filters
    assert ("eq", "client_id", 100001) in query.filters
    assert ("eq", "feature_name", "AMT_CREDIT") in query.filters
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("eq", "source_table", "snapshot") in query.filters
    assert ("ge", "snapshot_timestamp", start) in query.filters
    assert ("lt", "snapshot_timestamp", end) in query.filters


def test_build_alerts_query_applies_filters(db: FakeSession) -> None:
    query = FakeQuery()
    db.set_query(FakeAlert, query)

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 2)

    result = cm._build_alerts_query(
        db,
        status="open",
        severity="high",
        alert_type="drift",
        model_name="xgb",
        model_version="v1",
        feature_name="AMT_CREDIT",
        created_after=start,
        created_before=end,
    )

    assert result is query
    assert ("eq", "status", "open") in query.filters
    assert ("eq", "severity", "high") in query.filters
    assert ("eq", "alert_type", "drift") in query.filters
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("eq", "feature_name", "AMT_CREDIT") in query.filters
    assert ("ge", "created_at", start) in query.filters
    assert ("lt", "created_at", end) in query.filters


# =============================================================================
# Model registry
# =============================================================================

def test_get_active_model_record(db: FakeSession) -> None:
    entity = FakeModelRegistry(model_name="xgb", model_version="v1", stage="Production")
    query = FakeQuery(first_result=entity)
    db.set_query(FakeModelRegistry, query)

    result = cm.get_active_model_record(db, model_name="xgb")

    assert result is entity
    assert ("is", "is_active", True) in query.filters
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("desc", "deployed_at") in query.order_by_args
    assert ("desc", "created_at") in query.order_by_args


def test_get_model_record_by_name_version(db: FakeSession) -> None:
    entity = FakeModelRegistry(model_name="xgb", model_version="v2")
    query = FakeQuery(first_result=entity)
    db.set_query(FakeModelRegistry, query)

    result = cm.get_model_record_by_name_version(
        db,
        model_name="xgb",
        model_version="v2",
    )

    assert result is entity
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v2") in query.filters


def test_list_model_records(db: FakeSession) -> None:
    rows = [
        FakeModelRegistry(model_name="xgb", model_version="v1"),
        FakeModelRegistry(model_name="xgb", model_version="v2"),
    ]
    query = FakeQuery(all_result=rows)
    db.set_query(FakeModelRegistry, query)

    result = cm.list_model_records(db, limit=2, model_name="xgb", is_active=True)

    assert result == rows
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("is", "is_active", True) in query.filters
    assert query.limit_value == 2


def test_create_model_record(db: FakeSession) -> None:
    entity = cm.create_model_record(
        db,
        model_name="xgb",
        model_version="v1",
        stage="Production",
        is_active=True,
    )

    assert entity.model_name == "xgb"
    assert entity.model_version == "v1"
    assert entity.stage == "Production"
    assert entity.is_active is True
    assert db.flush_count == 1
    assert entity in db.added


def test_update_model_record(db: FakeSession) -> None:
    entity = FakeModelRegistry(
        id=1,
        model_name="xgb",
        model_version="v1",
        stage="Staging",
        is_active=False,
    )

    result = cm.update_model_record(
        db,
        entity=entity,
        stage="Production",
        run_id="run_1",
        source_path="artifacts/model.joblib",
        training_data_version="train_v1",
        feature_list=["f1", "f2"],
        hyperparameters={"depth": 4},
        metrics={"roc_auc": 0.8},
        deployed_at=datetime(2026, 4, 23),
        is_active=True,
    )

    assert result.stage == "Production"
    assert result.run_id == "run_1"
    assert result.is_active is True
    assert db.flush_count == 1


def test_deactivate_other_model_versions(db: FakeSession) -> None:
    query = FakeQuery()
    db.set_query(FakeModelRegistry, query)

    cm.deactivate_other_model_versions(db, model_name="xgb", keep_model_id=7)

    assert ("eq", "model_name", "xgb") in query.filters
    assert ("ne", "id", 7) in query.filters
    assert ("is", "is_active", True) in query.filters
    assert query.updated_with == {"is_active": False}
    assert query.synchronize_session is False
    assert db.flush_count == 1


# =============================================================================
# Drift metrics
# =============================================================================

def test_create_drift_metric_record(db: FakeSession) -> None:
    entity = cm.create_drift_metric_record(
        db,
        model_name="xgb",
        model_version="v1",
        feature_name="AMT_CREDIT",
        metric_name="psi",
        metric_value=0.22,
        threshold_value=0.1,
        drift_detected=True,
        details={"bucket_count": 10},
        computed_at=datetime(2026, 4, 23),
    )

    assert entity.feature_name == "AMT_CREDIT"
    assert entity.metric_name == "psi"
    assert entity.drift_detected is True
    assert entity in db.added
    assert db.flush_count == 1


def test_list_drift_metrics(db: FakeSession) -> None:
    rows = [FakeDriftMetric(id=1), FakeDriftMetric(id=2)]
    query = FakeQuery(all_result=rows)
    db.set_query(FakeDriftMetric, query)

    result = cm.list_drift_metrics(db, limit=2, model_name="xgb")

    assert result == rows
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("desc", "computed_at") in query.order_by_args
    assert query.limit_value == 2


def test_count_drift_metrics(db: FakeSession) -> None:
    query = FakeQuery(count_result=5)
    db.set_query(FakeDriftMetric, query)

    result = cm.count_drift_metrics(db, model_name="xgb", drift_detected=True)

    assert result == 5
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("is", "drift_detected", True) in query.filters


def test_get_latest_drift_metric(db: FakeSession) -> None:
    entity = FakeDriftMetric(id=9)
    query = FakeQuery(first_result=entity)
    db.set_query(FakeDriftMetric, query)

    result = cm.get_latest_drift_metric(db, model_name="xgb")

    assert result is entity
    assert ("desc", "computed_at") in query.order_by_args


# =============================================================================
# Evaluation metrics
# =============================================================================

def test_create_evaluation_metric_record(db: FakeSession) -> None:
    entity = cm.create_evaluation_metric_record(
        db,
        model_name="xgb",
        model_version="v1",
        dataset_name="prod",
        roc_auc=0.81,
        recall_score=0.66,
        sample_size=100,
        computed_at=datetime(2026, 4, 23),
    )

    assert entity.dataset_name == "prod"
    assert entity.roc_auc == 0.81
    assert entity.sample_size == 100
    assert entity in db.added
    assert db.flush_count == 1


def test_list_evaluation_metrics(db: FakeSession) -> None:
    rows = [FakeEvaluationMetric(id=1)]
    query = FakeQuery(all_result=rows)
    db.set_query(FakeEvaluationMetric, query)

    result = cm.list_evaluation_metrics(db, limit=10, dataset_name="prod")

    assert result == rows
    assert ("eq", "dataset_name", "prod") in query.filters
    assert ("desc", "computed_at") in query.order_by_args
    assert query.limit_value == 10


def test_get_latest_evaluation_metric(db: FakeSession) -> None:
    entity = FakeEvaluationMetric(id=4)
    query = FakeQuery(first_result=entity)
    db.set_query(FakeEvaluationMetric, query)

    result = cm.get_latest_evaluation_metric(db, model_name="xgb")

    assert result is entity
    assert ("desc", "computed_at") in query.order_by_args


# =============================================================================
# Feature store monitoring
# =============================================================================

def test_create_feature_store_record(db: FakeSession) -> None:
    entity = cm.create_feature_store_record(
        db,
        request_id="req_1",
        client_id=100001,
        model_name="xgb",
        model_version="v1",
        feature_name="AMT_CREDIT",
        feature_value="50000.0",
        feature_type="float",
        source_table="snapshot",
        snapshot_timestamp=datetime(2026, 4, 23),
    )

    assert entity.request_id == "req_1"
    assert entity.client_id == 100001
    assert entity.feature_name == "AMT_CREDIT"
    assert entity in db.added
    assert db.flush_count == 1


def test_create_feature_store_records_batch(db: FakeSession) -> None:
    cm.create_feature_store_records(
        db,
        records=[
            {
                "request_id": "req_1",
                "client_id": 1,
                "model_name": "xgb",
                "model_version": "v1",
                "feature_name": "A",
                "feature_value": "1",
                "feature_type": "int",
                "source_table": "snapshot",
            },
            {
                "request_id": "req_2",
                "client_id": 2,
                "model_name": "xgb",
                "model_version": "v1",
                "feature_name": "B",
                "feature_value": "2",
                "feature_type": "int",
                "source_table": "snapshot",
            },
        ],
        timestamp=datetime(2026, 4, 23),
    )

    assert len(db.added_all) == 2
    assert db.flush_count == 1
    assert db.added_all[0].feature_name == "A"
    assert db.added_all[1].feature_name == "B"


def test_create_feature_store_records_empty(db: FakeSession) -> None:
    cm.create_feature_store_records(
        db,
        records=[],
        timestamp=datetime(2026, 4, 23),
    )

    assert db.added_all == []
    assert db.flush_count == 0


def test_list_feature_store_records(db: FakeSession) -> None:
    rows = [FakeFeatureStoreMonitoring(id=1)]
    query = FakeQuery(all_result=rows)
    db.set_query(FakeFeatureStoreMonitoring, query)

    result = cm.list_feature_store_records(db, limit=5, request_id="req_1")

    assert result == rows
    assert ("eq", "request_id", "req_1") in query.filters
    assert ("desc", "snapshot_timestamp") in query.order_by_args
    assert query.limit_value == 5


def test_count_feature_store_records(db: FakeSession) -> None:
    query = FakeQuery(count_result=7)
    db.set_query(FakeFeatureStoreMonitoring, query)

    result = cm.count_feature_store_records(db, model_name="xgb")

    assert result == 7
    assert ("eq", "model_name", "xgb") in query.filters


def test_get_latest_feature_store_record(db: FakeSession) -> None:
    entity = FakeFeatureStoreMonitoring(id=3)
    query = FakeQuery(first_result=entity)
    db.set_query(FakeFeatureStoreMonitoring, query)

    result = cm.get_latest_feature_store_record(db, client_id=100001)

    assert result is entity
    assert ("eq", "client_id", 100001) in query.filters
    assert ("desc", "snapshot_timestamp") in query.order_by_args


# =============================================================================
# Alerts
# =============================================================================

def test_create_alert_record(db: FakeSession) -> None:
    entity = cm.create_alert_record(
        db,
        alert_type="drift",
        severity="high",
        title="Drift detected",
        message="Feature drift detected",
        model_name="xgb",
        model_version="v1",
        feature_name="AMT_CREDIT",
        context={"psi": 0.22},
        status="open",
        created_at=datetime(2026, 4, 23),
    )

    assert entity.alert_type == "drift"
    assert entity.severity == "high"
    assert entity.status == "open"
    assert entity in db.added
    assert db.flush_count == 1


def test_get_alert_by_id(db: FakeSession) -> None:
    entity = FakeAlert(id=42)
    query = FakeQuery(first_result=entity)
    db.set_query(FakeAlert, query)

    result = cm.get_alert_by_id(db, alert_id=42)

    assert result is entity
    assert ("eq", "id", 42) in query.filters


def test_list_alert_records(db: FakeSession) -> None:
    rows = [FakeAlert(id=1), FakeAlert(id=2)]
    query = FakeQuery(all_result=rows)
    db.set_query(FakeAlert, query)

    result = cm.list_alert_records(db, limit=2, status="open", severity="high")

    assert result == rows
    assert ("eq", "status", "open") in query.filters
    assert ("eq", "severity", "high") in query.filters
    assert ("desc", "created_at") in query.order_by_args
    assert query.limit_value == 2


def test_count_alert_records(db: FakeSession) -> None:
    query = FakeQuery(count_result=4)
    db.set_query(FakeAlert, query)

    result = cm.count_alert_records(db, status="open")

    assert result == 4
    assert ("eq", "status", "open") in query.filters


def test_update_alert_status(db: FakeSession) -> None:
    alert = FakeAlert(id=7, status="open")
    acknowledged_at = datetime(2026, 4, 23, 10, 0, 0)
    resolved_at = datetime(2026, 4, 23, 11, 0, 0)

    result = cm.update_alert_status(
        db,
        alert=alert,
        status="resolved",
        acknowledged_at=acknowledged_at,
        resolved_at=resolved_at,
    )

    assert result.status == "resolved"
    assert result.acknowledged_at == acknowledged_at
    assert result.resolved_at == resolved_at
    assert db.flush_count == 1