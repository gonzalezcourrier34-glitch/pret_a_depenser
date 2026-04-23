# tests/test_crud_prediction.py

from __future__ import annotations

from datetime import datetime

import pytest

from app.crud import prediction as cp


# =============================================================================
# Faux outils ORM
# =============================================================================

class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __ge__(self, other):
        return ("ge", self.name, other)

    def __lt__(self, other):
        return ("lt", self.name, other)

    def is_not(self, other):
        return ("is_not", self.name, other)

    def desc(self):
        return ("desc", self.name)

    def asc(self):
        return ("asc", self.name)


class FakeFunc:
    @staticmethod
    def avg(value):
        return ("avg", value)


class FakeEntity:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakePredictionLog(FakeEntity):
    id = FakeColumn("id")
    request_id = FakeColumn("request_id")
    client_id = FakeColumn("client_id")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    error_message = FakeColumn("error_message")
    prediction_timestamp = FakeColumn("prediction_timestamp")
    latency_ms = FakeColumn("latency_ms")


class FakePredictionFeatureSnapshot(FakeEntity):
    id = FakeColumn("id")
    request_id = FakeColumn("request_id")
    feature_name = FakeColumn("feature_name")


class FakeGroundTruthLabel(FakeEntity):
    id = FakeColumn("id")
    request_id = FakeColumn("request_id")
    client_id = FakeColumn("client_id")
    true_label = FakeColumn("true_label")
    label_source = FakeColumn("label_source")


class FakeQuery:
    def __init__(
        self,
        *,
        first_result=None,
        all_result=None,
        count_result=0,
        scalar_result=None,
    ):
        self.first_result = first_result
        self.all_result = all_result or []
        self.count_result = count_result
        self.scalar_result = scalar_result
        self.filters = []
        self.order_by_args = []
        self.limit_value = None
        self.entities_args = None

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

    def with_entities(self, *args):
        self.entities_args = args
        return self

    def scalar(self):
        return self.scalar_result


class FakeSession:
    def __init__(self):
        self.queries = {}
        self.added = []
        self.added_all = []
        self.refresh_calls = []
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

    def refresh(self, entity):
        self.refresh_calls.append(entity)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def patch_models_and_func(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cp, "PredictionLog", FakePredictionLog)
    monkeypatch.setattr(cp, "PredictionFeatureSnapshot", FakePredictionFeatureSnapshot)
    monkeypatch.setattr(cp, "GroundTruthLabel", FakeGroundTruthLabel)
    monkeypatch.setattr(cp, "func", FakeFunc)


@pytest.fixture
def db() -> FakeSession:
    return FakeSession()


# =============================================================================
# Helpers query builder
# =============================================================================

def test_build_prediction_logs_query_applies_all_filters(db: FakeSession) -> None:
    query = FakeQuery()
    db.set_query(FakePredictionLog, query)

    start = datetime(2026, 4, 1, 10, 0, 0)
    end = datetime(2026, 4, 2, 10, 0, 0)

    result = cp._build_prediction_logs_query(
        db,
        client_id=100001,
        model_name="xgb",
        model_version="v1",
        only_errors=True,
        window_start=start,
        window_end=end,
    )

    assert result is query
    assert ("eq", "client_id", 100001) in query.filters
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("is_not", "error_message", None) in query.filters
    assert ("ge", "prediction_timestamp", start) in query.filters
    assert ("lt", "prediction_timestamp", end) in query.filters


def test_build_prediction_logs_query_without_optional_filters(db: FakeSession) -> None:
    query = FakeQuery()
    db.set_query(FakePredictionLog, query)

    result = cp._build_prediction_logs_query(db)

    assert result is query
    assert query.filters == []


# =============================================================================
# Prediction logs
# =============================================================================

def test_create_prediction_log(db: FakeSession) -> None:
    ts = datetime(2026, 4, 23, 10, 0, 0)

    entity = cp.create_prediction_log(
        db,
        request_id="req_1",
        client_id=100001,
        model_name="xgb",
        model_version="v1",
        prediction=1,
        score=0.87,
        threshold_used=0.5,
        latency_ms=12.3,
        input_data={"A": 1},
        output_data={"prediction": 1},
        prediction_timestamp=ts,
        status_code=200,
        error_message=None,
    )

    assert entity.request_id == "req_1"
    assert entity.client_id == 100001
    assert entity.model_name == "xgb"
    assert entity.model_version == "v1"
    assert entity.prediction == 1
    assert entity.score == 0.87
    assert entity.threshold_used == 0.5
    assert entity.latency_ms == 12.3
    assert entity.input_data == {"A": 1}
    assert entity.output_data == {"prediction": 1}
    assert entity.prediction_timestamp == ts
    assert entity.status_code == 200
    assert entity.error_message is None
    assert entity in db.added
    assert db.flush_count == 1


def test_get_prediction_log_by_request_id(db: FakeSession) -> None:
    entity = FakePredictionLog(request_id="req_42")
    query = FakeQuery(first_result=entity)
    db.set_query(FakePredictionLog, query)

    result = cp.get_prediction_log_by_request_id(db, request_id="req_42")

    assert result is entity
    assert ("eq", "request_id", "req_42") in query.filters


def test_list_prediction_logs(db: FakeSession) -> None:
    rows = [FakePredictionLog(request_id="req_1"), FakePredictionLog(request_id="req_2")]
    query = FakeQuery(all_result=rows)
    db.set_query(FakePredictionLog, query)

    result = cp.list_prediction_logs(
        db,
        limit=2,
        client_id=100001,
        model_name="xgb",
        model_version="v1",
        only_errors=True,
    )

    assert result == rows
    assert ("eq", "client_id", 100001) in query.filters
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("is_not", "error_message", None) in query.filters
    assert ("desc", "prediction_timestamp") in query.order_by_args
    assert query.limit_value == 2


def test_count_prediction_logs(db: FakeSession) -> None:
    query = FakeQuery(count_result=7)
    db.set_query(FakePredictionLog, query)

    result = cp.count_prediction_logs(
        db,
        client_id=100001,
        model_name="xgb",
        model_version="v1",
    )

    assert result == 7
    assert ("eq", "client_id", 100001) in query.filters
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters


def test_count_prediction_errors(db: FakeSession) -> None:
    query = FakeQuery(count_result=3)
    db.set_query(FakePredictionLog, query)

    result = cp.count_prediction_errors(
        db,
        client_id=100001,
        model_name="xgb",
        model_version="v1",
    )

    assert result == 3
    assert ("eq", "client_id", 100001) in query.filters
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("is_not", "error_message", None) in query.filters


def test_get_latest_prediction_log(db: FakeSession) -> None:
    entity = FakePredictionLog(request_id="req_latest")
    query = FakeQuery(first_result=entity)
    db.set_query(FakePredictionLog, query)

    result = cp.get_latest_prediction_log(
        db,
        model_name="xgb",
        model_version="v1",
    )

    assert result is entity
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("desc", "prediction_timestamp") in query.order_by_args


def test_get_average_latency_ms_success(db: FakeSession) -> None:
    query = FakeQuery(scalar_result=15.7)
    db.set_query(FakePredictionLog, query)

    result = cp.get_average_latency_ms(
        db,
        model_name="xgb",
        model_version="v1",
    )

    assert result == 15.7
    assert query.entities_args == (("avg", FakePredictionLog.latency_ms),)


def test_get_average_latency_ms_none(db: FakeSession) -> None:
    query = FakeQuery(scalar_result=None)
    db.set_query(FakePredictionLog, query)

    result = cp.get_average_latency_ms(db, model_name="xgb")

    assert result is None


def test_get_average_latency_ms_cast_error(db: FakeSession) -> None:
    query = FakeQuery(scalar_result=object())
    db.set_query(FakePredictionLog, query)

    result = cp.get_average_latency_ms(db, model_name="xgb")

    assert result is None


# =============================================================================
# Prediction feature snapshots
# =============================================================================

def test_create_feature_snapshots_with_records(db: FakeSession) -> None:
    ts = datetime(2026, 4, 23, 11, 0, 0)

    cp.create_feature_snapshots(
        db,
        records=[
            {
                "request_id": "req_1",
                "client_id": 100001,
                "model_name": "xgb",
                "model_version": "v1",
                "feature_name": "AMT_CREDIT",
                "feature_value": "50000.0",
                "feature_type": "float",
            },
            {
                "request_id": "req_1",
                "client_id": 100001,
                "model_name": "xgb",
                "model_version": "v1",
                "feature_name": "EXT_SOURCE_2",
                "feature_value": "0.7",
                "feature_type": "float",
            },
        ],
        timestamp=ts,
    )

    assert len(db.added_all) == 2
    assert db.flush_count == 1
    assert db.added_all[0].feature_name == "AMT_CREDIT"
    assert db.added_all[0].snapshot_timestamp == ts
    assert db.added_all[1].feature_name == "EXT_SOURCE_2"


def test_create_feature_snapshots_with_empty_records(db: FakeSession) -> None:
    cp.create_feature_snapshots(
        db,
        records=[],
        timestamp=datetime(2026, 4, 23, 11, 0, 0),
    )

    assert db.added_all == []
    assert db.flush_count == 0


def test_list_feature_snapshots_by_request_id(db: FakeSession) -> None:
    rows = [
        FakePredictionFeatureSnapshot(request_id="req_1", feature_name="A"),
        FakePredictionFeatureSnapshot(request_id="req_1", feature_name="B"),
    ]
    query = FakeQuery(all_result=rows)
    db.set_query(FakePredictionFeatureSnapshot, query)

    result = cp.list_feature_snapshots_by_request_id(db, request_id="req_1")

    assert result == rows
    assert ("eq", "request_id", "req_1") in query.filters
    assert ("asc", "feature_name") in query.order_by_args


# =============================================================================
# Ground truth
# =============================================================================

def test_create_ground_truth_label(db: FakeSession) -> None:
    observed_at = datetime(2026, 4, 23, 12, 0, 0)

    entity = cp.create_ground_truth_label(
        db,
        request_id="req_gt_1",
        client_id=100001,
        true_label=1,
        label_source="manual_review",
        observed_at=observed_at,
        notes="confirmed default",
    )

    assert entity.request_id == "req_gt_1"
    assert entity.client_id == 100001
    assert entity.true_label == 1
    assert entity.label_source == "manual_review"
    assert entity.observed_at == observed_at
    assert entity.notes == "confirmed default"
    assert entity in db.added
    assert db.flush_count == 1
    assert db.refresh_calls == [entity]