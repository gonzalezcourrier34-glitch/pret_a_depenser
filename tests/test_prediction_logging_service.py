from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.services import prediction_logging_service as pls


# =============================================================================
# Faux objets
# =============================================================================

@dataclass
class FakePredictionLog:
    id: int
    request_id: str
    model_name: str
    model_version: str
    prediction: int
    score: float
    threshold_used: float | None
    latency_ms: float | None
    input_data: dict | None
    output_data: dict | None
    status_code: int | None
    error_message: str | None
    client_id: int | None = None
    prediction_timestamp: datetime | None = None
    event_time: datetime | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __init__(self, id: int, **kwargs: Any) -> None:
        self.id = id
        self.request_id = kwargs.pop("request_id")
        self.model_name = kwargs.pop("model_name")
        self.model_version = kwargs.pop("model_version")
        self.prediction = kwargs.pop("prediction")
        self.score = kwargs.pop("score")
        self.threshold_used = kwargs.pop("threshold_used", None)
        self.latency_ms = kwargs.pop("latency_ms", None)
        self.input_data = kwargs.pop("input_data", None)
        self.output_data = kwargs.pop("output_data", None)
        self.status_code = kwargs.pop("status_code", None)
        self.error_message = kwargs.pop("error_message", None)
        self.client_id = kwargs.pop("client_id", None)

        prediction_timestamp = kwargs.pop("prediction_timestamp", None)
        event_time = kwargs.pop("event_time", None)

        self.prediction_timestamp = prediction_timestamp or event_time
        self.event_time = event_time

        self.extra_fields = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fake_db():
    return object()


@pytest.fixture
def service(fake_db) -> pls.PredictionLoggingService:
    return pls.PredictionLoggingService(db=fake_db)


@pytest.fixture
def one_row_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "AMT_CREDIT": 50000.0,
                "EXT_SOURCE_2": 0.75,
                "DAYS_BIRTH": -12000,
            }
        ]
    )


# =============================================================================
# Helpers
# =============================================================================

def test_is_missing_cases() -> None:
    assert pls._is_missing(None) is True
    assert pls._is_missing(np.nan) is True
    assert pls._is_missing(pd.NA) is True
    assert pls._is_missing(0) is False
    assert pls._is_missing("x") is False


def test_to_python_scalar_none_and_nan() -> None:
    assert pls._to_python_scalar(None) is None
    assert pls._to_python_scalar(np.nan) is None


def test_to_python_scalar_numpy_scalar() -> None:
    result = pls._to_python_scalar(np.int64(5))
    assert result == 5
    assert isinstance(result, int)


def test_to_python_scalar_timestamp() -> None:
    ts = pd.Timestamp("2026-04-23T12:00:00")
    result = pls._to_python_scalar(ts)

    assert isinstance(result, datetime)
    assert result.year == 2026


def test_to_python_scalar_float_nan_inf() -> None:
    assert pls._to_python_scalar(float("nan")) is None
    assert pls._to_python_scalar(float("inf")) is None
    assert pls._to_python_scalar(float("-inf")) is None


def test_to_python_scalar_regular_value() -> None:
    assert pls._to_python_scalar(3.14) == 3.14
    assert pls._to_python_scalar("abc") == "abc"


def test_to_json_compatible_scalar_and_datetime() -> None:
    dt = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)

    assert pls._to_json_compatible(1) == 1
    assert pls._to_json_compatible(dt) == dt.isoformat()


def test_to_json_compatible_nested_structure() -> None:
    dt = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)

    result = pls._to_json_compatible(
        {
            "a": np.int64(1),
            "b": [1, np.nan, dt],
            "c": {"x": pd.Timestamp("2026-04-23T11:00:00")},
        }
    )

    assert result["a"] == 1
    assert result["b"][0] == 1
    assert result["b"][1] is None
    assert result["b"][2] == dt.isoformat()
    assert isinstance(result["c"]["x"], str)


def test_resolve_source_table_explicit() -> None:
    assert pls._resolve_source_table("my_source") == "my_source"
    assert pls._resolve_source_table("  my_source  ") == "my_source"


def test_resolve_source_table_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pls, "APPLICATION_CSV", "data/application_test.csv")
    assert pls._resolve_source_table(None) == "application_test.csv"


def test_dataframe_row_to_feature_records_success(one_row_df: pd.DataFrame) -> None:
    records = pls.dataframe_row_to_feature_records(
        one_row_df,
        request_id="req_1",
        model_name="xgb",
        model_version="v1",
        client_id=100001,
        source_table="features_cache",
    )

    assert len(records) == 3
    assert records[0]["request_id"] == "req_1"
    assert records[0]["client_id"] == 100001
    assert records[0]["model_name"] == "xgb"
    assert records[0]["model_version"] == "v1"
    assert records[0]["source_table"] == "features_cache"

    names = {r["feature_name"] for r in records}
    assert names == {"AMT_CREDIT", "EXT_SOURCE_2", "DAYS_BIRTH"}


def test_dataframe_row_to_feature_records_missing_values() -> None:
    df = pd.DataFrame([{"A": np.nan, "B": 2}])

    records = pls.dataframe_row_to_feature_records(
        df,
        request_id="req_1",
        model_name="xgb",
        model_version="v1",
    )

    rec_a = next(r for r in records if r["feature_name"] == "A")
    rec_b = next(r for r in records if r["feature_name"] == "B")

    assert rec_a["feature_value"] is None
    assert rec_a["feature_type"] is None
    assert float(rec_b["feature_value"]) == 2.0
    assert rec_b["feature_type"] in {"int", "float"}


def test_dataframe_row_to_feature_records_raises_type_error() -> None:
    with pytest.raises(TypeError, match="doit être un DataFrame pandas"):
        pls.dataframe_row_to_feature_records(  # type: ignore[arg-type]
            {"A": 1},
            request_id="req_1",
            model_name="xgb",
            model_version="v1",
        )


def test_dataframe_row_to_feature_records_raises_value_error() -> None:
    df = pd.DataFrame([{"A": 1}, {"A": 2}])

    with pytest.raises(ValueError, match="exactement une ligne"):
        pls.dataframe_row_to_feature_records(
            df,
            request_id="req_1",
            model_name="xgb",
            model_version="v1",
        )


# =============================================================================
# Service methods
# =============================================================================

def test_log_prediction_success(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
) -> None:
    fixed_now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)

    def fake_create_prediction_log(db, **kwargs):
        return FakePredictionLog(id=1, **kwargs)

    monkeypatch.setattr(
        pls.prediction_crud,
        "create_prediction_log",
        fake_create_prediction_log,
    )

    result = service.log_prediction(
        request_id="req_1",
        model_name="xgb",
        model_version="v1",
        prediction=1,
        score=0.82,
        threshold_used=0.5,
        latency_ms=12.3,
        input_data={"a": np.int64(1)},
        output_data={"score": 0.82},
        client_id=100001,
        status_code=200,
        error_message=None,
        event_time=fixed_now,
    )

    assert result.id == 1
    assert result.request_id == "req_1"
    assert result.prediction == 1
    assert result.score == 0.82
    assert result.threshold_used == 0.5
    assert result.latency_ms == 12.3
    assert result.client_id == 100001
    assert result.status_code == 200
    assert result.error_message is None
    assert result.prediction_timestamp == fixed_now
    assert result.input_data == {"a": 1}


def test_log_prediction_default_input_data_to_empty_dict(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
) -> None:
    fixed_now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)

    def fake_create_prediction_log(db, **kwargs):
        return FakePredictionLog(id=2, **kwargs)

    monkeypatch.setattr(
        pls.prediction_crud,
        "create_prediction_log",
        fake_create_prediction_log,
    )

    result = service.log_prediction(
        request_id="req_2",
        model_name="xgb",
        model_version="v1",
        prediction=0,
        score=0.1,
        threshold_used=None,
        latency_ms=None,
        input_data=None,
        output_data=None,
        event_time=fixed_now,
    )

    assert result.input_data == {}
    assert result.output_data is None
    assert result.threshold_used is None
    assert result.latency_ms is None


def test_log_prediction_error(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
) -> None:
    fixed_now = datetime(2026, 4, 23, 13, 0, tzinfo=timezone.utc)

    called = {}

    def fake_log_prediction(**kwargs):
        called.update(kwargs)
        return FakePredictionLog(id=3, **kwargs)

    monkeypatch.setattr(service, "log_prediction", fake_log_prediction)

    result = service.log_prediction_error(
        request_id="req_err",
        model_name="xgb",
        model_version="v1",
        input_data={"x": 1},
        error_message="boom",
        client_id=100001,
        status_code=500,
        latency_ms=9.5,
        event_time=fixed_now,
    )

    assert result.id == 3
    assert called["prediction"] == 0
    assert called["score"] == 0.0
    assert called["threshold_used"] is None
    assert called["status_code"] == 500
    assert called["error_message"] == "boom"
    assert called["output_data"] == {"status": "error", "message": "boom"}
    assert called["event_time"] == fixed_now


def test_log_prediction_features_snapshot_success(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
) -> None:
    fixed_now = datetime(2026, 4, 23, 14, 0, tzinfo=timezone.utc)
    called = {}

    def fake_create_feature_snapshots(db, records, timestamp):
        called["db"] = db
        called["records"] = records
        called["timestamp"] = timestamp

    monkeypatch.setattr(
        pls.prediction_crud,
        "create_feature_snapshots",
        fake_create_feature_snapshots,
    )

    records = [
        {
            "request_id": "req_1",
            "client_id": 1,
            "model_name": "xgb",
            "model_version": "v1",
            "feature_name": "A",
            "feature_value": "1",
            "feature_type": "int",
            "source_table": "csv",
        }
    ]

    service.log_prediction_features_snapshot(records, event_time=fixed_now)

    assert called["records"] == records
    assert called["timestamp"] == fixed_now


def test_log_prediction_features_snapshot_raises_on_empty(
    service: pls.PredictionLoggingService,
) -> None:
    with pytest.raises(ValueError, match="Aucun feature record"):
        service.log_prediction_features_snapshot([])


def test_log_feature_store_monitoring_success(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
) -> None:
    fixed_now = datetime(2026, 4, 23, 15, 0, tzinfo=timezone.utc)
    called = {}

    def fake_create_feature_store_records(db, records, timestamp):
        called["db"] = db
        called["records"] = records
        called["timestamp"] = timestamp

    monkeypatch.setattr(
        pls.monitoring_crud,
        "create_feature_store_records",
        fake_create_feature_store_records,
    )

    records = [
        {
            "request_id": "req_2",
            "client_id": 2,
            "model_name": "xgb",
            "model_version": "v1",
            "feature_name": "B",
            "feature_value": "2",
            "feature_type": "int",
            "source_table": "csv",
        }
    ]

    service.log_feature_store_monitoring(records, event_time=fixed_now)

    assert called["records"] == records
    assert called["timestamp"] == fixed_now


def test_log_feature_store_monitoring_raises_on_empty(
    service: pls.PredictionLoggingService,
) -> None:
    with pytest.raises(ValueError, match="Aucun feature record"):
        service.log_feature_store_monitoring([])


def test_log_full_prediction_event_success_with_feature_store(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
    one_row_df: pd.DataFrame,
) -> None:
    fixed_now = datetime(2026, 4, 23, 16, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(pls, "_utc_now", lambda: fixed_now)
    monkeypatch.setattr(pls, "APPLICATION_CSV", "data/application_test.csv")

    calls = {
        "snapshot": None,
        "feature_store": None,
        "prediction": None,
    }

    def fake_log_snapshot(feature_records, event_time=None):
        calls["snapshot"] = {
            "feature_records": feature_records,
            "event_time": event_time,
        }

    def fake_log_feature_store(feature_records, event_time=None):
        calls["feature_store"] = {
            "feature_records": feature_records,
            "event_time": event_time,
        }

    def fake_log_prediction(**kwargs):
        calls["prediction"] = kwargs
        return FakePredictionLog(id=10, **kwargs)

    monkeypatch.setattr(service, "log_prediction_features_snapshot", fake_log_snapshot)
    monkeypatch.setattr(service, "log_feature_store_monitoring", fake_log_feature_store)
    monkeypatch.setattr(service, "log_prediction", fake_log_prediction)

    service.log_full_prediction_event(
        request_id="req_full",
        model_name="xgb",
        model_version="v1",
        features_df=one_row_df,
        raw_input_data={"raw": "input"},
        prediction=1,
        score=0.95,
        threshold_used=0.5,
        latency_ms=33.2,
        client_id=100001,
        write_feature_store_monitoring=True,
        source_table="features_ready_cache",
        output_data=None,
        status_code=200,
    )

    assert calls["snapshot"] is not None
    assert calls["feature_store"] is not None
    assert calls["prediction"] is not None

    assert calls["snapshot"]["event_time"] == fixed_now
    assert calls["feature_store"]["event_time"] == fixed_now

    pred_kwargs = calls["prediction"]
    assert pred_kwargs["request_id"] == "req_full"
    assert pred_kwargs["prediction"] == 1
    assert pred_kwargs["score"] == 0.95
    assert pred_kwargs["threshold_used"] == 0.5
    assert pred_kwargs["latency_ms"] == 33.2
    assert pred_kwargs["client_id"] == 100001
    assert pred_kwargs["status_code"] == 200
    assert pred_kwargs["event_time"] == fixed_now
    assert pred_kwargs["input_data"] == {"raw": "input"}

    output_data = pred_kwargs["output_data"]
    assert output_data["request_id"] == "req_full"
    assert output_data["prediction"] == 1
    assert output_data["score"] == 0.95
    assert output_data["threshold_used"] == 0.5
    assert output_data["model_name"] == "xgb"
    assert output_data["model_version"] == "v1"
    assert output_data["latency_ms"] == 33.2
    assert output_data["source_csv"] == "application_test.csv"


def test_log_full_prediction_event_without_feature_store(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
    one_row_df: pd.DataFrame,
) -> None:
    fixed_now = datetime(2026, 4, 23, 17, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(pls, "_utc_now", lambda: fixed_now)

    called = {
        "feature_store_called": False,
        "prediction_called": False,
    }

    monkeypatch.setattr(
        service,
        "log_prediction_features_snapshot",
        lambda feature_records, event_time=None: None,
    )

    def fake_feature_store(feature_records, event_time=None):
        called["feature_store_called"] = True

    def fake_prediction(**kwargs):
        called["prediction_called"] = True
        return FakePredictionLog(id=11, **kwargs)

    monkeypatch.setattr(service, "log_feature_store_monitoring", fake_feature_store)
    monkeypatch.setattr(service, "log_prediction", fake_prediction)

    service.log_full_prediction_event(
        request_id="req_no_store",
        model_name="xgb",
        model_version="v1",
        features_df=one_row_df,
        raw_input_data={"x": 1},
        prediction=0,
        score=0.12,
        threshold_used=0.5,
        latency_ms=10.0,
        write_feature_store_monitoring=False,
    )

    assert called["feature_store_called"] is False
    assert called["prediction_called"] is True


def test_log_full_prediction_event_with_custom_output_data(
    monkeypatch: pytest.MonkeyPatch,
    service: pls.PredictionLoggingService,
    one_row_df: pd.DataFrame,
) -> None:
    fixed_now = datetime(2026, 4, 23, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(pls, "_utc_now", lambda: fixed_now)

    captured = {}

    monkeypatch.setattr(
        service,
        "log_prediction_features_snapshot",
        lambda feature_records, event_time=None: None,
    )
    monkeypatch.setattr(
        service,
        "log_feature_store_monitoring",
        lambda feature_records, event_time=None: None,
    )

    def fake_log_prediction(**kwargs):
        captured.update(kwargs)
        return FakePredictionLog(id=12, **kwargs)

    monkeypatch.setattr(service, "log_prediction", fake_log_prediction)

    custom_output = {"custom": "value"}

    service.log_full_prediction_event(
        request_id="req_custom",
        model_name="xgb",
        model_version="v1",
        features_df=one_row_df,
        raw_input_data={"x": 1},
        prediction=1,
        score=0.66,
        threshold_used=0.4,
        latency_ms=7.0,
        output_data=custom_output,
    )

    assert captured["output_data"] == {"custom": "value"}


def test_log_full_prediction_event_raises_if_not_dataframe(
    service: pls.PredictionLoggingService,
) -> None:
    with pytest.raises(TypeError, match="doit être un DataFrame pandas"):
        service.log_full_prediction_event(  # type: ignore[arg-type]
            request_id="req_bad",
            model_name="xgb",
            model_version="v1",
            features_df={"A": 1},
            raw_input_data=None,
            prediction=1,
            score=0.9,
            threshold_used=0.5,
            latency_ms=1.0,
        )


def test_log_full_prediction_event_raises_if_not_one_row(
    service: pls.PredictionLoggingService,
) -> None:
    df = pd.DataFrame([{"A": 1}, {"A": 2}])

    with pytest.raises(ValueError, match="exactement une ligne"):
        service.log_full_prediction_event(
            request_id="req_bad_rows",
            model_name="xgb",
            model_version="v1",
            features_df=df,
            raw_input_data=None,
            prediction=1,
            score=0.9,
            threshold_used=0.5,
            latency_ms=1.0,
        )