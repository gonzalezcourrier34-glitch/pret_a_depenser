# tests/services/test_history_service.py

from __future__ import annotations

from typing import Any

import pytest

from app.services import history_service as hs


# =============================================================================
# Faux objets SQLAlchemy
# =============================================================================

class FakeMappingsResult:
    def __init__(
        self,
        *,
        all_rows: list[dict[str, Any]] | None = None,
        first_row: dict[str, Any] | None = None,
    ) -> None:
        self._all_rows = all_rows or []
        self._first_row = first_row

    def all(self) -> list[dict[str, Any]]:
        return self._all_rows

    def first(self) -> dict[str, Any] | None:
        return self._first_row


class FakeExecuteResult:
    def __init__(
        self,
        *,
        all_rows: list[dict[str, Any]] | None = None,
        first_row: dict[str, Any] | None = None,
    ) -> None:
        self._all_rows = all_rows or []
        self._first_row = first_row

    def mappings(self) -> FakeMappingsResult:
        return FakeMappingsResult(
            all_rows=self._all_rows,
            first_row=self._first_row,
        )


class FakeSession:
    def __init__(
        self,
        *,
        all_rows: list[dict[str, Any]] | None = None,
        first_row: dict[str, Any] | None = None,
    ) -> None:
        self._all_rows = all_rows or []
        self._first_row = first_row
        self.last_sql = None
        self.last_params = None

    def execute(self, sql: Any, params: dict[str, Any] | None = None) -> FakeExecuteResult:
        self.last_sql = sql
        self.last_params = params or {}
        return FakeExecuteResult(
            all_rows=self._all_rows,
            first_row=self._first_row,
        )


# =============================================================================
# Tests helpers
# =============================================================================

@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (0, "accepted"),
        (1, "refused"),
        (None, None),
        (3, None),
    ],
)
def test_prediction_label_from_value(prediction: int | None, expected: str | None) -> None:
    assert hs._prediction_label_from_value(prediction) == expected


@pytest.mark.parametrize(
    ("error_message", "status_code", "expected"),
    [
        ("boom", 200, "error"),
        ("boom", 500, "error"),
        (None, None, None),
        (None, 200, "success"),
        (None, 201, "success"),
        (None, 204, "success"),
        (None, 400, "error"),
        (None, 500, "error"),
    ],
)
def test_status_from_row(
    error_message: str | None,
    status_code: int | None,
    expected: str | None,
) -> None:
    assert hs._status_from_row(error_message, status_code) == expected


# =============================================================================
# Tests get_prediction_history
# =============================================================================

def test_get_prediction_history_basic() -> None:
    rows = [
        {
            "id": 1,
            "request_id": "req_1",
            "client_id": 100001,
            "model_name": "xgb",
            "model_version": "v1",
            "prediction": 0,
            "score": 0.12,
            "threshold_used": 0.5,
            "latency_ms": 42.0,
            "prediction_timestamp": "2026-04-23T10:00:00",
            "status_code": 200,
            "error_message": None,
        },
        {
            "id": 2,
            "request_id": "req_2",
            "client_id": 100002,
            "model_name": "xgb",
            "model_version": "v1",
            "prediction": 1,
            "score": 0.91,
            "threshold_used": 0.5,
            "latency_ms": 55.0,
            "prediction_timestamp": "2026-04-23T10:01:00",
            "status_code": 500,
            "error_message": "internal error",
        },
    ]
    db = FakeSession(all_rows=rows)

    result = hs.get_prediction_history(db, limit=100, offset=0)

    assert result["count"] == 2
    assert result["limit"] == 100
    assert result["offset"] == 0
    assert len(result["items"]) == 2

    assert result["items"][0]["prediction_label"] == "accepted"
    assert result["items"][0]["status"] == "success"

    assert result["items"][1]["prediction_label"] == "refused"
    assert result["items"][1]["status"] == "error"


def test_get_prediction_history_with_filters_in_params() -> None:
    db = FakeSession(all_rows=[])

    hs.get_prediction_history(
        db,
        limit=10,
        offset=20,
        client_id=123,
        model_name="lgbm",
        model_version="v3",
        only_errors=True,
        prediction_value=1,
    )

    assert db.last_params["limit"] == 10
    assert db.last_params["offset"] == 20
    assert db.last_params["client_id"] == 123
    assert db.last_params["model_name"] == "lgbm"
    assert db.last_params["model_version"] == "v3"
    assert db.last_params["prediction_value"] == 1


def test_get_prediction_history_empty() -> None:
    db = FakeSession(all_rows=[])

    result = hs.get_prediction_history(db)

    assert result == {
        "count": 0,
        "limit": 100,
        "offset": 0,
        "items": [],
    }


def test_get_prediction_history_prediction_none_and_status_none() -> None:
    rows = [
        {
            "id": 3,
            "request_id": "req_3",
            "client_id": 100003,
            "model_name": "catboost",
            "model_version": "v2",
            "prediction": None,
            "score": None,
            "threshold_used": None,
            "latency_ms": None,
            "prediction_timestamp": "2026-04-23T10:02:00",
            "status_code": None,
            "error_message": None,
        }
    ]
    db = FakeSession(all_rows=rows)

    result = hs.get_prediction_history(db)

    item = result["items"][0]
    assert item["prediction_label"] is None
    assert item["status"] is None


# =============================================================================
# Tests get_prediction_detail
# =============================================================================

def test_get_prediction_detail_success() -> None:
    row = {
        "id": 1,
        "request_id": "req_detail",
        "client_id": 100001,
        "model_name": "xgb",
        "model_version": "v1",
        "prediction": 1,
        "score": 0.88,
        "threshold_used": 0.5,
        "latency_ms": 30.5,
        "input_data": {"a": 1},
        "output_data": {"prediction": 1},
        "prediction_timestamp": "2026-04-23T11:00:00",
        "status_code": 200,
        "error_message": None,
    }
    db = FakeSession(first_row=row)

    result = hs.get_prediction_detail(db, request_id="req_detail")

    assert result is not None
    assert result["request_id"] == "req_detail"
    assert result["prediction_label"] == "refused"
    assert result["status"] == "success"
    assert result["input_data"] == {"a": 1}
    assert result["output_data"] == {"prediction": 1}
    assert db.last_params == {"request_id": "req_detail"}


def test_get_prediction_detail_returns_none_when_not_found() -> None:
    db = FakeSession(first_row=None)

    result = hs.get_prediction_detail(db, request_id="missing_req")

    assert result is None


def test_get_prediction_detail_error_status() -> None:
    row = {
        "id": 2,
        "request_id": "req_error",
        "client_id": 100002,
        "model_name": "xgb",
        "model_version": "v1",
        "prediction": 0,
        "score": 0.25,
        "threshold_used": 0.5,
        "latency_ms": 70.0,
        "input_data": {},
        "output_data": {},
        "prediction_timestamp": "2026-04-23T11:05:00",
        "status_code": 500,
        "error_message": "db timeout",
    }
    db = FakeSession(first_row=row)

    result = hs.get_prediction_detail(db, request_id="req_error")

    assert result is not None
    assert result["prediction_label"] == "accepted"
    assert result["status"] == "error"
    assert result["error_message"] == "db timeout"


# =============================================================================
# Tests get_ground_truth_history
# =============================================================================

def test_get_ground_truth_history_basic() -> None:
    rows = [
        {
            "id": 1,
            "request_id": "req_1",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual_review",
            "observed_at": "2026-04-22T10:00:00",
            "notes": "confirmed default",
        },
        {
            "id": 2,
            "request_id": "req_2",
            "client_id": 100002,
            "true_label": 0,
            "label_source": "batch_update",
            "observed_at": "2026-04-21T10:00:00",
            "notes": None,
        },
    ]
    db = FakeSession(all_rows=rows)

    result = hs.get_ground_truth_history(db, limit=50, offset=10)

    assert result["count"] == 2
    assert result["limit"] == 50
    assert result["offset"] == 10
    assert len(result["items"]) == 2
    assert result["items"][0]["label_source"] == "manual_review"
    assert result["items"][1]["true_label"] == 0


def test_get_ground_truth_history_with_filters() -> None:
    db = FakeSession(all_rows=[])

    hs.get_ground_truth_history(
        db,
        limit=5,
        offset=15,
        client_id=789,
        request_id="req_x",
    )

    assert db.last_params["limit"] == 5
    assert db.last_params["offset"] == 15
    assert db.last_params["client_id"] == 789
    assert db.last_params["request_id"] == "req_x"


def test_get_ground_truth_history_empty() -> None:
    db = FakeSession(all_rows=[])

    result = hs.get_ground_truth_history(db)

    assert result == {
        "count": 0,
        "limit": 100,
        "offset": 0,
        "items": [],
    }


# =============================================================================
# Tests get_prediction_features_snapshot
# =============================================================================

def test_get_prediction_features_snapshot_success() -> None:
    rows = [
        {
            "request_id": "req_snap",
            "client_id": 100001,
            "model_name": "xgb",
            "model_version": "v1",
            "feature_name": "AMT_CREDIT",
            "feature_value": 50000.0,
            "feature_type": "float",
            "snapshot_timestamp": "2026-04-23T12:00:00",
        },
        {
            "request_id": "req_snap",
            "client_id": 100001,
            "model_name": "xgb",
            "model_version": "v1",
            "feature_name": "EXT_SOURCE_2",
            "feature_value": 0.5,
            "feature_type": "float",
            "snapshot_timestamp": "2026-04-23T12:00:00",
        },
    ]
    db = FakeSession(all_rows=rows)

    result = hs.get_prediction_features_snapshot(db, request_id="req_snap")

    assert result is not None
    assert result["request_id"] == "req_snap"
    assert result["client_id"] == 100001
    assert result["model_name"] == "xgb"
    assert result["model_version"] == "v1"
    assert result["snapshot_timestamp"] == "2026-04-23T12:00:00"
    assert result["feature_count"] == 2
    assert result["items"] == [
        {
            "feature_name": "AMT_CREDIT",
            "feature_value": 50000.0,
            "feature_type": "float",
        },
        {
            "feature_name": "EXT_SOURCE_2",
            "feature_value": 0.5,
            "feature_type": "float",
        },
    ]
    assert db.last_params == {"request_id": "req_snap"}


def test_get_prediction_features_snapshot_returns_none_when_empty() -> None:
    db = FakeSession(all_rows=[])

    result = hs.get_prediction_features_snapshot(db, request_id="missing_snap")

    assert result is None