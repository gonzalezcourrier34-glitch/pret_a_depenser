# tests/test_route_history.py

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import route_history as rh


# =============================================================================
# Faux objets
# =============================================================================

class FakeDB:
    pass


# =============================================================================
# App de test
# =============================================================================

def create_test_client(fake_db: FakeDB) -> TestClient:
    app = FastAPI()
    app.include_router(rh.router)

    app.dependency_overrides[rh.get_db] = lambda: fake_db
    app.dependency_overrides[rh.verify_api_key] = lambda: None

    return TestClient(app)


# =============================================================================
# Helper métier
# =============================================================================

def test_resolve_decision_to_prediction_value_none() -> None:
    assert rh.resolve_decision_to_prediction_value(None) is None


def test_resolve_decision_to_prediction_value_accepted() -> None:
    assert rh.resolve_decision_to_prediction_value("accepted") == rh.ACCEPTED_PREDICTION_VALUE


def test_resolve_decision_to_prediction_value_refused() -> None:
    assert rh.resolve_decision_to_prediction_value("refused") == rh.REFUSED_PREDICTION_VALUE


# =============================================================================
# GET /history/predictions
# =============================================================================

def test_get_prediction_history_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_history(
        db,
        *,
        limit,
        offset,
        client_id,
        model_name,
        model_version,
        only_errors,
        prediction_value,
    ):
        assert db is fake_db
        assert limit == 50
        assert offset == 10
        assert client_id == 123
        assert model_name == "xgb"
        assert model_version == "v1"
        assert only_errors is True
        assert prediction_value == 0

        return {
            "count": 1,
            "limit": 50,
            "offset": 10,
            "items": [
                {
                    "id": 1,
                    "request_id": "req_1",
                    "client_id": 123,
                    "model_name": "xgb",
                    "model_version": "v1",
                    "prediction": 0,
                    "prediction_label": "accepted",
                    "score": 0.12,
                    "threshold_used": 0.5,
                    "latency_ms": 11.2,
                    "prediction_timestamp": "2026-04-23T10:00:00",
                    "status_code": 200,
                    "status": "success",
                    "error_message": None,
                }
            ],
        }

    monkeypatch.setattr(rh.history_service, "get_prediction_history", fake_get_prediction_history)

    response = client.get(
        "/history/predictions",
        params={
            "limit": 50,
            "offset": 10,
            "client_id": 123,
            "model_name": "xgb",
            "model_version": "v1",
            "only_errors": "true",
            "decision": "accepted",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["limit"] == 50
    assert body["offset"] == 10
    assert body["items"][0]["request_id"] == "req_1"
    assert body["items"][0]["prediction_label"] == "accepted"


def test_get_prediction_history_value_error_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_history(*args, **kwargs):
        raise ValueError("bad filter")

    monkeypatch.setattr(rh.history_service, "get_prediction_history", fake_get_prediction_history)

    response = client.get("/history/predictions")

    assert response.status_code == 400
    assert "Erreur lors de la récupération de l'historique des prédictions" in response.json()["detail"]
    assert "bad filter" in response.json()["detail"]


def test_get_prediction_history_unexpected_error_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_history(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rh.history_service, "get_prediction_history", fake_get_prediction_history)

    response = client.get("/history/predictions")

    assert response.status_code == 500
    assert "Erreur lors de la récupération de l'historique des prédictions" in response.json()["detail"]
    assert "boom" in response.json()["detail"]


# =============================================================================
# GET /history/predictions/{request_id}
# =============================================================================

def test_get_prediction_detail_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_detail(db, *, request_id):
        assert db is fake_db
        assert request_id == "req_123"
        return {
            "id": 1,
            "request_id": "req_123",
            "client_id": 100001,
            "model_name": "xgb",
            "model_version": "v1",
            "prediction": 1,
            "prediction_label": "refused",
            "score": 0.91,
            "threshold_used": 0.5,
            "latency_ms": 15.5,
            "input_data": {"A": 1},
            "output_data": {"prediction": 1},
            "prediction_timestamp": "2026-04-23T11:00:00",
            "status_code": 200,
            "status": "success",
            "error_message": None,
        }

    monkeypatch.setattr(rh.history_service, "get_prediction_detail", fake_get_prediction_detail)

    response = client.get("/history/predictions/req_123")

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req_123"
    assert body["prediction_label"] == "refused"
    assert body["input_data"] == {"A": 1}


def test_get_prediction_detail_not_found(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        rh.history_service,
        "get_prediction_detail",
        lambda db, *, request_id: None,
    )

    response = client.get("/history/predictions/missing_req")

    assert response.status_code == 404
    assert "Aucune prédiction trouvée" in response.json()["detail"]


def test_get_prediction_detail_value_error_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_detail(*args, **kwargs):
        raise ValueError("bad request id")

    monkeypatch.setattr(rh.history_service, "get_prediction_detail", fake_get_prediction_detail)

    response = client.get("/history/predictions/req_bad")

    assert response.status_code == 400
    assert "Erreur lors de la récupération du détail de prédiction" in response.json()["detail"]
    assert "bad request id" in response.json()["detail"]


def test_get_prediction_detail_unexpected_error_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_detail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rh.history_service, "get_prediction_detail", fake_get_prediction_detail)

    response = client.get("/history/predictions/req_boom")

    assert response.status_code == 500
    assert "Erreur lors de la récupération du détail de prédiction" in response.json()["detail"]
    assert "boom" in response.json()["detail"]


# =============================================================================
# GET /history/ground-truth
# =============================================================================

def test_get_ground_truth_history_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_ground_truth_history(db, *, limit, offset, client_id, request_id):
        assert db is fake_db
        assert limit == 20
        assert offset == 5
        assert client_id == 100001
        assert request_id == "req_1"
        return {
            "count": 1,
            "limit": 20,
            "offset": 5,
            "items": [
                {
                    "id": 1,
                    "request_id": "req_1",
                    "client_id": 100001,
                    "true_label": 1,
                    "label_source": "manual",
                    "observed_at": "2026-04-23T09:00:00",
                    "notes": "confirmed",
                }
            ],
        }

    monkeypatch.setattr(rh.history_service, "get_ground_truth_history", fake_get_ground_truth_history)

    response = client.get(
        "/history/ground-truth",
        params={
            "limit": 20,
            "offset": 5,
            "client_id": 100001,
            "request_id": "req_1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["true_label"] == 1
    assert body["items"][0]["label_source"] == "manual"


def test_get_ground_truth_history_value_error_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_ground_truth_history(*args, **kwargs):
        raise ValueError("bad ground truth query")

    monkeypatch.setattr(rh.history_service, "get_ground_truth_history", fake_get_ground_truth_history)

    response = client.get("/history/ground-truth")

    assert response.status_code == 400
    assert "Erreur lors de la récupération des vérités terrain" in response.json()["detail"]
    assert "bad ground truth query" in response.json()["detail"]


def test_get_ground_truth_history_unexpected_error_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_ground_truth_history(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rh.history_service, "get_ground_truth_history", fake_get_ground_truth_history)

    response = client.get("/history/ground-truth")

    assert response.status_code == 500
    assert "Erreur lors de la récupération des vérités terrain" in response.json()["detail"]
    assert "boom" in response.json()["detail"]


# =============================================================================
# GET /history/features/{request_id}
# =============================================================================

def test_get_prediction_features_snapshot_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_features_snapshot(db, *, request_id):
        assert db is fake_db
        assert request_id == "req_features"
        return {
            "request_id": "req_features",
            "client_id": 100001,
            "model_name": "xgb",
            "model_version": "v1",
            "snapshot_timestamp": "2026-04-23T12:00:00",
            "feature_count": 2,
            "items": [
                {
                    "feature_name": "AMT_CREDIT",
                    "feature_value": 50000.0,
                    "feature_type": "float",
                },
                {
                    "feature_name": "EXT_SOURCE_2",
                    "feature_value": 0.7,
                    "feature_type": "float",
                },
            ],
        }

    monkeypatch.setattr(
        rh.history_service,
        "get_prediction_features_snapshot",
        fake_get_prediction_features_snapshot,
    )

    response = client.get("/history/features/req_features")

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req_features"
    assert body["feature_count"] == 2
    assert body["items"][0]["feature_name"] == "AMT_CREDIT"


def test_get_prediction_features_snapshot_not_found(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        rh.history_service,
        "get_prediction_features_snapshot",
        lambda db, *, request_id: None,
    )

    response = client.get("/history/features/missing_req")

    assert response.status_code == 404
    assert "Aucun snapshot de features trouvé" in response.json()["detail"]


def test_get_prediction_features_snapshot_value_error_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_features_snapshot(*args, **kwargs):
        raise ValueError("bad snapshot request")

    monkeypatch.setattr(
        rh.history_service,
        "get_prediction_features_snapshot",
        fake_get_prediction_features_snapshot,
    )

    response = client.get("/history/features/req_bad")

    assert response.status_code == 400
    assert "Erreur lors de la récupération du snapshot de features" in response.json()["detail"]
    assert "bad snapshot request" in response.json()["detail"]


def test_get_prediction_features_snapshot_unexpected_error_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_get_prediction_features_snapshot(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        rh.history_service,
        "get_prediction_features_snapshot",
        fake_get_prediction_features_snapshot,
    )

    response = client.get("/history/features/req_boom")

    assert response.status_code == 500
    assert "Erreur lors de la récupération du snapshot de features" in response.json()["detail"]
    assert "boom" in response.json()["detail"]