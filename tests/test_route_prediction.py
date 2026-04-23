# tests/test_route_prediction.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import route_prediction as rp


# =============================================================================
# Faux objets
# =============================================================================

class FakeDB:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.executed = []

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def execute(self, query: Any) -> None:
        self.executed.append(query)


# =============================================================================
# App de test
# =============================================================================

def create_test_client(fake_db: FakeDB) -> TestClient:
    app = FastAPI()
    app.include_router(rp.router)

    app.dependency_overrides[rp.get_db] = lambda: fake_db
    app.dependency_overrides[rp.verify_api_key] = lambda: None

    return TestClient(app)


# =============================================================================
# Helpers payloads
# =============================================================================

def build_predict_payload(
    *,
    client_id: int | None = 100001,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Construit un payload de prédiction valide.
    """
    if features is None:
        features = {
            "AMT_CREDIT": 50000.0,
            "AMT_INCOME_TOTAL": 120000.0,
            "CNT_CHILDREN": 1,
        }

    return {
        "SK_ID_CURR": client_id,
        "features": features,
    }


def build_batch_item(
    *,
    request_id: str,
    client_id: int | None,
    prediction: int | None,
    score: float | None,
    status: str = "success",
    error_message: str | None = None,
) -> dict[str, Any]:
    """
    Construit un item de réponse batch compatible avec le schéma API.
    """
    return {
        "request_id": request_id,
        "client_id": client_id,
        "prediction": prediction,
        "score": score,
        "threshold_used": 0.0593,
        "model_name": "credit_scoring_model",
        "model_version": "v1",
        "latency_ms": 12.3,
        "status": status,
        "error_message": error_message,
    }


# =============================================================================
# /predict/health
# =============================================================================

def test_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rp, "get_model", lambda: object())
    monkeypatch.setattr(rp, "get_threshold", lambda: 0.0593)

    response = client.get("/predict/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 0
    assert len(fake_db.executed) == 1


def test_health_returns_503_if_model_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rp, "get_model", lambda: (_ for _ in ()).throw(RuntimeError("boom model")))
    monkeypatch.setattr(rp, "get_threshold", lambda: 0.0593)

    response = client.get("/predict/health")

    assert response.status_code == 503
    assert "Healthcheck failed" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 0


# =============================================================================
# /predict
# =============================================================================

def test_predict_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_make_prediction(**kwargs):
        return {
            "request_id": "req_123",
            "prediction": 1,
            "score": 0.82,
            "model_version": "v1",
            "latency_ms": 15.4,
        }

    monkeypatch.setattr(rp, "make_prediction", fake_make_prediction)

    response = client.post("/predict", json=build_predict_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req_123"
    assert body["prediction"] == 1
    assert body["score"] == 0.82
    assert body["model_version"] == "v1"
    assert body["latency_ms"] == 15.4
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_predict_returns_400_on_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_make_prediction(**kwargs):
        raise ValueError("features invalides")

    monkeypatch.setattr(rp, "make_prediction", fake_make_prediction)

    response = client.post("/predict", json=build_predict_payload())

    assert response.status_code == 400
    assert response.json()["detail"] == "Prediction failed: features invalides"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_predict_returns_500_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_make_prediction(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "make_prediction", fake_make_prediction)

    response = client.post("/predict", json=build_predict_payload())

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed: boom"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_predict_returns_500_if_service_returns_non_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rp, "make_prediction", lambda **kwargs: ["not", "a", "dict"])

    response = client.post("/predict", json=build_predict_payload())

    assert response.status_code == 500
    assert "Prediction failed:" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /predict/batch
# =============================================================================

def test_predict_batch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_batch_prediction(**kwargs):
        return {
            "batch_size": 2,
            "success_count": 2,
            "error_count": 0,
            "model_name": "credit_scoring_model",
            "model_version": "v1",
            "batch_latency_ms": 48.2,
            "items": [
                build_batch_item(
                    request_id="req_1",
                    client_id=100001,
                    prediction=1,
                    score=0.71,
                ),
                build_batch_item(
                    request_id="req_2",
                    client_id=100002,
                    prediction=0,
                    score=0.12,
                ),
            ],
            "selected_client_ids": [100001, 100002],
        }

    monkeypatch.setattr(rp, "run_batch_prediction", fake_run_batch_prediction)

    response = client.post(
        "/predict/batch",
        json=[
            build_predict_payload(client_id=100001),
            build_predict_payload(client_id=100002),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 2
    assert body["success_count"] == 2
    assert body["error_count"] == 0
    assert len(body["items"]) == 2
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_predict_batch_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_batch_prediction(**kwargs):
        raise ValueError("batch invalide")

    monkeypatch.setattr(rp, "run_batch_prediction", fake_run_batch_prediction)

    response = client.post(
        "/predict/batch",
        json=[build_predict_payload(client_id=100001)],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Batch prediction failed: batch invalide"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_predict_batch_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_batch_prediction(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "run_batch_prediction", fake_run_batch_prediction)

    response = client.post(
        "/predict/batch",
        json=[build_predict_payload(client_id=100001)],
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Batch prediction failed: boom"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /predict/{client_id}
# =============================================================================

def test_predict_from_client_id_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_make_prediction_from_client_id(**kwargs):
        return {
            "request_id": "req_client_1",
            "prediction": 0,
            "score": 0.19,
            "model_version": "v1",
            "latency_ms": 10.8,
        }

    monkeypatch.setattr(
        rp,
        "make_prediction_from_client_id",
        fake_make_prediction_from_client_id,
    )

    response = client.get("/predict/100001")

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req_client_1"
    assert body["prediction"] == 0
    assert body["score"] == 0.19
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_predict_from_client_id_returns_404_on_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_make_prediction_from_client_id(**kwargs):
        raise ValueError("Client 100001 introuvable.")

    monkeypatch.setattr(
        rp,
        "make_prediction_from_client_id",
        fake_make_prediction_from_client_id,
    )

    response = client.get("/predict/100001")

    assert response.status_code == 404
    assert response.json()["detail"] == "Client 100001 introuvable."
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_predict_from_client_id_returns_500_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_make_prediction_from_client_id(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        rp,
        "make_prediction_from_client_id",
        fake_make_prediction_from_client_id,
    )

    response = client.get("/predict/100001")

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de la prédiction : boom"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /predict/ground-truth
# =============================================================================

def test_create_prediction_ground_truth_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    observed_at = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)

    def fake_create_ground_truth_label(**kwargs):
        return {
            "id": 1,
            "request_id": "req_123",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual_review",
            "observed_at": observed_at.isoformat(),
            "notes": "confirmé",
        }

    monkeypatch.setattr(rp, "create_ground_truth_label", fake_create_ground_truth_label)

    response = client.post(
        "/predict/ground-truth",
        json={
            "request_id": "req_123",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual_review",
            "observed_at": observed_at.isoformat(),
            "notes": "confirmé",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["request_id"] == "req_123"
    assert body["client_id"] == 100001
    assert body["true_label"] == 1
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_create_prediction_ground_truth_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_create_ground_truth_label(**kwargs):
        raise ValueError("request_id inconnu")

    monkeypatch.setattr(rp, "create_ground_truth_label", fake_create_ground_truth_label)

    response = client.post(
        "/predict/ground-truth",
        json={
            "request_id": "req_404",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual_review",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Création du ground truth impossible : request_id inconnu"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_create_prediction_ground_truth_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_create_ground_truth_label(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "create_ground_truth_label", fake_create_ground_truth_label)

    response = client.post(
        "/predict/ground-truth",
        json={
            "request_id": "req_123",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual_review",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de la création du ground truth : boom"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /predict/simulate/real-sample
# =============================================================================

def test_simulate_real_sample_predictions_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_real_client_simulation(**kwargs):
        return {
            "batch_size": 2,
            "success_count": 2,
            "error_count": 0,
            "model_name": "credit_scoring_model",
            "model_version": "v1",
            "batch_latency_ms": 33.4,
            "items": [
                build_batch_item(
                    request_id="req_real_1",
                    client_id=100001,
                    prediction=1,
                    score=0.61,
                ),
                build_batch_item(
                    request_id="req_real_2",
                    client_id=100002,
                    prediction=0,
                    score=0.21,
                ),
            ],
            "selected_client_ids": [100001, 100002],
        }

    monkeypatch.setattr(
        rp,
        "run_real_client_simulation",
        fake_run_real_client_simulation,
    )

    response = client.post("/predict/simulate/real-sample", params={"limit": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 2
    assert body["success_count"] == 2
    assert len(body["items"]) == 2
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_simulate_real_sample_predictions_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_real_client_simulation(**kwargs):
        raise ValueError("aucun client disponible")

    monkeypatch.setattr(
        rp,
        "run_real_client_simulation",
        fake_run_real_client_simulation,
    )

    response = client.post("/predict/simulate/real-sample", params={"limit": 2})

    assert response.status_code == 400
    assert response.json()["detail"] == "Simulation sur données réelles impossible: aucun client disponible"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_simulate_real_sample_predictions_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_real_client_simulation(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        rp,
        "run_real_client_simulation",
        fake_run_real_client_simulation,
    )

    response = client.post("/predict/simulate/real-sample", params={"limit": 2})

    assert response.status_code == 500
    assert response.json()["detail"] == "Simulation sur données réelles impossible: boom"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


# =============================================================================
# /predict/simulate/random
# =============================================================================

def test_simulate_random_predictions_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_random_feature_simulation(**kwargs):
        return {
            "batch_size": 3,
            "success_count": 3,
            "error_count": 0,
            "model_name": "credit_scoring_model",
            "model_version": "v1",
            "batch_latency_ms": 21.7,
            "items": [
                build_batch_item(
                    request_id="req_rand_1",
                    client_id=None,
                    prediction=1,
                    score=0.66,
                ),
                build_batch_item(
                    request_id="req_rand_2",
                    client_id=None,
                    prediction=0,
                    score=0.14,
                ),
                build_batch_item(
                    request_id="req_rand_3",
                    client_id=None,
                    prediction=1,
                    score=0.72,
                ),
            ],
            "selected_client_ids": None,
        }

    monkeypatch.setattr(
        rp,
        "run_random_feature_simulation",
        fake_run_random_feature_simulation,
    )

    response = client.post("/predict/simulate/random", params={"limit": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 3
    assert body["success_count"] == 3
    assert len(body["items"]) == 3
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_simulate_random_predictions_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_random_feature_simulation(**kwargs):
        raise ValueError("simulation impossible")

    monkeypatch.setattr(
        rp,
        "run_random_feature_simulation",
        fake_run_random_feature_simulation,
    )

    response = client.post("/predict/simulate/random", params={"limit": 3})

    assert response.status_code == 400
    assert response.json()["detail"] == "Simulation sur données aléatoires impossible: simulation impossible"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_simulate_random_predictions_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def fake_run_random_feature_simulation(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        rp,
        "run_random_feature_simulation",
        fake_run_random_feature_simulation,
    )

    response = client.post("/predict/simulate/random", params={"limit": 3})

    assert response.status_code == 500
    assert response.json()["detail"] == "Simulation sur données aléatoires impossible: boom"
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1