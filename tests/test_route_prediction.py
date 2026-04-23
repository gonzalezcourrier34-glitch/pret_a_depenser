# tests/test_route_prediction.py

from __future__ import annotations

from datetime import datetime

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

    def execute(self, query):
        self.executed.append(query)
        return 1

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


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
# /predict/health
# =============================================================================

def test_health_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rp, "get_model", lambda: object())
    monkeypatch.setattr(rp, "get_threshold", lambda: 0.5)

    response = client.get("/predict/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(fake_db.executed) == 1


def test_health_failure_returns_503(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken_model():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(rp, "get_model", broken_model)

    response = client.get("/predict/health")

    assert response.status_code == 503
    assert "Healthcheck failed" in response.json()["detail"]


# =============================================================================
# POST /predict
# =============================================================================

def test_predict_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        rp,
        "make_prediction",
        lambda **kwargs: {
            "request_id": "req_1",
            "prediction": 1,
            "score": 0.91,
            "model_version": "v1",
            "latency_ms": 12.5,
        },
    )

    response = client.post(
        "/predict",
        json={
            "SK_ID_CURR": 100001,
            "features": {
                "AMT_CREDIT": 50000.0,
                "AMT_INCOME_TOTAL": 100000.0,
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "req_1",
        "prediction": 1,
        "score": 0.91,
        "model_version": "v1",
        "latency_ms": 12.5,
    }
    assert fake_db.commits == 1
    assert fake_db.rollbacks == 0


def test_predict_returns_400_on_value_error(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken_predict(**kwargs):
        raise ValueError("invalid features")

    monkeypatch.setattr(rp, "make_prediction", broken_predict)

    response = client.post(
        "/predict",
        json={
            "SK_ID_CURR": 100001,
            "features": {
                "AMT_CREDIT": 50000.0,
                "AMT_INCOME_TOTAL": 100000.0,
            },
        },
    )

    assert response.status_code == 400
    assert "Prediction failed: invalid features" == response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_predict_returns_500_on_non_dict_result(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rp, "make_prediction", lambda **kwargs: (1, 0.8, 0.5))

    response = client.post(
        "/predict",
        json={
            "SK_ID_CURR": 100001,
            "features": {
                "AMT_CREDIT": 50000.0,
                "AMT_INCOME_TOTAL": 100000.0,
            },
        },
    )

    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]
    assert fake_db.commits == 0
    assert fake_db.rollbacks == 1


def test_predict_returns_500_on_unexpected_error(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken_predict(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "make_prediction", broken_predict)

    response = client.post(
        "/predict",
        json={
            "SK_ID_CURR": 100001,
            "features": {
                "AMT_CREDIT": 50000.0,
                "AMT_INCOME_TOTAL": 100000.0,
            },
        },
    )

    assert response.status_code == 500
    assert "Prediction failed: boom" == response.json()["detail"]
    assert fake_db.rollbacks == 1


# =============================================================================
# POST /predict/batch
# =============================================================================

def test_predict_batch_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        rp,
        "run_batch_prediction",
        lambda **kwargs: {
            "batch_size": 2,
            "success_count": 2,
            "error_count": 0,
            "model_name": "credit_model",
            "model_version": "v1",
            "batch_latency_ms": 33.0,
            "items": [
                {
                    "request_id": "req_1",
                    "client_id": 1,
                    "prediction": 1,
                    "score": 0.8,
                    "threshold_used": 0.5,
                    "model_name": "credit_model",
                    "model_version": "v1",
                    "latency_ms": 10.0,
                    "status": "success",
                },
                {
                    "request_id": "req_2",
                    "client_id": 2,
                    "prediction": 0,
                    "score": 0.2,
                    "threshold_used": 0.5,
                    "model_name": "credit_model",
                    "model_version": "v1",
                    "latency_ms": 11.0,
                    "status": "success",
                },
            ],
        },
    )

    response = client.post(
        "/predict/batch",
        json=[
            {
                "SK_ID_CURR": 1,
                "features": {
                    "AMT_CREDIT": 50000.0,
                    "AMT_INCOME_TOTAL": 100000.0,
                },
            },
            {
                "SK_ID_CURR": 2,
                "features": {
                    "AMT_CREDIT": 20000.0,
                    "AMT_INCOME_TOTAL": 50000.0,
                },
            },
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


def test_predict_batch_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken_batch(**kwargs):
        raise ValueError("bad batch")

    monkeypatch.setattr(rp, "run_batch_prediction", broken_batch)

    response = client.post(
        "/predict/batch",
        json=[
            {
                "SK_ID_CURR": 1,
                "features": {
                    "AMT_CREDIT": 50000.0,
                    "AMT_INCOME_TOTAL": 100000.0,
                },
            }
        ],
    )

    assert response.status_code == 400
    assert "Batch prediction failed: bad batch" == response.json()["detail"]
    assert fake_db.rollbacks == 1


def test_predict_batch_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken_batch(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "run_batch_prediction", broken_batch)

    response = client.post(
        "/predict/batch",
        json=[
            {
                "SK_ID_CURR": 1,
                "features": {
                    "AMT_CREDIT": 50000.0,
                    "AMT_INCOME_TOTAL": 100000.0,
                },
            }
        ],
    )

    assert response.status_code == 500
    assert "Batch prediction failed: boom" == response.json()["detail"]
    assert fake_db.rollbacks == 1


# =============================================================================
# GET /predict/{client_id}
# =============================================================================

def test_predict_from_client_id_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        rp,
        "make_prediction_from_client_id",
        lambda **kwargs: {
            "request_id": "req_client",
            "prediction": 1,
            "score": 0.88,
            "model_version": "v1",
            "latency_ms": 8.5,
        },
    )

    response = client.get("/predict/12345")

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "req_client",
        "prediction": 1,
        "score": 0.88,
        "model_version": "v1",
        "latency_ms": 8.5,
    }
    assert fake_db.commits == 1


def test_predict_from_client_id_returns_404(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def not_found(**kwargs):
        raise ValueError("Client 12345 introuvable.")

    monkeypatch.setattr(rp, "make_prediction_from_client_id", not_found)

    response = client.get("/predict/12345")

    assert response.status_code == 404
    assert response.json()["detail"] == "Client 12345 introuvable."
    assert fake_db.rollbacks == 1


def test_predict_from_client_id_returns_500_on_non_dict(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(rp, "make_prediction_from_client_id", lambda **kwargs: (1, 0.8, 0.5))

    response = client.get("/predict/12345")

    assert response.status_code == 500
    assert "Erreur interne lors de la prédiction" in response.json()["detail"]
    assert fake_db.rollbacks == 1


def test_predict_from_client_id_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "make_prediction_from_client_id", broken)

    response = client.get("/predict/12345")

    assert response.status_code == 500
    assert "Erreur interne lors de la prédiction : boom" == response.json()["detail"]
    assert fake_db.rollbacks == 1


# =============================================================================
# POST /predict/ground-truth
# =============================================================================

def test_create_prediction_ground_truth_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    observed_at = "2026-04-23T10:00:00"

    monkeypatch.setattr(
        rp,
        "create_ground_truth_label",
        lambda **kwargs: {
            "id": 1,
            "request_id": "req_1",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual",
            "observed_at": observed_at,
            "notes": "ok",
            "message": "Ground truth enregistré avec succès.",
        },
    )

    response = client.post(
        "/predict/ground-truth",
        json={
            "request_id": "req_1",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual",
            "observed_at": observed_at,
            "notes": "ok",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["request_id"] == "req_1"
    assert body["client_id"] == 100001
    assert body["true_label"] == 1
    assert fake_db.commits == 1


def test_create_prediction_ground_truth_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken(**kwargs):
        raise ValueError("true_label invalide")

    monkeypatch.setattr(rp, "create_ground_truth_label", broken)

    response = client.post(
        "/predict/ground-truth",
        json={
            "request_id": "req_1",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual",
            "observed_at": "2026-04-23T10:00:00",
            "notes": "ok",
        },
    )

    assert response.status_code == 400
    assert "Création du ground truth impossible : true_label invalide" == response.json()["detail"]
    assert fake_db.rollbacks == 1


def test_create_prediction_ground_truth_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "create_ground_truth_label", broken)

    response = client.post(
        "/predict/ground-truth",
        json={
            "request_id": "req_1",
            "client_id": 100001,
            "true_label": 1,
            "label_source": "manual",
            "observed_at": "2026-04-23T10:00:00",
            "notes": "ok",
        },
    )

    assert response.status_code == 500
    assert "Erreur interne lors de la création du ground truth : boom" == response.json()["detail"]
    assert fake_db.rollbacks == 1


# =============================================================================
# POST /predict/simulate/real-sample
# =============================================================================

def test_simulate_real_sample_predictions_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        rp,
        "run_real_client_simulation",
        lambda **kwargs: {
            "batch_size": 2,
            "success_count": 2,
            "error_count": 0,
            "model_name": "credit_model",
            "model_version": "v1",
            "batch_latency_ms": 50.0,
            "items": [],
        },
    )

    response = client.post(
        "/predict/simulate/real-sample",
        params={"limit": 2, "random_seed": 42},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 2
    assert body["success_count"] == 2
    assert fake_db.commits == 1


def test_simulate_real_sample_predictions_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken(**kwargs):
        raise ValueError("pas assez de clients")

    monkeypatch.setattr(rp, "run_real_client_simulation", broken)

    response = client.post("/predict/simulate/real-sample", params={"limit": 2})

    assert response.status_code == 400
    assert "Simulation sur données réelles impossible: pas assez de clients" == response.json()["detail"]
    assert fake_db.rollbacks == 1


def test_simulate_real_sample_predictions_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "run_real_client_simulation", broken)

    response = client.post("/predict/simulate/real-sample", params={"limit": 2})

    assert response.status_code == 500
    assert "Simulation sur données réelles impossible: boom" == response.json()["detail"]
    assert fake_db.rollbacks == 1


# =============================================================================
# POST /predict/simulate/random
# =============================================================================

def test_simulate_random_predictions_success(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    monkeypatch.setattr(
        rp,
        "run_random_feature_simulation",
        lambda **kwargs: {
            "batch_size": 3,
            "success_count": 3,
            "error_count": 0,
            "model_name": "credit_model",
            "model_version": "v1",
            "batch_latency_ms": 40.0,
            "items": [],
        },
    )

    response = client.post("/predict/simulate/random", params={"limit": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 3
    assert body["success_count"] == 3
    assert fake_db.commits == 1


def test_simulate_random_predictions_returns_400(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken(**kwargs):
        raise ValueError("génération impossible")

    monkeypatch.setattr(rp, "run_random_feature_simulation", broken)

    response = client.post("/predict/simulate/random", params={"limit": 3})

    assert response.status_code == 400
    assert "Simulation sur données aléatoires impossible: génération impossible" == response.json()["detail"]
    assert fake_db.rollbacks == 1


def test_simulate_random_predictions_returns_500(monkeypatch) -> None:
    fake_db = FakeDB()
    client = create_test_client(fake_db)

    def broken(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp, "run_random_feature_simulation", broken)

    response = client.post("/predict/simulate/random", params={"limit": 3})

    assert response.status_code == 500
    assert "Simulation sur données aléatoires impossible: boom" == response.json()["detail"]
    assert fake_db.rollbacks == 1