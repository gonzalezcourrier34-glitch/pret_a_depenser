"""
Routes FastAPI liées à la prédiction.

Ce module expose les endpoints principaux de l'API de scoring crédit.
Il gère :

- la vérification de l'état de santé de l'application
- la sécurisation des appels via une clé API
- la réception des données de prédiction
- l'appel aux services métier de prédiction
- la journalisation indirecte via les services
- la simulation de prédictions unitaires ou massives pour les tests

Principe d'architecture
-----------------------
Les routes FastAPI doivent rester fines :
- elles valident et reçoivent les entrées HTTP
- elles appellent les services métier
- elles gèrent la transaction HTTP lorsque nécessaire
- elles retournent la réponse HTTP

La logique métier ne doit pas être implémentée directement ici.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import SIMULATION_DEFAULT_ITEMS, SIMULATION_MAX_ITEMS
from app.core.db import get_db
from app.core.schemas import (
    HealthResponse,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from app.core.security import verify_api_key
from app.services.model_loading_service import get_model, get_threshold
from app.services.prediction_service import (
    make_prediction,
    make_prediction_from_client_id,
    run_batch_prediction,
    run_random_feature_simulation,
    run_real_client_simulation,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Initialisation du routeur
# =============================================================================

router = APIRouter(
    prefix="/predict",
    tags=["Predict"],
)


# =============================================================================
# Endpoint de santé
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier l'état de santé de l'API",
)
def health(
    db: Session = Depends(get_db),
) -> HealthResponse:
    """
    Vérifie que l'API, PostgreSQL, le modèle et le seuil sont accessibles.
    """
    logger.info(
        "Healthcheck requested",
        extra={
            "extra_data": {
                "event": "predict_health_start",
            }
        },
    )

    try:
        db.execute(text("SELECT 1"))
        get_model()
        get_threshold()

        logger.info(
            "Healthcheck succeeded",
            extra={
                "extra_data": {
                    "event": "predict_health_success",
                    "status": "ok",
                }
            },
        )

        return HealthResponse(status="ok")

    except Exception as exc:
        logger.exception(
            "Healthcheck failed",
            extra={
                "extra_data": {
                    "event": "predict_health_exception",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Healthcheck failed: {exc}",
        ) from exc


# =============================================================================
# Endpoint principal de prédiction unitaire
# =============================================================================

@router.post(
    "",
    response_model=PredictResponse,
    summary="Réaliser une prédiction de risque de défaut",
)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    """
    Réalise une prédiction unitaire à partir des données d'entrée.
    """
    logger.info(
        "Single prediction requested",
        extra={
            "extra_data": {
                "event": "predict_single_start",
                "client_id": payload.SK_ID_CURR,
                "source_table": "api_request",
            }
        },
    )

    try:
        features = payload.features.model_dump(by_alias=True)

        result = make_prediction(
            features=features,
            client_id=payload.SK_ID_CURR,
            db=db,
            source_table="api_request",
        )

        if not isinstance(result, dict):
            raise TypeError(
                "Le service de prédiction devait retourner un dictionnaire."
            )

        db.commit()

        logger.info(
            "Single prediction succeeded",
            extra={
                "extra_data": {
                    "event": "predict_single_success",
                    "client_id": payload.SK_ID_CURR,
                    "request_id": result.get("request_id"),
                    "prediction": result.get("prediction"),
                    "score": result.get("score"),
                    "model_version": result.get("model_version"),
                    "latency_ms": result.get("latency_ms"),
                    "source_table": "api_request",
                }
            },
        )

        return PredictResponse(
            request_id=result["request_id"],
            prediction=result["prediction"],
            score=result["score"],
            model_version=result["model_version"],
            latency_ms=result["latency_ms"],
        )

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Single prediction rejected",
            extra={
                "extra_data": {
                    "event": "predict_single_value_error",
                    "client_id": payload.SK_ID_CURR,
                    "source_table": "api_request",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected error during single prediction",
            extra={
                "extra_data": {
                    "event": "predict_single_exception",
                    "client_id": payload.SK_ID_CURR,
                    "source_table": "api_request",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


# =============================================================================
# Endpoint batch de prédiction
# =============================================================================

@router.post(
    "/batch",
    response_model=PredictBatchResponse,
    summary="Réaliser un batch de prédictions",
)
def predict_batch(
    payloads: list[PredictRequest] = Body(
        ...,
        min_length=1,
        max_length=SIMULATION_MAX_ITEMS,
        description="Liste de requêtes de prédiction à traiter en batch.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictBatchResponse:
    """
    Réalise un batch de prédictions à partir d'une liste d'entrées.
    """
    logger.info(
        "Batch prediction requested",
        extra={
            "extra_data": {
                "event": "predict_batch_start",
                "batch_size": len(payloads),
                "source_table": "api_batch_request",
            }
        },
    )

    try:
        normalized_payloads = [
            {
                "client_id": item.SK_ID_CURR,
                "features": item.features.model_dump(by_alias=True),
            }
            for item in payloads
        ]

        result = run_batch_prediction(
            payloads=normalized_payloads,
            db=db,
            source_table="api_batch_request",
        )

        db.commit()

        logger.info(
            "Batch prediction succeeded",
            extra={
                "extra_data": {
                    "event": "predict_batch_success",
                    "batch_size": len(payloads),
                    "count": result.get("count"),
                    "source_table": "api_batch_request",
                }
            },
        )

        return PredictBatchResponse(**result)

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Batch prediction rejected",
            extra={
                "extra_data": {
                    "event": "predict_batch_value_error",
                    "batch_size": len(payloads),
                    "source_table": "api_batch_request",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected error during batch prediction",
            extra={
                "extra_data": {
                    "event": "predict_batch_exception",
                    "batch_size": len(payloads),
                    "source_table": "api_batch_request",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc


# =============================================================================
# Endpoint de prédiction à partir d'un identifiant client
# =============================================================================

@router.get(
    "/{client_id}",
    response_model=PredictResponse,
    summary="Réaliser une prédiction à partir d'un identifiant client",
)
def predict_from_client_id(
    client_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    """
    Réalise une prédiction unitaire à partir d'un identifiant client
    existant dans le cache de features prêtes.
    """
    logger.info(
        "Prediction from client_id requested",
        extra={
            "extra_data": {
                "event": "predict_client_id_start",
                "client_id": client_id,
                "source_table": "features_ready_cache",
            }
        },
    )

    try:
        result = make_prediction_from_client_id(
            client_id=client_id,
            db=db,
            source_table="features_ready_cache",
        )

        if not isinstance(result, dict):
            raise TypeError(
                "Le service de prédiction devait retourner un dictionnaire."
            )

        db.commit()

        logger.info(
            "Prediction from client_id succeeded",
            extra={
                "extra_data": {
                    "event": "predict_client_id_success",
                    "client_id": client_id,
                    "request_id": result.get("request_id"),
                    "prediction": result.get("prediction"),
                    "score": result.get("score"),
                    "model_version": result.get("model_version"),
                    "latency_ms": result.get("latency_ms"),
                    "source_table": "features_ready_cache",
                }
            },
        )

        return PredictResponse(
            request_id=result["request_id"],
            prediction=result["prediction"],
            score=result["score"],
            model_version=result["model_version"],
            latency_ms=result["latency_ms"],
        )

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Prediction from client_id not possible",
            extra={
                "extra_data": {
                    "event": "predict_client_id_value_error",
                    "client_id": client_id,
                    "source_table": "features_ready_cache",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected error during prediction from client_id",
            extra={
                "extra_data": {
                    "event": "predict_client_id_exception",
                    "client_id": client_id,
                    "source_table": "features_ready_cache",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne lors de la prédiction : {exc}",
        ) from exc


# =============================================================================
# Endpoint simulation à partir de données réelles
# =============================================================================

@router.post(
    "/simulate/real-sample",
    response_model=PredictBatchResponse,
    summary="Lancer jusqu'à 200 prédictions à partir de données réelles",
)
def simulate_real_sample_predictions(
    limit: int = Query(
        default=SIMULATION_DEFAULT_ITEMS,
        ge=1,
        le=SIMULATION_MAX_ITEMS,
        description="Nombre de prédictions à lancer.",
    ),
    random_seed: int | None = Query(
        default=None,
        description="Graine aléatoire optionnelle pour rendre le tirage reproductible.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictBatchResponse:
    """
    Lance une simulation de prédictions à partir de clients réels.
    """
    logger.info(
        "Real sample simulation requested",
        extra={
            "extra_data": {
                "event": "predict_simulate_real_start",
                "limit": limit,
                "random_seed": random_seed,
                "source_table": "simulate_real_sample",
            }
        },
    )

    try:
        result = run_real_client_simulation(
            limit=limit,
            random_seed=random_seed,
            db=db,
            source_table="simulate_real_sample",
        )

        db.commit()

        logger.info(
            "Real sample simulation succeeded",
            extra={
                "extra_data": {
                    "event": "predict_simulate_real_success",
                    "limit": limit,
                    "random_seed": random_seed,
                    "count": result.get("count"),
                    "source_table": "simulate_real_sample",
                }
            },
        )

        return PredictBatchResponse(**result)

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Real sample simulation rejected",
            extra={
                "extra_data": {
                    "event": "predict_simulate_real_value_error",
                    "limit": limit,
                    "random_seed": random_seed,
                    "source_table": "simulate_real_sample",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation sur données réelles impossible: {exc}",
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected error during real sample simulation",
            extra={
                "extra_data": {
                    "event": "predict_simulate_real_exception",
                    "limit": limit,
                    "random_seed": random_seed,
                    "source_table": "simulate_real_sample",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation sur données réelles impossible: {exc}",
        ) from exc


# =============================================================================
# Endpoint simulation à partir de données aléatoires
# =============================================================================

@router.post(
    "/simulate/random",
    response_model=PredictBatchResponse,
    summary="Lancer jusqu'à 200 prédictions à partir de données aléatoires",
)
def simulate_random_predictions(
    limit: int = Query(
        default=SIMULATION_DEFAULT_ITEMS,
        ge=1,
        le=SIMULATION_MAX_ITEMS,
        description="Nombre de prédictions à lancer.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictBatchResponse:
    """
    Lance une simulation de prédictions à partir de données artificielles.
    """
    logger.info(
        "Random simulation requested",
        extra={
            "extra_data": {
                "event": "predict_simulate_random_start",
                "limit": limit,
                "source_table": "simulate_random",
            }
        },
    )

    try:
        result = run_random_feature_simulation(
            limit=limit,
            db=db,
            source_table="simulate_random",
        )

        db.commit()

        logger.info(
            "Random simulation succeeded",
            extra={
                "extra_data": {
                    "event": "predict_simulate_random_success",
                    "limit": limit,
                    "count": result.get("count"),
                    "source_table": "simulate_random",
                }
            },
        )

        return PredictBatchResponse(**result)

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Random simulation rejected",
            extra={
                "extra_data": {
                    "event": "predict_simulate_random_value_error",
                    "limit": limit,
                    "source_table": "simulate_random",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation sur données aléatoires impossible: {exc}",
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected error during random simulation",
            extra={
                "extra_data": {
                    "event": "predict_simulate_random_exception",
                    "limit": limit,
                    "source_table": "simulate_random",
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation sur données aléatoires impossible: {exc}",
        ) from exc