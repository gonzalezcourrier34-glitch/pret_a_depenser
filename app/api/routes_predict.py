"""
Routes FastAPI liées à la prédiction.

Ce module expose les endpoints principaux de l'API de scoring crédit.
Il gère :

- la vérification de l'état de santé de l'application
- la sécurisation des appels via une clé API
- la réception des données de prédiction
- l'appel au modèle de machine learning
- la journalisation des résultats en base PostgreSQL

Endpoints
---------
- GET /health
    Vérifie que l'API répond correctement.
- POST /predict
    Réalise une prédiction à partir des variables d'entrée, puis
    enregistre le résultat dans la base de données.
"""

from __future__ import annotations

import time
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.schemas import HealthResponse, PredictRequest, PredictResponse
from app.core.security import verify_api_key
from app.core.config import MODEL_NAME, MODEL_VERSION
from app.crud.prediction import LoggingService
from app.services.model_loader import get_model, get_threshold
from app.services.predictor import make_prediction


# Initialisation du routeur FastAPI
router = APIRouter(
    prefix="",
    tags=["Prediction"],
)


# Endpoint de santé
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier l'état de santé de l'API",
)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Vérifie que l'API, la base et le modèle sont accessibles.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy injectée par FastAPI.

    Returns
    -------
    HealthResponse
        Objet indiquant l'état de santé de l'application.
    """
    try:
        db.execute(text("SELECT 1"))
        get_model()
        get_threshold()

        return HealthResponse(status="ok")

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Healthcheck failed: {exc}",
        )


# Endpoint principal de prédiction
@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Réaliser une prédiction de risque de défaut",
)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    """
    Réalise une prédiction à partir des données d'entrée du client.

    Cet endpoint :
    1. valide les données reçues via Pydantic,
    2. convertit les features au format attendu par le modèle,
    3. appelle le service de prédiction,
    4. mesure le temps d'inférence,
    5. journalise le résultat dans PostgreSQL,
    6. retourne la prédiction au client.

    Parameters
    ----------
    payload : PredictRequest
        Corps de la requête contenant l'identifiant client optionnel
        et les variables d'entrée du modèle.
    db : Session
        Session SQLAlchemy injectée par FastAPI.
    _ : None
        Validation de la clé API via la dépendance `verify_api_key`.

    Returns
    -------
    PredictResponse
        Objet de réponse contenant :
        - l'identifiant unique de requête,
        - la classe prédite,
        - le score,
        - la version du modèle,
        - la latence d'inférence.
    """
    request_id = str(uuid.uuid4())
    features = payload.features.model_dump(by_alias=True)
    logger = LoggingService(db=db)

    start_time = time.perf_counter()

    try:
        prediction, score, threshold_used = make_prediction(features)
        latency_ms = (time.perf_counter() - start_time) * 1000

        features_df = pd.DataFrame([features])

        logger.log_full_prediction_event(
            request_id=request_id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            features_df=features_df,
            raw_input_data=features,
            prediction=prediction,
            score=score,
            threshold_used=threshold_used,
            latency_ms=latency_ms,
            client_id=payload.SK_ID_CURR,
            write_feature_store_monitoring=True,
            source_table="api_request",
            output_data={
                "request_id": request_id,
                "prediction": prediction,
                "score": score,
                "threshold_used": threshold_used,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "latency_ms": latency_ms,
            },
            status_code=200,
        )

        logger.commit()

        return PredictResponse(
            request_id=request_id,
            prediction=prediction,
            score=score,
            model_version=MODEL_VERSION,
            latency_ms=latency_ms,
        )

    except HTTPException:
        logger.rollback()
        raise

    except Exception as exc:
        logger.rollback()
        latency_ms = (time.perf_counter() - start_time) * 1000

        try:
            logger.log_prediction_error(
                request_id=request_id,
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                input_data=features,
                error_message=str(exc),
                client_id=payload.SK_ID_CURR,
                status_code=500,
                latency_ms=latency_ms,
            )
            logger.commit()

        except Exception:
            logger.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )