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
- elles retournent la réponse HTTP

La logique métier ne doit pas être implémentée directement ici.

Endpoints
---------
- GET /predict/health
    Vérifie que l'API, la base et le modèle sont accessibles.

- POST /predict
    Réalise une prédiction unitaire à partir des données d'entrée.

- POST /predict/batch
    Réalise plusieurs prédictions en une seule requête.

- POST /predict/simulate/real-sample
    Lance jusqu'à 200 prédictions à partir de clients réels
    tirés aléatoirement depuis la source de données.

- POST /predict/simulate/random
    Lance jusqu'à 200 prédictions à partir de données artificielles
    générées à partir du profil des features observées.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import SIMULATION_DEFAULT_ITEMS, SIMULATION_MAX_ITEMS
from app.core.db import get_db
from app.core.schemas import HealthResponse, PredictRequest, PredictResponse
from app.core.security import verify_api_key
from app.services.model_loader_service import get_model, get_threshold
from app.services.prediction_service import (
    make_prediction,
    run_batch_prediction,
    run_random_feature_simulation,
    run_real_client_simulation,
)


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

    Cette route délègue la logique métier au service de prédiction.

    Parameters
    ----------
    payload : PredictRequest
        Requête de prédiction contenant les features d'entrée.
    db : Session
        Session SQLAlchemy active.

    Returns
    -------
    PredictResponse
        Résultat de la prédiction.
    """
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

        return PredictResponse(
            request_id=result["request_id"],
            prediction=result["prediction"],
            score=result["score"],
            model_version=result["model_version"],
            latency_ms=result["latency_ms"],
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


# =============================================================================
# Endpoint batch de prédiction
# =============================================================================

@router.post(
    "/batch",
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
) -> dict[str, Any]:
    """
    Réalise un batch de prédictions à partir d'une liste d'entrées.

    Cette route délègue la logique métier au service de prédiction batch.

    Parameters
    ----------
    payloads : list[PredictRequest]
        Liste des entrées à scorer.
    db : Session
        Session SQLAlchemy active.

    Returns
    -------
    dict[str, Any]
        Résumé global du batch et résultats détaillés.
    """
    try:
        normalized_payloads = [
            {
                "client_id": item.SK_ID_CURR,
                "features": item.features.model_dump(by_alias=True),
            }
            for item in payloads
        ]

        return run_batch_prediction(
            payloads=normalized_payloads,
            db=db,
            source_table="api_batch_request",
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc


# =============================================================================
# Endpoint simulation à partir de données réelles
# =============================================================================

@router.post(
    "/simulate/real-sample",
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
) -> dict[str, Any]:
    """
    Lance une simulation de prédictions à partir de clients réels.

    Toute la logique métier est déléguée au service de simulation.

    Parameters
    ----------
    limit : int
        Nombre de clients à traiter.
    random_seed : int | None
        Graine aléatoire optionnelle.
    db : Session
        Session SQLAlchemy active.

    Returns
    -------
    dict[str, Any]
        Résumé global de la simulation et détails.
    """
    try:
        return run_real_client_simulation(
            limit=limit,
            random_seed=random_seed,
            db=db,
            source_table="simulate_real_sample",
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation sur données réelles impossible: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation sur données réelles impossible: {exc}",
        ) from exc


# =============================================================================
# Endpoint simulation à partir de données aléatoires
# =============================================================================

@router.post(
    "/simulate/random",
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
) -> dict[str, Any]:
    """
    Lance une simulation de prédictions à partir de données artificielles.

    Toute la logique métier est déléguée au service de simulation.

    Parameters
    ----------
    limit : int
        Nombre de lignes artificielles à générer.
    db : Session
        Session SQLAlchemy active.

    Returns
    -------
    dict[str, Any]
        Résumé global de la simulation et détails.
    """
    try:
        return run_random_feature_simulation(
            limit=limit,
            db=db,
            source_table="simulate_random",
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation sur données aléatoires impossible: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation sur données aléatoires impossible: {exc}",
        ) from exc