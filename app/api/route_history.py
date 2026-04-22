"""
Routes FastAPI liées à l'historique des prédictions.

Ce module expose les endpoints permettant de consulter :

- l'historique des prédictions réalisées
- les vérités terrain observées
- les snapshots de features enregistrés au moment de l'inférence

Endpoints
---------
- GET /history/predictions
    Retourne l'historique des prédictions, avec possibilité
    de filtrer les décisions acceptées ou refusées.
- GET /history/predictions/{request_id}
    Retourne le détail d'une prédiction.
- GET /history/ground-truth
    Retourne les vérités terrain historisées.
- GET /history/features/{request_id}
    Retourne le snapshot des features d'une requête.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.schemas import (
    GroundTruthHistoryResponse,
    PredictionDetailResponse,
    PredictionFeaturesSnapshotResponse,
    PredictionHistoryResponse,
)
from app.core.security import verify_api_key
from app.services import history_service


# =============================================================================
# Initialisation du routeur
# =============================================================================

router = APIRouter(
    prefix="/history",
    tags=["History"],
)


# =============================================================================
# Convention métier sur la colonne prediction
# =============================================================================
# Convention actuelle :
# - 0 = crédit accepté
# - 1 = crédit refusé
ACCEPTED_PREDICTION_VALUE = 0
REFUSED_PREDICTION_VALUE = 1


# =============================================================================
# Helpers
# =============================================================================

def resolve_decision_to_prediction_value(
    decision: Literal["accepted", "refused"] | None,
) -> int | None:
    """
    Convertit une décision métier en valeur de prédiction binaire.

    Parameters
    ----------
    decision : Literal["accepted", "refused"] | None
        Valeur textuelle attendue parmi :
        - "accepted"
        - "refused"
        - None

    Returns
    -------
    int | None
        Valeur binaire correspondante dans la colonne `prediction`,
        ou None si aucun filtre n'est demandé.
    """
    if decision is None:
        return None

    if decision == "accepted":
        return ACCEPTED_PREDICTION_VALUE

    if decision == "refused":
        return REFUSED_PREDICTION_VALUE

    return None


# =============================================================================
# Historique des prédictions
# =============================================================================

@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
    summary="Consulter l'historique des prédictions",
)
def get_prediction_history(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    client_id: int | None = Query(default=None),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    only_errors: bool = Query(default=False),
    decision: Literal["accepted", "refused"] | None = Query(
        default=None,
        description="Filtre métier sur la décision de crédit : accepted ou refused.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictionHistoryResponse:
    """
    Retourne l'historique des prédictions journalisées.

    Filtres disponibles
    -------------------
    - client_id
    - model_name
    - model_version
    - only_errors
    - decision : accepted / refused

    Notes
    -----
    Le filtre `decision` est converti en valeur binaire de la colonne
    `prediction` selon la convention métier définie dans ce module.
    """
    try:
        prediction_value = resolve_decision_to_prediction_value(decision)

        payload = history_service.get_prediction_history(
            db,
            limit=limit,
            offset=offset,
            client_id=client_id,
            model_name=model_name,
            model_version=model_version,
            only_errors=only_errors,
            prediction_value=prediction_value,
        )

        return PredictionHistoryResponse.model_validate(payload)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Erreur lors de la récupération de l'historique des prédictions : "
                f"{exc}"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Erreur lors de la récupération de l'historique des prédictions : "
                f"{exc}"
            ),
        ) from exc


@router.get(
    "/predictions/{request_id}",
    response_model=PredictionDetailResponse,
    summary="Consulter le détail d'une prédiction",
)
def get_prediction_detail(
    request_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictionDetailResponse:
    """
    Retourne le détail d'une prédiction pour un request_id donné.
    """
    try:
        result = history_service.get_prediction_detail(
            db,
            request_id=request_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Aucune prédiction trouvée pour request_id={request_id}",
            )

        return PredictionDetailResponse.model_validate(result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la récupération du détail de prédiction : {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du détail de prédiction : {exc}",
        ) from exc


# =============================================================================
# Vérités terrain
# =============================================================================

@router.get(
    "/ground-truth",
    response_model=GroundTruthHistoryResponse,
    summary="Consulter les vérités terrain",
)
def get_ground_truth_history(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    client_id: int | None = Query(default=None),
    request_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> GroundTruthHistoryResponse:
    """
    Retourne l'historique des vérités terrain enregistrées.
    """
    try:
        payload = history_service.get_ground_truth_history(
            db,
            limit=limit,
            offset=offset,
            client_id=client_id,
            request_id=request_id,
        )

        return GroundTruthHistoryResponse.model_validate(payload)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la récupération des vérités terrain : {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des vérités terrain : {exc}",
        ) from exc


# =============================================================================
# Snapshot des features
# =============================================================================

@router.get(
    "/features/{request_id}",
    response_model=PredictionFeaturesSnapshotResponse,
    summary="Consulter le snapshot des features d'une requête",
)
def get_prediction_features_snapshot(
    request_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PredictionFeaturesSnapshotResponse:
    """
    Retourne le snapshot des features enregistré pour une requête.
    """
    try:
        result = history_service.get_prediction_features_snapshot(
            db,
            request_id=request_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Aucun snapshot de features trouvé pour request_id={request_id}",
            )

        return PredictionFeaturesSnapshotResponse.model_validate(result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la récupération du snapshot de features : {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du snapshot de features : {exc}",
        ) from exc