"""
Routes FastAPI liées à l'historique des prédictions.

Ce module expose les endpoints permettant de consulter :

- l'historique des prédictions réalisées
- les vérités terrain observées
- les snapshots de features enregistrés au moment de l'inférence

Endpoints
---------
- GET /history/predictions
    Retourne l'historique des prédictions.
- GET /history/predictions/{request_id}
    Retourne le détail d'une prédiction.
- GET /history/ground-truth
    Retourne les vérités terrain historisées.
- GET /history/features/{request_id}
    Retourne le snapshot des features d'une requête.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.db import get_db
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
# Historique des prédictions
# =============================================================================

@router.get(
    "/predictions",
    summary="Consulter l'historique des prédictions",
)
def get_prediction_history(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    client_id: int | None = Query(default=None),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    only_errors: bool = Query(default=False),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> dict:
    """
    Retourne l'historique des prédictions journalisées.
    """
    return history_service.get_prediction_history(
        db,
        limit=limit,
        offset=offset,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        only_errors=only_errors,
    )


@router.get(
    "/predictions/{request_id}",
    summary="Consulter le détail d'une prédiction",
)
def get_prediction_detail(
    request_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> dict:
    """
    Retourne le détail d'une prédiction pour un request_id donné.
    """
    result = history_service.get_prediction_detail(db, request_id=request_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucune prédiction trouvée pour request_id={request_id}",
        )

    return result


# =============================================================================
# Vérités terrain
# =============================================================================

@router.get(
    "/ground-truth",
    summary="Consulter les vérités terrain",
)
def get_ground_truth_history(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    client_id: int | None = Query(default=None),
    request_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> dict:
    """
    Retourne l'historique des vérités terrain enregistrées.
    """
    return history_service.get_ground_truth_history(
        db,
        limit=limit,
        offset=offset,
        client_id=client_id,
        request_id=request_id,
    )


# =============================================================================
# Snapshot des features
# =============================================================================

@router.get(
    "/features/{request_id}",
    summary="Consulter le snapshot des features d'une requête",
)
def get_prediction_features_snapshot(
    request_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> dict:
    """
    Retourne le snapshot des features enregistré pour une requête.
    """
    result = history_service.get_prediction_features_snapshot(
        db,
        request_id=request_id,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucun snapshot de features trouvé pour request_id={request_id}",
        )

    return result