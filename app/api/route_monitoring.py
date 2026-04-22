"""
Routes FastAPI liées au monitoring du modèle.

Ce module expose les endpoints permettant de :
- consulter le modèle actif
- consulter le registre des modèles
- enregistrer une version de modèle
- consulter les métriques de drift
- consulter les métriques d'évaluation
- consulter le feature store de monitoring
- consulter les alertes
- reconnaître ou résoudre une alerte
- récupérer une synthèse de monitoring
- vérifier rapidement l'état du monitoring

Objectif
--------
Fournir une interface API simple pour exploiter les données de monitoring
depuis le dashboard, un outil d'administration ou des jobs externes.

Principe d'architecture
-----------------------
Les routes FastAPI restent fines :
- elles valident les paramètres HTTP,
- elles appellent le service métier,
- elles retournent la réponse.

La logique métier et l'accès aux données sont portés
par `MonitoringService`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import verify_api_key
from app.core.schemas import (
    ActiveModelResponse,
    AlertActionResponse,
    AlertListResponse,
    AlertResponse,
    GenericItemsResponse,
    ModelRegistryCreateRequest,
    ModelRegistryListResponse,
    ModelRegistryRegisterResponse,
    MonitoringHealthResponse,
    MonitoringSummaryResponse,
)
from app.services.monitoring_service import MonitoringService


# =============================================================================
# Initialisation du routeur
# =============================================================================

router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"],
)


# =============================================================================
# Helpers
# =============================================================================

def _validate_window(
    window_start: datetime | None,
    window_end: datetime | None,
) -> None:
    """
    Vérifie la cohérence des bornes temporelles.
    """
    if (window_start is None) ^ (window_end is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="window_start et window_end doivent être fournis ensemble.",
        )

    if (
        window_start is not None
        and window_end is not None
        and window_end <= window_start
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="window_end doit être strictement supérieur à window_start.",
        )


def _serialize_active_model(entity: Any) -> ActiveModelResponse:
    """
    Convertit une entité de modèle actif en schéma de réponse.
    """
    return ActiveModelResponse(
        model_name=entity.model_name,
        model_version=entity.model_version,
        stage=entity.stage,
        run_id=entity.run_id,
        source_path=entity.source_path,
        training_data_version=entity.training_data_version,
        feature_list=entity.feature_list,
        hyperparameters=entity.hyperparameters,
        metrics=entity.metrics,
        deployed_at=entity.deployed_at,
        is_active=entity.is_active,
        created_at=entity.created_at,
    )


def _serialize_alert(alert: Any) -> AlertResponse:
    """
    Convertit une entité ORM Alert en schéma de réponse.
    """
    return AlertResponse(
        id=alert.id,
        alert_type=alert.alert_type,
        severity=alert.severity,
        model_name=alert.model_name,
        model_version=alert.model_version,
        feature_name=alert.feature_name,
        title=alert.title,
        message=alert.message,
        context=alert.context,
        status=alert.status,
        created_at=alert.created_at,
        acknowledged_at=alert.acknowledged_at,
        resolved_at=alert.resolved_at,
    )


# =============================================================================
# Routes modèle
# =============================================================================

@router.get(
    "/active-model",
    response_model=ActiveModelResponse,
    summary="Retourne le modèle actif",
)
def get_active_model(
    model_name: str | None = Query(
        default=None,
        description="Nom du modèle à filtrer. Si absent, retourne le dernier actif.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> ActiveModelResponse:
    """
    Retourne la version active d'un modèle.
    """
    service = MonitoringService(db)

    try:
        entity = service.get_active_model(model_name=model_name)

        if entity is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aucun modèle actif trouvé.",
            )

        return _serialize_active_model(entity)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du modèle actif : {exc}",
        ) from exc


@router.get(
    "/models",
    response_model=ModelRegistryListResponse,
    summary="Retourne le registre des modèles",
)
def get_models(
    limit: int = Query(default=200, ge=1, le=1000),
    model_name: str | None = Query(default=None),
    is_active: bool | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> ModelRegistryListResponse:
    """
    Retourne les versions de modèles enregistrées.
    """
    service = MonitoringService(db)

    try:
        payload = service.get_models(
            limit=limit,
            model_name=model_name,
            is_active=is_active,
        )
        return ModelRegistryListResponse(**payload)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du registre des modèles : {exc}",
        ) from exc


@router.post(
    "/models/register",
    response_model=ModelRegistryRegisterResponse,
    summary="Enregistre ou met à jour une version de modèle",
)
def register_model_version(
    payload: ModelRegistryCreateRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> ModelRegistryRegisterResponse:
    """
    Enregistre ou met à jour une version de modèle dans le registry.
    """
    service = MonitoringService(db)

    try:
        result = service.register_model_version(
            model_name=payload.model_name,
            model_version=payload.model_version,
            stage=payload.stage,
            run_id=payload.run_id,
            source_path=payload.source_path,
            training_data_version=payload.training_data_version,
            feature_list=payload.feature_list,
            hyperparameters=payload.hyperparameters,
            metrics=payload.metrics,
            deployed_at=payload.deployed_at,
            is_active=payload.is_active,
        )
        db.commit()
        return ModelRegistryRegisterResponse(**result)

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'enregistrement du modèle : {exc}",
        ) from exc


# =============================================================================
# Routes drift / évaluation / feature store
# =============================================================================

@router.get(
    "/drift",
    response_model=GenericItemsResponse,
    summary="Retourne les métriques de drift",
)
def get_drift_metrics(
    limit: int = Query(default=200, ge=1, le=2000),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    feature_name: str | None = Query(default=None),
    metric_name: str | None = Query(default=None),
    drift_detected: bool | None = Query(default=None),
    window_start: datetime | None = Query(default=None),
    window_end: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> GenericItemsResponse:
    """
    Retourne les métriques de drift avec filtres optionnels.
    """
    _validate_window(window_start, window_end)
    service = MonitoringService(db)

    try:
        payload = service.get_drift_metrics(
            limit=limit,
            model_name=model_name,
            model_version=model_version,
            feature_name=feature_name,
            metric_name=metric_name,
            drift_detected=drift_detected,
            window_start=window_start,
            window_end=window_end,
        )
        return GenericItemsResponse(**payload)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des métriques de drift : {exc}",
        ) from exc


@router.get(
    "/evaluation",
    response_model=GenericItemsResponse,
    summary="Retourne les métriques d'évaluation",
)
def get_evaluation_metrics(
    limit: int = Query(default=200, ge=1, le=2000),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    dataset_name: str | None = Query(default=None),
    window_start: datetime | None = Query(default=None),
    window_end: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> GenericItemsResponse:
    """
    Retourne les métriques d'évaluation avec filtres optionnels.
    """
    _validate_window(window_start, window_end)
    service = MonitoringService(db)

    try:
        payload = service.get_evaluation_metrics(
            limit=limit,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            window_start=window_start,
            window_end=window_end,
        )
        return GenericItemsResponse(**payload)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des métriques d'évaluation : {exc}",
        ) from exc


@router.get(
    "/feature-store",
    response_model=GenericItemsResponse,
    summary="Retourne le feature store de monitoring",
)
def get_feature_store(
    limit: int = Query(default=200, ge=1, le=5000),
    request_id: str | None = Query(default=None),
    client_id: int | None = Query(default=None),
    feature_name: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    source_table: str | None = Query(default=None),
    window_start: datetime | None = Query(default=None),
    window_end: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> GenericItemsResponse:
    """
    Retourne les snapshots de features stockés pour le monitoring.
    """
    _validate_window(window_start, window_end)
    service = MonitoringService(db)

    try:
        payload = service.get_feature_store(
            limit=limit,
            request_id=request_id,
            client_id=client_id,
            feature_name=feature_name,
            model_name=model_name,
            model_version=model_version,
            source_table=source_table,
            window_start=window_start,
            window_end=window_end,
        )
        return GenericItemsResponse(**payload)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du feature store : {exc}",
        ) from exc


# =============================================================================
# Routes alertes
# =============================================================================

@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="Retourne les alertes récentes",
)
def get_recent_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    alert_status: str | None = Query(default=None, alias="status"),
    severity: str | None = Query(default=None),
    alert_type: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    feature_name: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> AlertListResponse:
    """
    Retourne les alertes de monitoring récentes avec filtres optionnels.
    """
    service = MonitoringService(db)

    try:
        alerts = service.get_recent_alerts(
            limit=limit,
            status=alert_status,
            severity=severity,
            alert_type=alert_type,
            model_name=model_name,
            model_version=model_version,
            feature_name=feature_name,
        )

        items = [_serialize_alert(alert) for alert in alerts]

        return AlertListResponse(
            count=len(items),
            items=items,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des alertes : {exc}",
        ) from exc


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertActionResponse,
    summary="Reconnaît une alerte",
)
def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> AlertActionResponse:
    """
    Marque une alerte comme reconnue.
    """
    service = MonitoringService(db)

    try:
        alert = service.acknowledge_alert(alert_id)

        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alerte introuvable.",
            )

        db.commit()
        db.refresh(alert)

        return AlertActionResponse(
            id=alert.id,
            status=alert.status,
            acknowledged_at=alert.acknowledged_at,
            resolved_at=alert.resolved_at,
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la reconnaissance de l'alerte : {exc}",
        ) from exc


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertActionResponse,
    summary="Résout une alerte",
)
def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> AlertActionResponse:
    """
    Marque une alerte comme résolue.
    """
    service = MonitoringService(db)

    try:
        alert = service.resolve_alert(alert_id)

        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alerte introuvable.",
            )

        db.commit()
        db.refresh(alert)

        return AlertActionResponse(
            id=alert.id,
            status=alert.status,
            acknowledged_at=alert.acknowledged_at,
            resolved_at=alert.resolved_at,
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la résolution de l'alerte : {exc}",
        ) from exc


# =============================================================================
# Routes synthèse
# =============================================================================

@router.get(
    "/summary",
    response_model=MonitoringSummaryResponse,
    summary="Retourne une synthèse de monitoring",
)
def get_monitoring_summary(
    model_name: str = Query(...),
    model_version: str | None = Query(default=None),
    window_start: datetime | None = Query(default=None),
    window_end: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> MonitoringSummaryResponse:
    """
    Retourne une synthèse du monitoring sur un modèle.
    """
    _validate_window(window_start, window_end)
    service = MonitoringService(db)

    try:
        payload = service.get_monitoring_summary(
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )
        return MonitoringSummaryResponse(**payload)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du calcul de la synthèse : {exc}",
        ) from exc


@router.get(
    "/health",
    response_model=MonitoringHealthResponse,
    summary="Retourne un état simple du monitoring",
)
def get_monitoring_health(
    model_name: str = Query(...),
    model_version: str | None = Query(default=None),
    window_start: datetime | None = Query(default=None),
    window_end: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> MonitoringHealthResponse:
    """
    Retourne un état simple et lisible du monitoring.
    """
    _validate_window(window_start, window_end)
    service = MonitoringService(db)

    try:
        payload = service.get_monitoring_health(
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )
        return MonitoringHealthResponse(**payload)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du calcul de l'état du monitoring : {exc}",
        ) from exc