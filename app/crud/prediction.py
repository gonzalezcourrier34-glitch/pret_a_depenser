"""
CRUD pour la journalisation des prédictions.

Ce module contient uniquement des opérations de persistance
liées aux prédictions et à leur traçabilité directe.

Objectif
--------
Isoler les écritures et lectures techniques liées à :
- prediction_logs
- prediction_features_snapshot

Notes
-----
- Aucune logique métier ne doit vivre ici.
- Le feature store de monitoring ne doit pas être géré ici.
- Les commits restent gérés par l'appelant.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.model.model_SQLalchemy import (
    PredictionFeatureSnapshot,
    PredictionLog,
)


# =============================================================================
# Helpers internes
# =============================================================================

def _build_prediction_logs_query(
    db: Session,
    *,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    only_errors: bool = False,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
):
    """
    Construit la requête ORM de base pour prediction_logs.
    """
    query = db.query(PredictionLog)

    if client_id is not None:
        query = query.filter(PredictionLog.client_id == client_id)

    if model_name is not None:
        query = query.filter(PredictionLog.model_name == model_name)

    if model_version is not None:
        query = query.filter(PredictionLog.model_version == model_version)

    if only_errors:
        query = query.filter(PredictionLog.error_message.is_not(None))

    if window_start is not None:
        query = query.filter(PredictionLog.prediction_timestamp >= window_start)

    if window_end is not None:
        query = query.filter(PredictionLog.prediction_timestamp < window_end)

    return query


# =============================================================================
# Prediction logs
# =============================================================================

def create_prediction_log(
    db: Session,
    *,
    request_id: str,
    client_id: int | None,
    model_name: str,
    model_version: str,
    prediction: int,
    score: float,
    threshold_used: float | None,
    latency_ms: float | None,
    input_data: dict[str, Any],
    output_data: dict[str, Any] | None,
    prediction_timestamp: datetime,
    status_code: int | None,
    error_message: str | None,
) -> PredictionLog:
    """
    Crée un enregistrement dans la table des logs de prédiction.
    """
    entity = PredictionLog(
        request_id=request_id,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        prediction=prediction,
        score=score,
        threshold_used=threshold_used,
        latency_ms=latency_ms,
        input_data=input_data,
        output_data=output_data,
        prediction_timestamp=prediction_timestamp,
        status_code=status_code,
        error_message=error_message,
    )
    db.add(entity)
    db.flush()
    return entity


def get_prediction_log_by_request_id(
    db: Session,
    *,
    request_id: str,
) -> PredictionLog | None:
    """
    Retourne un log de prédiction à partir du request_id.
    """
    return (
        db.query(PredictionLog)
        .filter(PredictionLog.request_id == request_id)
        .first()
    )


def list_prediction_logs(
    db: Session,
    *,
    limit: int = 100,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    only_errors: bool = False,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> list[PredictionLog]:
    """
    Retourne une liste de logs de prédiction filtrés.
    """
    query = _build_prediction_logs_query(
        db,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        only_errors=only_errors,
        window_start=window_start,
        window_end=window_end,
    )

    return (
        query.order_by(PredictionLog.prediction_timestamp.desc())
        .limit(limit)
        .all()
    )


def count_prediction_logs(
    db: Session,
    *,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> int:
    """
    Compte les logs de prédiction.
    """
    query = _build_prediction_logs_query(
        db,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        only_errors=False,
        window_start=window_start,
        window_end=window_end,
    )
    return query.count()


def count_prediction_errors(
    db: Session,
    *,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> int:
    """
    Compte les logs de prédiction en erreur.
    """
    query = _build_prediction_logs_query(
        db,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        only_errors=True,
        window_start=window_start,
        window_end=window_end,
    )
    return query.count()


def get_latest_prediction_log(
    db: Session,
    *,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> PredictionLog | None:
    """
    Retourne le log de prédiction le plus récent.
    """
    query = _build_prediction_logs_query(
        db,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        only_errors=False,
        window_start=window_start,
        window_end=window_end,
    )

    return query.order_by(PredictionLog.prediction_timestamp.desc()).first()


def get_average_latency_ms(
    db: Session,
    *,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> float | None:
    """
    Retourne la latence moyenne d'inférence en millisecondes.
    """
    query = _build_prediction_logs_query(
        db,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        only_errors=False,
        window_start=window_start,
        window_end=window_end,
    )

    value = query.with_entities(func.avg(PredictionLog.latency_ms)).scalar()

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


# =============================================================================
# Prediction feature snapshots
# =============================================================================

def create_feature_snapshots(
    db: Session,
    *,
    records: list[dict[str, Any]],
    timestamp: datetime,
) -> None:
    """
    Crée les snapshots de features utilisés pour une prédiction.
    """
    entities = [
        PredictionFeatureSnapshot(
            request_id=r["request_id"],
            client_id=r.get("client_id"),
            model_name=r["model_name"],
            model_version=r["model_version"],
            feature_name=r["feature_name"],
            feature_value=r.get("feature_value"),
            feature_type=r.get("feature_type"),
            snapshot_timestamp=timestamp,
        )
        for r in records
    ]

    if entities:
        db.add_all(entities)
        db.flush()


def list_feature_snapshots_by_request_id(
    db: Session,
    *,
    request_id: str,
) -> list[PredictionFeatureSnapshot]:
    """
    Retourne les snapshots de features pour une requête donnée.
    """
    return (
        db.query(PredictionFeatureSnapshot)
        .filter(PredictionFeatureSnapshot.request_id == request_id)
        .order_by(PredictionFeatureSnapshot.feature_name.asc())
        .all()
    )