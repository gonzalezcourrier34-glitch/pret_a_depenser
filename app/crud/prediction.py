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

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.model.model_SQLalchemy import (
    PredictionFeatureSnapshot,
    PredictionLog,
    GroundTruthLabel
)


logger = logging.getLogger(__name__)


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
    inference_latency_ms: float | None,
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
        inference_latency_ms=inference_latency_ms,
        input_data=input_data,
        output_data=output_data,
        prediction_timestamp=prediction_timestamp,
        status_code=status_code,
        error_message=error_message,
    )
    db.add(entity)
    db.flush()

    logger.debug(
        "CRUD prediction create_prediction_log done",
        extra={
            "extra_data": {
                "event": "crud_prediction_create_log_success",
                "id": entity.id,
                "request_id": entity.request_id,
                "client_id": entity.client_id,
                "model_name": entity.model_name,
                "model_version": entity.model_version,
                "prediction": entity.prediction,
                "status_code": entity.status_code,
                "has_error": entity.error_message is not None,
            }
        },
    )

    return entity


def get_prediction_log_by_request_id(
    db: Session,
    *,
    request_id: str,
) -> PredictionLog | None:
    """
    Retourne un log de prédiction à partir du request_id.
    """
    entity = (
        db.query(PredictionLog)
        .filter(PredictionLog.request_id == request_id)
        .first()
    )

    logger.debug(
        "CRUD prediction get_prediction_log_by_request_id done",
        extra={
            "extra_data": {
                "event": "crud_prediction_get_log_by_request_id_success",
                "request_id": request_id,
                "found": entity is not None,
            }
        },
    )

    return entity


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

    rows = (
        query.order_by(PredictionLog.prediction_timestamp.desc())
        .limit(limit)
        .all()
    )

    logger.debug(
        "CRUD prediction list_prediction_logs done",
        extra={
            "extra_data": {
                "event": "crud_prediction_list_logs_success",
                "limit": limit,
                "client_id": client_id,
                "model_name": model_name,
                "model_version": model_version,
                "only_errors": only_errors,
                "count": len(rows),
            }
        },
    )

    return rows


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

    count = query.count()

    logger.debug(
        "CRUD prediction count_prediction_logs done",
        extra={
            "extra_data": {
                "event": "crud_prediction_count_logs_success",
                "client_id": client_id,
                "model_name": model_name,
                "model_version": model_version,
                "count": count,
            }
        },
    )

    return count


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

    count = query.count()

    logger.debug(
        "CRUD prediction count_prediction_errors done",
        extra={
            "extra_data": {
                "event": "crud_prediction_count_errors_success",
                "client_id": client_id,
                "model_name": model_name,
                "model_version": model_version,
                "count": count,
            }
        },
    )

    return count


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

    entity = query.order_by(PredictionLog.prediction_timestamp.desc()).first()

    logger.debug(
        "CRUD prediction get_latest_prediction_log done",
        extra={
            "extra_data": {
                "event": "crud_prediction_get_latest_log_success",
                "client_id": client_id,
                "model_name": model_name,
                "model_version": model_version,
                "found": entity is not None,
            }
        },
    )

    return entity


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
        logger.debug(
            "CRUD prediction get_average_latency_ms done with no value",
            extra={
                "extra_data": {
                    "event": "crud_prediction_get_avg_latency_empty",
                    "client_id": client_id,
                    "model_name": model_name,
                    "model_version": model_version,
                }
            },
        )
        return None

    try:
        result = float(value)
    except Exception:
        logger.debug(
            "CRUD prediction get_average_latency_ms failed to cast value",
            extra={
                "extra_data": {
                    "event": "crud_prediction_get_avg_latency_cast_error",
                    "client_id": client_id,
                    "model_name": model_name,
                    "model_version": model_version,
                }
            },
        )
        return None

    logger.debug(
        "CRUD prediction get_average_latency_ms done",
        extra={
            "extra_data": {
                "event": "crud_prediction_get_avg_latency_success",
                "client_id": client_id,
                "model_name": model_name,
                "model_version": model_version,
                "avg_latency_ms": result,
            }
        },
    )

    return result


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

    logger.debug(
        "CRUD prediction create_feature_snapshots done",
        extra={
            "extra_data": {
                "event": "crud_prediction_create_feature_snapshots_success",
                "count": len(entities),
                "timestamp": timestamp.isoformat() if timestamp else None,
            }
        },
    )


def list_feature_snapshots_by_request_id(
    db: Session,
    *,
    request_id: str,
) -> list[PredictionFeatureSnapshot]:
    """
    Retourne les snapshots de features pour une requête donnée.
    """
    rows = (
        db.query(PredictionFeatureSnapshot)
        .filter(PredictionFeatureSnapshot.request_id == request_id)
        .order_by(PredictionFeatureSnapshot.feature_name.asc())
        .all()
    )

    logger.debug(
        "CRUD prediction list_feature_snapshots_by_request_id done",
        extra={
            "extra_data": {
                "event": "crud_prediction_list_feature_snapshots_success",
                "request_id": request_id,
                "count": len(rows),
            }
        },
    )

    return rows

def create_ground_truth_label(
    db: Session,
    *,
    request_id: str | None,
    client_id: int | None,
    true_label: int,
    label_source: str | None,
    observed_at: datetime,
    notes: str | None,
) -> GroundTruthLabel:
    """
    Crée une vérité terrain dans `ground_truth_labels`.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    request_id : str | None
        Identifiant de requête éventuel.
    client_id : int | None
        Identifiant client éventuel.
    true_label : int
        Label réel observé.
    label_source : str | None
        Source métier du label.
    observed_at : datetime
        Date d'observation.
    notes : str | None
        Notes complémentaires.

    Returns
    -------
    GroundTruthLabel
        Entité SQLAlchemy ajoutée à la session.
    """
    entity = GroundTruthLabel(
        request_id=request_id,
        client_id=client_id,
        true_label=true_label,
        label_source=label_source,
        observed_at=observed_at,
        notes=notes,
    )
    db.add(entity)
    db.flush()
    db.refresh(entity)

    logger.debug(
        "CRUD prediction create_ground_truth_label done",
        extra={
            "extra_data": {
                "event": "crud_prediction_create_ground_truth_success",
                "id": entity.id,
                "request_id": entity.request_id,
                "client_id": entity.client_id,
                "true_label": entity.true_label,
                "label_source": entity.label_source,
            }
        },
    )

    return entity