"""
Service métier lié à l'historique des prédictions.

Ce module centralise les lectures SQL utilisées par les routes
d'historique de l'API.

Objectif
--------
Séparer la logique d'accès aux données des routes FastAPI afin de :
- rendre les routes plus lisibles
- centraliser les requêtes SQL
- faciliter les tests et la maintenance

Fonctionnalités
---------------
- lecture de l'historique des prédictions
- lecture du détail d'une prédiction
- lecture des vérités terrain
- lecture du snapshot de features d'une requête
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _prediction_label_from_value(prediction: int | None) -> str | None:
    """
    Convertit une valeur binaire de prédiction en libellé métier.
    """
    if prediction == 0:
        return "accepted"
    if prediction == 1:
        return "refused"
    return None


def _status_from_row(error_message: str | None, status_code: int | None) -> str | None:
    """
    Déduit un statut lisible à partir du code HTTP et/ou du message d'erreur.
    """
    if error_message:
        return "error"

    if status_code is None:
        return None

    if 200 <= status_code < 300:
        return "success"

    return "error"


# =============================================================================
# Historique des prédictions
# =============================================================================

def get_prediction_history(
    db: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    only_errors: bool = False,
    prediction_value: int | None = None,
) -> dict[str, Any]:
    """
    Retourne l'historique des prédictions journalisées.
    """
    logger.info(
        "History service loading prediction history",
        extra={
            "extra_data": {
                "event": "history_service_predictions_start",
                "limit": limit,
                "offset": offset,
                "client_id": client_id,
                "model_name": model_name,
                "model_version": model_version,
                "only_errors": only_errors,
                "prediction_value": prediction_value,
            }
        },
    )

    sql = """
    SELECT
        id,
        request_id,
        client_id,
        model_name,
        model_version,
        prediction,
        score,
        threshold_used,
        latency_ms,
        inference_latency_ms,
        prediction_timestamp,
        status_code,
        error_message
    FROM prediction_logs
    WHERE 1 = 1
    """

    params: dict[str, object] = {
        "limit": limit,
        "offset": offset,
    }

    if client_id is not None:
        sql += " AND client_id = :client_id"
        params["client_id"] = client_id

    if model_name is not None:
        sql += " AND model_name = :model_name"
        params["model_name"] = model_name

    if model_version is not None:
        sql += " AND model_version = :model_version"
        params["model_version"] = model_version

    if only_errors:
        sql += " AND error_message IS NOT NULL"

    if prediction_value is not None:
        sql += " AND prediction = :prediction_value"
        params["prediction_value"] = prediction_value

    sql += """
    ORDER BY prediction_timestamp DESC
    LIMIT :limit OFFSET :offset
    """

    rows = db.execute(text(sql), params).mappings().all()

    items = []
    for row in rows:
        prediction = row["prediction"]
        error_message = row["error_message"]
        status_code = row["status_code"]

        items.append(
            {
                "id": row["id"],
                "request_id": row["request_id"],
                "client_id": row["client_id"],
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "prediction": prediction,
                "prediction_label": _prediction_label_from_value(prediction),
                "score": row["score"],
                "threshold_used": row["threshold_used"],
                "latency_ms": row["latency_ms"],
                "inference_latency_ms": row.get("inference_latency_ms"),
                "prediction_timestamp": row["prediction_timestamp"],
                "status_code": status_code,
                "status": _status_from_row(error_message, status_code),
                "error_message": error_message,
            }
        )

    logger.info(
        "History service loaded prediction history successfully",
        extra={
            "extra_data": {
                "event": "history_service_predictions_success",
                "returned_items": len(items),
                "limit": limit,
                "offset": offset,
            }
        },
    )

    return {
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "items": items,
    }


def get_prediction_detail(
    db: Session,
    *,
    request_id: str,
) -> dict[str, Any] | None:
    """
    Retourne le détail complet d'une prédiction.
    """
    logger.info(
        "History service loading prediction detail",
        extra={
            "extra_data": {
                "event": "history_service_prediction_detail_start",
                "request_id": request_id,
            }
        },
    )

    sql = text(
        """
        SELECT
            id,
            request_id,
            client_id,
            model_name,
            model_version,
            prediction,
            score,
            threshold_used,
            latency_ms,
            inference_latency_ms,
            input_data,
            output_data,
            prediction_timestamp,
            status_code,
            error_message
        FROM prediction_logs
        WHERE request_id = :request_id
        LIMIT 1
        """
    )

    row = db.execute(sql, {"request_id": request_id}).mappings().first()

    if row is None:
        logger.warning(
            "History service did not find prediction detail",
            extra={
                "extra_data": {
                    "event": "history_service_prediction_detail_not_found",
                    "request_id": request_id,
                }
            },
        )
        return None

    result = {
        "id": row["id"],
        "request_id": row["request_id"],
        "client_id": row["client_id"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "prediction": row["prediction"],
        "prediction_label": _prediction_label_from_value(row["prediction"]),
        "score": row["score"],
        "threshold_used": row["threshold_used"],
        "latency_ms": row["latency_ms"],
        "inference_latency_ms": row.get("inference_latency_ms"),
        "input_data": row["input_data"],
        "output_data": row["output_data"],
        "prediction_timestamp": row["prediction_timestamp"],
        "status_code": row["status_code"],
        "status": _status_from_row(row["error_message"], row["status_code"]),
        "error_message": row["error_message"],
    }

    logger.info(
        "History service loaded prediction detail successfully",
        extra={
            "extra_data": {
                "event": "history_service_prediction_detail_success",
                "request_id": request_id,
            }
        },
    )

    return result


# =============================================================================
# Vérités terrain
# =============================================================================

def get_ground_truth_history(
    db: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    client_id: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Retourne l'historique des vérités terrain enregistrées.
    """
    logger.info(
        "History service loading ground truth history",
        extra={
            "extra_data": {
                "event": "history_service_ground_truth_start",
                "limit": limit,
                "offset": offset,
                "client_id": client_id,
                "request_id": request_id,
            }
        },
    )

    sql = """
    SELECT
        id,
        request_id,
        client_id,
        true_label,
        label_source,
        observed_at,
        notes
    FROM ground_truth_labels
    WHERE 1 = 1
    """

    params: dict[str, object] = {
        "limit": limit,
        "offset": offset,
    }

    if client_id is not None:
        sql += " AND client_id = :client_id"
        params["client_id"] = client_id

    if request_id is not None:
        sql += " AND request_id = :request_id"
        params["request_id"] = request_id

    sql += """
    ORDER BY observed_at DESC
    LIMIT :limit OFFSET :offset
    """

    rows = db.execute(text(sql), params).mappings().all()

    items = [
        {
            "id": row["id"],
            "request_id": row["request_id"],
            "client_id": row["client_id"],
            "true_label": row["true_label"],
            "label_source": row["label_source"],
            "observed_at": row["observed_at"],
            "notes": row["notes"],
        }
        for row in rows
    ]

    logger.info(
        "History service loaded ground truth history successfully",
        extra={
            "extra_data": {
                "event": "history_service_ground_truth_success",
                "returned_items": len(items),
                "limit": limit,
                "offset": offset,
            }
        },
    )

    return {
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "items": items,
    }


# =============================================================================
# Snapshot des features
# =============================================================================

def get_prediction_features_snapshot(
    db: Session,
    *,
    request_id: str,
) -> dict[str, Any] | None:
    """
    Retourne le snapshot des features enregistré pour une requête.
    """
    logger.info(
        "History service loading prediction feature snapshot",
        extra={
            "extra_data": {
                "event": "history_service_feature_snapshot_start",
                "request_id": request_id,
            }
        },
    )

    sql = text(
        """
        SELECT
            request_id,
            client_id,
            model_name,
            model_version,
            feature_name,
            feature_value,
            feature_type,
            snapshot_timestamp
        FROM prediction_features_snapshot
        WHERE request_id = :request_id
        ORDER BY feature_name
        """
    )

    rows = db.execute(sql, {"request_id": request_id}).mappings().all()

    if not rows:
        logger.warning(
            "History service did not find feature snapshot",
            extra={
                "extra_data": {
                    "event": "history_service_feature_snapshot_not_found",
                    "request_id": request_id,
                }
            },
        )
        return None

    first_row = rows[0]

    result = {
        "request_id": request_id,
        "client_id": first_row["client_id"],
        "model_name": first_row["model_name"],
        "model_version": first_row["model_version"],
        "snapshot_timestamp": first_row["snapshot_timestamp"],
        "feature_count": len(rows),
        "items": [
            {
                "feature_name": row["feature_name"],
                "feature_value": row["feature_value"],
                "feature_type": row["feature_type"],
            }
            for row in rows
        ],
    }

    logger.info(
        "History service loaded feature snapshot successfully",
        extra={
            "extra_data": {
                "event": "history_service_feature_snapshot_success",
                "request_id": request_id,
                "feature_count": len(rows),
            }
        },
    )

    return result