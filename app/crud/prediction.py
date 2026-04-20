"""
CRUD pour la journalisation des prédictions.

Ce module contient uniquement des opérations de persistance.
Aucune logique métier.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.model.model_SQLalchemy import (
    FeatureStoreMonitoring,
    PredictionFeatureSnapshot,
    PredictionLog,
)


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

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    request_id : str
        Identifiant unique de requête.
    client_id : int | None
        Identifiant client éventuel.
    model_name : str
        Nom du modèle utilisé.
    model_version : str
        Version du modèle utilisé.
    prediction : int
        Classe prédite.
    score : float
        Score de probabilité de la classe positive.
    threshold_used : float | None
        Seuil de décision appliqué.
    latency_ms : float | None
        Temps d'inférence en millisecondes.
    input_data : dict[str, Any]
        Données d'entrée sérialisées.
    output_data : dict[str, Any] | None
        Données de sortie sérialisées.
    prediction_timestamp : datetime
        Horodatage de la prédiction.
    status_code : int | None
        Code logique HTTP ou technique.
    error_message : str | None
        Message d'erreur éventuel.

    Returns
    -------
    PredictionLog
        Entité SQLAlchemy ajoutée à la session.
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


def create_feature_snapshots(
    db: Session,
    records: list[dict[str, Any]],
    timestamp: datetime,
) -> None:
    """
    Crée les snapshots de features utilisés pour une prédiction.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    records : list[dict[str, Any]]
        Liste des enregistrements à insérer.
    timestamp : datetime
        Horodatage commun aux snapshots.
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


def create_feature_store_records(
    db: Session,
    records: list[dict[str, Any]],
    timestamp: datetime,
) -> None:
    """
    Crée les enregistrements dans la table de feature store monitoring.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    records : list[dict[str, Any]]
        Liste des enregistrements à insérer.
    timestamp : datetime
        Horodatage commun aux snapshots.
    """
    entities = [
        FeatureStoreMonitoring(
            request_id=r["request_id"],
            client_id=r.get("client_id"),
            model_name=r["model_name"],
            model_version=r["model_version"],
            feature_name=r["feature_name"],
            feature_value=r.get("feature_value"),
            feature_type=r.get("feature_type"),
            source_table=r.get("source_table"),
            snapshot_timestamp=timestamp,
        )
        for r in records
    ]

    if entities:
        db.add_all(entities)