"""
Service de journalisation des prédictions et snapshots de features.

Ce module centralise l'écriture en base des événements liés à l'inférence :
- log principal de prédiction dans `prediction_logs`
- snapshot détaillé des features dans `prediction_features_snapshot`
- alimentation optionnelle de `feature_store_monitoring`

Objectif
--------
Séparer la logique de persistance de la logique métier de prédiction
afin de garder des routes FastAPI lisibles et faciles à maintenir.

Notes
-----
- Ce service utilise les modèles SQLAlchemy ORM du projet.
- Il suppose que la session SQLAlchemy est injectée depuis la couche API.
- Les commits sont laissés au choix de l'appelant pour garder le contrôle
  transactionnel.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.model.model import (
    FeatureStoreMonitoring,
    PredictionFeatureSnapshot,
    PredictionLog,
)


# Helpers
def _utc_now() -> datetime:
    """
    Retourne l'heure actuelle en UTC.

    Returns
    -------
    datetime
        Horodatage timezone-aware.
    """
    return datetime.now(timezone.utc)


def _is_missing(value: Any) -> bool:
    """
    Indique si une valeur doit être considérée comme manquante.

    Parameters
    ----------
    value : Any
        Valeur à tester.

    Returns
    -------
    bool
        True si la valeur est manquante, sinon False.
    """
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _to_python_scalar(value: Any) -> Any:
    """
    Convertit une valeur pandas / NumPy vers un type Python natif.

    Parameters
    ----------
    value : Any
        Valeur source.

    Returns
    -------
    Any
        Valeur convertie.
    """
    if _is_missing(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def _to_json_compatible(value: Any) -> Any:
    """
    Rend une structure compatible JSON / JSONB.

    Parameters
    ----------
    value : Any
        Valeur ou structure à convertir.

    Returns
    -------
    Any
        Structure compatible JSON.
    """
    value = _to_python_scalar(value)

    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def dataframe_row_to_feature_records(
    features_df: pd.DataFrame,
    *,
    request_id: str,
    model_name: str,
    model_version: str,
    client_id: int | None = None,
    source_table: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convertit une ligne de DataFrame en liste d'enregistrements feature.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame contenant exactement une ligne de features.
    request_id : str
        Identifiant unique de requête.
    model_name : str
        Nom du modèle.
    model_version : str
        Version du modèle.
    client_id : int | None, optional
        Identifiant client.
    source_table : str | None, optional
        Table source éventuelle.

    Returns
    -------
    list[dict[str, Any]]
        Liste de dictionnaires, une entrée par feature.

    Raises
    ------
    ValueError
        Si le DataFrame ne contient pas exactement une ligne.
    """
    if len(features_df) != 1:
        raise ValueError("Le DataFrame de features doit contenir exactement une ligne.")

    row = features_df.iloc[0].to_dict()
    records: list[dict[str, Any]] = []

    for feature_name, raw_value in row.items():
        value = _to_python_scalar(raw_value)

        records.append(
            {
                "request_id": request_id,
                "client_id": client_id,
                "model_name": model_name,
                "model_version": model_version,
                "feature_name": str(feature_name),
                "feature_value": None if value is None else str(value),
                "feature_type": type(value).__name__ if value is not None else None,
                "source_table": source_table,
            }
        )

    return records


# Service principal
class LoggingService:
    """
    Service applicatif de journalisation des inférences.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def log_prediction(
        self,
        *,
        request_id: str,
        model_name: str,
        model_version: str,
        prediction: int,
        score: float,
        threshold_used: float | None,
        latency_ms: float | None,
        input_data: dict[str, Any],
        output_data: dict[str, Any] | None,
        client_id: int | None = None,
        status_code: int | None = 200,
        error_message: str | None = None,
    ) -> PredictionLog:
        """
        Crée un log de prédiction.

        Parameters
        ----------
        request_id : str
            Identifiant unique de la requête.
        model_name : str
            Nom du modèle.
        model_version : str
            Version du modèle.
        prediction : int
            Classe prédite.
        score : float
            Score du modèle.
        threshold_used : float | None
            Seuil de décision utilisé.
        latency_ms : float | None
            Temps d'inférence.
        input_data : dict[str, Any]
            Payload d'entrée.
        output_data : dict[str, Any] | None
            Payload de sortie.
        client_id : int | None, optional
            Identifiant du client.
        status_code : int | None, optional
            Code logique HTTP.
        error_message : str | None, optional
            Message d'erreur éventuel.

        Returns
        -------
        PredictionLog
            Instance ORM ajoutée à la session.
        """
        entity = PredictionLog(
            request_id=request_id,
            client_id=client_id,
            model_name=model_name,
            model_version=model_version,
            prediction=int(prediction),
            score=float(score),
            threshold_used=None if threshold_used is None else float(threshold_used),
            latency_ms=None if latency_ms is None else float(latency_ms),
            input_data=_to_json_compatible(input_data),
            output_data=_to_json_compatible(output_data),
            prediction_timestamp=_utc_now(),
            status_code=status_code,
            error_message=error_message,
        )

        self.db.add(entity)
        return entity

    def log_prediction_error(
        self,
        *,
        request_id: str,
        model_name: str,
        model_version: str,
        input_data: dict[str, Any],
        error_message: str,
        client_id: int | None = None,
        status_code: int | None = 500,
        latency_ms: float | None = None,
    ) -> PredictionLog:
        """
        Crée un log d'erreur de prédiction.

        Parameters
        ----------
        request_id : str
            Identifiant unique de la requête.
        model_name : str
            Nom du modèle.
        model_version : str
            Version du modèle.
        input_data : dict[str, Any]
            Données d'entrée.
        error_message : str
            Message d'erreur.
        client_id : int | None, optional
            Identifiant client.
        status_code : int | None, optional
            Code logique HTTP.
        latency_ms : float | None, optional
            Temps écoulé avant l'erreur.

        Returns
        -------
        PredictionLog
            Instance ORM ajoutée à la session.
        """
        return self.log_prediction(
            request_id=request_id,
            model_name=model_name,
            model_version=model_version,
            prediction=-1,
            score=0.0,
            threshold_used=None,
            latency_ms=latency_ms,
            input_data=input_data,
            output_data={"status": "error", "message": error_message},
            client_id=client_id,
            status_code=status_code,
            error_message=error_message,
        )

    def log_prediction_features_snapshot(
        self,
        feature_records: list[dict[str, Any]],
    ) -> list[PredictionFeatureSnapshot]:
        """
        Enregistre un snapshot détaillé des features de prédiction.

        Parameters
        ----------
        feature_records : list[dict[str, Any]]
            Liste des features à enregistrer.

        Returns
        -------
        list[PredictionFeatureSnapshot]
            Entités ajoutées à la session.
        """
        entities: list[PredictionFeatureSnapshot] = []

        for record in feature_records:
            entity = PredictionFeatureSnapshot(
                request_id=record["request_id"],
                client_id=record.get("client_id"),
                model_name=record["model_name"],
                model_version=record["model_version"],
                feature_name=record["feature_name"],
                feature_value=record.get("feature_value"),
                feature_type=record.get("feature_type"),
                snapshot_timestamp=_utc_now(),
            )
            entities.append(entity)

        self.db.add_all(entities)
        return entities

    def log_feature_store_monitoring(
        self,
        feature_records: list[dict[str, Any]],
    ) -> list[FeatureStoreMonitoring]:
        """
        Enregistre les features dans la table de monitoring.

        Parameters
        ----------
        feature_records : list[dict[str, Any]]
            Liste des features à historiser.

        Returns
        -------
        list[FeatureStoreMonitoring]
            Entités ajoutées à la session.
        """
        entities: list[FeatureStoreMonitoring] = []

        for record in feature_records:
            entity = FeatureStoreMonitoring(
                request_id=record["request_id"],
                client_id=record.get("client_id"),
                model_name=record["model_name"],
                model_version=record["model_version"],
                feature_name=record["feature_name"],
                feature_value=record.get("feature_value"),
                feature_type=record.get("feature_type"),
                source_table=record.get("source_table"),
                snapshot_timestamp=_utc_now(),
            )
            entities.append(entity)

        self.db.add_all(entities)
        return entities

    def log_full_prediction_event(
        self,
        *,
        request_id: str,
        model_name: str,
        model_version: str,
        features_df: pd.DataFrame,
        raw_input_data: dict[str, Any],
        prediction: int,
        score: float,
        threshold_used: float | None,
        latency_ms: float | None,
        client_id: int | None = None,
        write_feature_store_monitoring: bool = True,
        source_table: str | None = None,
        output_data: dict[str, Any] | None = None,
        status_code: int | None = 200,
    ) -> None:
        """
        Orchestration complète de journalisation d'une prédiction réussie.

        Parameters
        ----------
        request_id : str
            Identifiant unique de requête.
        model_name : str
            Nom du modèle.
        model_version : str
            Version du modèle.
        features_df : pd.DataFrame
            DataFrame final utilisé pour l'inférence.
        raw_input_data : dict[str, Any]
            Données d'entrée API.
        prediction : int
            Classe prédite.
        score : float
            Score prédit.
        threshold_used : float | None
            Seuil appliqué.
        latency_ms : float | None
            Temps d'inférence.
        client_id : int | None, optional
            Identifiant client.
        write_feature_store_monitoring : bool, optional
            Si True, alimente aussi la table de monitoring des features.
        source_table : str | None, optional
            Table source éventuelle.
        output_data : dict[str, Any] | None, optional
            Payload de sortie.
        status_code : int | None, optional
            Code logique HTTP.
        """
        feature_records = dataframe_row_to_feature_records(
            features_df,
            request_id=request_id,
            model_name=model_name,
            model_version=model_version,
            client_id=client_id,
            source_table=source_table,
        )

        self.log_prediction_features_snapshot(feature_records)

        if write_feature_store_monitoring:
            self.log_feature_store_monitoring(feature_records)

        final_output = output_data or {
            "request_id": request_id,
            "prediction": int(prediction),
            "score": float(score),
            "threshold_used": threshold_used,
            "model_name": model_name,
            "model_version": model_version,
        }

        self.log_prediction(
            request_id=request_id,
            model_name=model_name,
            model_version=model_version,
            prediction=prediction,
            score=score,
            threshold_used=threshold_used,
            latency_ms=latency_ms,
            input_data=raw_input_data,
            output_data=final_output,
            client_id=client_id,
            status_code=status_code,
            error_message=None,
        )

    def commit(self) -> None:
        """
        Valide la transaction courante.
        """
        self.db.commit()

    def rollback(self) -> None:
        """
        Annule la transaction courante.
        """
        self.db.rollback()