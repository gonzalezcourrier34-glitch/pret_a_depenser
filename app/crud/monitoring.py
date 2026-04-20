"""
Couche CRUD pour le monitoring.

Ce module contient uniquement les opérations d'accès aux données :
- lecture des tables de monitoring
- insertion des métriques
- insertion et mise à jour des alertes
- lecture / mise à jour du registre des modèles

Objectif
--------
Isoler les accès ORM/SQLAlchemy de la logique métier.

Notes
-----
- Aucune logique métier complexe ne doit vivre ici.
- Les commits sont laissés à l'appelant.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Query, Session

from app.model.model_SQLalchemy import (
    Alert,
    DriftMetric,
    EvaluationMetric,
    FeatureStoreMonitoring,
    ModelRegistry,
    PredictionLog,
)


# =============================================================================
# Model registry
# =============================================================================

def get_active_model_record(
    db: Session,
    *,
    model_name: str | None = None,
) -> ModelRegistry | None:
    """
    Retourne le modèle actif le plus récent.
    """
    query = db.query(ModelRegistry).filter(ModelRegistry.is_active.is_(True))

    if model_name is not None:
        query = query.filter(ModelRegistry.model_name == model_name)

    return (
        query.order_by(
            ModelRegistry.deployed_at.desc(),
            ModelRegistry.created_at.desc(),
        )
        .first()
    )


def get_model_record_by_name_version(
    db: Session,
    *,
    model_name: str,
    model_version: str,
) -> ModelRegistry | None:
    """
    Retourne une version précise du modèle si elle existe.
    """
    return (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.model_version == model_version,
        )
        .first()
    )


def list_model_records(
    db: Session,
    *,
    limit: int = 200,
    model_name: str | None = None,
    is_active: bool | None = None,
) -> list[ModelRegistry]:
    """
    Retourne les versions de modèles enregistrées.
    """
    query = db.query(ModelRegistry)

    if model_name is not None:
        query = query.filter(ModelRegistry.model_name == model_name)

    if is_active is not None:
        query = query.filter(ModelRegistry.is_active.is_(is_active))

    return (
        query.order_by(
            ModelRegistry.created_at.desc(),
            ModelRegistry.deployed_at.desc(),
        )
        .limit(limit)
        .all()
    )


def create_model_record(
    db: Session,
    *,
    model_name: str,
    model_version: str,
    stage: str,
    run_id: str | None = None,
    source_path: str | None = None,
    training_data_version: str | None = None,
    feature_list: list[str] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    deployed_at: datetime | None = None,
    is_active: bool = False,
) -> ModelRegistry:
    """
    Crée une nouvelle ligne dans le registre des modèles.
    """
    entity = ModelRegistry(
        model_name=model_name,
        model_version=model_version,
        stage=stage,
        run_id=run_id,
        source_path=source_path,
        training_data_version=training_data_version,
        feature_list=feature_list,
        hyperparameters=hyperparameters,
        metrics=metrics,
        deployed_at=deployed_at,
        is_active=is_active,
    )
    db.add(entity)
    db.flush()
    return entity


def deactivate_other_model_versions(
    db: Session,
    *,
    model_name: str,
    keep_model_id: int,
) -> None:
    """
    Désactive les autres versions actives du même modèle.
    """
    (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.id != keep_model_id,
            ModelRegistry.is_active.is_(True),
        )
        .update({"is_active": False}, synchronize_session=False)
    )


# =============================================================================
# Drift metrics
# =============================================================================

def create_drift_metric_record(
    db: Session,
    *,
    model_name: str,
    model_version: str,
    feature_name: str,
    metric_name: str,
    metric_value: float,
    threshold_value: float | None = None,
    drift_detected: bool = False,
    details: dict[str, Any] | None = None,
    reference_window_start: datetime | None = None,
    reference_window_end: datetime | None = None,
    current_window_start: datetime | None = None,
    current_window_end: datetime | None = None,
    computed_at: datetime | None = None,
) -> DriftMetric:
    """
    Crée une métrique de drift.
    """
    entity = DriftMetric(
        model_name=model_name,
        model_version=model_version,
        feature_name=feature_name,
        metric_name=metric_name,
        reference_window_start=reference_window_start,
        reference_window_end=reference_window_end,
        current_window_start=current_window_start,
        current_window_end=current_window_end,
        metric_value=metric_value,
        threshold_value=threshold_value,
        drift_detected=drift_detected,
        details=details,
        computed_at=computed_at,
    )
    db.add(entity)
    db.flush()
    return entity


def build_drift_metrics_query(
    db: Session,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    feature_name: str | None = None,
    drift_detected: bool | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> Query:
    """
    Construit la requête ORM pour les métriques de drift.
    """
    query = db.query(DriftMetric)

    if model_name is not None:
        query = query.filter(DriftMetric.model_name == model_name)

    if model_version is not None:
        query = query.filter(DriftMetric.model_version == model_version)

    if feature_name is not None:
        query = query.filter(DriftMetric.feature_name == feature_name)

    if drift_detected is not None:
        query = query.filter(DriftMetric.drift_detected.is_(drift_detected))

    if window_start is not None:
        query = query.filter(DriftMetric.computed_at >= window_start)

    if window_end is not None:
        query = query.filter(DriftMetric.computed_at < window_end)

    return query


# =============================================================================
# Evaluation metrics
# =============================================================================

def create_evaluation_metric_record(
    db: Session,
    *,
    model_name: str,
    model_version: str,
    dataset_name: str,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    roc_auc: float | None = None,
    pr_auc: float | None = None,
    precision_score: float | None = None,
    recall_score: float | None = None,
    f1_score: float | None = None,
    fbeta_score: float | None = None,
    business_cost: float | None = None,
    tn: int | None = None,
    fp: int | None = None,
    fn: int | None = None,
    tp: int | None = None,
    sample_size: int | None = None,
    computed_at: datetime | None = None,
) -> EvaluationMetric:
    """
    Crée une métrique d'évaluation.
    """
    entity = EvaluationMetric(
        model_name=model_name,
        model_version=model_version,
        dataset_name=dataset_name,
        window_start=window_start,
        window_end=window_end,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision_score=precision_score,
        recall_score=recall_score,
        f1_score=f1_score,
        fbeta_score=fbeta_score,
        business_cost=business_cost,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        sample_size=sample_size,
        computed_at=computed_at,
    )
    db.add(entity)
    db.flush()
    return entity


def build_evaluation_metrics_query(
    db: Session,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    dataset_name: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> Query:
    """
    Construit la requête ORM pour les métriques d'évaluation.
    """
    query = db.query(EvaluationMetric)

    if model_name is not None:
        query = query.filter(EvaluationMetric.model_name == model_name)

    if model_version is not None:
        query = query.filter(EvaluationMetric.model_version == model_version)

    if dataset_name is not None:
        query = query.filter(EvaluationMetric.dataset_name == dataset_name)

    if window_start is not None:
        query = query.filter(EvaluationMetric.computed_at >= window_start)

    if window_end is not None:
        query = query.filter(EvaluationMetric.computed_at < window_end)

    return query


# =============================================================================
# Feature store monitoring
# =============================================================================

def create_feature_store_record(
    db: Session,
    *,
    request_id: str | None = None,
    client_id: int | None = None,
    model_name: str,
    model_version: str,
    feature_name: str,
    feature_value: str | None = None,
    feature_type: str | None = None,
    source_table: str | None = None,
    snapshot_timestamp: datetime | None = None,
) -> FeatureStoreMonitoring:
    """
    Crée un enregistrement dans le feature store de monitoring.
    """
    entity = FeatureStoreMonitoring(
        request_id=request_id,
        client_id=client_id,
        model_name=model_name,
        model_version=model_version,
        feature_name=feature_name,
        feature_value=feature_value,
        feature_type=feature_type,
        source_table=source_table,
        snapshot_timestamp=snapshot_timestamp,
    )
    db.add(entity)
    db.flush()
    return entity


def build_feature_store_query(
    db: Session,
    *,
    request_id: str | None = None,
    client_id: int | None = None,
    feature_name: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> Query:
    """
    Construit la requête ORM pour le feature store de monitoring.
    """
    query = db.query(FeatureStoreMonitoring)

    if request_id is not None:
        query = query.filter(FeatureStoreMonitoring.request_id == request_id)

    if client_id is not None:
        query = query.filter(FeatureStoreMonitoring.client_id == client_id)

    if feature_name is not None:
        query = query.filter(FeatureStoreMonitoring.feature_name == feature_name)

    if model_name is not None:
        query = query.filter(FeatureStoreMonitoring.model_name == model_name)

    if model_version is not None:
        query = query.filter(FeatureStoreMonitoring.model_version == model_version)

    if window_start is not None:
        query = query.filter(FeatureStoreMonitoring.snapshot_timestamp >= window_start)

    if window_end is not None:
        query = query.filter(FeatureStoreMonitoring.snapshot_timestamp < window_end)

    return query


# =============================================================================
# Alertes
# =============================================================================

def create_alert_record(
    db: Session,
    *,
    alert_type: str,
    severity: str,
    title: str,
    message: str,
    model_name: str | None = None,
    model_version: str | None = None,
    feature_name: str | None = None,
    context: dict[str, Any] | None = None,
    status: str = "open",
    created_at: datetime | None = None,
) -> Alert:
    """
    Crée une alerte.
    """
    entity = Alert(
        alert_type=alert_type,
        severity=severity,
        model_name=model_name,
        model_version=model_version,
        feature_name=feature_name,
        title=title,
        message=message,
        context=context,
        status=status,
        created_at=created_at,
    )
    db.add(entity)
    db.flush()
    return entity


def get_alert_by_id(
    db: Session,
    *,
    alert_id: int,
) -> Alert | None:
    """
    Retourne une alerte par identifiant.
    """
    return db.query(Alert).filter(Alert.id == alert_id).first()


def build_alerts_query(
    db: Session,
    *,
    status: str | None = None,
    severity: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
) -> Query:
    """
    Construit la requête ORM pour les alertes.
    """
    query = db.query(Alert)

    if status is not None:
        query = query.filter(Alert.status == status)

    if severity is not None:
        query = query.filter(Alert.severity == severity)

    if model_name is not None:
        query = query.filter(Alert.model_name == model_name)

    if model_version is not None:
        query = query.filter(Alert.model_version == model_version)

    return query


# =============================================================================
# Prediction logs
# =============================================================================

def build_prediction_logs_query(
    db: Session,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> Query:
    """
    Construit la requête ORM pour prediction_logs.
    """
    query = db.query(PredictionLog)

    if model_name is not None:
        query = query.filter(PredictionLog.model_name == model_name)

    if model_version is not None:
        query = query.filter(PredictionLog.model_version == model_version)

    if window_start is not None:
        query = query.filter(PredictionLog.prediction_timestamp >= window_start)

    if window_end is not None:
        query = query.filter(PredictionLog.prediction_timestamp < window_end)

    return query