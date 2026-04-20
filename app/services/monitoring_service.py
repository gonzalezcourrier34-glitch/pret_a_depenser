"""
Service métier de monitoring.

Ce module orchestre les opérations de monitoring :
- lecture du registre des modèles
- enregistrement des versions de modèle
- lecture des métriques de drift
- lecture des métriques d'évaluation
- lecture du feature store
- gestion des alertes
- synthèse et healthcheck de monitoring

Objectif
--------
Porter la logique métier et utiliser la couche CRUD comme
couche technique d'accès à la base.

Notes
-----
- Les commits restent gérés par l'appelant.
- Cette couche ne doit pas contenir de SQLAlchemy query complexes
  directement si elles existent déjà dans crud.monitoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.crud import monitoring as monitoring_crud
from app.model.model_SQLalchemy import (
    Alert,
    DriftMetric,
    EvaluationMetric,
    FeatureStoreMonitoring,
    ModelRegistry,
    PredictionLog,
)


# =============================================================================
# Helpers
# =============================================================================

def _utc_now() -> datetime:
    """
    Retourne l'heure actuelle en UTC.
    """
    return datetime.now(timezone.utc)


# =============================================================================
# Service
# =============================================================================

class MonitoringService:
    """
    Service métier de monitoring.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # =========================================================================
    # Registry modèle
    # =========================================================================

    def get_active_model(
        self,
        model_name: str | None = None,
    ) -> ModelRegistry | None:
        """
        Retourne le modèle actif le plus récent.
        """
        return monitoring_crud.get_active_model_record(
            self.db,
            model_name=model_name,
        )

    def get_models(
        self,
        *,
        limit: int = 200,
        model_name: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any]:
        """
        Retourne les versions de modèles enregistrées.
        """
        rows = monitoring_crud.list_model_records(
            self.db,
            limit=limit,
            model_name=model_name,
            is_active=is_active,
        )

        return {
            "count": len(rows),
            "items": [
                {
                    "id": row.id,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "stage": row.stage,
                    "run_id": row.run_id,
                    "source_path": row.source_path,
                    "training_data_version": row.training_data_version,
                    "feature_list": row.feature_list,
                    "hyperparameters": row.hyperparameters,
                    "metrics": row.metrics,
                    "deployed_at": row.deployed_at,
                    "is_active": row.is_active,
                    "created_at": row.created_at,
                }
                for row in rows
            ],
        }

    def register_model_version(
        self,
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
    ) -> dict[str, Any]:
        """
        Enregistre ou met à jour une version de modèle.
        """
        entity = monitoring_crud.get_model_record_by_name_version(
            self.db,
            model_name=model_name,
            model_version=model_version,
        )

        if entity is None:
            entity = monitoring_crud.create_model_record(
                self.db,
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
        else:
            entity.stage = stage
            entity.run_id = run_id
            entity.source_path = source_path
            entity.training_data_version = training_data_version
            entity.feature_list = feature_list
            entity.hyperparameters = hyperparameters
            entity.metrics = metrics
            entity.deployed_at = deployed_at
            entity.is_active = is_active
            self.db.flush()

        if is_active:
            monitoring_crud.deactivate_other_model_versions(
                self.db,
                model_name=model_name,
                keep_model_id=entity.id,
            )

        self.db.flush()
        self.db.refresh(entity)

        return {
            "message": "Version de modèle enregistrée avec succès.",
            "model_name": entity.model_name,
            "model_version": entity.model_version,
            "stage": entity.stage,
            "is_active": entity.is_active,
            "deployed_at": entity.deployed_at,
        }

    # =========================================================================
    # Drift metrics
    # =========================================================================

    def log_drift_metric(
        self,
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
    ) -> DriftMetric:
        """
        Enregistre une métrique de drift.
        """
        return monitoring_crud.create_drift_metric_record(
            self.db,
            model_name=model_name,
            model_version=model_version,
            feature_name=feature_name,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=threshold_value,
            drift_detected=drift_detected,
            details=details,
            reference_window_start=reference_window_start,
            reference_window_end=reference_window_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
            computed_at=_utc_now(),
        )

    def get_drift_metrics(
        self,
        *,
        limit: int = 200,
        model_name: str | None = None,
        model_version: str | None = None,
        feature_name: str | None = None,
        drift_detected: bool | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retourne les métriques de drift.
        """
        query = monitoring_crud.build_drift_metrics_query(
            self.db,
            model_name=model_name,
            model_version=model_version,
            feature_name=feature_name,
            drift_detected=drift_detected,
            window_start=window_start,
            window_end=window_end,
        )

        rows = query.order_by(DriftMetric.computed_at.desc()).limit(limit).all()

        return {
            "count": len(rows),
            "items": [
                {
                    "id": row.id,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "feature_name": row.feature_name,
                    "metric_name": row.metric_name,
                    "reference_window_start": row.reference_window_start,
                    "reference_window_end": row.reference_window_end,
                    "current_window_start": row.current_window_start,
                    "current_window_end": row.current_window_end,
                    "metric_value": row.metric_value,
                    "threshold_value": row.threshold_value,
                    "drift_detected": row.drift_detected,
                    "details": row.details,
                    "computed_at": row.computed_at,
                }
                for row in rows
            ],
        }

    # =========================================================================
    # Evaluation metrics
    # =========================================================================

    def log_evaluation_metrics(
        self,
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
    ) -> EvaluationMetric:
        """
        Enregistre des métriques d'évaluation.
        """
        return monitoring_crud.create_evaluation_metric_record(
            self.db,
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
            computed_at=_utc_now(),
        )

    def get_evaluation_metrics(
        self,
        *,
        limit: int = 200,
        model_name: str | None = None,
        model_version: str | None = None,
        dataset_name: str | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retourne les métriques d'évaluation.
        """
        query = monitoring_crud.build_evaluation_metrics_query(
            self.db,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            window_start=window_start,
            window_end=window_end,
        )

        rows = query.order_by(EvaluationMetric.computed_at.desc()).limit(limit).all()

        return {
            "count": len(rows),
            "items": [
                {
                    "id": row.id,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "dataset_name": row.dataset_name,
                    "window_start": row.window_start,
                    "window_end": row.window_end,
                    "roc_auc": row.roc_auc,
                    "pr_auc": row.pr_auc,
                    "precision_score": row.precision_score,
                    "recall_score": row.recall_score,
                    "f1_score": row.f1_score,
                    "fbeta_score": row.fbeta_score,
                    "business_cost": row.business_cost,
                    "tn": row.tn,
                    "fp": row.fp,
                    "fn": row.fn,
                    "tp": row.tp,
                    "sample_size": row.sample_size,
                    "computed_at": row.computed_at,
                }
                for row in rows
            ],
        }

    # =========================================================================
    # Feature store
    # =========================================================================

    def get_feature_store(
        self,
        *,
        limit: int = 200,
        request_id: str | None = None,
        client_id: int | None = None,
        feature_name: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retourne le feature store de monitoring.
        """
        query = monitoring_crud.build_feature_store_query(
            self.db,
            request_id=request_id,
            client_id=client_id,
            feature_name=feature_name,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        rows = query.order_by(FeatureStoreMonitoring.snapshot_timestamp.desc()).limit(limit).all()

        return {
            "count": len(rows),
            "items": [
                {
                    "id": row.id,
                    "request_id": row.request_id,
                    "client_id": row.client_id,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "feature_name": row.feature_name,
                    "feature_value": row.feature_value,
                    "feature_type": row.feature_type,
                    "source_table": row.source_table,
                    "snapshot_timestamp": row.snapshot_timestamp,
                }
                for row in rows
            ],
        }

    # =========================================================================
    # Alertes
    # =========================================================================

    def create_alert(
        self,
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
    ) -> Alert:
        """
        Crée une alerte de monitoring.
        """
        return monitoring_crud.create_alert_record(
            self.db,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            model_name=model_name,
            model_version=model_version,
            feature_name=feature_name,
            context=context,
            status=status,
            created_at=_utc_now(),
        )

    def get_recent_alerts(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        severity: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[Alert]:
        """
        Retourne les alertes récentes.
        """
        query = monitoring_crud.build_alerts_query(
            self.db,
            status=status,
            severity=severity,
            model_name=model_name,
            model_version=model_version,
        )

        return query.order_by(Alert.created_at.desc()).limit(limit).all()

    def acknowledge_alert(self, alert_id: int) -> Alert | None:
        """
        Marque une alerte comme reconnue.
        """
        alert = monitoring_crud.get_alert_by_id(self.db, alert_id=alert_id)

        if alert is None:
            return None

        if alert.status == "resolved":
            return alert

        alert.status = "acknowledged"
        alert.acknowledged_at = _utc_now()
        self.db.flush()

        return alert

    def resolve_alert(self, alert_id: int) -> Alert | None:
        """
        Marque une alerte comme résolue.
        """
        alert = monitoring_crud.get_alert_by_id(self.db, alert_id=alert_id)

        if alert is None:
            return None

        if alert.acknowledged_at is None:
            alert.acknowledged_at = _utc_now()

        alert.status = "resolved"
        alert.resolved_at = _utc_now()
        self.db.flush()

        return alert

    # =========================================================================
    # Synthèse monitoring
    # =========================================================================

    def get_monitoring_summary(
        self,
        *,
        model_name: str,
        model_version: str | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retourne une synthèse complète du monitoring.
        """
        prediction_query = monitoring_crud.build_prediction_logs_query(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        drift_query = monitoring_crud.build_drift_metrics_query(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        evaluation_query = monitoring_crud.build_evaluation_metrics_query(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        alert_query = monitoring_crud.build_alerts_query(
            self.db,
            model_name=model_name,
            model_version=model_version,
        )

        if window_start is not None:
            alert_query = alert_query.filter(Alert.created_at >= window_start)

        if window_end is not None:
            alert_query = alert_query.filter(Alert.created_at < window_end)

        total_predictions = prediction_query.count()
        total_errors = prediction_query.filter(PredictionLog.error_message.is_not(None)).count()

        last_prediction = prediction_query.order_by(
            PredictionLog.prediction_timestamp.desc()
        ).first()

        last_drift = drift_query.order_by(DriftMetric.computed_at.desc()).first()

        latest_evaluation = evaluation_query.order_by(
            EvaluationMetric.computed_at.desc()
        ).first()

        total_drift_metrics = drift_query.count()
        detected_drifts = drift_query.filter(DriftMetric.drift_detected.is_(True)).count()

        open_alerts = alert_query.filter(Alert.status == "open").count()
        acknowledged_alerts = alert_query.filter(Alert.status == "acknowledged").count()
        resolved_alerts = alert_query.filter(Alert.status == "resolved").count()

        latest_evaluation_payload = None
        if latest_evaluation is not None:
            latest_evaluation_payload = {
                "id": latest_evaluation.id,
                "dataset_name": latest_evaluation.dataset_name,
                "window_start": latest_evaluation.window_start,
                "window_end": latest_evaluation.window_end,
                "roc_auc": latest_evaluation.roc_auc,
                "pr_auc": latest_evaluation.pr_auc,
                "precision_score": latest_evaluation.precision_score,
                "recall_score": latest_evaluation.recall_score,
                "f1_score": latest_evaluation.f1_score,
                "fbeta_score": latest_evaluation.fbeta_score,
                "business_cost": latest_evaluation.business_cost,
                "tn": latest_evaluation.tn,
                "fp": latest_evaluation.fp,
                "fn": latest_evaluation.fn,
                "tp": latest_evaluation.tp,
                "sample_size": latest_evaluation.sample_size,
                "computed_at": latest_evaluation.computed_at,
            }

        return {
            "model_name": model_name,
            "model_version": model_version,
            "window_start": window_start,
            "window_end": window_end,
            "predictions": {
                "total_predictions": total_predictions,
                "total_errors": total_errors,
                "error_rate": (total_errors / total_predictions) if total_predictions > 0 else 0.0,
                "last_prediction_at": (
                    last_prediction.prediction_timestamp if last_prediction is not None else None
                ),
            },
            "drift": {
                "total_drift_metrics": total_drift_metrics,
                "detected_drifts": detected_drifts,
                "drift_rate": (
                    detected_drifts / total_drift_metrics if total_drift_metrics > 0 else 0.0
                ),
                "last_drift_at": (
                    last_drift.computed_at if last_drift is not None else None
                ),
            },
            "latest_evaluation": latest_evaluation_payload,
            "alerts": {
                "open_alerts": open_alerts,
                "acknowledged_alerts": acknowledged_alerts,
                "resolved_alerts": resolved_alerts,
            },
        }

    def get_monitoring_health(
        self,
        *,
        model_name: str,
        model_version: str | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retourne un état simple et lisible du monitoring.
        """
        summary = self.get_monitoring_summary(
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        latest_evaluation = summary.get("latest_evaluation")
        predictions = summary.get("predictions", {})
        drift = summary.get("drift", {})
        alerts = summary.get("alerts", {})

        return {
            "model_name": summary.get("model_name"),
            "model_version": summary.get("model_version"),
            "window_start": summary.get("window_start"),
            "window_end": summary.get("window_end"),
            "has_predictions": predictions.get("total_predictions", 0) > 0,
            "has_drift_metrics": drift.get("total_drift_metrics", 0) > 0,
            "has_latest_evaluation": latest_evaluation is not None,
            "open_alerts": alerts.get("open_alerts", 0),
            "last_prediction_at": predictions.get("last_prediction_at"),
            "last_drift_at": drift.get("last_drift_at"),
            "latest_evaluation_at": (
                latest_evaluation.get("computed_at")
                if latest_evaluation is not None
                else None
            ),
        }