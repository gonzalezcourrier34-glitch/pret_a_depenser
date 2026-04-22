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
Porter la logique métier et utiliser les couches CRUD
comme couches techniques d'accès à la base.

Notes
-----
- Les commits restent gérés par l'appelant.
- Cette couche ne doit pas faire d'accès ORM direct.
- Cette couche ne doit pas écrire directement dans les tables.
- Les lectures / écritures passent par crud.monitoring
  et, pour les logs de prédiction, par crud.prediction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.crud import monitoring as monitoring_crud
from app.crud import prediction as prediction_crud
from app.model.model_SQLalchemy import Alert, ModelRegistry


# =============================================================================
# Helpers
# =============================================================================

def _utc_now() -> datetime:
    """
    Retourne l'heure actuelle en UTC.
    """
    return datetime.now(timezone.utc)


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """
    Effectue une division sécurisée.
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


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
            entity = monitoring_crud.update_model_record(
                self.db,
                entity=entity,
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

        if is_active:
            monitoring_crud.deactivate_other_model_versions(
                self.db,
                model_name=model_name,
                keep_model_id=entity.id,
            )

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
    ) -> dict[str, Any]:
        """
        Journalise une métrique de drift via la couche CRUD.
        """
        entity = monitoring_crud.create_drift_metric_record(
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

        return {
            "id": entity.id,
            "model_name": entity.model_name,
            "model_version": entity.model_version,
            "feature_name": entity.feature_name,
            "metric_name": entity.metric_name,
            "drift_detected": entity.drift_detected,
            "computed_at": entity.computed_at,
        }

    def get_drift_metrics(
        self,
        *,
        limit: int = 200,
        model_name: str | None = None,
        model_version: str | None = None,
        feature_name: str | None = None,
        metric_name: str | None = None,
        drift_detected: bool | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retourne les métriques de drift.
        """
        rows = monitoring_crud.list_drift_metrics(
            self.db,
            limit=limit,
            model_name=model_name,
            model_version=model_version,
            feature_name=feature_name,
            metric_name=metric_name,
            drift_detected=drift_detected,
            window_start=window_start,
            window_end=window_end,
        )

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
    ) -> dict[str, Any]:
        """
        Enregistre des métriques d'évaluation via la couche CRUD.
        """
        entity = monitoring_crud.create_evaluation_metric_record(
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

        return {
            "id": entity.id,
            "model_name": entity.model_name,
            "model_version": entity.model_version,
            "dataset_name": entity.dataset_name,
            "computed_at": entity.computed_at,
        }

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
        rows = monitoring_crud.list_evaluation_metrics(
            self.db,
            limit=limit,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            window_start=window_start,
            window_end=window_end,
        )

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
        source_table: str | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Retourne le feature store de monitoring.
        """
        rows = monitoring_crud.list_feature_store_records(
            self.db,
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
        Crée une alerte de monitoring via la couche CRUD.
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
        alert_type: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        feature_name: str | None = None,
    ) -> list[Alert]:
        """
        Retourne les alertes récentes.
        """
        return monitoring_crud.list_alert_records(
            self.db,
            limit=limit,
            status=status,
            severity=severity,
            alert_type=alert_type,
            model_name=model_name,
            model_version=model_version,
            feature_name=feature_name,
        )

    def acknowledge_alert(self, alert_id: int) -> Alert | None:
        """
        Marque une alerte comme reconnue.
        """
        alert = monitoring_crud.get_alert_by_id(
            self.db,
            alert_id=alert_id,
        )

        if alert is None:
            return None

        if alert.status == "resolved":
            return alert

        return monitoring_crud.update_alert_status(
            self.db,
            alert=alert,
            status="acknowledged",
            acknowledged_at=_utc_now(),
        )

    def resolve_alert(self, alert_id: int) -> Alert | None:
        """
        Marque une alerte comme résolue.
        """
        alert = monitoring_crud.get_alert_by_id(
            self.db,
            alert_id=alert_id,
        )

        if alert is None:
            return None

        acknowledged_at = alert.acknowledged_at or _utc_now()

        return monitoring_crud.update_alert_status(
            self.db,
            alert=alert,
            status="resolved",
            acknowledged_at=acknowledged_at,
            resolved_at=_utc_now(),
        )

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
        total_predictions = prediction_crud.count_prediction_logs(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        total_errors = prediction_crud.count_prediction_errors(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        last_prediction = prediction_crud.get_latest_prediction_log(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        avg_latency_ms = prediction_crud.get_average_latency_ms(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        total_drift_metrics = monitoring_crud.count_drift_metrics(
            self.db,
            model_name=model_name,
            model_version=model_version,
            drift_detected=None,
            window_start=window_start,
            window_end=window_end,
        )

        detected_drifts = monitoring_crud.count_drift_metrics(
            self.db,
            model_name=model_name,
            model_version=model_version,
            drift_detected=True,
            window_start=window_start,
            window_end=window_end,
        )

        last_drift = monitoring_crud.get_latest_drift_metric(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        latest_evaluation = monitoring_crud.get_latest_evaluation_metric(
            self.db,
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        open_alerts = monitoring_crud.count_alert_records(
            self.db,
            status="open",
            model_name=model_name,
            model_version=model_version,
            created_after=window_start,
            created_before=window_end,
        )

        acknowledged_alerts = monitoring_crud.count_alert_records(
            self.db,
            status="acknowledged",
            model_name=model_name,
            model_version=model_version,
            created_after=window_start,
            created_before=window_end,
        )

        resolved_alerts = monitoring_crud.count_alert_records(
            self.db,
            status="resolved",
            model_name=model_name,
            model_version=model_version,
            created_after=window_start,
            created_before=window_end,
        )

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
                "error_rate": _safe_divide(total_errors, total_predictions),
                "avg_latency_ms": avg_latency_ms,
                "last_prediction_at": (
                    last_prediction.prediction_timestamp
                    if last_prediction is not None
                    else None
                ),
            },
            "drift": {
                "total_drift_metrics": total_drift_metrics,
                "detected_drifts": detected_drifts,
                "drift_rate": _safe_divide(detected_drifts, total_drift_metrics),
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
            "avg_latency_ms": predictions.get("avg_latency_ms"),
            "last_prediction_at": predictions.get("last_prediction_at"),
            "last_drift_at": drift.get("last_drift_at"),
            "latest_evaluation_at": (
                latest_evaluation.get("computed_at")
                if latest_evaluation is not None
                else None
            ),
        }