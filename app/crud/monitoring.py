"""
Service de monitoring du modèle en production.

Ce module centralise :
- l'enregistrement des versions de modèle
- l'écriture des métriques de dérive
- l'écriture des métriques de performance
- la création et gestion des alertes
- la lecture de synthèses de monitoring

Objectif
--------
Fournir une couche de service claire pour les jobs batch,
les endpoints de monitoring et le dashboard.

Notes
-----
- Ce service utilise les modèles SQLAlchemy ORM du projet.
- Les commits sont laissés à l'appelant afin de garder un contrôle
  explicite sur les transactions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.model.model import Alert, DriftMetric, EvaluationMetric, ModelRegistry
from app.model.model import PredictionLog


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


# Service principal
class MonitoringService:
    """
    Service applicatif dédié au monitoring du modèle.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

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
    ) -> ModelRegistry:
        """
        Enregistre ou met à jour une version de modèle.

        Parameters
        ----------
        model_name : str
            Nom du modèle.
        model_version : str
            Version du modèle.
        stage : str
            Stade du modèle.
        run_id : str | None, optional
            Identifiant d'exécution MLflow.
        source_path : str | None, optional
            Chemin de l'artefact.
        training_data_version : str | None, optional
            Version des données d'entraînement.
        feature_list : list[str] | None, optional
            Liste des features attendues.
        hyperparameters : dict[str, Any] | None, optional
            Hyperparamètres d'entraînement.
        metrics : dict[str, Any] | None, optional
            Métriques associées.
        deployed_at : datetime | None, optional
            Date de déploiement.
        is_active : bool, optional
            Indique si la version doit être active.

        Returns
        -------
        ModelRegistry
            Instance ORM enregistrée ou mise à jour.
        """
        existing = (
            self.db.query(ModelRegistry)
            .filter(
                ModelRegistry.model_name == model_name,
                ModelRegistry.model_version == model_version,
            )
            .first()
        )

        if is_active:
            (
                self.db.query(ModelRegistry)
                .filter(ModelRegistry.model_name == model_name)
                .update({"is_active": False}, synchronize_session=False)
            )

        if existing:
            existing.stage = stage
            existing.run_id = run_id
            existing.source_path = source_path
            existing.training_data_version = training_data_version
            existing.feature_list = feature_list
            existing.hyperparameters = hyperparameters
            existing.metrics = metrics
            existing.deployed_at = deployed_at
            existing.is_active = is_active
            return existing

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
            created_at=_utc_now(),
        )

        self.db.add(entity)
        return entity

    def get_active_model(self, model_name: str) -> ModelRegistry | None:
        """
        Retourne la version active d'un modèle.

        Parameters
        ----------
        model_name : str
            Nom du modèle.

        Returns
        -------
        ModelRegistry | None
            Version active si trouvée.
        """
        return (
            self.db.query(ModelRegistry)
            .filter(
                ModelRegistry.model_name == model_name,
                ModelRegistry.is_active.is_(True),
            )
            .order_by(ModelRegistry.deployed_at.desc(), ModelRegistry.created_at.desc())
            .first()
        )

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
        reference_window_start: datetime | None = None,
        reference_window_end: datetime | None = None,
        current_window_start: datetime | None = None,
        current_window_end: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> DriftMetric:
        """
        Enregistre une métrique de drift.

        Returns
        -------
        DriftMetric
            Instance ORM ajoutée à la session.
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
            metric_value=float(metric_value),
            threshold_value=threshold_value,
            drift_detected=drift_detected,
            details=details,
            computed_at=_utc_now(),
        )

        self.db.add(entity)
        return entity

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
        Enregistre une ligne de métriques de performance.

        Returns
        -------
        EvaluationMetric
            Instance ORM ajoutée à la session.
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
            computed_at=_utc_now(),
        )

        self.db.add(entity)
        return entity

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

        Returns
        -------
        Alert
            Instance ORM ajoutée à la session.
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
            created_at=_utc_now(),
        )

        self.db.add(entity)
        return entity

    def acknowledge_alert(self, alert_id: int) -> Alert | None:
        """
        Marque une alerte comme reconnue.

        Parameters
        ----------
        alert_id : int
            Identifiant de l'alerte.

        Returns
        -------
        Alert | None
            Alerte mise à jour ou None si absente.
        """
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()

        if alert is None:
            return None

        alert.status = "acknowledged"
        alert.acknowledged_at = _utc_now()
        return alert

    def resolve_alert(self, alert_id: int) -> Alert | None:
        """
        Marque une alerte comme résolue.

        Parameters
        ----------
        alert_id : int
            Identifiant de l'alerte.

        Returns
        -------
        Alert | None
            Alerte mise à jour ou None si absente.
        """
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()

        if alert is None:
            return None

        alert.status = "resolved"
        alert.resolved_at = _utc_now()
        return alert

    def get_recent_alerts(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        severity: str | None = None,
    ) -> list[Alert]:
        """
        Retourne les alertes récentes.

        Parameters
        ----------
        limit : int, optional
            Nombre maximal de résultats.
        status : str | None, optional
            Filtre de statut.
        severity : str | None, optional
            Filtre de sévérité.

        Returns
        -------
        list[Alert]
            Liste d'alertes.
        """
        query = self.db.query(Alert)

        if status is not None:
            query = query.filter(Alert.status == status)

        if severity is not None:
            query = query.filter(Alert.severity == severity)

        return query.order_by(Alert.created_at.desc()).limit(limit).all()

    def get_monitoring_summary(
        self,
        *,
        model_name: str,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Construit une synthèse simple de monitoring.

        Parameters
        ----------
        model_name : str
            Nom du modèle.
        model_version : str | None, optional
            Version filtrée.

        Returns
        -------
        dict[str, Any]
            Dictionnaire de synthèse.
        """
        pred_query = self.db.query(PredictionLog).filter(
            PredictionLog.model_name == model_name,
            PredictionLog.error_message.is_(None),
        )

        if model_version is not None:
            pred_query = pred_query.filter(PredictionLog.model_version == model_version)

        predictions = pred_query.all()

        total_predictions = len(predictions)
        avg_score = (
            sum(p.score for p in predictions) / total_predictions
            if total_predictions > 0 else None
        )
        avg_latency_ms = (
            sum((p.latency_ms or 0.0) for p in predictions) / total_predictions
            if total_predictions > 0 else None
        )
        last_prediction_at = (
            max((p.prediction_timestamp for p in predictions), default=None)
            if predictions else None
        )

        drift_query = self.db.query(DriftMetric).filter(
            DriftMetric.model_name == model_name
        )
        if model_version is not None:
            drift_query = drift_query.filter(DriftMetric.model_version == model_version)

        drift_metrics = drift_query.all()
        total_drift_metrics = len(drift_metrics)
        total_drift_detected = sum(1 for d in drift_metrics if d.drift_detected)
        last_drift_at = max((d.computed_at for d in drift_metrics), default=None) if drift_metrics else None

        eval_query = self.db.query(EvaluationMetric).filter(
            EvaluationMetric.model_name == model_name
        )
        if model_version is not None:
            eval_query = eval_query.filter(EvaluationMetric.model_version == model_version)

        last_eval = eval_query.order_by(EvaluationMetric.computed_at.desc()).first()

        alert_query = self.db.query(Alert).filter(
            Alert.model_name == model_name,
            Alert.status == "open",
        )
        if model_version is not None:
            alert_query = alert_query.filter(Alert.model_version == model_version)

        open_alerts = alert_query.count()

        return {
            "model_name": model_name,
            "model_version": model_version,
            "predictions": {
                "total_predictions": total_predictions,
                "avg_score": avg_score,
                "avg_latency_ms": avg_latency_ms,
                "last_prediction_at": last_prediction_at,
            },
            "drift": {
                "total_drift_metrics": total_drift_metrics,
                "total_drift_detected": total_drift_detected,
                "last_drift_at": last_drift_at,
            },
            "latest_evaluation": None if last_eval is None else {
                "dataset_name": last_eval.dataset_name,
                "roc_auc": last_eval.roc_auc,
                "pr_auc": last_eval.pr_auc,
                "precision_score": last_eval.precision_score,
                "recall_score": last_eval.recall_score,
                "f1_score": last_eval.f1_score,
                "fbeta_score": last_eval.fbeta_score,
                "business_cost": last_eval.business_cost,
                "sample_size": last_eval.sample_size,
                "computed_at": last_eval.computed_at,
            },
            "alerts": {
                "open_alerts": open_alerts,
            },
        }

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