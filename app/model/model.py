"""
Modèles SQLAlchemy pour la base PostgreSQL du projet de scoring crédit.

Ce module définit les tables utilisées pour :
- tracer les prédictions de l'API
- stocker les vérités terrain
- historiser les snapshots de features
- suivre les versions de modèles
- enregistrer les métriques de drift
- enregistrer les métriques de performance
- centraliser les alertes

Tables couvertes
----------------
- PredictionLog
- GroundTruthLabel
- PredictionFeatureSnapshot
- ModelRegistry
- FeatureStoreMonitoring
- DriftMetric
- EvaluationMetric
- Alert

Notes
-----
- Ce fichier est aligné avec les scripts SQL de création de tables PostgreSQL.
- Les colonnes JSONB sont utilisées pour les données semi-structurées.
- Les timestamps utilisent `server_default=func.now()` pour laisser PostgreSQL
  gérer automatiquement l'horodatage.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    BigInteger,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.core.db import Base


# Tables de prédiction
class PredictionLog(Base):
    """
    Journal des prédictions réalisées par l'API.

    Une ligne correspond à une inférence.
    """

    __tablename__ = "prediction_logs"

    id = Column(BigInteger, primary_key=True, index=True)
    request_id = Column(Text, unique=True, nullable=True, index=True)

    client_id = Column(BigInteger, nullable=True, index=True)

    model_name = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)

    prediction = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    threshold_used = Column(Float, nullable=True)

    latency_ms = Column(Float, nullable=True)

    input_data = Column(JSONB, nullable=False)
    output_data = Column(JSONB, nullable=True)

    prediction_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    status_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_prediction_logs_model_version", "model_name", "model_version"),
    )


class GroundTruthLabel(Base):
    """
    Vérités terrain observées après la prédiction.

    Permet de comparer les prédictions aux résultats réels.
    """

    __tablename__ = "ground_truth_labels"

    id = Column(BigInteger, primary_key=True, index=True)

    request_id = Column(Text, nullable=True, index=True)
    client_id = Column(BigInteger, nullable=True, index=True)

    true_label = Column(Integer, nullable=False)
    label_source = Column(Text, nullable=True)

    observed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    notes = Column(Text, nullable=True)


class PredictionFeatureSnapshot(Base):
    """
    Snapshot des features observées au moment de l'inférence.

    Une ligne correspond à une feature pour une requête.
    """

    __tablename__ = "prediction_features_snapshot"

    id = Column(BigInteger, primary_key=True, index=True)

    request_id = Column(Text, nullable=False, index=True)
    client_id = Column(BigInteger, nullable=True, index=True)

    model_name = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)

    feature_name = Column(Text, nullable=False, index=True)
    feature_value = Column(Text, nullable=True)
    feature_type = Column(Text, nullable=True)

    snapshot_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )


# Tables de monitoring
class ModelRegistry(Base):
    """
    Registre des versions de modèles déployées ou historisées.

    Cette table centralise les métadonnées utiles au suivi du cycle de vie
    du modèle.
    """

    __tablename__ = "model_registry"

    id = Column(BigInteger, primary_key=True, index=True)

    model_name = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)

    stage = Column(Text, nullable=False)
    run_id = Column(Text, nullable=True)
    source_path = Column(Text, nullable=True)
    training_data_version = Column(Text, nullable=True)

    feature_list = Column(JSONB, nullable=True)
    hyperparameters = Column(JSONB, nullable=True)
    metrics = Column(JSONB, nullable=True)

    deployed_at = Column(DateTime(timezone=True), nullable=True)

    is_active = Column(Boolean, nullable=False, default=False, server_default="false")

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_registry_name_version"),
        Index("idx_model_registry_name_version", "model_name", "model_version"),
        Index("idx_model_registry_active", "is_active"),
    )


class FeatureStoreMonitoring(Base):
    """
    Historique des features observées pour le monitoring.

    Cette table peut être utilisée pour stocker les features réellement vues
    en production, afin d'analyser les distributions et détecter la dérive.
    """

    __tablename__ = "feature_store_monitoring"

    id = Column(BigInteger, primary_key=True, index=True)

    request_id = Column(Text, nullable=True, index=True)
    client_id = Column(BigInteger, nullable=True, index=True)

    model_name = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)

    feature_name = Column(Text, nullable=False, index=True)
    feature_value = Column(Text, nullable=True)
    feature_type = Column(Text, nullable=True)
    source_table = Column(Text, nullable=True)

    snapshot_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )


class DriftMetric(Base):
    """
    Métriques de dérive calculées sur les features.

    Une ligne correspond à une métrique de drift calculée pour une feature,
    une version de modèle et une fenêtre temporelle.
    """

    __tablename__ = "drift_metrics"

    id = Column(BigInteger, primary_key=True, index=True)

    model_name = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)

    feature_name = Column(Text, nullable=False, index=True)
    metric_name = Column(Text, nullable=False)

    reference_window_start = Column(DateTime(timezone=True), nullable=True)
    reference_window_end = Column(DateTime(timezone=True), nullable=True)

    current_window_start = Column(DateTime(timezone=True), nullable=True)
    current_window_end = Column(DateTime(timezone=True), nullable=True)

    metric_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=True)

    drift_detected = Column(Boolean, nullable=False, default=False, server_default="false")

    details = Column(JSONB, nullable=True)

    computed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("idx_drift_metrics_model_version", "model_name", "model_version"),
        Index("idx_drift_metrics_detected", "drift_detected"),
    )


class EvaluationMetric(Base):
    """
    Métriques agrégées de performance du modèle.

    Ces métriques sont calculées sur une période ou un dataset donné.
    """

    __tablename__ = "evaluation_metrics"

    id = Column(BigInteger, primary_key=True, index=True)

    model_name = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)

    dataset_name = Column(Text, nullable=False, index=True)

    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)

    roc_auc = Column(Float, nullable=True)
    pr_auc = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    fbeta_score = Column(Float, nullable=True)

    business_cost = Column(Float, nullable=True)

    tn = Column(Integer, nullable=True)
    fp = Column(Integer, nullable=True)
    fn = Column(Integer, nullable=True)
    tp = Column(Integer, nullable=True)

    sample_size = Column(Integer, nullable=True)

    computed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("idx_evaluation_metrics_model_version", "model_name", "model_version"),
    )


class Alert(Base):
    """
    Alertes générées par le système de monitoring.

    Peut contenir des alertes de drift, de performance, de qualité de données
    ou d'incidents techniques.
    """

    __tablename__ = "alerts"

    id = Column(BigInteger, primary_key=True, index=True)

    alert_type = Column(Text, nullable=False)
    severity = Column(Text, nullable=False, index=True)

    model_name = Column(Text, nullable=True)
    model_version = Column(Text, nullable=True)
    feature_name = Column(Text, nullable=True)

    title = Column(Text, nullable=False)
    message = Column(Text, nullable=False)

    context = Column(JSONB, nullable=True)

    status = Column(Text, nullable=False, default="open", server_default="open", index=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)