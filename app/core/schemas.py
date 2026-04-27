"""
Schémas Pydantic centralisés de l'application.

Ce module regroupe les schémas réutilisables exposés par l'API :
- prédiction
- historique
- monitoring
- Evidently

Objectif
--------
Centraliser les schémas HTTP pour éviter les doublons entre routes
et garder une seule source de vérité pour la validation / sérialisation.

Notes
-----
- Ces schémas sont pensés pour être réutilisés dans plusieurs routes.
- Ils reflètent les structures effectivement renvoyées
  par les services métier et les routes FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Base commune
# =============================================================================

class StrictSchema(BaseModel):
    """
    Base commune stricte pour tous les schémas de l'API.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        from_attributes=True,
    )


# =============================================================================
# Helpers de réponse génériques
# =============================================================================

class CountItemsResponse(StrictSchema):
    """
    Réponse générique minimale de type :
    {
        "count": int,
        "items": [...]
    }
    """

    count: int


class GenericItemsResponse(CountItemsResponse):
    """
    Réponse générique pour les endpoints qui renvoient :
    - count
    - items (liste de dictionnaires souples)

    Notes
    -----
    Utilisé pour les routes génériques de monitoring.
    """

    items: list[dict[str, Any]]


class PaginatedItemsResponse(CountItemsResponse):
    """
    Base générique paginée.
    """

    limit: int
    offset: int


# =============================================================================
# Predict
# =============================================================================

class HealthResponse(StrictSchema):
    """
    Réponse du healthcheck.
    """

    status: str


class PredictRequest(StrictSchema):
    """
    Payload de prédiction unitaire.
    """

    SK_ID_CURR: int | None = Field(default=None, description="Identifiant client")
    features: dict[str, Any] = Field(..., description="Features prêtes pour le modèle")


class PredictResponse(StrictSchema):
    """
    Réponse de prédiction unitaire.
    """

    request_id: str
    prediction: int
    score: float
    model_version: str
    latency_ms: float
    inference_latency_ms: float | None = None

class PredictBatchItemResponse(StrictSchema):
    """
    Item unitaire d'un batch de prédictions.
    """

    request_id: str
    client_id: int | None = None
    prediction: int | None = None
    score: float | None = None
    threshold_used: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    latency_ms: float | None = None
    inference_latency_ms: float | None = None
    status: str
    error_message: str | None = None


class PredictBatchResponse(StrictSchema):
    """
    Réponse batch de prédictions.
    """

    batch_size: int
    success_count: int
    error_count: int
    model_name: str
    model_version: str
    batch_latency_ms: float
    batch_inference_latency_ms: float | None = None
    items: list[PredictBatchItemResponse]
    selected_client_ids: list[int] | None = None


# =============================================================================
# Registry modèle
# =============================================================================

class ModelRegistryCreateRequest(StrictSchema):
    """
    Payload d'enregistrement ou de mise à jour d'une version de modèle.
    """

    model_name: str = Field(..., description="Nom du modèle")
    model_version: str = Field(..., description="Version du modèle")
    stage: Literal["dev", "staging", "production", "archived"] = Field(
        ...,
        description="Stade du modèle",
    )
    run_id: str | None = Field(
        default=None,
        description="Identifiant du run MLflow",
    )
    source_path: str | None = Field(
        default=None,
        description="Chemin de l'artefact ou du modèle",
    )
    training_data_version: str | None = Field(
        default=None,
        description="Version des données d'entraînement",
    )
    feature_list: list[str] | None = Field(
        default=None,
        description="Liste des features attendues par le modèle",
    )
    hyperparameters: dict[str, Any] | None = Field(
        default=None,
        description="Hyperparamètres du modèle",
    )
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Métriques associées au modèle",
    )
    deployed_at: datetime | None = Field(
        default=None,
        description="Date de déploiement",
    )
    is_active: bool = Field(
        default=False,
        description="Indique si cette version devient active",
    )


class ModelRegistryBaseResponse(StrictSchema):
    """
    Base commune de réponse pour les objets du registre des modèles.
    """

    model_name: str
    model_version: str
    stage: str
    run_id: str | None = None
    source_path: str | None = None
    training_data_version: str | None = None
    feature_list: list[str] | None = None
    hyperparameters: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    deployed_at: datetime | None = None
    is_active: bool


class ActiveModelResponse(ModelRegistryBaseResponse):
    """
    Réponse décrivant le modèle actif.
    """

    created_at: datetime


class ModelRegistryItemResponse(ModelRegistryBaseResponse):
    """
    Représentation d'une ligne du registre des modèles.
    """

    id: int
    created_at: datetime


class ModelRegistryListResponse(CountItemsResponse):
    """
    Réponse listant les versions de modèles enregistrées.
    """

    items: list[ModelRegistryItemResponse]


class ModelRegistryRegisterResponse(StrictSchema):
    """
    Réponse après enregistrement ou mise à jour d'une version de modèle.
    """

    message: str
    model_name: str
    model_version: str
    stage: str
    is_active: bool
    deployed_at: datetime | None = None


# =============================================================================
# Alertes
# =============================================================================

class AlertBaseResponse(StrictSchema):
    """
    Base commune de réponse pour les alertes de monitoring.
    """

    alert_type: str
    severity: str
    model_name: str | None = None
    model_version: str | None = None
    feature_name: str | None = None
    title: str
    message: str
    context: dict[str, Any] | None = None
    status: str


class AlertResponse(AlertBaseResponse):
    """
    Représentation complète d'une alerte de monitoring.
    """

    id: int
    created_at: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


class AlertListResponse(CountItemsResponse):
    """
    Réponse listant les alertes récentes.
    """

    items: list[AlertResponse]


class AlertActionResponse(StrictSchema):
    """
    Réponse standard pour les actions sur une alerte.
    """

    id: int
    status: str
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


# =============================================================================
# Monitoring summary / health
# =============================================================================

class MonitoringPredictionsSummary(StrictSchema):
    """
    Bloc synthétique sur les prédictions.
    """

    total_predictions: int
    total_errors: int
    error_rate: float
    avg_latency_ms: float | None = None
    last_prediction_at: datetime | None = None


class MonitoringDriftSummary(StrictSchema):
    """
    Bloc synthétique sur la dérive.
    """

    total_drift_metrics: int
    detected_drifts: int
    drift_rate: float
    last_drift_at: datetime | None = None


class MonitoringAlertsSummary(StrictSchema):
    """
    Bloc synthétique sur les alertes.
    """

    open_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int


class EvaluationMetricSummary(StrictSchema):
    """
    Dernière métrique d'évaluation disponible.
    """

    id: int
    dataset_name: str
    window_start: datetime | None = None
    window_end: datetime | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    precision_score: float | None = None
    recall_score: float | None = None
    f1_score: float | None = None
    fbeta_score: float | None = None
    business_cost: float | None = None
    tn: int | None = None
    fp: int | None = None
    fn: int | None = None
    tp: int | None = None
    sample_size: int | None = None
    computed_at: datetime


class MonitoringSummaryResponse(StrictSchema):
    """
    Réponse de synthèse complète du monitoring.
    """

    model_name: str
    model_version: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    predictions: MonitoringPredictionsSummary
    drift: MonitoringDriftSummary
    latest_evaluation: EvaluationMetricSummary | None = None
    alerts: MonitoringAlertsSummary


class MonitoringHealthResponse(StrictSchema):
    """
    Réponse d'état simple et lisible du monitoring.
    """

    model_name: str
    model_version: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    has_predictions: bool
    has_drift_metrics: bool
    has_latest_evaluation: bool
    open_alerts: int
    avg_latency_ms: float | None = None
    last_prediction_at: datetime | None = None
    last_drift_at: datetime | None = None
    latest_evaluation_at: datetime | None = None


# =============================================================================
# History / historique des prédictions
# =============================================================================

class PredictionHistoryItemResponse(StrictSchema):
    """
    Représentation d'une ligne d'historique de prédiction.

    Notes
    -----
    Ce schéma est aligné sur ce que renvoient aujourd'hui
    `history_service` et `route_history`.
    """

    id: int
    request_id: str
    client_id: int | None = None
    model_name: str | None = None
    model_version: str | None = None
    prediction: int | None = None
    prediction_label: str | None = None
    score: float | None = None
    threshold_used: float | None = None
    latency_ms: float | None = None
    inference_latency_ms: float | None = None
    prediction_timestamp: datetime | None = None
    status_code: int | None = None
    error_message: str | None = None
    status: str | None = None


class PredictionHistoryResponse(PaginatedItemsResponse):
    """
    Réponse listant l'historique des prédictions.
    """

    items: list[PredictionHistoryItemResponse]
    total: int | None = None


class PredictionDetailResponse(BaseModel):
    """
    Réponse détaillée pour une prédiction donnée.

    Notes
    -----
    On garde `extra="allow"` pour rester compatible avec les champs
    réellement renvoyés par le service.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        from_attributes=True,
    )

    id: int
    request_id: str
    client_id: int | None = None
    model_name: str | None = None
    model_version: str | None = None
    prediction: int | None = None
    prediction_label: str | None = None
    score: float | None = None
    threshold_used: float | None = None
    latency_ms: float | None = None
    inference_latency_ms: float | None = None
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    prediction_timestamp: datetime | None = None
    status_code: int | None = None
    error_message: str | None = None
    status: str | None = None


class GroundTruthItemResponse(StrictSchema):
    """
    Représentation d'une ligne de vérité terrain.
    """

    id: int
    request_id: str | None = None
    client_id: int | None = None
    true_label: int
    label_source: str | None = None
    observed_at: datetime | None = None
    notes: str | None = None


class GroundTruthHistoryResponse(PaginatedItemsResponse):
    """
    Réponse listant les vérités terrain historisées.
    """

    items: list[GroundTruthItemResponse]
    total: int | None = None


class PredictionFeatureSnapshotItemResponse(StrictSchema):
    """
    Représentation d'une feature enregistrée dans un snapshot.
    """

    feature_name: str
    feature_value: Any | None = None
    feature_type: str | None = None


class PredictionFeaturesSnapshotResponse(StrictSchema):
    """
    Réponse contenant le snapshot des features pour une requête.
    """

    request_id: str
    client_id: int | None = None
    model_name: str | None = None
    model_version: str | None = None
    snapshot_timestamp: datetime | None = None
    feature_count: int
    items: list[PredictionFeatureSnapshotItemResponse]


# =============================================================================
# Monitoring list payloads
# =============================================================================

class DriftMetricResponse(StrictSchema):
    """
    Représentation d'une métrique de drift.
    """

    id: int
    model_name: str
    model_version: str
    feature_name: str
    metric_name: str
    reference_window_start: datetime | None = None
    reference_window_end: datetime | None = None
    current_window_start: datetime | None = None
    current_window_end: datetime | None = None
    metric_value: float
    threshold_value: float | None = None
    drift_detected: bool
    details: dict[str, Any] | None = None
    computed_at: datetime


class DriftMetricListResponse(CountItemsResponse):
    """
    Réponse listant les métriques de drift.
    """

    items: list[DriftMetricResponse]


class EvaluationMetricResponse(StrictSchema):
    """
    Représentation d'une métrique d'évaluation.
    """

    id: int
    model_name: str
    model_version: str
    dataset_name: str
    window_start: datetime | None = None
    window_end: datetime | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    precision_score: float | None = None
    recall_score: float | None = None
    f1_score: float | None = None
    fbeta_score: float | None = None
    business_cost: float | None = None
    tn: int | None = None
    fp: int | None = None
    fn: int | None = None
    tp: int | None = None
    sample_size: int | None = None
    computed_at: datetime


class EvaluationMetricListResponse(CountItemsResponse):
    """
    Réponse listant les métriques d'évaluation.
    """

    items: list[EvaluationMetricResponse]


class FeatureStoreMonitoringItemResponse(StrictSchema):
    """
    Représentation d'une ligne du feature store monitoring.
    """

    id: int
    request_id: str | None = None
    client_id: int | None = None
    model_name: str
    model_version: str
    feature_name: str
    feature_value: str | None = None
    feature_type: str | None = None
    source_table: str | None = None
    snapshot_timestamp: datetime


class FeatureStoreMonitoringListResponse(CountItemsResponse):
    """
    Réponse listant le feature store monitoring.
    """

    items: list[FeatureStoreMonitoringItemResponse]


# =============================================================================
# Evidently
# =============================================================================

class EvidentlyRunResponse(StrictSchema):
    """
    Réponse de lancement d'une analyse Evidently.
    """

    success: bool
    message: str
    model_name: str | None = None
    model_version: str | None = None
    reference_kind: str | None = None
    current_kind: str | None = None
    logged_metrics: int
    html_report_path: str | None = None
    reference_rows: int
    current_rows: int
    analyzed_columns: list[str]
    report: dict[str, Any] | None = None

# =============================================================================
# Evaluation
# =============================================================================

class MonitoringEvaluationRunResponse(StrictSchema):
    success: bool
    message: str
    model_name: str
    model_version: str
    dataset_name: str
    logged_metrics: int
    sample_size: int
    matched_rows: int
    threshold_used: float | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)

class GroundTruthCreateRequest(StrictSchema):
    """
    Payload d'ajout d'une vérité terrain.
    """

    request_id: str
    client_id: int | None = None
    true_label: Literal[0, 1]
    label_source: str | None = None
    observed_at: datetime | None = None
    notes: str | None = None


class GroundTruthCreateResponse(StrictSchema):
    """
    Réponse renvoyée après création d'une vérité terrain.
    """

    id: int
    request_id: str | None = None
    client_id: int | None = None
    true_label: int
    label_source: str | None = None
    observed_at: datetime
    notes: str | None = None