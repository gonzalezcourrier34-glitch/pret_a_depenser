"""
Requêtes SQL centralisées pour le dashboard Streamlit.

Ce module regroupe toutes les requêtes utilisées par le dashboard
de scoring crédit et de monitoring MLOps.

Objectif
--------
Centraliser les requêtes SQL pour :
- améliorer la lisibilité du dashboard
- éviter de dupliquer les chaînes SQL
- faciliter la maintenance
- rendre les pages Streamlit plus courtes et plus claires

Tables concernées
-----------------
Prédiction / historique
- prediction_logs
- ground_truth_labels
- prediction_features_snapshot

Monitoring
- model_registry
- feature_store_monitoring
- drift_metrics
- evaluation_metrics
- alerts

Features
- features_client_test
- features_client_test_enriched
"""

from __future__ import annotations


# =============================================================================
# Requêtes techniques génériques
# =============================================================================

TABLE_EXISTS_QUERY = """
SELECT EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = :table_name
) AS exists_flag
"""


TABLE_COLUMNS_QUERY = """
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = :table_name
ORDER BY ordinal_position
"""


# =============================================================================
# Requêtes de chargement de features client
# =============================================================================

def get_client_features_query(table_name: str) -> str:
    """
    Construit la requête SQL permettant de charger un client
    depuis une table de features autorisée.

    Parameters
    ----------
    table_name : str
        Nom de la table source.

    Returns
    -------
    str
        Requête SQL formatée.

    Notes
    -----
    Le nom de table n'est pas paramétrable via :param SQLAlchemy,
    donc cette fonction doit être utilisée uniquement avec une
    liste blanche de tables autorisées côté dashboard.
    """
    return f'''
    SELECT *
    FROM "{table_name}"
    WHERE "SK_ID_CURR" = :client_id
    LIMIT 1
    '''


def get_table_preview_query(table_name: str) -> str:
    """
    Construit une requête SQL d'aperçu d'une table.

    Parameters
    ----------
    table_name : str
        Nom de la table à prévisualiser.

    Returns
    -------
    str
        Requête SQL formatée.
    """
    return f'''
    SELECT *
    FROM "{table_name}"
    LIMIT :limit_rows
    '''


# =============================================================================
# Requêtes prediction_logs
# =============================================================================

PREDICTION_LOGS_QUERY = """
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
    input_data,
    output_data,
    prediction_timestamp,
    status_code,
    error_message
FROM prediction_logs
ORDER BY prediction_timestamp DESC
LIMIT :limit_rows
"""


PREDICTION_LOGS_LIGHT_QUERY = """
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
    prediction_timestamp,
    status_code,
    error_message
FROM prediction_logs
ORDER BY prediction_timestamp DESC
LIMIT :limit_rows
"""


PREDICTION_LOG_BY_REQUEST_ID_QUERY = """
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
    input_data,
    output_data,
    prediction_timestamp,
    status_code,
    error_message
FROM prediction_logs
WHERE request_id = :request_id
ORDER BY prediction_timestamp DESC
"""


PREDICTION_LOGS_TIMESERIES_QUERY = """
SELECT
    request_id,
    client_id,
    prediction,
    score,
    latency_ms,
    prediction_timestamp,
    status_code
FROM prediction_logs
ORDER BY prediction_timestamp ASC
LIMIT :limit_rows
"""


# =============================================================================
# Requêtes ground truth
# =============================================================================

GROUND_TRUTH_LABELS_QUERY = """
SELECT
    id,
    request_id,
    client_id,
    true_label,
    label_source,
    observed_at,
    notes
FROM ground_truth_labels
ORDER BY observed_at DESC
LIMIT :limit_rows
"""


GROUND_TRUTH_BY_REQUEST_ID_QUERY = """
SELECT
    id,
    request_id,
    client_id,
    true_label,
    label_source,
    observed_at,
    notes
FROM ground_truth_labels
WHERE request_id = :request_id
ORDER BY observed_at DESC
"""


GROUND_TRUTH_BY_CLIENT_ID_QUERY = """
SELECT
    id,
    request_id,
    client_id,
    true_label,
    label_source,
    observed_at,
    notes
FROM ground_truth_labels
WHERE client_id = :client_id
ORDER BY observed_at DESC
"""


# =============================================================================
# Requêtes snapshot de features de prédiction
# =============================================================================

PREDICTION_FEATURES_SNAPSHOT_QUERY = """
SELECT
    id,
    request_id,
    client_id,
    model_name,
    model_version,
    feature_name,
    feature_value,
    feature_type,
    snapshot_timestamp
FROM prediction_features_snapshot
ORDER BY snapshot_timestamp DESC
LIMIT :limit_rows
"""


PREDICTION_FEATURES_BY_REQUEST_ID_QUERY = """
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


PREDICTION_FEATURES_BY_CLIENT_ID_QUERY = """
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
WHERE client_id = :client_id
ORDER BY snapshot_timestamp DESC, feature_name
LIMIT :limit_rows
"""


# =============================================================================
# Requêtes model registry
# =============================================================================

MODEL_REGISTRY_QUERY = """
SELECT
    id,
    model_name,
    model_version,
    stage,
    run_id,
    source_path,
    training_data_version,
    feature_list,
    hyperparameters,
    metrics,
    deployed_at,
    is_active,
    created_at
FROM model_registry
ORDER BY created_at DESC
"""


ACTIVE_MODEL_QUERY = """
SELECT
    id,
    model_name,
    model_version,
    stage,
    run_id,
    source_path,
    training_data_version,
    feature_list,
    hyperparameters,
    metrics,
    deployed_at,
    is_active,
    created_at
FROM model_registry
WHERE is_active = TRUE
ORDER BY created_at DESC
LIMIT 1
"""


# =============================================================================
# Requêtes evaluation metrics
# =============================================================================

EVALUATION_METRICS_QUERY = """
SELECT
    id,
    model_name,
    model_version,
    dataset_name,
    window_start,
    window_end,
    roc_auc,
    pr_auc,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    business_cost,
    tn,
    fp,
    fn,
    tp,
    sample_size,
    computed_at
FROM evaluation_metrics
ORDER BY computed_at DESC
LIMIT :limit_rows
"""


LATEST_EVALUATION_METRICS_QUERY = """
SELECT
    id,
    model_name,
    model_version,
    dataset_name,
    window_start,
    window_end,
    roc_auc,
    pr_auc,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    business_cost,
    tn,
    fp,
    fn,
    tp,
    sample_size,
    computed_at
FROM evaluation_metrics
ORDER BY computed_at DESC
LIMIT 1
"""


# =============================================================================
# Requêtes drift metrics
# =============================================================================

DRIFT_METRICS_QUERY = """
SELECT
    id,
    model_name,
    model_version,
    feature_name,
    metric_name,
    reference_window_start,
    reference_window_end,
    current_window_start,
    current_window_end,
    metric_value,
    threshold_value,
    drift_detected,
    details,
    computed_at
FROM drift_metrics
ORDER BY computed_at DESC
LIMIT :limit_rows
"""


DRIFT_ONLY_DETECTED_QUERY = """
SELECT
    id,
    model_name,
    model_version,
    feature_name,
    metric_name,
    reference_window_start,
    reference_window_end,
    current_window_start,
    current_window_end,
    metric_value,
    threshold_value,
    drift_detected,
    details,
    computed_at
FROM drift_metrics
WHERE drift_detected = TRUE
ORDER BY computed_at DESC
LIMIT :limit_rows
"""


# =============================================================================
# Requêtes alerts
# =============================================================================

ALERTS_QUERY = """
SELECT
    id,
    alert_type,
    severity,
    model_name,
    model_version,
    feature_name,
    title,
    message,
    context,
    status,
    created_at,
    acknowledged_at,
    resolved_at
FROM alerts
ORDER BY created_at DESC
LIMIT :limit_rows
"""


OPEN_ALERTS_QUERY = """
SELECT
    id,
    alert_type,
    severity,
    model_name,
    model_version,
    feature_name,
    title,
    message,
    context,
    status,
    created_at,
    acknowledged_at,
    resolved_at
FROM alerts
WHERE status = 'open'
ORDER BY created_at DESC
LIMIT :limit_rows
"""


# =============================================================================
# Requêtes feature_store_monitoring
# =============================================================================

FEATURE_STORE_MONITORING_QUERY = """
SELECT
    id,
    request_id,
    client_id,
    model_name,
    model_version,
    feature_name,
    feature_value,
    feature_type,
    source_table,
    snapshot_timestamp
FROM feature_store_monitoring
ORDER BY snapshot_timestamp DESC
LIMIT :limit_rows
"""


FEATURE_STORE_BY_REQUEST_ID_QUERY = """
SELECT
    id,
    request_id,
    client_id,
    model_name,
    model_version,
    feature_name,
    feature_value,
    feature_type,
    source_table,
    snapshot_timestamp
FROM feature_store_monitoring
WHERE request_id = :request_id
ORDER BY feature_name
"""


FEATURE_STORE_BY_CLIENT_ID_QUERY = """
SELECT
    id,
    request_id,
    client_id,
    model_name,
    model_version,
    feature_name,
    feature_value,
    feature_type,
    source_table,
    snapshot_timestamp
FROM feature_store_monitoring
WHERE client_id = :client_id
ORDER BY snapshot_timestamp DESC, feature_name
LIMIT :limit_rows
"""