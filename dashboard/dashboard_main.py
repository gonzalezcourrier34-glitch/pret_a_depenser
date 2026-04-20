"""
Point d'entrée principal du dashboard Streamlit.

Ce script orchestre :
- la configuration générale
- les helpers communs
- les chargements partagés via l'API FastAPI
- l'affichage des 3 grands modules du dashboard

Architecture
------------
Le dashboard ne lit jamais directement PostgreSQL.
Toutes les données sont récupérées via l'API FastAPI.

Architecture actuelle
---------------------
- les prédictions utilisent des données issues exclusivement de
  `application_test.csv` côté API
- le dashboard ne lit plus de table SQL de features
- PostgreSQL sert uniquement à l'historique, au logging et au monitoring
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard_config import (
    API_KEY,
    API_URL,
    DEFAULT_LIMIT,
    MODEL_NAME,
    MODEL_VERSION,
)
from dashboard_monitoring import render_monitoring_page
from dashboard_predictions import render_predictions_page
from dashboard_request import (
    build_preview_map,
    build_tables_status_dataframe,
    call_predict_api,
    call_predict_batch_api,
    call_predict_fully_random_batch_api,
    call_predict_real_random_batch_api,
    get_active_model,
    get_alerts,
    get_drift_metrics,
    get_evaluation_metrics,
    get_feature_store_monitoring,
    get_ground_truth_by_request_id,
    get_ground_truth_history,
    get_health,
    get_models,
    get_monitoring_summary,
    get_prediction_detail,
    get_prediction_features_snapshot,
    get_prediction_history,
    load_client_features,
)
from dashboard_systeme import render_systeme_page


# =============================================================================
# Configuration Streamlit
# =============================================================================

st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALLOWED_FEATURES_TABLES: set[str] = {
    "application_test",
}
# =============================================================================
# Style
# =============================================================================

def inject_css() -> None:
    """
    Injecte un style CSS léger pour le dashboard.
    """
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .subtitle {
                color: #6B7280;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Helpers
# =============================================================================

def dataframe_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    """
    Convertit une ligne de DataFrame en payload compatible avec l'API /predict.

    Notes
    -----
    Le payload attendu par l'API est :
    {
        "SK_ID_CURR": ...,
        "features": {...}
    }

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins une ligne.

    Returns
    -------
    dict[str, Any]
        Payload API prêt à envoyer.
    """
    if df.empty:
        return {}

    row = df.iloc[0]
    features: dict[str, Any] = {}
    client_id = None

    for key, value in row.items():
        python_value: Any

        if pd.isna(value):
            python_value = None
        elif isinstance(value, pd.Timestamp):
            python_value = value.isoformat()
        else:
            python_value = value.item() if hasattr(value, "item") else value

        if key == "SK_ID_CURR":
            client_id = python_value
        elif key != "TARGET":
            features[key] = python_value

    return {
        "SK_ID_CURR": client_id,
        "features": features,
    }


def metric_safe_number(
    df: pd.DataFrame,
    col: str,
    agg: str = "mean",
    default: float | None = 0.0,
) -> float | None:
    """
    Calcule une statistique simple sur une colonne numérique.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    col : str
        Colonne à agréger.
    agg : str, default="mean"
        Type d'agrégation.
    default : float | None, default=0.0
        Valeur de repli.

    Returns
    -------
    float | None
        Statistique calculée ou valeur par défaut.
    """
    if df.empty or col not in df.columns:
        return default

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return default

    if agg == "mean":
        return float(series.mean())
    if agg == "sum":
        return float(series.sum())
    if agg == "max":
        return float(series.max())
    if agg == "min":
        return float(series.min())
    if agg == "p95":
        return float(series.quantile(0.95))
    if agg == "p99":
        return float(series.quantile(0.99))

    return default


def safe_dataframe(value: Any) -> pd.DataFrame:
    """
    Retourne un DataFrame vide si la valeur reçue n'est pas un DataFrame.

    Parameters
    ----------
    value : Any
        Valeur à sécuriser.

    Returns
    -------
    pd.DataFrame
        DataFrame valide.
    """
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def safe_dict(value: Any) -> dict[str, Any]:
    """
    Retourne un dictionnaire vide si la valeur reçue n'est pas un dict.

    Parameters
    ----------
    value : Any
        Valeur à sécuriser.

    Returns
    -------
    dict[str, Any]
        Dictionnaire valide.
    """
    return value if isinstance(value, dict) else {}


# =============================================================================
# Wrappers API pour les pages
# =============================================================================

def call_predict_api_wrapper(payload: dict[str, Any]) -> tuple[bool, Any]:
    """
    Wrapper local pour appeler /predict avec la configuration du dashboard.
    """
    return call_predict_api(
        payload,
        base_url=API_URL,
        api_key=API_KEY,
    )


def call_predict_batch_api_wrapper(payloads: list[dict[str, Any]]) -> tuple[bool, Any]:
    """
    Wrapper local pour appeler /predict/batch avec la configuration du dashboard.
    """
    return call_predict_batch_api(
        payloads,
        base_url=API_URL,
        api_key=API_KEY,
    )


def call_predict_real_random_batch_api_wrapper(
    batch_size: int,
    random_seed: int | None = None,
) -> tuple[bool, Any]:
    """
    Wrapper local pour appeler la route de simulation basée sur des données réelles.
    """
    return call_predict_real_random_batch_api(
        batch_size=batch_size,
        random_seed=random_seed,
        base_url=API_URL,
        api_key=API_KEY,
    )


def call_predict_fully_random_batch_api_wrapper(batch_size: int) -> tuple[bool, Any]:
    """
    Wrapper local pour appeler la route de simulation basée sur des données aléatoires.
    """
    return call_predict_fully_random_batch_api(
        batch_size=batch_size,
        base_url=API_URL,
        api_key=API_KEY,
    )


def get_prediction_detail_wrapper(request_id: str) -> dict[str, Any] | None:
    """
    Wrapper local pour récupérer le détail d'une prédiction.
    """
    return get_prediction_detail(
        request_id,
        base_url=API_URL,
        api_key=API_KEY,
    )


def get_prediction_features_snapshot_wrapper(request_id: str) -> dict[str, Any] | None:
    """
    Wrapper local pour récupérer le snapshot de features d'une requête.
    """
    return get_prediction_features_snapshot(
        request_id,
        base_url=API_URL,
        api_key=API_KEY,
    )


def get_ground_truth_by_request_id_wrapper(
    request_id: str,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wrapper local pour récupérer une vérité terrain par request_id
    à partir du DataFrame déjà chargé.
    """
    return get_ground_truth_by_request_id(
        request_id,
        ground_truth_df=ground_truth_df,
    )

def load_client_features_wrapper(
    client_id: int,
    source_table: str,
) -> pd.DataFrame:
    """
    Wrapper local pour charger les features d'un client via l'API.
    """
    return load_client_features(
        client_id,
        source_table,
        base_url=API_URL,
        api_key=API_KEY,
    )
# =============================================================================
# Chargements communs via l'API
# =============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def load_shared_data(limit: int) -> dict[str, Any]:
    """
    Charge toutes les données partagées du dashboard via l'API.

    Parameters
    ----------
    limit : int
        Nombre maximal de lignes à demander aux endpoints de liste.

    Returns
    -------
    dict[str, Any]
        Données mutualisées pour toutes les pages du dashboard.
    """
    health_data = safe_dict(get_health(base_url=API_URL))

    prediction_logs_df = safe_dataframe(
        get_prediction_history(
            base_url=API_URL,
            api_key=API_KEY,
            limit=limit,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION or None,
        )
    )

    ground_truth_df = safe_dataframe(
        get_ground_truth_history(
            base_url=API_URL,
            api_key=API_KEY,
            limit=limit,
        )
    )

    monitoring_summary = safe_dict(
        get_monitoring_summary(
            base_url=API_URL,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION or None,
        )
    )

    model_registry_df = safe_dataframe(
        get_models(
            base_url=API_URL,
            api_key=API_KEY,
            limit=limit,
            model_name=MODEL_NAME,
        )
    )

    active_model_df = safe_dataframe(
        get_active_model(
            base_url=API_URL,
            api_key=API_KEY,
            model_name=MODEL_NAME,
        )
    )

    evaluation_metrics_df = safe_dataframe(
        get_evaluation_metrics(
            base_url=API_URL,
            api_key=API_KEY,
            limit=limit,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION or None,
        )
    )

    drift_metrics_df = safe_dataframe(
        get_drift_metrics(
            base_url=API_URL,
            api_key=API_KEY,
            limit=limit,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION or None,
        )
    )

    alerts_df = safe_dataframe(
        get_alerts(
            base_url=API_URL,
            api_key=API_KEY,
            limit=limit,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION or None,
        )
    )

    feature_store_monitoring_df = safe_dataframe(
        get_feature_store_monitoring(
            base_url=API_URL,
            api_key=API_KEY,
            limit=limit,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION or None,
        )
    )

    tables_status_df = build_tables_status_dataframe(
        prediction_logs_df=prediction_logs_df,
        ground_truth_df=ground_truth_df,
        model_registry_df=model_registry_df,
        feature_store_monitoring_df=feature_store_monitoring_df,
        drift_metrics_df=drift_metrics_df,
        evaluation_metrics_df=evaluation_metrics_df,
        alerts_df=alerts_df,
    )

    preview_map = build_preview_map(
        prediction_logs_df=prediction_logs_df,
        ground_truth_df=ground_truth_df,
        model_registry_df=model_registry_df,
        feature_store_monitoring_df=feature_store_monitoring_df,
        drift_metrics_df=drift_metrics_df,
        evaluation_metrics_df=evaluation_metrics_df,
        alerts_df=alerts_df,
        max_rows=min(limit, 200),
    )

    return {
        "health_data": health_data,
        "prediction_logs_df": prediction_logs_df,
        "ground_truth_df": ground_truth_df,
        "monitoring_summary": monitoring_summary,
        "model_registry_df": model_registry_df,
        "active_model_df": active_model_df,
        "evaluation_metrics_df": evaluation_metrics_df,
        "drift_metrics_df": drift_metrics_df,
        "alerts_df": alerts_df,
        "feature_store_monitoring_df": feature_store_monitoring_df,
        "tables_status_df": tables_status_df,
        "preview_map": preview_map,
    }


# =============================================================================
# UI
# =============================================================================

inject_css()

st.markdown(
    '<div class="main-title">Dashboard scoring crédit</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Prédiction, traçabilité et monitoring MLOps dans une seule interface.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Navigation")

    st.caption(f"API : {API_URL}")
    st.caption(f"Modèle : {MODEL_NAME}")
    if MODEL_VERSION:
        st.caption(f"Version ciblée : {MODEL_VERSION}")

    page = st.radio(
        "Aller vers",
        [
            "Système / Données",
            "Prédictions / Traçabilité",
            "Monitoring",
        ],
        index=1,
    )

    history_limit = st.slider(
        "Nombre max de lignes",
        min_value=20,
        max_value=1000,
        value=DEFAULT_LIMIT,
        step=20,
    )

    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# =============================================================================
# Chargement principal
# =============================================================================

shared = load_shared_data(history_limit)

health_data = safe_dict(shared.get("health_data"))
prediction_logs_df = safe_dataframe(shared.get("prediction_logs_df"))
ground_truth_df = safe_dataframe(shared.get("ground_truth_df"))
monitoring_summary = safe_dict(shared.get("monitoring_summary"))
model_registry_df = safe_dataframe(shared.get("model_registry_df"))
active_model_df = safe_dataframe(shared.get("active_model_df"))
evaluation_metrics_df = safe_dataframe(shared.get("evaluation_metrics_df"))
drift_metrics_df = safe_dataframe(shared.get("drift_metrics_df"))
alerts_df = safe_dataframe(shared.get("alerts_df"))
feature_store_monitoring_df = safe_dataframe(shared.get("feature_store_monitoring_df"))
tables_status_df = safe_dataframe(shared.get("tables_status_df"))
preview_map = shared.get("preview_map", {})

if not isinstance(preview_map, dict):
    preview_map = {}


# =============================================================================
# Routage
# =============================================================================

if page == "Système / Données":
    render_systeme_page(
        health_data=health_data,
        tables_status_df=tables_status_df,
        selected_table_preview_map=preview_map,
        available_tables=list(preview_map.keys()),
        api_key=API_KEY,
    )

elif page == "Prédictions / Traçabilité":
    render_predictions_page(
        ALLOWED_FEATURES_TABLES=ALLOWED_FEATURES_TABLES,
        prediction_logs_df=prediction_logs_df,
        ground_truth_df=ground_truth_df,
        load_client_features=load_client_features_wrapper,
        dataframe_to_payload=dataframe_to_payload,
        call_predict_api=call_predict_api_wrapper,
        call_predict_batch_api=call_predict_batch_api_wrapper,
        get_prediction_detail=get_prediction_detail_wrapper,
        get_prediction_features_snapshot=get_prediction_features_snapshot_wrapper,
        get_ground_truth_by_request_id=get_ground_truth_by_request_id_wrapper,
        metric_safe_number=metric_safe_number,
        call_predict_real_random_batch_api=call_predict_real_random_batch_api_wrapper,
        call_predict_fully_random_batch_api=call_predict_fully_random_batch_api_wrapper,
    )
    
elif page == "Monitoring":
    render_monitoring_page(
        monitoring_summary=monitoring_summary,
        prediction_logs_df=prediction_logs_df,
        model_registry_df=model_registry_df,
        active_model_df=active_model_df,
        evaluation_metrics_df=evaluation_metrics_df,
        drift_metrics_df=drift_metrics_df,
        alerts_df=alerts_df,
        feature_store_monitoring_df=feature_store_monitoring_df,
        metric_safe_number=metric_safe_number,
    )