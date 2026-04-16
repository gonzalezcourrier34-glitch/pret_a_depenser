"""
Point d'entrée principal du dashboard Streamlit.

Ce script orchestre :
- la configuration générale
- les accès base de données
- les helpers communs
- les chargements partagés
- l'affichage des 3 grands modules du dashboard
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from dashboard.dashboard_request import (
    ACTIVE_MODEL_QUERY,
    ALERTS_QUERY,
    DRIFT_METRICS_QUERY,
    EVALUATION_METRICS_QUERY,
    FEATURE_STORE_MONITORING_QUERY,
    GROUND_TRUTH_BY_REQUEST_ID_QUERY,
    GROUND_TRUTH_LABELS_QUERY,
    MODEL_REGISTRY_QUERY,
    PREDICTION_FEATURES_BY_REQUEST_ID_QUERY,
    PREDICTION_LOGS_LIGHT_QUERY,
    TABLE_COLUMNS_QUERY,
    TABLE_EXISTS_QUERY,
    get_client_features_query,
    get_table_preview_query,
)
from dashboard.dashboard_systeme import render_systeme_page
from dashboard.dashboard_predictions import render_predictions_page
from dashboard.dashboard_monitoring import render_monitoring_page


# =============================================================================
# Configuration Streamlit
# =============================================================================

st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Configuration applicative
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")
FEATURES_TABLE = os.getenv("FEATURES_TABLE", "features_client_test_enriched")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "200"))

ALLOWED_FEATURES_TABLES = {
    "features_client_test",
    "features_client_test_enriched",
}


# =============================================================================
# Style
# =============================================================================

def inject_css() -> None:
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
# Accès base
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine | None:
    if not DATABASE_URL:
        return None
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=30, show_spinner=False)
def run_query(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    with engine.begin() as connection:
        return pd.read_sql(text(query), connection, params=params or {})


@st.cache_data(ttl=30, show_spinner=False)
def table_exists(table_name: str) -> bool:
    if not DATABASE_URL:
        return False

    df = run_query(TABLE_EXISTS_QUERY, {"table_name": table_name})
    if df.empty:
        return False
    return bool(df.iloc[0]["exists_flag"])


@st.cache_data(ttl=30, show_spinner=False)
def get_table_columns(table_name: str) -> list[str]:
    if not table_exists(table_name):
        return []

    df = run_query(TABLE_COLUMNS_QUERY, {"table_name": table_name})
    if df.empty:
        return []

    return df["column_name"].tolist()


@st.cache_data(ttl=30, show_spinner=False)
def load_client_features(client_id: int, table_name: str) -> pd.DataFrame:
    if table_name not in ALLOWED_FEATURES_TABLES:
        return pd.DataFrame()

    if not table_exists(table_name):
        return pd.DataFrame()

    query = get_client_features_query(table_name)
    return run_query(query, {"client_id": client_id})


def query_if_table_exists(
    table_name: str,
    query: str,
    params: dict[str, Any] | None = None,
    datetime_cols: list[str] | None = None,
) -> pd.DataFrame:
    if not table_exists(table_name):
        return pd.DataFrame()

    df = run_query(query, params=params)

    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# =============================================================================
# Helpers
# =============================================================================

def dataframe_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}

    row = df.iloc[0]
    payload: dict[str, Any] = {}

    for key, value in row.items():
        if pd.isna(value):
            payload[key] = None
        elif isinstance(value, pd.Timestamp):
            payload[key] = value.isoformat()
        else:
            payload[key] = value.item() if hasattr(value, "item") else value

    payload.pop("TARGET", None)
    return payload


def call_predict_api(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    url = f"{API_URL.rstrip('/')}/predict"
    headers = {"Content-Type": "application/json"}

    if API_KEY:
        headers["X-API-Key"] = API_KEY

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return True, response.json()

    except requests.HTTPError:
        try:
            details = response.json()
        except Exception:
            details = {"error": response.text}

        return False, {
            "status_code": response.status_code,
            "details": details,
            "url_called": url,
        }

    except Exception as exc:
        return False, {
            "error": str(exc),
            "url_called": url,
        }


def metric_safe_number(
    df: pd.DataFrame,
    col: str,
    agg: str = "mean",
    default: float = 0.0,
) -> float:
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


def show_table_status(table_name: str) -> None:
    if table_exists(table_name):
        st.success(f"Table disponible : `{table_name}`")
    else:
        st.warning(f"Table absente : `{table_name}`")


# =============================================================================
# UI
# =============================================================================

inject_css()

st.markdown('<div class="main-title">Dashboard scoring crédit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Prédiction, traçabilité et monitoring MLOps dans une seule interface.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🧭 Navigation")

    page = st.radio(
        "Aller vers",
        [
            "💻 Système / Données",
            "⚡ Prédictions / Traçabilité",
            "📊 Monitoring",
        ],
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


# =============================================================================
# Chargements communs
# =============================================================================

prediction_logs_df = query_if_table_exists(
    "prediction_logs",
    PREDICTION_LOGS_LIGHT_QUERY,
    {"limit_rows": history_limit},
    ["prediction_timestamp"],
)

ground_truth_df = query_if_table_exists(
    "ground_truth_labels",
    GROUND_TRUTH_LABELS_QUERY,
    {"limit_rows": history_limit},
    ["observed_at"],
)

model_registry_df = query_if_table_exists(
    "model_registry",
    MODEL_REGISTRY_QUERY,
    datetime_cols=["deployed_at", "created_at"],
)

active_model_df = query_if_table_exists(
    "model_registry",
    ACTIVE_MODEL_QUERY,
    datetime_cols=["deployed_at", "created_at"],
)

evaluation_metrics_df = query_if_table_exists(
    "evaluation_metrics",
    EVALUATION_METRICS_QUERY,
    {"limit_rows": history_limit},
    ["window_start", "window_end", "computed_at"],
)

drift_metrics_df = query_if_table_exists(
    "drift_metrics",
    DRIFT_METRICS_QUERY,
    {"limit_rows": history_limit},
    [
        "reference_window_start",
        "reference_window_end",
        "current_window_start",
        "current_window_end",
        "computed_at",
    ],
)

alerts_df = query_if_table_exists(
    "alerts",
    ALERTS_QUERY,
    {"limit_rows": history_limit},
    ["created_at", "acknowledged_at", "resolved_at"],
)

feature_store_monitoring_df = query_if_table_exists(
    "feature_store_monitoring",
    FEATURE_STORE_MONITORING_QUERY,
    {"limit_rows": history_limit},
    ["snapshot_timestamp"],
)


# =============================================================================
# Routage
# =============================================================================

if page == "💻 Système / Données":
    render_systeme_page(
        DATABASE_URL=DATABASE_URL,
        FEATURES_TABLE=FEATURES_TABLE,
        table_exists=table_exists,
        get_table_columns=get_table_columns,
        show_table_status=show_table_status,
        run_query=run_query,
        get_table_preview_query=get_table_preview_query,
        history_limit=history_limit,
    )

elif page == "⚡ Prédictions / Traçabilité":
    render_predictions_page(
        ALLOWED_FEATURES_TABLES=ALLOWED_FEATURES_TABLES,
        prediction_logs_df=prediction_logs_df,
        load_client_features=load_client_features,
        dataframe_to_payload=dataframe_to_payload,
        call_predict_api=call_predict_api,
        metric_safe_number=metric_safe_number,
        query_if_table_exists=query_if_table_exists,
        PREDICTION_FEATURES_BY_REQUEST_ID_QUERY=PREDICTION_FEATURES_BY_REQUEST_ID_QUERY,
        GROUND_TRUTH_BY_REQUEST_ID_QUERY=GROUND_TRUTH_BY_REQUEST_ID_QUERY,
    )

elif page == "📊 Monitoring":
    render_monitoring_page(
        prediction_logs_df=prediction_logs_df,
        ground_truth_df=ground_truth_df,
        model_registry_df=model_registry_df,
        active_model_df=active_model_df,
        evaluation_metrics_df=evaluation_metrics_df,
        drift_metrics_df=drift_metrics_df,
        alerts_df=alerts_df,
        feature_store_monitoring_df=feature_store_monitoring_df,
        metric_safe_number=metric_safe_number,
    )