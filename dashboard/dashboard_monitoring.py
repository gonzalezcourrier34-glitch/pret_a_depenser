"""
Page Streamlit : monitoring MLOps.

Cette page regroupe :
- une vue d'ensemble du système de monitoring
- les métriques de performance du modèle
- l'analyse de dérive des données
- les alertes et le registre des modèles

Notes
-----
Cette page ne lit pas directement PostgreSQL.
Les données affichées doivent être récupérées via l'API FastAPI
puis converties en DataFrames dans la couche dashboard_request.

Tables métier visées
--------------------
- prediction_logs
- model_registry
- evaluation_metrics
- drift_metrics
- alerts
- feature_store_monitoring

Architecture
------------
- Le dashboard reçoit déjà des objets Python / pandas préparés.
- Cette page ne fait que :
    1. sécuriser les données reçues,
    2. calculer quelques KPI,
    3. afficher les visualisations.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import streamlit as st


# =============================================================================
# Helpers locaux
# =============================================================================

def _safe_dict(value: Any) -> dict[str, Any]:
    """
    Retourne un dictionnaire vide si la valeur n'est pas un dict.
    """
    return value if isinstance(value, dict) else {}


def _safe_dataframe(value: Any) -> pd.DataFrame:
    """
    Garantit le retour d'un DataFrame.

    Si la valeur reçue n'est pas un DataFrame, on renvoie un DataFrame vide.
    """
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _safe_timestamp_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convertit une colonne en datetime si elle existe.
    """
    if df.empty or col not in df.columns:
        return df

    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convertit une colonne en numérique si elle existe.
    """
    if df.empty or col not in df.columns:
        return df

    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _coerce_columns_to_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Convertit plusieurs colonnes en datetime si elles existent.
    """
    out = df.copy()
    for col in columns:
        out = _safe_timestamp_series(out, col)
    return out


def _coerce_columns_to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Convertit plusieurs colonnes en numérique si elles existent.
    """
    out = df.copy()
    for col in columns:
        out = _safe_numeric_series(out, col)
    return out


def _render_card(title: str, value: Any, subtitle: str = "") -> None:
    """
    Affiche une carte KPI légère.
    """
    st.markdown(
        f"""
        <div style="
            padding: 16px 18px;
            border-radius: 16px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            min-height: 120px;
        ">
            <div style="font-size: 0.9rem; color: #9CA3AF; margin-bottom: 8px;">
                {title}
            </div>
            <div style="font-size: 1.9rem; font-weight: 700; color: white; margin-bottom: 8px;">
                {value}
            </div>
            <div style="font-size: 0.85rem; color: #D1D5DB;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_badge(label: str, color: str) -> str:
    """
    Retourne un badge HTML simple.
    """
    return (
        f"<span style='"
        f"display:inline-block;"
        f"padding:6px 12px;"
        f"border-radius:999px;"
        f"background:{color};"
        f"color:white;"
        f"font-size:0.82rem;"
        f"font-weight:600;'>"
        f"{label}</span>"
    )


def _render_section_title(title: str, subtitle: str = "") -> None:
    """
    Affiche un titre de section plus visuel.
    """
    st.markdown(
        f"""
        <div style="margin-top: 8px; margin-bottom: 14px;">
            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
                {title}
            </div>
            <div style="color: #6B7280; font-size: 0.95rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_metric_value(value: Any, decimals: int = 3) -> str:
    """
    Formate proprement une valeur numérique de KPI.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Convertit une valeur en entier de façon sûre.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Convertit une valeur en float de façon sûre.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    """
    Convertit une valeur en booléen de façon robuste.
    """
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "oui"}

    try:
        return bool(value)
    except Exception:
        return False


def _pick_latest_row(
    df: pd.DataFrame,
    date_cols: list[str],
) -> pd.Series | None:
    """
    Retourne la ligne la plus récente en fonction de la première colonne date
    disponible dans `date_cols`.
    """
    if df.empty:
        return None

    working = df.copy()

    for col in date_cols:
        if col in working.columns:
            working[col] = pd.to_datetime(working[col], errors="coerce")
            working = working.sort_values(col, ascending=False, na_position="last")
            if not working.empty:
                return working.iloc[0]

    return working.iloc[0] if not working.empty else None


def _choose_existing_columns(df: pd.DataFrame, preferred_cols: list[str]) -> list[str]:
    """
    Retourne uniquement les colonnes présentes dans le DataFrame.
    """
    return [col for col in preferred_cols if col in df.columns]


# =============================================================================
# Page principale
# =============================================================================

def render_monitoring_page(
    *,
    monitoring_summary: dict,
    monitoring_health: dict,
    prediction_logs_df: pd.DataFrame,
    model_registry_df: pd.DataFrame,
    active_model_df: pd.DataFrame,
    evaluation_metrics_df: pd.DataFrame,
    drift_metrics_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    feature_store_monitoring_df: pd.DataFrame,
    metric_safe_number,
    run_evidently_analysis,
) -> None:
    """
    Affiche la page de monitoring du modèle.
    """
    summary = _safe_dict(monitoring_summary)
    health = _safe_dict(monitoring_health)

    predictions_summary = _safe_dict(summary.get("predictions"))
    drift_summary = _safe_dict(summary.get("drift"))
    latest_eval_summary = _safe_dict(summary.get("latest_evaluation"))
    alerts_summary = _safe_dict(summary.get("alerts"))

    prediction_logs_df = _safe_dataframe(prediction_logs_df)
    model_registry_df = _safe_dataframe(model_registry_df)
    active_model_df = _safe_dataframe(active_model_df)
    evaluation_metrics_df = _safe_dataframe(evaluation_metrics_df)
    drift_metrics_df = _safe_dataframe(drift_metrics_df)
    alerts_df = _safe_dataframe(alerts_df)
    feature_store_monitoring_df = _safe_dataframe(feature_store_monitoring_df)

    evaluation_metrics_df = _coerce_columns_to_datetime(
        evaluation_metrics_df,
        ["computed_at", "window_start", "window_end"],
    )
    evaluation_metrics_df = _coerce_columns_to_numeric(
        evaluation_metrics_df,
        [
            "roc_auc",
            "pr_auc",
            "precision_score",
            "recall_score",
            "f1_score",
            "fbeta_score",
            "business_cost",
            "sample_size",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )

    drift_metrics_df = _coerce_columns_to_datetime(
        drift_metrics_df,
        [
            "computed_at",
            "reference_window_start",
            "reference_window_end",
            "current_window_start",
            "current_window_end",
        ],
    )
    drift_metrics_df = _coerce_columns_to_numeric(
        drift_metrics_df,
        ["metric_value", "threshold_value"],
    )

    prediction_logs_df = _coerce_columns_to_datetime(
        prediction_logs_df,
        ["prediction_timestamp"],
    )
    prediction_logs_df = _coerce_columns_to_numeric(
        prediction_logs_df,
        ["score", "threshold", "latency_ms", "prediction"],
    )

    alerts_df = _coerce_columns_to_datetime(
        alerts_df,
        ["created_at", "acknowledged_at", "resolved_at"],
    )

    model_registry_df = _coerce_columns_to_datetime(
        model_registry_df,
        ["deployed_at", "created_at"],
    )

    active_model_df = _coerce_columns_to_datetime(
        active_model_df,
        ["deployed_at", "created_at"],
    )

    feature_store_monitoring_df = _coerce_columns_to_datetime(
        feature_store_monitoring_df,
        ["snapshot_timestamp"],
    )

    # -------------------------------------------------------------------------
    # Calculs de synthèse
    # -------------------------------------------------------------------------
    detected_count_df = 0
    if not drift_metrics_df.empty and "drift_detected" in drift_metrics_df.columns:
        detected_count_df = int(
            drift_metrics_df["drift_detected"].apply(_safe_bool).sum()
        )

    open_alerts_df = 0
    if not alerts_df.empty and "status" in alerts_df.columns:
        open_alerts_df = int(
            alerts_df["status"].fillna("").astype(str).str.lower().eq("open").sum()
        )

    critical_alerts_df = 0
    if not alerts_df.empty and "severity" in alerts_df.columns:
        critical_alerts_df = int(
            alerts_df["severity"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(["critical", "high"])
            .sum()
        )

    recall_value = latest_eval_summary.get("recall_score")
    roc_auc_value = latest_eval_summary.get("roc_auc")
    avg_latency_summary = predictions_summary.get("avg_latency_ms")

    if avg_latency_summary is None:
        avg_latency_summary = metric_safe_number(
            prediction_logs_df,
            "latency_ms",
            "mean",
            None,
        )

    latest_active_model_row = _pick_latest_row(
        active_model_df,
        ["deployed_at", "created_at"],
    )
    latest_registry_row = _pick_latest_row(
        model_registry_df,
        ["deployed_at", "created_at"],
    )
    latest_eval_row = _pick_latest_row(
        evaluation_metrics_df,
        ["computed_at"],
    )

    has_predictions = _safe_bool(health.get("has_predictions"))
    has_drift_metrics = _safe_bool(health.get("has_drift_metrics"))
    has_latest_evaluation = _safe_bool(health.get("has_latest_evaluation"))

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    st.markdown("## Monitoring MLOps")
    st.caption(
        "Centre de pilotage du modèle en production : performance, dérive, alertes, latence et cycle de vie des versions."
    )

    # -------------------------------------------------------------------------
    # Bandeau KPI principal
    # -------------------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        _render_card(
            "Alertes ouvertes",
            alerts_summary.get("open_alerts", open_alerts_df),
            "Incidents actifs à surveiller",
        )
    with k2:
        _render_card(
            "Drifts détectés",
            drift_summary.get("detected_drifts", detected_count_df),
            "Features en dérive détectées",
        )
    with k3:
        _render_card(
            "Recall",
            _safe_metric_value(recall_value, 3),
            "Sensibilité du modèle",
        )
    with k4:
        _render_card(
            "ROC AUC",
            _safe_metric_value(roc_auc_value, 3),
            "Qualité de séparation",
        )
    with k5:
        latency_label = (
            f"{float(avg_latency_summary):.1f} ms"
            if avg_latency_summary is not None and not pd.isna(avg_latency_summary)
            else "N/A"
        )
        _render_card(
            "Latence moyenne",
            latency_label,
            "Temps moyen d'inférence",
        )

    st.markdown("")

    # -------------------------------------------------------------------------
    # Bloc modèle actif + statut
    # -------------------------------------------------------------------------
    left, right = st.columns([2, 1])

    with left:
        _render_section_title(
            "État courant du modèle",
            "Vue rapide sur la version active et la dernière évaluation connue.",
        )

        row = (
            latest_active_model_row
            if latest_active_model_row is not None
            else latest_registry_row
        )

        if row is not None:
            model_name = row.get("model_name", "N/A")
            model_version = row.get("model_version", "N/A")
            stage = row.get("stage", "N/A")
            deployed_at = row.get("deployed_at", None)
            run_id = row.get("run_id", "N/A")

            deployed_label = (
                deployed_at.strftime("%Y-%m-%d %H:%M:%S")
                if deployed_at is not None and pd.notna(deployed_at)
                else "N/A"
            )

            st.markdown(
                f"""
                <div style="
                    padding: 18px;
                    border-radius: 18px;
                    background: #F8FAFC;
                    border: 1px solid #E5E7EB;
                    margin-bottom: 12px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;">
                        {model_name} | version {model_version}
                    </div>
                    <div style="margin-bottom: 8px;">
                        {_status_badge(f"Stage : {stage}", "#2563EB")}
                    </div>
                    <div style="color: #4B5563; font-size: 0.92rem; margin-bottom: 4px;">
                        Déployé le : {deployed_label}
                    </div>
                    <div style="color: #6B7280; font-size: 0.85rem;">
                        Run ID : {run_id}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Aucun modèle actif remonté par l'API.")

    with right:
        _render_section_title(
            "Signal global",
            "Lecture rapide de la situation.",
        )

        health_color = "#16A34A"
        health_label = "Stable"

        if critical_alerts_df > 0:
            health_color = "#DC2626"
            health_label = "Critique"
        elif open_alerts_df > 0 or detected_count_df > 0:
            health_color = "#D97706"
            health_label = "Sous surveillance"

        if not has_predictions:
            health_label = "Peu de signal"
            health_color = "#6B7280"

        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 18px;
                background: {health_color};
                color: white;
                text-align: center;
                font-weight: 700;
                font-size: 1.2rem;
                margin-top: 6px;
            ">
                {health_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown(
            f"""
            <div style="font-size: 0.9rem; color: #6B7280;">
                Prédictions : {"Oui" if has_predictions else "Non"}<br>
                Drift disponible : {"Oui" if has_drift_metrics else "Non"}<br>
                Évaluation dispo : {"Oui" if has_latest_evaluation else "Non"}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------------
    tabs = st.tabs(
        [
            "Vue d'ensemble",
            "Performance",
            "Dérive",
            "Alertes & Modèles",
        ]
    )

    # =====================================================================
    # ONGLET 1 - VUE D'ENSEMBLE
    # =====================================================================
    with tabs[0]:
        _render_section_title(
            "Vue d'ensemble",
            "Panorama consolidé du monitoring actuel.",
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Prédictions",
            predictions_summary.get("total_predictions", len(prediction_logs_df)),
        )
        c2.metric(
            "Lignes de drift",
            drift_summary.get("total_drift_metrics", len(drift_metrics_df)),
        )
        c3.metric("Modèles enregistrés", len(model_registry_df))
        c4.metric("Alertes critiques / hautes", critical_alerts_df)

        st.markdown("")

        col_a, col_b = st.columns([1.4, 1])

        with col_a:
            st.markdown("#### Résumé de monitoring")
            if summary:
                st.json(summary)
            else:
                st.info("Aucun résumé global disponible.")

            st.markdown("#### Santé de monitoring")
            if health:
                st.json(health)
            else:
                st.info("Aucun état de santé monitoring disponible.")

        with col_b:
            st.markdown("#### Dernière évaluation")
            if latest_eval_row is not None:
                st.write(f"**Dataset** : {latest_eval_row.get('dataset_name', 'N/A')}")
                st.write(f"**Recall** : {_safe_metric_value(latest_eval_row.get('recall_score'))}")
                st.write(f"**Precision** : {_safe_metric_value(latest_eval_row.get('precision_score'))}")
                st.write(f"**F1** : {_safe_metric_value(latest_eval_row.get('f1_score'))}")
                st.write(f"**Fbeta** : {_safe_metric_value(latest_eval_row.get('fbeta_score'))}")
                st.write(f"**ROC AUC** : {_safe_metric_value(latest_eval_row.get('roc_auc'))}")
                st.write(f"**Business cost** : {_safe_metric_value(latest_eval_row.get('business_cost'), 2)}")
                st.write(f"**Taille échantillon** : {_safe_int(latest_eval_row.get('sample_size'))}")
            elif latest_eval_summary:
                st.write(f"**Dataset** : {latest_eval_summary.get('dataset_name', 'N/A')}")
                st.write(f"**Recall** : {_safe_metric_value(latest_eval_summary.get('recall_score'))}")
                st.write(f"**Precision** : {_safe_metric_value(latest_eval_summary.get('precision_score'))}")
                st.write(f"**F1** : {_safe_metric_value(latest_eval_summary.get('f1_score'))}")
                st.write(f"**Business cost** : {_safe_metric_value(latest_eval_summary.get('business_cost'), 2)}")
            else:
                st.info("Aucune synthèse d'évaluation disponible.")

    # =====================================================================
    # ONGLET 2 - PERFORMANCE
    # =====================================================================
    with tabs[1]:
        _render_section_title(
            "Performance du modèle",
            "Évolution des métriques métier et comportement de latence.",
        )

        if evaluation_metrics_df.empty:
            st.info("Aucune métrique disponible via `/monitoring/evaluation`.")
        else:
            latest_eval_df = evaluation_metrics_df.copy()
            if "computed_at" in latest_eval_df.columns:
                latest_eval_df = latest_eval_df.sort_values("computed_at", ascending=False)
            latest_eval_df = latest_eval_df.head(1)

            if not latest_eval_df.empty:
                row = latest_eval_df.iloc[0]

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("ROC AUC", _safe_metric_value(row.get("roc_auc"), 3))
                m2.metric("PR AUC", _safe_metric_value(row.get("pr_auc"), 3))
                m3.metric("Precision", _safe_metric_value(row.get("precision_score"), 3))
                m4.metric("Recall", _safe_metric_value(row.get("recall_score"), 3))
                m5.metric("Fbeta", _safe_metric_value(row.get("fbeta_score"), 3))

            metric_options = [
                c for c in [
                    "roc_auc",
                    "pr_auc",
                    "precision_score",
                    "recall_score",
                    "f1_score",
                    "fbeta_score",
                    "business_cost",
                    "sample_size",
                ]
                if c in evaluation_metrics_df.columns
            ]

            if metric_options:
                selected_metric = st.selectbox(
                    "Métrique à tracer",
                    options=metric_options,
                    key="monitoring_metric_selector",
                )

                plot_df = evaluation_metrics_df.copy()

                if selected_metric in plot_df.columns:
                    plot_df = plot_df.dropna(subset=[selected_metric])

                if "computed_at" in plot_df.columns:
                    plot_df = plot_df.sort_values("computed_at")

                if not plot_df.empty and "computed_at" in plot_df.columns:
                    st.line_chart(plot_df.set_index("computed_at")[[selected_metric]])
                elif not plot_df.empty:
                    st.dataframe(plot_df[[selected_metric]], width="stretch")

            with st.expander("Voir le détail complet des métriques"):
                preferred_cols = _choose_existing_columns(
                    evaluation_metrics_df,
                    [
                        "computed_at",
                        "dataset_name",
                        "model_name",
                        "model_version",
                        "roc_auc",
                        "pr_auc",
                        "precision_score",
                        "recall_score",
                        "f1_score",
                        "fbeta_score",
                        "business_cost",
                        "tn",
                        "fp",
                        "fn",
                        "tp",
                        "sample_size",
                    ],
                )
                st.dataframe(
                    evaluation_metrics_df[preferred_cols] if preferred_cols else evaluation_metrics_df,
                    width="stretch",
                )

        st.markdown("")
        st.markdown("#### Latence d'inférence")

        if prediction_logs_df.empty:
            st.info("Aucune donnée de prédiction disponible via `/history/predictions`.")
        elif "prediction_timestamp" not in prediction_logs_df.columns:
            st.info("La colonne `prediction_timestamp` est absente des données reçues.")
        elif "latency_ms" not in prediction_logs_df.columns:
            st.info("La colonne `latency_ms` est absente des données reçues.")
        else:
            latency_df = (
                prediction_logs_df
                .dropna(subset=["prediction_timestamp"])
                .sort_values("prediction_timestamp")
            )

            if not latency_df.empty:
                st.line_chart(
                    latency_df.set_index("prediction_timestamp")[["latency_ms"]]
                )

                l1, l2, l3 = st.columns(3)
                l1.metric(
                    "Latence moyenne",
                    round(_safe_float(metric_safe_number(latency_df, "latency_ms", "mean", 0), 0.0), 2),
                )
                l2.metric(
                    "Latence p95",
                    round(_safe_float(metric_safe_number(latency_df, "latency_ms", "p95", 0), 0.0), 2),
                )
                l3.metric(
                    "Latence p99",
                    round(_safe_float(metric_safe_number(latency_df, "latency_ms", "p99", 0), 0.0), 2),
                )

            with st.expander("Voir les logs de latence"):
                preferred_cols = _choose_existing_columns(
                    latency_df,
                    [
                        "prediction_timestamp",
                        "request_id",
                        "client_id",
                        "model_name",
                        "model_version",
                        "score",
                        "prediction",
                        "threshold",
                        "latency_ms",
                        "status",
                        "error_message",
                    ],
                )
                st.dataframe(
                    latency_df[preferred_cols] if preferred_cols else latency_df,
                    width="stretch",
                )

    # =====================================================================
    # ONGLET 3 - DÉRIVE
    # =====================================================================
    with tabs[2]:
        _render_section_title(
            "Dérive des données",
            "Détection des signaux de changement dans les features de production.",
        )

        if drift_metrics_df.empty:
            st.info("Aucune donnée disponible via `/monitoring/drift`.")
        else:
            d1, d2, d3 = st.columns(3)

            detected_count = 0
            if "drift_detected" in drift_metrics_df.columns:
                detected_count = int(
                    drift_metrics_df["drift_detected"].apply(_safe_bool).sum()
                )

            d1.metric("Lignes drift", len(drift_metrics_df))
            d2.metric("Drifts détectés", detected_count)
            d3.metric(
                "Features monitorées",
                drift_metrics_df["feature_name"].nunique()
                if "feature_name" in drift_metrics_df.columns
                else 0,
            )

            filter_col1, filter_col2 = st.columns([1, 1])

            with filter_col1:
                only_drift = st.toggle(
                    "Uniquement les drifts détectés",
                    value=True,
                    key="toggle_only_detected_drift",
                )

            with filter_col2:
                metric_names = (
                    sorted(
                        drift_metrics_df["metric_name"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if "metric_name" in drift_metrics_df.columns
                    else []
                )
                selected_metric_name = st.selectbox(
                    "Métrique de drift",
                    options=["Toutes"] + metric_names,
                    key="drift_metric_name_selector",
                )

            drift_view = drift_metrics_df.copy()

            if only_drift and "drift_detected" in drift_view.columns:
                drift_view = drift_view[drift_view["drift_detected"].apply(_safe_bool)]

            if selected_metric_name != "Toutes" and "metric_name" in drift_view.columns:
                drift_view = drift_view[
                    drift_view["metric_name"].astype(str) == selected_metric_name
                ]

            if not drift_view.empty and "feature_name" in drift_view.columns:
                top_features = drift_view["feature_name"].value_counts().head(15)
                st.markdown("#### Features les plus souvent remontées")
                st.bar_chart(top_features)

            if (
                not drift_view.empty
                and "computed_at" in drift_view.columns
                and "metric_value" in drift_view.columns
            ):
                timeline_df = (
                    drift_view
                    .dropna(subset=["computed_at", "metric_value"])
                    .sort_values("computed_at")
                )
                if not timeline_df.empty:
                    grouped = (
                        timeline_df.groupby("computed_at", as_index=True)["metric_value"]
                        .mean()
                        .to_frame("metric_value_mean")
                    )
                    st.markdown("#### Évolution moyenne du score de drift")
                    st.line_chart(grouped)

            with st.expander(
                "Voir le détail complet des métriques de drift",
                expanded=True,
            ):
                preferred_cols = _choose_existing_columns(
                    drift_view,
                    [
                        "computed_at",
                        "model_name",
                        "model_version",
                        "feature_name",
                        "metric_name",
                        "metric_value",
                        "threshold_value",
                        "drift_detected",
                        "reference_window_start",
                        "reference_window_end",
                        "current_window_start",
                        "current_window_end",
                        "details",
                    ],
                )
                st.dataframe(
                    drift_view[preferred_cols] if preferred_cols else drift_view,
                    width="stretch",
                )

        st.markdown("")
        st.markdown("#### Exécution Evidently")

        default_model_name = "credit_scoring_model"
        default_model_version = None

        if latest_active_model_row is not None:
            model_name_value = latest_active_model_row.get("model_name")
            if model_name_value is not None and pd.notna(model_name_value):
                default_model_name = str(model_name_value)

            model_version_value = latest_active_model_row.get("model_version")
            if model_version_value is not None and pd.notna(model_version_value):
                default_model_version = str(model_version_value)

        run_col1, run_col2, run_col3 = st.columns([1, 1, 1])

        with run_col1:
            evidently_reference_kind = st.selectbox(
                "Reference kind",
                options=["transformed", "raw"],
                index=0,
                key="evidently_reference_kind",
            )

        with run_col2:
            evidently_current_kind = st.selectbox(
                "Current kind",
                options=["transformed", "raw"],
                index=0,
                key="evidently_current_kind",
            )

        with run_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_evidently_clicked = st.button(
                "Lancer Evidently",
                type="primary",
                use_container_width=True,
            )

        if run_evidently_clicked:
            if evidently_reference_kind != evidently_current_kind:
                st.error(
                    "reference_kind et current_kind doivent être identiques."
                )
            else:
                with st.spinner("Analyse Evidently en cours..."):
                    ok, result = run_evidently_analysis(
                        model_name=default_model_name,
                        model_version=default_model_version,
                        reference_kind=evidently_reference_kind,
                        current_kind=evidently_current_kind,
                        save_html_path="artifacts/evidently/report.html",
                    )

                if ok:
                    if isinstance(result, dict) and result.get("success", False):
                        st.success(
                            result.get(
                                "message",
                                "Analyse Evidently exécutée avec succès.",
                            )
                        )

                        html_report_path = result.get("html_report_path")
                        if html_report_path:
                            st.info(f"Rapport HTML sauvegardé : {html_report_path}")

                        analyzed_columns = result.get("analyzed_columns")
                        if isinstance(analyzed_columns, list):
                            st.caption(f"Colonnes analysées : {len(analyzed_columns)}")

                        with st.expander("Voir le résultat Evidently", expanded=False):
                            st.json(result)

                        st.cache_data.clear()
                        st.rerun()

                    else:
                        detail = (
                            result.get("message")
                            if isinstance(result, dict)
                            else result
                        )
                        st.error(f"Analyse Evidently non réussie : {detail}")
                else:
                    detail = (
                        result.get("detail")
                        if isinstance(result, dict)
                        else result
                    )
                    st.error(f"Erreur API Evidently : {detail}")

        st.markdown("")
        st.markdown("#### Feature store monitoring")

        if feature_store_monitoring_df.empty:
            st.info("Aucune donnée disponible via `/monitoring/feature-store`.")
        else:
            fs1, fs2, fs3 = st.columns(3)
            fs1.metric("Snapshots", len(feature_store_monitoring_df))
            fs2.metric(
                "Request IDs",
                feature_store_monitoring_df["request_id"].nunique()
                if "request_id" in feature_store_monitoring_df.columns
                else 0,
            )
            fs3.metric(
                "Features distinctes",
                feature_store_monitoring_df["feature_name"].nunique()
                if "feature_name" in feature_store_monitoring_df.columns
                else 0,
            )

            vis_col1, vis_col2 = st.columns(2)

            with vis_col1:
                if "feature_name" in feature_store_monitoring_df.columns:
                    st.markdown("##### Top features observées")
                    top_fs_features = (
                        feature_store_monitoring_df["feature_name"]
                        .fillna("unknown")
                        .value_counts()
                        .head(15)
                    )
                    st.bar_chart(top_fs_features)

            with vis_col2:
                if "source_table" in feature_store_monitoring_df.columns:
                    st.markdown("##### Répartition par source_table")
                    source_counts = (
                        feature_store_monitoring_df["source_table"]
                        .fillna("unknown")
                        .value_counts()
                        .head(15)
                    )
                    st.bar_chart(source_counts)

            with st.expander("Voir le feature store", expanded=False):
                preferred_cols = _choose_existing_columns(
                    feature_store_monitoring_df,
                    [
                        "snapshot_timestamp",
                        "request_id",
                        "client_id",
                        "model_name",
                        "model_version",
                        "feature_name",
                        "feature_value",
                        "feature_type",
                        "source_table",
                    ],
                )
                st.dataframe(
                    feature_store_monitoring_df[preferred_cols]
                    if preferred_cols else feature_store_monitoring_df,
                    width="stretch",
                )

    # =====================================================================
    # ONGLET 4 - ALERTES ET MODÈLES
    # =====================================================================
    with tabs[3]:
        _render_section_title(
            "Alertes et registre des modèles",
            "Lecture des incidents de monitoring et du cycle de vie des versions.",
        )

        st.markdown("#### Alertes")

        if alerts_df.empty:
            st.info("Aucune alerte disponible via `/monitoring/alerts`.")
        else:
            a1, a2, a3 = st.columns(3)

            a1.metric("Total alertes", len(alerts_df))
            a2.metric(
                "Alertes ouvertes",
                int(
                    alerts_df["status"]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .eq("open")
                    .sum()
                )
                if "status" in alerts_df.columns
                else 0,
            )
            a3.metric(
                "Sévérités",
                alerts_df["severity"].nunique()
                if "severity" in alerts_df.columns
                else 0,
            )

            if "severity" in alerts_df.columns:
                severity_counts = alerts_df["severity"].fillna("unknown").value_counts()
                st.bar_chart(severity_counts)

            with st.expander("Voir les alertes", expanded=True):
                preferred_cols = _choose_existing_columns(
                    alerts_df,
                    [
                        "created_at",
                        "alert_type",
                        "severity",
                        "status",
                        "model_name",
                        "model_version",
                        "feature_name",
                        "title",
                        "message",
                        "acknowledged_at",
                        "resolved_at",
                        "context",
                    ],
                )
                st.dataframe(
                    alerts_df[preferred_cols] if preferred_cols else alerts_df,
                    width="stretch",
                )

        st.markdown("")
        st.markdown("#### Registre des modèles")

        if model_registry_df.empty:
            st.info("Aucune donnée disponible via `/monitoring/models`.")
        else:
            registry_view = model_registry_df.copy()

            if "deployed_at" in registry_view.columns:
                registry_view = registry_view.sort_values(
                    "deployed_at",
                    ascending=False,
                    na_position="last",
                )
            elif "created_at" in registry_view.columns:
                registry_view = registry_view.sort_values(
                    "created_at",
                    ascending=False,
                    na_position="last",
                )

            with st.expander("Voir le registre des modèles", expanded=True):
                preferred_cols = _choose_existing_columns(
                    registry_view,
                    [
                        "model_name",
                        "model_version",
                        "stage",
                        "is_active",
                        "run_id",
                        "training_data_version",
                        "deployed_at",
                        "created_at",
                        "source_path",
                        "metrics",
                        "hyperparameters",
                    ],
                )
                st.dataframe(
                    registry_view[preferred_cols] if preferred_cols else registry_view,
                    width="stretch",
                )

        st.markdown("")
        st.markdown("#### Modèle actif")

        if active_model_df.empty:
            st.warning("Aucun modèle actif disponible via `/monitoring/active-model`.")
        else:
            row = active_model_df.iloc[0]
            st.success(
                f"Modèle actif : {row.get('model_name', 'N/A')} | "
                f"version {row.get('model_version', 'N/A')} | "
                f"stage {row.get('stage', 'N/A')}"
            )

            preferred_cols = _choose_existing_columns(
                active_model_df,
                [
                    "model_name",
                    "model_version",
                    "stage",
                    "is_active",
                    "run_id",
                    "training_data_version",
                    "deployed_at",
                    "created_at",
                    "source_path",
                    "metrics",
                    "hyperparameters",
                ],
            )
            st.dataframe(
                active_model_df[preferred_cols] if preferred_cols else active_model_df,
                width="stretch",
            )