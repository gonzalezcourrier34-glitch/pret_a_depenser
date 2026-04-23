"""
Page Streamlit : monitoring MLOps.

Cette page regroupe :
- une vue d'ensemble du système de monitoring
- les métriques de performance du modèle
- l'analyse de dérive des données
- les alertes et le registre des modèles
- le déclenchement manuel des analyses avancées

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
    3. afficher les visualisations,
    4. déclencher certaines analyses via l'API.
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


def _coerce_columns_to_datetime(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """
    Convertit plusieurs colonnes en datetime si elles existent.
    """
    out = df.copy()
    for col in columns:
        out = _safe_timestamp_series(out, col)
    return out


def _coerce_columns_to_numeric(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
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


def _extract_drifted_columns_count(details: Any) -> float | None:
    """
    Extrait le nombre de colonnes en drift depuis la colonne `details`
    si le format correspond à un dictionnaire.
    """
    if not isinstance(details, dict):
        return None

    for key in [
        "number_of_drifted_columns",
        "drifted_columns",
        "n_drifted_columns",
    ]:
        value = details.get(key)
        try:
            if value is not None and not pd.isna(value):
                return float(value)
        except Exception:
            continue

    return None


def _resolve_default_model_context(
    latest_active_model_row: pd.Series | None,
    latest_registry_row: pd.Series | None,
) -> tuple[str | None, str | None]:
    """
    Résout le modèle et la version par défaut à utiliser pour lancer
    les analyses depuis la page monitoring.
    """
    default_model_name: str | None = None
    default_model_version: str | None = None

    source_row = latest_active_model_row if latest_active_model_row is not None else latest_registry_row

    if source_row is not None:
        model_name_value = source_row.get("model_name")
        if model_name_value is not None and pd.notna(model_name_value):
            default_model_name = str(model_name_value)

        model_version_value = source_row.get("model_version")
        if model_version_value is not None and pd.notna(model_version_value):
            default_model_version = str(model_version_value)

    return default_model_name, default_model_version


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
    run_monitoring_evaluation_analysis,
) -> None:
    """
    Affiche la page de monitoring du modèle.

    Parameters
    ----------
    monitoring_summary : dict
        Résumé global de monitoring renvoyé par l'API.
    monitoring_health : dict
        État synthétique de santé du monitoring.
    prediction_logs_df : pd.DataFrame
        Historique des prédictions.
    model_registry_df : pd.DataFrame
        Registre des modèles.
    active_model_df : pd.DataFrame
        Modèle actuellement actif.
    evaluation_metrics_df : pd.DataFrame
        Historique des métriques d'évaluation.
    drift_metrics_df : pd.DataFrame
        Historique des métriques de drift.
    alerts_df : pd.DataFrame
        Historique des alertes.
    feature_store_monitoring_df : pd.DataFrame
        Feature store de monitoring.
    metric_safe_number : callable
        Helper partagé pour calculer des métriques numériques de façon sûre.
    run_evidently_analysis : callable
        Fonction déclenchant l'analyse Evidently via l'API.
    run_monitoring_evaluation_analysis : callable
        Fonction déclenchant l'évaluation monitoring via l'API.
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

    # -------------------------------------------------------------------------
    # Normalisation des colonnes
    # -------------------------------------------------------------------------
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
            "metric_value",
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
        ["prediction_timestamp", "created_at"],
    )
    prediction_logs_df = _coerce_columns_to_numeric(
        prediction_logs_df,
        ["score", "threshold", "threshold_used", "latency_ms", "prediction"],
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
    # KPI globaux
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

    default_model_name, default_model_version = _resolve_default_model_context(
        latest_active_model_row=latest_active_model_row,
        latest_registry_row=latest_registry_row,
    )

    # -------------------------------------------------------------------------
    # En-tête page
    # -------------------------------------------------------------------------
    st.markdown("## Monitoring MLOps")
    st.caption(
        "Centre de pilotage du modèle en production : performance, dérive, alertes, latence et cycle de vie des versions."
    )

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

    tabs = st.tabs(
        [
            "Vue d'ensemble",
            "Performance",
            "Dérive",
            "Alertes & Modèles",
        ]
    )

    # =========================================================================
    # Onglet 1 - Vue d'ensemble
    # =========================================================================
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

    # =========================================================================
    # Onglet 2 - Performance
    # =========================================================================
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
        st.markdown("#### Exécution évaluation monitoring")

        eval_col1, eval_col2, eval_col3, eval_col4 = st.columns([1, 1, 1, 1])

        with eval_col1:
            evaluation_dataset_name = st.text_input(
                "Dataset name",
                value="scoring_prod",
                key="monitoring_eval_dataset_name",
            )

        with eval_col2:
            evaluation_beta = st.number_input(
                "Beta",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="monitoring_eval_beta",
            )

        with eval_col3:
            evaluation_cost_fn = st.number_input(
                "Coût FN",
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="monitoring_eval_cost_fn",
            )

        with eval_col4:
            evaluation_cost_fp = st.number_input(
                "Coût FP",
                min_value=0.0,
                value=1.0,
                step=1.0,
                key="monitoring_eval_cost_fp",
            )

        run_eval_clicked = st.button(
            "Lancer l'évaluation monitoring",
            type="primary",
            use_container_width=True,
            key="run_monitoring_evaluation_button",
        )

        if run_eval_clicked:
            if not default_model_name:
                st.error("Impossible de lancer l'évaluation monitoring : aucun modèle actif ou enregistré n'a été trouvé.")
            else:
                with st.spinner("Évaluation monitoring en cours..."):
                    ok, result = run_monitoring_evaluation_analysis(
                        model_name=default_model_name,
                        model_version=default_model_version,
                        dataset_name=evaluation_dataset_name,
                        beta=float(evaluation_beta),
                        cost_fn=float(evaluation_cost_fn),
                        cost_fp=float(evaluation_cost_fp),
                    )

                if ok:
                    if isinstance(result, dict) and result.get("success", False):
                        st.success(
                            result.get(
                                "message",
                                "Évaluation monitoring exécutée avec succès.",
                            )
                        )

                        sample_size = result.get("sample_size")
                        matched_rows = result.get("matched_rows")
                        threshold_used = result.get("threshold_used")

                        info_parts = []
                        if sample_size is not None:
                            info_parts.append(f"sample_size={sample_size}")
                        if matched_rows is not None:
                            info_parts.append(f"matched_rows={matched_rows}")
                        if threshold_used is not None:
                            info_parts.append(f"threshold_used={threshold_used}")

                        if info_parts:
                            st.caption(" | ".join(info_parts))

                        with st.expander("Voir le résultat de l'évaluation", expanded=False):
                            st.json(result)

                        st.cache_data.clear()
                        st.rerun()
                    else:
                        detail = (
                            result.get("message")
                            if isinstance(result, dict)
                            else result
                        )
                        st.error(f"Évaluation monitoring non réussie : {detail}")
                else:
                    detail = (
                        result.get("detail")
                        if isinstance(result, dict)
                        else result
                    )
                    st.error(f"Erreur API évaluation monitoring : {detail}")

        st.markdown("")
        st.markdown("#### Latence d'inférence")

        if prediction_logs_df.empty:
            st.info("Aucune donnée de prédiction disponible via `/history/predictions`.")
        elif "prediction_timestamp" not in prediction_logs_df.columns and "created_at" not in prediction_logs_df.columns:
            st.info("Aucune colonne temporelle exploitable n'est présente dans les logs de prédiction.")
        elif "latency_ms" not in prediction_logs_df.columns:
            st.info("La colonne `latency_ms` est absente des données reçues.")
        else:
            latency_time_col = (
                "prediction_timestamp"
                if "prediction_timestamp" in prediction_logs_df.columns
                else "created_at"
            )

            latency_df = (
                prediction_logs_df
                .dropna(subset=[latency_time_col])
                .sort_values(latency_time_col)
            )

            if not latency_df.empty:
                st.line_chart(
                    latency_df.set_index(latency_time_col)[["latency_ms"]]
                )

                latency_mean = round(
                    _safe_float(metric_safe_number(latency_df, "latency_ms", "mean", 0), 0.0),
                    2,
                )
                latency_p95 = round(
                    _safe_float(metric_safe_number(latency_df, "latency_ms", "p95", 0), 0.0),
                    2,
                )
                latency_p99 = round(
                    _safe_float(metric_safe_number(latency_df, "latency_ms", "p99", 0), 0.0),
                    2,
                )

                l1, l2, l3 = st.columns(3)
                l1.metric("Latence moyenne", f"{latency_mean} ms")
                l2.metric("Latence p95", f"{latency_p95} ms")
                l3.metric("Latence p99", f"{latency_p99} ms")

            with st.expander("Voir les logs de latence"):
                preferred_cols = _choose_existing_columns(
                    latency_df,
                    [
                        latency_time_col,
                        "request_id",
                        "client_id",
                        "model_name",
                        "model_version",
                        "score",
                        "prediction",
                        "threshold",
                        "threshold_used",
                        "latency_ms",
                        "status",
                        "error_message",
                    ],
                )
                st.dataframe(
                    latency_df[preferred_cols] if preferred_cols else latency_df,
                    width="stretch",
                )

    # =========================================================================
    # Onglet 3 - Dérive
    # =========================================================================
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

            st.markdown("")
            st.markdown("#### Visualisations de drift")

            drift_graph_df = drift_metrics_df.copy()

            if "computed_at" in drift_graph_df.columns:
                drift_graph_df = drift_graph_df.dropna(subset=["computed_at"])

            if drift_graph_df.empty:
                st.info("Les métriques de drift ne contiennent pas de date exploitable.")
            else:
                drift_graph_df = drift_graph_df.sort_values("computed_at")

                dataset_level_df = drift_graph_df.copy()
                if "feature_name" in dataset_level_df.columns:
                    dataset_level_df = dataset_level_df[
                        dataset_level_df["feature_name"].astype(str).eq("__dataset__")
                    ]

                share_plot_ready = pd.DataFrame()
                drifted_count_plot_ready = pd.DataFrame()

                if not dataset_level_df.empty:
                    if "metric_name" in dataset_level_df.columns and "metric_value" in dataset_level_df.columns:
                        share_candidates = dataset_level_df[
                            dataset_level_df["metric_name"]
                            .astype(str)
                            .isin(
                                [
                                    "share_of_drifted_columns",
                                    "dataset_drift_share",
                                    "drift_share",
                                ]
                            )
                        ].copy()

                        if not share_candidates.empty:
                            share_plot_ready = (
                                share_candidates
                                .groupby("computed_at", as_index=True)["metric_value"]
                                .mean()
                                .to_frame("share_of_drifted_columns")
                            )

                    if "details" in dataset_level_df.columns:
                        dataset_details_df = dataset_level_df.copy()
                        dataset_details_df["number_of_drifted_columns"] = dataset_details_df["details"].apply(
                            _extract_drifted_columns_count
                        )
                        dataset_details_df = dataset_details_df.dropna(
                            subset=["number_of_drifted_columns"]
                        )

                        if not dataset_details_df.empty:
                            drifted_count_plot_ready = (
                                dataset_details_df
                                .groupby("computed_at", as_index=True)["number_of_drifted_columns"]
                                .mean()
                                .to_frame("number_of_drifted_columns")
                            )

                g1, g2 = st.columns(2)

                with g1:
                    st.markdown("##### Part des colonnes en dérive dans le temps")
                    if not share_plot_ready.empty:
                        st.line_chart(share_plot_ready)
                    else:
                        st.info(
                            "Aucune métrique `share_of_drifted_columns` disponible pour l'instant."
                        )

                with g2:
                    st.markdown("##### Nombre de colonnes en dérive dans le temps")
                    if not drifted_count_plot_ready.empty:
                        st.line_chart(drifted_count_plot_ready)
                    else:
                        st.info(
                            "Aucun nombre de colonnes en dérive disponible pour l'instant."
                        )

                per_feature_df = drift_graph_df.copy()
                if "feature_name" in per_feature_df.columns:
                    per_feature_df = per_feature_df[
                        ~per_feature_df["feature_name"].astype(str).eq("__dataset__")
                    ]

                if not per_feature_df.empty and "drift_detected" in per_feature_df.columns:
                    drift_only_features_df = per_feature_df[
                        per_feature_df["drift_detected"].apply(_safe_bool)
                    ].copy()

                    st.markdown("##### Features les plus souvent en dérive")

                    if not drift_only_features_df.empty and "feature_name" in drift_only_features_df.columns:
                        top_drift_features = (
                            drift_only_features_df["feature_name"]
                            .fillna("unknown")
                            .astype(str)
                            .value_counts()
                            .head(15)
                        )
                        st.bar_chart(top_drift_features)
                    else:
                        st.info("Aucune feature marquée en dérive pour l'instant.")

                if (
                    not per_feature_df.empty
                    and "computed_at" in per_feature_df.columns
                    and "metric_value" in per_feature_df.columns
                    and "feature_name" in per_feature_df.columns
                ):
                    latest_computed_at = per_feature_df["computed_at"].max()

                    latest_feature_drift_df = per_feature_df[
                        per_feature_df["computed_at"] == latest_computed_at
                    ].copy()

                    latest_feature_drift_df = latest_feature_drift_df.dropna(
                        subset=["metric_value"]
                    )

                    if not latest_feature_drift_df.empty:
                        latest_feature_drift_df["feature_name"] = (
                            latest_feature_drift_df["feature_name"]
                            .fillna("unknown")
                            .astype(str)
                        )

                        top_latest_feature_scores = (
                            latest_feature_drift_df
                            .groupby("feature_name", as_index=True)["metric_value"]
                            .mean()
                            .sort_values(ascending=False)
                            .head(15)
                            .to_frame("metric_value")
                        )

                        st.markdown("##### Top scores de drift par feature sur la dernière exécution")
                        st.bar_chart(top_latest_feature_scores)

                if (
                    not per_feature_df.empty
                    and "computed_at" in per_feature_df.columns
                    and "metric_value" in per_feature_df.columns
                ):
                    mean_drift_timeline_df = (
                        per_feature_df
                        .dropna(subset=["computed_at", "metric_value"])
                        .groupby("computed_at", as_index=True)["metric_value"]
                        .mean()
                        .to_frame("mean_feature_drift_score")
                    )

                    if not mean_drift_timeline_df.empty:
                        st.markdown("##### Évolution du score moyen de drift")
                        st.line_chart(mean_drift_timeline_df)

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

        run_col1, run_col2, run_col3, run_col4 = st.columns([1, 1, 1, 1])

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
            evidently_max_rows = st.number_input(
                "Max rows",
                min_value=1000,
                max_value=100000,
                value=20000,
                step=1000,
                key="evidently_max_rows",
            )

        with run_col4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_evidently_clicked = st.button(
                "Lancer Evidently",
                type="primary",
                use_container_width=True,
                key="run_evidently_button",
            )

        if run_evidently_clicked:
            if not default_model_name:
                st.error("Impossible de lancer Evidently : aucun modèle actif ou enregistré n'a été trouvé.")
            elif evidently_reference_kind != evidently_current_kind:
                st.error("reference_kind et current_kind doivent être identiques.")
            else:
                with st.spinner("Analyse Evidently en cours..."):
                    ok, result = run_evidently_analysis(
                        model_name=default_model_name,
                        model_version=default_model_version,
                        reference_kind=evidently_reference_kind,
                        current_kind=evidently_current_kind,
                        max_rows=int(evidently_max_rows),
                    )

                if ok:
                    if isinstance(result, dict) and result.get("success", False):
                        st.success(
                            result.get(
                                "message",
                                "Analyse Evidently exécutée avec succès.",
                            )
                        )

                        analyzed_columns = result.get("analyzed_columns")
                        if isinstance(analyzed_columns, list):
                            st.caption(f"Colonnes analysées : {len(analyzed_columns)}")

                        reference_rows = result.get("reference_rows")
                        current_rows = result.get("current_rows")
                        if reference_rows is not None or current_rows is not None:
                            st.caption(
                                f"reference_rows={reference_rows} | current_rows={current_rows}"
                            )

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

    # =========================================================================
    # Onglet 4 - Alertes & Modèles
    # =========================================================================
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

        active_row = _pick_latest_row(active_model_df, ["deployed_at", "created_at"])

        if active_row is None:
            st.warning("Aucun modèle actif disponible via `/monitoring/active-model`.")
        else:
            st.success(
                f"Modèle actif : {active_row.get('model_name', 'N/A')} | "
                f"version {active_row.get('model_version', 'N/A')} | "
                f"stage {active_row.get('stage', 'N/A')}"
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