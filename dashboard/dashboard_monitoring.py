"""
Page Streamlit : monitoring MLOps.

Cette page regroupe :
- métriques d'évaluation
- analyse de latence
- data drift
- alertes
- registre des modèles
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_monitoring_page(
    *,
    prediction_logs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    model_registry_df: pd.DataFrame,
    active_model_df: pd.DataFrame,
    evaluation_metrics_df: pd.DataFrame,
    drift_metrics_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    feature_store_monitoring_df: pd.DataFrame,
    metric_safe_number,
) -> None:
    """
    Affiche la page monitoring.
    """
    st.markdown("### Monitoring du modèle")

    monitor_tabs = st.tabs(["📈 Métriques", "⏱️ Latence", "🧪 Data Drift", "🚨 Alertes"])

    with monitor_tabs[0]:
        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Prédictions", len(prediction_logs_df))
        top2.metric("Vérités terrain", len(ground_truth_df))
        top3.metric("Versions de modèle", len(model_registry_df))
        top4.metric("Évaluations", len(evaluation_metrics_df))

        st.markdown("### Métriques ML")

        if evaluation_metrics_df.empty:
            st.info("Aucune métrique disponible dans `evaluation_metrics`.")
        else:
            latest_eval = evaluation_metrics_df.sort_values("computed_at", ascending=False).head(1)

            if not latest_eval.empty:
                row = latest_eval.iloc[0]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("ROC AUC", round(float(row.get("roc_auc", 0) or 0), 3))
                c2.metric("PR AUC", round(float(row.get("pr_auc", 0) or 0), 3))
                c3.metric("Precision", round(float(row.get("precision_score", 0) or 0), 3))
                c4.metric("Recall", round(float(row.get("recall_score", 0) or 0), 3))
                c5.metric("Fbeta", round(float(row.get("fbeta_score", 0) or 0), 3))

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
                selected_metric = st.selectbox("Tracer une métrique", options=metric_options)
                plot_df = evaluation_metrics_df.dropna(subset=[selected_metric]).sort_values("computed_at")

                if not plot_df.empty:
                    st.line_chart(plot_df.set_index("computed_at")[[selected_metric]])

            st.dataframe(evaluation_metrics_df, use_container_width=True)

    with monitor_tabs[1]:
        st.markdown("### Analyse de latence")

        if prediction_logs_df.empty or "prediction_timestamp" not in prediction_logs_df.columns:
            st.info("Aucune donnée suffisante dans `prediction_logs`.")
        else:
            latency_df = prediction_logs_df.dropna(subset=["prediction_timestamp"]).sort_values("prediction_timestamp")

            if not latency_df.empty and "latency_ms" in latency_df.columns:
                st.line_chart(latency_df.set_index("prediction_timestamp")[["latency_ms"]])

                l1, l2, l3 = st.columns(3)
                l1.metric("Latence moyenne", round(metric_safe_number(latency_df, "latency_ms", "mean"), 2))
                l2.metric("Latence p95", round(metric_safe_number(latency_df, "latency_ms", "p95"), 2))
                l3.metric("Latence p99", round(metric_safe_number(latency_df, "latency_ms", "p99"), 2))

            st.dataframe(latency_df, use_container_width=True)

    with monitor_tabs[2]:
        st.markdown("### Data Drift")

        if drift_metrics_df.empty:
            st.info("Aucune donnée dans `drift_metrics`.")
        else:
            d1, d2, d3 = st.columns(3)

            detected_count = 0
            if "drift_detected" in drift_metrics_df.columns:
                detected_count = int(drift_metrics_df["drift_detected"].fillna(False).sum())

            d1.metric("Lignes drift", len(drift_metrics_df))
            d2.metric("Drifts détectés", detected_count)
            d3.metric(
                "Features monitorées",
                drift_metrics_df["feature_name"].nunique() if "feature_name" in drift_metrics_df.columns else 0,
            )

            only_drift = st.toggle("Afficher uniquement les drifts détectés", value=True)

            drift_view = drift_metrics_df.copy()
            if only_drift and "drift_detected" in drift_view.columns:
                drift_view = drift_view[drift_view["drift_detected"] == True]

            if not drift_view.empty and "feature_name" in drift_view.columns:
                top_features = drift_view["feature_name"].value_counts().head(15)
                st.bar_chart(top_features)

            st.dataframe(drift_view, use_container_width=True)

            with st.expander("Feature store monitoring"):
                if feature_store_monitoring_df.empty:
                    st.info("Aucune donnée dans `feature_store_monitoring`.")
                else:
                    st.dataframe(feature_store_monitoring_df, use_container_width=True)

    with monitor_tabs[3]:
        st.markdown("### Alertes")

        if alerts_df.empty:
            st.info("Aucune alerte dans `alerts`.")
        else:
            a1, a2, a3 = st.columns(3)

            a1.metric("Total alertes", len(alerts_df))
            a2.metric(
                "Alertes ouvertes",
                int((alerts_df["status"].fillna("") == "open").sum()) if "status" in alerts_df.columns else 0,
            )
            a3.metric(
                "Sévérités",
                alerts_df["severity"].nunique() if "severity" in alerts_df.columns else 0,
            )

            if "severity" in alerts_df.columns:
                severity_counts = alerts_df["severity"].fillna("unknown").value_counts()
                st.bar_chart(severity_counts)

            st.dataframe(alerts_df, use_container_width=True)

        st.markdown("### Registre des modèles")
        if model_registry_df.empty:
            st.info("Aucune donnée dans `model_registry`.")
        else:
            st.dataframe(model_registry_df, use_container_width=True)

            if not active_model_df.empty:
                row = active_model_df.iloc[0]
                st.success(
                    f"Modèle actif : {row['model_name']} | version {row['model_version']} | stage {row['stage']}"
                )