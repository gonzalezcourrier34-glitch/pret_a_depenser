"""
Page Streamlit : prédictions et traçabilité.

Cette page regroupe :
- la prédiction via un client chargé en base
- la prédiction via un JSON libre
- la consultation de l'historique des prédictions
- les snapshots de features et vérités terrain
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st


def render_predictions_page(
    *,
    ALLOWED_FEATURES_TABLES: set[str],
    prediction_logs_df: pd.DataFrame,
    load_client_features,
    dataframe_to_payload,
    call_predict_api,
    metric_safe_number,
    query_if_table_exists,
    PREDICTION_FEATURES_BY_REQUEST_ID_QUERY: str,
    GROUND_TRUTH_BY_REQUEST_ID_QUERY: str,
) -> None:
    """
    Affiche la page prédictions / traçabilité.
    """
    tab1, tab2 = st.tabs(["⚡ Prédictions", "📜 Traçabilité"])

    with tab1:
        st.markdown("### Lancer une prédiction")

        tabs = st.tabs(["Depuis un client en base", "Depuis un JSON libre"])

        with tabs[0]:
            col1, col2 = st.columns(2)

            with col1:
                client_id = st.number_input(
                    "SK_ID_CURR",
                    min_value=100000,
                    max_value=999999,
                    value=100001,
                    step=1,
                )

            with col2:
                source_table = st.selectbox(
                    "Table source",
                    options=sorted(ALLOWED_FEATURES_TABLES),
                    index=0,
                )

            if st.button("Charger le client", use_container_width=True):
                client_df = load_client_features(int(client_id), source_table)

                if client_df.empty:
                    st.error("Aucune ligne trouvée pour ce client.")
                else:
                    st.success("Client chargé avec succès.")
                    st.dataframe(client_df, use_container_width=True)

                    payload = dataframe_to_payload(client_df)

                    with st.expander("Payload JSON envoyé à l'API"):
                        st.json(payload)

                    if st.button("Envoyer à l'API /predict", key="predict_db", use_container_width=True):
                        ok, result = call_predict_api(payload)

                        if ok:
                            st.success("Prédiction reçue.")
                            st.json(result)
                        else:
                            st.error("Échec de l'appel API.")
                            st.json(result)

        with tabs[1]:
            template = {
                "SK_ID_CURR": 100001,
                "AMT_INCOME_TOTAL": 135000.0,
                "AMT_CREDIT": 312682.5,
                "AMT_ANNUITY": 16893.0,
                "EXT_SOURCE_2": 0.65,
            }

            json_text = st.text_area(
                "Payload JSON",
                value=json.dumps(template, ensure_ascii=False, indent=2),
                height=260,
            )

            validate_col, send_col = st.columns(2)
            parsed_payload: dict[str, Any] | None = None

            with validate_col:
                validate_btn = st.button("Valider le JSON", use_container_width=True)

            with send_col:
                send_btn = st.button("Envoyer à l'API", use_container_width=True)

            if validate_btn or send_btn:
                try:
                    parsed_payload = json.loads(json_text)
                    st.success("JSON valide.")
                    st.json(parsed_payload)
                except json.JSONDecodeError as exc:
                    st.error(f"JSON invalide : {exc}")

            if send_btn and parsed_payload is not None:
                ok, result = call_predict_api(parsed_payload)

                if ok:
                    st.success("Prédiction reçue.")
                    st.json(result)
                else:
                    st.error("Échec de l'appel API.")
                    st.json(result)

        st.markdown("### Résumé opérationnel")

        m1, m2, m3, m4, m5 = st.columns(5)

        total_preds = len(prediction_logs_df)
        mean_latency = metric_safe_number(prediction_logs_df, "latency_ms", "mean", 0.0)
        p95_latency = metric_safe_number(prediction_logs_df, "latency_ms", "p95", 0.0)
        p99_latency = metric_safe_number(prediction_logs_df, "latency_ms", "p99", 0.0)

        error_rate = 0.0
        if not prediction_logs_df.empty and "status_code" in prediction_logs_df.columns:
            error_rate = float((prediction_logs_df["status_code"].fillna(200) >= 400).mean()) * 100

        m1.metric("Total prédictions", total_preds)
        m2.metric("Latence moy. (ms)", round(mean_latency, 2))
        m3.metric("Latence p95 (ms)", round(p95_latency, 2))
        m4.metric("Latence p99 (ms)", round(p99_latency, 2))
        m5.metric("Erreur %", f"{error_rate:.1f}%")

        if not prediction_logs_df.empty and "prediction_timestamp" in prediction_logs_df.columns:
            chart_df = prediction_logs_df.dropna(subset=["prediction_timestamp"]).sort_values("prediction_timestamp")
            if not chart_df.empty and "score" in chart_df.columns:
                st.line_chart(chart_df.set_index("prediction_timestamp")[["score"]])

    with tab2:
        st.markdown("### Traçabilité des prédictions")

        if prediction_logs_df.empty:
            st.info("Aucune donnée dans `prediction_logs`.")
        else:
            st.dataframe(prediction_logs_df, use_container_width=True)

            request_ids = prediction_logs_df["request_id"].dropna().astype(str).tolist()

            if request_ids:
                selected_request_id = st.selectbox(
                    "Sélectionner une requête",
                    options=request_ids,
                )

                selected_log_df = prediction_logs_df[
                    prediction_logs_df["request_id"].astype(str) == selected_request_id
                ]

                st.markdown("#### Détail de la prédiction")
                st.dataframe(selected_log_df, use_container_width=True)

                snapshot_df = query_if_table_exists(
                    "prediction_features_snapshot",
                    PREDICTION_FEATURES_BY_REQUEST_ID_QUERY,
                    {"request_id": selected_request_id},
                    ["snapshot_timestamp"],
                )

                st.markdown("#### Snapshot de features")
                if snapshot_df.empty:
                    st.info("Aucun snapshot de features trouvé pour cette requête.")
                else:
                    st.dataframe(snapshot_df, use_container_width=True)

                gt_selected_df = query_if_table_exists(
                    "ground_truth_labels",
                    GROUND_TRUTH_BY_REQUEST_ID_QUERY,
                    {"request_id": selected_request_id},
                    ["observed_at"],
                )

                st.markdown("#### Vérité terrain associée")
                if gt_selected_df.empty:
                    st.info("Aucune vérité terrain trouvée pour cette requête.")
                else:
                    st.dataframe(gt_selected_df, use_container_width=True)