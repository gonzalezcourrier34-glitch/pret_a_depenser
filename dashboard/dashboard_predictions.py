"""
Page Streamlit : prédictions, données stockées et snapshot des features.

Cette page regroupe :
- la prédiction via un client existant déjà connu de l'API
- la prédiction via un JSON libre
- la prédiction batch via JSON
- le lancement de batches de simulation
- l'historique des prédictions
- la consultation des données stockées liées à une requête
- la consultation du snapshot exact des features utilisées

Notes
-----
Cette page ne lit pas directement PostgreSQL.
Toutes les données doivent être récupérées via l'API FastAPI.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd
import streamlit as st


# =============================================================================
# Helpers UI
# =============================================================================

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
            min-height: 115px;
        ">
            <div style="font-size: 0.9rem; color: #9CA3AF; margin-bottom: 8px;">
                {title}
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 8px;">
                {value}
            </div>
            <div style="font-size: 0.85rem; color: #D1D5DB;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_title(title: str, subtitle: str = "") -> None:
    """
    Affiche un titre de section plus visuel.
    """
    st.markdown(
        f"""
        <div style="margin-top: 8px; margin-bottom: 14px;">
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 4px;">
                {title}
            </div>
            <div style="color: #6B7280; font-size: 0.95rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_prediction_result_card(result: dict[str, Any]) -> None:
    """
    Affiche un résultat de prédiction unitaire de façon lisible.
    """
    if not isinstance(result, dict) or not result:
        st.info("Aucun résultat de prédiction à afficher.")
        return

    prediction = result.get("prediction")
    score = result.get("score")
    model_version = result.get("model_version", "unknown")
    latency_ms = result.get("latency_ms")
    request_id = result.get("request_id", "-")

    if prediction == 0:
        decision_label = "Crédit accepté"
        decision_color = "#10B981"
    elif prediction == 1:
        decision_label = "Crédit refusé"
        decision_color = "#EF4444"
    else:
        decision_label = "Décision inconnue"
        decision_color = "#6B7280"

    score_display = (
        f"{_safe_float(score):.4f}" if score is not None else "N/A"
    )
    latency_display = (
        f"{_safe_float(latency_ms):.2f} ms" if latency_ms is not None else "N/A"
    )

    st.markdown(
        f"""
        <div style="
            padding: 20px 22px;
            border-radius: 18px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 22px rgba(0,0,0,0.10);
            margin-bottom: 14px;
        ">
            <div style="font-size: 0.95rem; color: #9CA3AF; margin-bottom: 8px;">
                Résultat de la prédiction
            </div>
            <div style="
                font-size: 1.6rem;
                font-weight: 800;
                color: {decision_color};
                margin-bottom: 12px;
            ">
                {decision_label}
            </div>
            <div style="color: white; font-size: 0.95rem; line-height: 1.8;">
                <strong>Score :</strong> {score_display}<br>
                <strong>Version modèle :</strong> {model_version}<br>
                <strong>Latence :</strong> {latency_display}<br>
                <strong>Request ID :</strong> {request_id}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_json_text(value: Any) -> str:
    """
    Sérialise proprement une structure Python en JSON lisible.
    """
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def _safe_dataframe(value: Any) -> pd.DataFrame:
    """
    Garantit le retour d'un DataFrame.
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convertit une valeur en float de façon robuste.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Convertit une valeur en entier de façon robuste.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _choose_existing_columns(
    df: pd.DataFrame,
    preferred_cols: list[str],
) -> list[str]:
    """
    Retourne uniquement les colonnes présentes dans le DataFrame.
    """
    return [col for col in preferred_cols if col in df.columns]


def _snapshot_to_dataframe(snapshot: Any) -> pd.DataFrame:
    """
    Convertit un retour de snapshot en DataFrame de façon robuste.
    """
    if isinstance(snapshot, pd.DataFrame):
        return snapshot.copy()

    if isinstance(snapshot, list):
        try:
            return pd.DataFrame(snapshot)
        except Exception:
            return pd.DataFrame()

    if not isinstance(snapshot, dict):
        return pd.DataFrame()

    if "features" in snapshot:
        features = snapshot["features"]

        if isinstance(features, list):
            return pd.DataFrame(features)

        if isinstance(features, dict):
            return pd.DataFrame([features])

    if "items" in snapshot and isinstance(snapshot["items"], list):
        return pd.DataFrame(snapshot["items"])

    return pd.DataFrame()


def _normalize_prediction_result(result: Any) -> dict[str, Any]:
    """
    Normalise un résultat de prédiction pour l'affichage.
    """
    if isinstance(result, dict):
        return result
    return {"result": result}


def _extract_batch_items(result: Any) -> pd.DataFrame:
    """
    Extrait les items d'un résultat batch/simulation.
    """
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            return pd.DataFrame(items)
    return pd.DataFrame()


# =============================================================================
# Page principale
# =============================================================================

def render_predictions_page(
    *,
    prediction_logs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    call_predict_client_api,
    call_predict_api,
    call_predict_batch_api,
    get_prediction_detail,
    get_prediction_features_snapshot,
    get_ground_truth_by_request_id,
    metric_safe_number,
    call_predict_real_random_batch_api=None,
    call_predict_fully_random_batch_api=None,
) -> None:
    """
    Affiche la page prédictions, données stockées et snapshot des features.
    """
    prediction_logs_df = _safe_dataframe(prediction_logs_df)
    ground_truth_df = _safe_dataframe(ground_truth_df)

    prediction_logs_df = _coerce_columns_to_datetime(
        prediction_logs_df,
        ["prediction_timestamp"],
    )
    prediction_logs_df = _coerce_columns_to_numeric(
        prediction_logs_df,
        ["prediction", "score", "threshold", "latency_ms"],
    )

    ground_truth_df = _coerce_columns_to_datetime(
        ground_truth_df,
        ["observed_at"],
    )
    ground_truth_df = _coerce_columns_to_numeric(
        ground_truth_df,
        ["ground_truth", "client_id"],
    )

    if "last_prediction_result" not in st.session_state:
        st.session_state["last_prediction_result"] = None

    if "last_batch_result" not in st.session_state:
        st.session_state["last_batch_result"] = None

    if "last_simulation_result" not in st.session_state:
        st.session_state["last_simulation_result"] = None

    st.markdown("## Prédictions et traçabilité")
    st.caption(
        "Lancer des prédictions, simuler des batches, consulter l’historique et inspecter les données stockées."
    )

    total_predictions = len(prediction_logs_df)
    mean_latency = metric_safe_number(prediction_logs_df, "latency_ms", "mean", 0)
    p95_latency = metric_safe_number(prediction_logs_df, "latency_ms", "p95", 0)
    p99_latency = metric_safe_number(prediction_logs_df, "latency_ms", "p99", 0)

    error_rate = 0.0
    if not prediction_logs_df.empty:
        if "error_message" in prediction_logs_df.columns:
            error_rate = float(
                prediction_logs_df["error_message"].notna().mean() * 100
            )
        elif "status" in prediction_logs_df.columns:
            error_rate = float(
                prediction_logs_df["status"]
                .fillna("")
                .astype(str)
                .str.lower()
                .eq("error")
                .mean() * 100
            )

    avg_score = None
    if not prediction_logs_df.empty and "score" in prediction_logs_df.columns:
        avg_score = metric_safe_number(prediction_logs_df, "score", "mean", None)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _render_card(
            "Total prédictions",
            total_predictions,
            "Historique actuellement chargé",
        )
    with k2:
        _render_card(
            "Latence moyenne",
            f"{_safe_float(mean_latency):.2f} ms",
            "Temps moyen d’inférence",
        )
    with k3:
        _render_card(
            "Latence p95",
            f"{_safe_float(p95_latency):.2f} ms",
            "Queue de distribution",
        )
    with k4:
        _render_card(
            "Latence p99",
            f"{_safe_float(p99_latency):.2f} ms",
            "Cas extrêmes",
        )
    with k5:
        _render_card(
            "Erreur",
            f"{error_rate:.1f} %",
            "Taux basé sur les logs",
        )

    if avg_score is not None:
        st.caption(
            f"Score moyen observé dans l’historique : {_safe_float(avg_score):.4f}"
        )

    tabs = st.tabs(
        [
            "Prédiction",
            "Batches & simulations",
            "Données stockées",
            "Snapshot des features",
        ]
    )

    # =====================================================================
    # ONGLET 1 - PRÉDICTION
    # =====================================================================
    with tabs[0]:
        _render_section_title(
            "Lancer une prédiction",
            "Tester le modèle depuis un client existant ou un payload JSON libre.",
        )

        subtabs = st.tabs(["Depuis un client", "Depuis un JSON libre"])

        with subtabs[0]:
            client_id = st.number_input(
                "SK_ID_CURR",
                min_value=100000,
                max_value=999999,
                value=100001,
                step=1,
            )

            if st.button(
                "Lancer la prédiction du client",
                key="predict_single_client",
                use_container_width=True,
            ):
                try:
                    ok, result = call_predict_client_api(int(client_id))

                    if ok:
                        st.session_state["last_prediction_result"] = _normalize_prediction_result(result)
                        st.success("Prédiction reçue.")
                    else:
                        error_result = _normalize_prediction_result(result)
                        error_message = error_result.get("detail", "Erreur API.")
                        st.error(error_message)
                        st.session_state["last_prediction_result"] = error_result

                except Exception as e:
                    st.error(f"Erreur lors de l'appel API : {e}")
                    st.session_state["last_prediction_result"] = {"error": str(e)}

            if st.session_state["last_prediction_result"] is not None:
                st.markdown("#### Résultat de la prédiction")
                prediction_result = st.session_state["last_prediction_result"]

                if isinstance(prediction_result, dict) and "error" not in prediction_result and "detail" not in prediction_result:
                    _render_prediction_result_card(prediction_result)

                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

                    with result_col1:
                        decision = prediction_result.get("prediction")
                        _render_card(
                            "Décision",
                            "Accepté" if decision == 0 else "Refusé" if decision == 1 else "Inconnu",
                            "Décision calculée par seuil",
                        )

                    with result_col2:
                        _render_card(
                            "Score",
                            f"{_safe_float(prediction_result.get('score')):.4f}",
                            "Probabilité de défaut",
                        )

                    with result_col3:
                        _render_card(
                            "Version modèle",
                            prediction_result.get("model_version", "unknown"),
                            "Version utilisée",
                        )

                    with result_col4:
                        _render_card(
                            "Latence",
                            f"{_safe_float(prediction_result.get('latency_ms')):.2f} ms",
                            "Temps d'inférence",
                        )

                    with st.expander("Voir la réponse JSON complète", expanded=False):
                        st.json(prediction_result)
                else:
                    st.error("Le client demandé n'est pas disponible pour la prédiction.")
                    st.json(prediction_result)

        with subtabs[1]:
            template = {
                "SK_ID_CURR": 100001,
                "features": {},
            }

            json_text = st.text_area(
                "Payload JSON",
                value=_safe_json_text(template),
                height=280,
                key="free_json_predict_payload",
            )

            parsed = None
            try:
                parsed = json.loads(json_text)
                st.success("JSON valide.")
            except Exception as e:
                st.error(f"Erreur JSON : {e}")

            if st.button(
                "Envoyer le JSON à l'API",
                key="send_free_json",
                use_container_width=True,
            ):
                if parsed is None:
                    st.error("Le JSON n'est pas valide.")
                else:
                    try:
                        ok, result = call_predict_api(parsed)

                        if ok:
                            st.session_state["last_prediction_result"] = (
                                _normalize_prediction_result(result)
                            )
                            st.success("Prédiction reçue.")
                        else:
                            st.error("Erreur API.")
                            st.session_state["last_prediction_result"] = (
                                _normalize_prediction_result(result)
                            )
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel API : {e}")
                        st.session_state["last_prediction_result"] = {"error": str(e)}

            if st.session_state["last_prediction_result"] is not None:
                st.markdown("#### Dernier résultat")
                prediction_result = st.session_state["last_prediction_result"]

                if (
                    isinstance(prediction_result, dict)
                    and "error" not in prediction_result
                ):
                    _render_prediction_result_card(prediction_result)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        decision = prediction_result.get("prediction")
                        _render_card(
                            "Décision",
                            (
                                "Accepté"
                                if decision == 0
                                else "Refusé"
                                if decision == 1
                                else "Inconnu"
                            ),
                        )
                    with c2:
                        _render_card(
                            "Score",
                            f"{_safe_float(prediction_result.get('score')):.4f}",
                        )
                    with c3:
                        _render_card(
                            "Version modèle",
                            prediction_result.get("model_version", "unknown"),
                        )
                    with c4:
                        _render_card(
                            "Latence",
                            f"{_safe_float(prediction_result.get('latency_ms')):.2f} ms",
                        )

                    with st.expander("Voir la réponse JSON complète", expanded=False):
                        st.json(prediction_result)
                else:
                    st.json(prediction_result)

        st.markdown("")
        _render_section_title(
            "Historique des prédictions",
            "Vue séparée des crédits acceptés et refusés déjà journalisés.",
        )

        if prediction_logs_df.empty:
            st.info("Aucune donnée d’historique disponible.")
        else:
            history_view = prediction_logs_df.copy()

            if "prediction_timestamp" in history_view.columns:
                history_view = history_view.sort_values(
                    "prediction_timestamp",
                    ascending=False,
                    na_position="last",
                )

            accepted_df = pd.DataFrame()
            refused_df = pd.DataFrame()

            if "prediction" in history_view.columns:
                accepted_df = history_view[history_view["prediction"] == 0].copy()
                refused_df = history_view[history_view["prediction"] == 1].copy()

                if not accepted_df.empty:
                    accepted_df["decision_credit"] = "Accepté"
                if not refused_df.empty:
                    refused_df["decision_credit"] = "Refusé"
            else:
                st.warning("La colonne `prediction` est absente de l’historique.")

            accepted_cols = _choose_existing_columns(
                accepted_df,
                [
                    "prediction_timestamp",
                    "request_id",
                    "client_id",
                    "model_name",
                    "model_version",
                    "decision_credit",
                    "prediction",
                    "score",
                    "threshold",
                    "latency_ms",
                    "status",
                    "error_message",
                ],
            )

            refused_cols = _choose_existing_columns(
                refused_df,
                [
                    "prediction_timestamp",
                    "request_id",
                    "client_id",
                    "model_name",
                    "model_version",
                    "decision_credit",
                    "prediction",
                    "score",
                    "threshold",
                    "latency_ms",
                    "status",
                    "error_message",
                ],
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### Demandes de crédit acceptées")
                st.caption(f"{len(accepted_df)} demande(s) acceptée(s)")

                if accepted_df.empty:
                    st.info("Aucune demande de crédit acceptée trouvée.")
                else:
                    st.dataframe(
                        accepted_df[accepted_cols] if accepted_cols else accepted_df,
                        width="stretch",
                    )

            with c2:
                st.markdown("### Demandes de crédit refusées")
                st.caption(f"{len(refused_df)} demande(s) refusée(s)")

                if refused_df.empty:
                    st.info("Aucune demande de crédit refusée trouvée.")
                else:
                    st.dataframe(
                        refused_df[refused_cols] if refused_cols else refused_df,
                        width="stretch",
                    )

    # =====================================================================
    # ONGLET 2 - BATCHES & SIMULATIONS
    # =====================================================================
    with tabs[1]:
        _render_section_title(
            "Batchs et simulations",
            "Tester des lots JSON manuels ou déclencher les nouvelles routes de simulation.",
        )

        subtabs_batch = st.tabs(
            [
                "Batch JSON libre",
                "Simulation réelle aléatoire",
                "Simulation totalement aléatoire",
            ]
        )

        with subtabs_batch[0]:
            batch_text = st.text_area(
                "Batch JSON",
                value="[]",
                height=280,
                key="manual_batch_json",
            )

            parsed_batch = None
            is_valid_batch = False

            try:
                parsed_batch = json.loads(batch_text)
                is_valid_batch = isinstance(parsed_batch, list)
                if is_valid_batch:
                    st.success("Batch JSON valide.")
                else:
                    st.error("Le contenu doit être une liste JSON.")
            except Exception as e:
                st.error(f"Erreur JSON : {e}")

            if st.button(
                "Envoyer le batch JSON",
                key="send_manual_batch",
                use_container_width=True,
            ):
                if not is_valid_batch:
                    st.error("Le batch JSON n'est pas valide.")
                else:
                    try:
                        ok, result = call_predict_batch_api(parsed_batch)

                        if ok:
                            st.session_state["last_batch_result"] = (
                                _normalize_prediction_result(result)
                            )
                            st.success("Batch traité avec succès.")
                        else:
                            st.error("Erreur batch.")
                            st.session_state["last_batch_result"] = (
                                _normalize_prediction_result(result)
                            )
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel batch : {e}")
                        st.session_state["last_batch_result"] = {"error": str(e)}

        with subtabs_batch[1]:
            st.info(
                "Cette simulation envoie un lot de prédictions construit à partir de données réelles "
                "prises aléatoirement dans la source disponible."
            )

            real_count = st.number_input(
                "Nombre de prédictions réelles aléatoires",
                min_value=1,
                max_value=200,
                value=200,
                step=1,
                key="real_random_batch_count",
            )

            real_seed = st.number_input(
                "Graine aléatoire (optionnelle)",
                min_value=0,
                value=0,
                step=1,
                key="real_random_batch_seed",
            )
            use_real_seed = st.checkbox(
                "Utiliser la graine aléatoire",
                value=False,
                key="use_real_random_seed",
            )

            if call_predict_real_random_batch_api is None:
                st.warning(
                    "La route batch réelle aléatoire n'est pas branchée côté dashboard."
                )
            else:
                if st.button(
                    "Lancer les prédictions réelles aléatoires",
                    key="launch_real_random_batch",
                    use_container_width=True,
                ):
                    try:
                        ok, result = call_predict_real_random_batch_api(
                            batch_size=int(real_count),
                            random_seed=int(real_seed) if use_real_seed else None,
                        )

                        if ok:
                            st.session_state["last_simulation_result"] = (
                                _normalize_prediction_result(result)
                            )
                            st.success("Simulation réelle aléatoire terminée.")
                        else:
                            st.error("Erreur lors de la simulation.")
                            st.session_state["last_simulation_result"] = (
                                _normalize_prediction_result(result)
                            )
                    except Exception as e:
                        st.error(f"Erreur lors de la simulation : {e}")
                        st.session_state["last_simulation_result"] = {"error": str(e)}

        with subtabs_batch[2]:
            st.info(
                "Cette simulation envoie un lot de prédictions généré à partir de données entièrement "
                "aléatoires, utile pour tester robustesse, monitoring et dérive."
            )

            random_count = st.number_input(
                "Nombre de prédictions totalement aléatoires",
                min_value=1,
                max_value=200,
                value=200,
                step=1,
                key="fully_random_batch_count",
            )

            if call_predict_fully_random_batch_api is None:
                st.warning(
                    "La route batch totalement aléatoire n'est pas branchée côté dashboard."
                )
            else:
                if st.button(
                    "Lancer les prédictions totalement aléatoires",
                    key="launch_fully_random_batch",
                    use_container_width=True,
                ):
                    try:
                        ok, result = call_predict_fully_random_batch_api(
                            batch_size=int(random_count)
                        )

                        if ok:
                            st.session_state["last_simulation_result"] = (
                                _normalize_prediction_result(result)
                            )
                            st.success("Simulation totalement aléatoire terminée.")
                        else:
                            st.error("Erreur lors de la simulation.")
                            st.session_state["last_simulation_result"] = (
                                _normalize_prediction_result(result)
                            )
                    except Exception as e:
                        st.error(f"Erreur lors de la simulation : {e}")
                        st.session_state["last_simulation_result"] = {"error": str(e)}

        st.markdown("#### Dernier résultat batch / simulation")

        if st.session_state["last_batch_result"] is not None:
            st.markdown("##### Dernier batch manuel")
            st.json(st.session_state["last_batch_result"])

            batch_items_df = _extract_batch_items(
                st.session_state["last_batch_result"]
            )
            if not batch_items_df.empty:
                st.dataframe(batch_items_df, width="stretch")

        if st.session_state["last_simulation_result"] is not None:
            st.markdown("##### Dernière simulation")
            st.json(st.session_state["last_simulation_result"])

            simulation_items_df = _extract_batch_items(
                st.session_state["last_simulation_result"]
            )
            if not simulation_items_df.empty:
                st.dataframe(simulation_items_df, width="stretch")

    # =====================================================================
    # ONGLET 3 - DONNÉES STOCKÉES
    # =====================================================================
    with tabs[2]:
        _render_section_title(
            "Données stockées",
            "Inspecter le détail d’une requête et les données associées.",
        )

        if prediction_logs_df.empty:
            st.info("Aucune donnée disponible.")
        else:
            if "request_id" not in prediction_logs_df.columns:
                st.warning("La colonne `request_id` est absente de l’historique.")
            else:
                request_ids = (
                    prediction_logs_df["request_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if not request_ids:
                    st.info("Aucun request_id disponible.")
                else:
                    selected = st.selectbox(
                        "Choisir une requête",
                        request_ids,
                        key="stored_data_request_id",
                    )

                    detail = None
                    try:
                        detail = get_prediction_detail(selected)
                    except Exception as e:
                        st.error(
                            f"Erreur lors de la récupération du détail : {e}"
                        )

                    if detail:
                        st.markdown("#### Détail de la prédiction")

                        if isinstance(detail, dict):
                            st.dataframe(pd.DataFrame([detail]), width="stretch")
                        elif isinstance(detail, pd.DataFrame):
                            st.dataframe(detail, width="stretch")
                        else:
                            st.json(detail)
                    else:
                        st.info("Aucun détail disponible pour cette requête.")

                    gt = None
                    try:
                        gt = get_ground_truth_by_request_id(
                            selected,
                            ground_truth_df=ground_truth_df,
                        )
                    except Exception as e:
                        st.error(
                            f"Erreur lors de la récupération de la vérité terrain : {e}"
                        )

                    st.markdown("#### Vérité terrain associée")

                    if isinstance(gt, pd.DataFrame) and not gt.empty:
                        gt_view = gt.copy()
                        preferred_cols = _choose_existing_columns(
                            gt_view,
                            [
                                "request_id",
                                "client_id",
                                "ground_truth",
                                "label_source",
                                "observed_at",
                                "notes",
                            ],
                        )
                        st.dataframe(
                            gt_view[preferred_cols] if preferred_cols else gt_view,
                            width="stretch",
                        )
                    elif isinstance(gt, dict) and gt:
                        st.dataframe(pd.DataFrame([gt]), width="stretch")
                    else:
                        st.info(
                            "Aucune vérité terrain trouvée pour cette requête."
                        )

    # =====================================================================
    # ONGLET 4 - SNAPSHOT DES FEATURES
    # =====================================================================
    with tabs[3]:
        _render_section_title(
            "Snapshot exact des features",
            "Retrouver les variables réellement utilisées pour une prédiction donnée.",
        )

        if prediction_logs_df.empty:
            st.info("Aucune donnée disponible.")
        else:
            if "request_id" not in prediction_logs_df.columns:
                st.warning("La colonne `request_id` est absente de l’historique.")
            else:
                request_ids = (
                    prediction_logs_df["request_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if not request_ids:
                    st.info("Aucun request_id disponible.")
                else:
                    selected = st.selectbox(
                        "Choisir une requête snapshot",
                        request_ids,
                        key="snapshot_request_id",
                    )

                    snapshot = None
                    try:
                        snapshot = get_prediction_features_snapshot(selected)
                    except Exception as e:
                        st.error(
                            f"Erreur lors de la récupération du snapshot : {e}"
                        )

                    df_snapshot = _snapshot_to_dataframe(snapshot)

                    if not df_snapshot.empty:
                        info1, info2, info3 = st.columns([1, 2, 1])

                        with info1:
                            st.metric("Nombre de lignes snapshot", len(df_snapshot))
                        with info2:
                            st.caption(f"request_id : {selected}")
                        with info3:
                            if "feature_name" in df_snapshot.columns:
                                st.metric(
                                    "Features distinctes",
                                    df_snapshot["feature_name"].nunique(),
                                )

                        preferred_cols = _choose_existing_columns(
                            df_snapshot,
                            [
                                "request_id",
                                "client_id",
                                "model_name",
                                "model_version",
                                "feature_name",
                                "feature_value",
                                "feature_type",
                                "snapshot_timestamp",
                            ],
                        )

                        st.dataframe(
                            df_snapshot[preferred_cols]
                            if preferred_cols
                            else df_snapshot,
                            width="stretch",
                        )

                        csv = df_snapshot.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Télécharger le snapshot CSV",
                            csv,
                            file_name=f"snapshot_{selected}.csv",
                            use_container_width=True,
                        )
                    else:
                        st.info("Aucun snapshot disponible.")