"""
Page Streamlit : prédictions, données stockées et snapshot des features.

Cette page regroupe :
- la prédiction via un client chargé depuis une source exposée par l'API
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

Tables métier visées
--------------------
- prediction_logs
- ground_truth_labels
- prediction_features_snapshot
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


def _choose_existing_columns(df: pd.DataFrame, preferred_cols: list[str]) -> list[str]:
    """
    Retourne uniquement les colonnes présentes dans le DataFrame.
    """
    return [col for col in preferred_cols if col in df.columns]


def _snapshot_to_dataframe(snapshot: Any) -> pd.DataFrame:
    """
    Convertit un retour de snapshot en DataFrame de façon robuste.

    Cas gérés
    ---------
    - dict avec clé "features" = list[dict]
    - dict avec clé "features" = dict
    - dict avec clé "items" = list[dict]
    - list[dict]
    - DataFrame déjà prêt
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

    Si le retour n'est pas un dict, on l'encapsule.
    """
    if isinstance(result, dict):
        return result
    return {"result": result}


# =============================================================================
# Page principale
# =============================================================================

def render_predictions_page(
    *,
    ALLOWED_FEATURES_TABLES: set[str],
    prediction_logs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    load_client_features,
    dataframe_to_payload,
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

    Parameters
    ----------
    ALLOWED_FEATURES_TABLES : set[str]
        Tables autorisées pour charger un client.
    prediction_logs_df : pd.DataFrame
        Historique des prédictions.
    ground_truth_df : pd.DataFrame
        Historique des vérités terrain.
    load_client_features :
        Fonction de chargement des features depuis l'API.
    dataframe_to_payload :
        Conversion d'un DataFrame vers un payload API.
    call_predict_api :
        Appel à l'endpoint /predict.
    call_predict_batch_api :
        Appel à l'endpoint /predict/batch.
    get_prediction_detail :
        Récupération du détail d'une prédiction.
    get_prediction_features_snapshot :
        Récupération du snapshot de features.
    get_ground_truth_by_request_id :
        Récupération de la vérité terrain.
    metric_safe_number :
        Calcul robuste de métriques.
    call_predict_real_random_batch_api :
        Appel optionnel à la route batch aléatoire basée sur données réelles.
    call_predict_fully_random_batch_api :
        Appel optionnel à la route batch complètement aléatoire.
    """
    prediction_logs_df = _safe_dataframe(prediction_logs_df)
    ground_truth_df = _safe_dataframe(ground_truth_df)

    prediction_logs_df = _coerce_columns_to_datetime(
        prediction_logs_df,
        ["prediction_timestamp"],
    )
    prediction_logs_df = _coerce_columns_to_numeric(
        prediction_logs_df,
        ["prediction", "score", "threshold_used", "latency_ms", "status_code"],
    )

    ground_truth_df = _coerce_columns_to_datetime(
        ground_truth_df,
        ["observed_at"],
    )
    ground_truth_df = _coerce_columns_to_numeric(
        ground_truth_df,
        ["true_label", "client_id"],
    )

    # ---------------------------------------------------------------------
    # Initialisation session_state
    # ---------------------------------------------------------------------
    if "last_loaded_client_df" not in st.session_state:
        st.session_state["last_loaded_client_df"] = pd.DataFrame()

    if "last_payload_preview" not in st.session_state:
        st.session_state["last_payload_preview"] = {}

    if "last_prediction_result" not in st.session_state:
        st.session_state["last_prediction_result"] = None

    if "last_batch_result" not in st.session_state:
        st.session_state["last_batch_result"] = None

    if "last_simulation_result" not in st.session_state:
        st.session_state["last_simulation_result"] = None

    # ---------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------
    st.markdown("## Prédictions et traçabilité")
    st.caption(
        "Lancer des prédictions, simuler des batches, consulter l’historique et inspecter les données stockées."
    )

    total_predictions = len(prediction_logs_df)
    mean_latency = metric_safe_number(prediction_logs_df, "latency_ms", "mean", 0)
    p95_latency = metric_safe_number(prediction_logs_df, "latency_ms", "p95", 0)
    p99_latency = metric_safe_number(prediction_logs_df, "latency_ms", "p99", 0)

    error_rate = 0.0
    if not prediction_logs_df.empty and "status_code" in prediction_logs_df.columns:
        status_codes = pd.to_numeric(prediction_logs_df["status_code"], errors="coerce")
        if status_codes.notna().any():
            error_rate = float((status_codes >= 400).mean() * 100)

    avg_score = None
    if not prediction_logs_df.empty and "score" in prediction_logs_df.columns:
        avg_score = metric_safe_number(prediction_logs_df, "score", "mean", None)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _render_card("Total prédictions", total_predictions, "Historique actuellement chargé")
    with k2:
        _render_card("Latence moyenne", f"{_safe_float(mean_latency):.2f} ms", "Temps moyen d’inférence")
    with k3:
        _render_card("Latence p95", f"{_safe_float(p95_latency):.2f} ms", "Queue de distribution")
    with k4:
        _render_card("Latence p99", f"{_safe_float(p99_latency):.2f} ms", "Cas extrêmes")
    with k5:
        _render_card("Erreur", f"{error_rate:.1f} %", "Taux basé sur les status_code")

    if avg_score is not None:
        st.caption(f"Score moyen observé dans l’historique : {_safe_float(avg_score):.4f}")

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

        # -----------------------------------------------------------------
        # Depuis un client
        # -----------------------------------------------------------------
        with subtabs[0]:
            box1, box2 = st.columns([1, 1])

            with box1:
                client_id = st.number_input(
                    "SK_ID_CURR",
                    min_value=100000,
                    max_value=999999,
                    value=100001,
                    step=1,
                )

            with box2:
                source_table = st.selectbox(
                    "Table source",
                    options=sorted(ALLOWED_FEATURES_TABLES) if ALLOWED_FEATURES_TABLES else [],
                )

            action_col1, action_col2 = st.columns([1, 1])

            with action_col1:
                if st.button("Charger le client", key="load_client", use_container_width=True):
                    try:
                        client_df = load_client_features(int(client_id), source_table)

                        if not isinstance(client_df, pd.DataFrame) or client_df.empty:
                            st.error("Aucune donnée trouvée.")
                            st.session_state["last_loaded_client_df"] = pd.DataFrame()
                            st.session_state["last_payload_preview"] = {}
                        else:
                            payload = dataframe_to_payload(client_df)
                            st.session_state["last_loaded_client_df"] = client_df
                            st.session_state["last_payload_preview"] = payload
                            st.success("Client chargé avec succès.")
                    except Exception as e:
                        st.error(f"Erreur lors du chargement client : {e}")
                        st.session_state["last_loaded_client_df"] = pd.DataFrame()
                        st.session_state["last_payload_preview"] = {}

            client_df = st.session_state["last_loaded_client_df"]
            payload = st.session_state["last_payload_preview"]

            if isinstance(client_df, pd.DataFrame) and not client_df.empty:
                st.markdown("#### Données chargées")
                st.dataframe(client_df, width="stretch")

                with st.expander("Voir le payload JSON généré", expanded=False):
                    st.json(payload)

                with action_col2:
                    if st.button(
                        "Envoyer à l'API",
                        key="predict_single_client",
                        use_container_width=True,
                    ):
                        try:
                            ok, result = call_predict_api(payload)

                            if ok:
                                st.session_state["last_prediction_result"] = _normalize_prediction_result(result)
                                st.success("Prédiction reçue.")
                            else:
                                st.error("Erreur API.")
                                st.session_state["last_prediction_result"] = _normalize_prediction_result(result)
                        except Exception as e:
                            st.error(f"Erreur lors de l'appel API : {e}")
                            st.session_state["last_prediction_result"] = {"error": str(e)}

            if st.session_state["last_prediction_result"] is not None:
                st.markdown("#### Dernier résultat")
                st.json(st.session_state["last_prediction_result"])

        # -----------------------------------------------------------------
        # JSON libre
        # -----------------------------------------------------------------
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

            if st.button("Envoyer le JSON à l'API", key="send_free_json", use_container_width=True):
                if parsed is None:
                    st.error("Le JSON n'est pas valide.")
                else:
                    try:
                        ok, result = call_predict_api(parsed)

                        if ok:
                            st.session_state["last_prediction_result"] = _normalize_prediction_result(result)
                            st.success("Prédiction reçue.")
                        else:
                            st.error("Erreur API.")
                            st.session_state["last_prediction_result"] = _normalize_prediction_result(result)
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel API : {e}")
                        st.session_state["last_prediction_result"] = {"error": str(e)}

            if st.session_state["last_prediction_result"] is not None:
                st.markdown("#### Dernier résultat")
                st.json(st.session_state["last_prediction_result"])

        st.markdown("")
        _render_section_title(
            "Historique des prédictions",
            "Vue consolidée des prédictions déjà journalisées.",
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

            preferred_cols = _choose_existing_columns(
                history_view,
                [
                    "prediction_timestamp",
                    "request_id",
                    "client_id",
                    "model_name",
                    "model_version",
                    "prediction",
                    "score",
                    "threshold_used",
                    "latency_ms",
                    "status_code",
                    "error_message",
                ],
            )

            st.dataframe(
                history_view[preferred_cols] if preferred_cols else history_view,
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
            ["Batch JSON libre", "Simulation réelle aléatoire", "Simulation totalement aléatoire"]
        )

        # -----------------------------------------------------------------
        # Batch JSON manuel
        # -----------------------------------------------------------------
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

            if st.button("Envoyer le batch JSON", key="send_manual_batch", use_container_width=True):
                if not is_valid_batch:
                    st.error("Le batch JSON n'est pas valide.")
                else:
                    try:
                        ok, result = call_predict_batch_api(parsed_batch)

                        if ok:
                            st.session_state["last_batch_result"] = _normalize_prediction_result(result)
                            st.success("Batch traité avec succès.")
                        else:
                            st.error("Erreur batch.")
                            st.session_state["last_batch_result"] = _normalize_prediction_result(result)
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel batch : {e}")
                        st.session_state["last_batch_result"] = {"error": str(e)}

        # -----------------------------------------------------------------
        # Simulation réelle aléatoire
        # -----------------------------------------------------------------
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
                st.warning("La route batch réelle aléatoire n'est pas branchée côté dashboard.")
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
                            st.session_state["last_simulation_result"] = _normalize_prediction_result(result)
                            st.success("Simulation réelle aléatoire terminée.")
                        else:
                            st.error("Erreur lors de la simulation.")
                            st.session_state["last_simulation_result"] = _normalize_prediction_result(result)
                    except Exception as e:
                        st.error(f"Erreur lors de la simulation : {e}")
                        st.session_state["last_simulation_result"] = {"error": str(e)}

        # -----------------------------------------------------------------
        # Simulation totalement aléatoire
        # -----------------------------------------------------------------
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
                st.warning("La route batch totalement aléatoire n'est pas branchée côté dashboard.")
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
                            st.session_state["last_simulation_result"] = _normalize_prediction_result(result)
                            st.success("Simulation totalement aléatoire terminée.")
                        else:
                            st.error("Erreur lors de la simulation.")
                            st.session_state["last_simulation_result"] = _normalize_prediction_result(result)
                    except Exception as e:
                        st.error(f"Erreur lors de la simulation : {e}")
                        st.session_state["last_simulation_result"] = {"error": str(e)}

        st.markdown("#### Dernier résultat batch / simulation")

        if st.session_state["last_batch_result"] is not None:
            st.markdown("##### Dernier batch manuel")
            st.json(st.session_state["last_batch_result"])

        if st.session_state["last_simulation_result"] is not None:
            st.markdown("##### Dernière simulation")
            st.json(st.session_state["last_simulation_result"])

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
                        st.error(f"Erreur lors de la récupération du détail : {e}")

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
                        st.error(f"Erreur lors de la récupération de la vérité terrain : {e}")

                    st.markdown("#### Vérité terrain associée")

                    if isinstance(gt, pd.DataFrame) and not gt.empty:
                        gt_view = gt.copy()
                        preferred_cols = _choose_existing_columns(
                            gt_view,
                            [
                                "request_id",
                                "client_id",
                                "true_label",
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
                        st.info("Aucune vérité terrain trouvée pour cette requête.")

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
                        st.error(f"Erreur lors de la récupération du snapshot : {e}")

                    df_snapshot = _snapshot_to_dataframe(snapshot)

                    if not df_snapshot.empty:
                        info1, info2 = st.columns([1, 2])
                        with info1:
                            st.metric("Nombre de lignes snapshot", len(df_snapshot))
                        with info2:
                            st.caption(f"request_id : {selected}")

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
                            df_snapshot[preferred_cols] if preferred_cols else df_snapshot,
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