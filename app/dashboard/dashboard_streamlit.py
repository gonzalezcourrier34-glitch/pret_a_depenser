"""
Dashboard Streamlit pour l'API de scoring crédit et le monitoring MLOps.

Ce dashboard propose trois grands espaces :

- Prédiction
    Lancer une prédiction soit depuis un JSON libre, soit depuis un client
    présent dans une table SQL de features prêtes pour le modèle.

- Historique
    Consulter les prédictions déjà journalisées par l'API et, si disponible,
    les snapshots de features ainsi que les vérités terrain.

- Monitoring
    Suivre les versions du modèle, les métriques d'évaluation, les métriques
    de drift, les alertes et le feature store de monitoring.

Configuration via variables d'environnement
-------------------------------------------
- DATABASE_URL : URL SQLAlchemy PostgreSQL
- API_URL : URL de base de l'API FastAPI
- API_KEY : clé API attendue par l'endpoint /predict
- FEATURES_TABLE : nom de la table SQL utilisée pour recharger un client
- DEFAULT_LIMIT : nombre de lignes par défaut dans les tableaux

Exemples
--------
En local :
- API_URL=http://127.0.0.1:8000

En Docker Compose :
- API_URL=http://api:8000

Notes
-----
- Le dashboard reste exploitable même si certaines tables de monitoring
  ou d'historique n'existent pas encore.
- Les requêtes SQL sont volontairement simples pour rester lisibles
  et faciles à maintenir.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =============================================================================
# Configuration Streamlit
# =============================================================================

st.set_page_config(
    page_title="Dashboard Scoring et Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Configuration applicative
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "")
API_URL = os.getenv("API_URL", "http://api:8000")
API_KEY = os.getenv("API_KEY", "")
FEATURES_TABLE = os.getenv("FEATURES_TABLE", "features_client_test_enriched")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "200"))

# Tables autorisées pour recharger un client depuis le dashboard.
# Cette liste évite qu'un nom de table arbitraire soit injecté depuis l'UI.
ALLOWED_FEATURES_TABLES = {
    "features_client_test_enriched",
    "features_client_test",
}


# =============================================================================
# Helpers UI
# =============================================================================

def inject_css() -> None:
    """
    Ajoute une couche légère de style pour améliorer la lisibilité.

    Notes
    -----
    L'objectif n'est pas de faire un design complexe, mais d'obtenir
    un rendu plus propre pour la démonstration et l'usage quotidien.
    """
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .subtitle {
                color: #6b7280;
                margin-bottom: 1rem;
            }
            .section-title {
                font-size: 1.2rem;
                font-weight: 600;
                margin-top: 0.6rem;
                margin-bottom: 0.8rem;
            }
            .small-muted {
                color: #6b7280;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Accès base de données
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine | None:
    """
    Retourne un moteur SQLAlchemy réutilisable.

    Returns
    -------
    Engine | None
        Moteur SQLAlchemy si DATABASE_URL est défini, sinon None.

    Notes
    -----
    `pool_pre_ping=True` permet de tester la connexion avant réutilisation
    d'une connexion du pool, ce qui réduit certains problèmes de connexions
    mortes dans Docker ou PostgreSQL.
    """
    if not DATABASE_URL:
        return None

    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=30, show_spinner=False)
def run_query(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Exécute une requête SQL et renvoie un DataFrame.

    Parameters
    ----------
    query : str
        Requête SQL à exécuter.
    params : dict[str, Any] | None, default=None
        Paramètres nommés de la requête.

    Returns
    -------
    pd.DataFrame
        Résultat de la requête sous forme de DataFrame.
        Retourne un DataFrame vide si aucune base n'est configurée.

    Raises
    ------
    Exception
        Lève l'exception SQLAlchemy/DBAPI si la requête échoue.
    """
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    with engine.begin() as connection:
        return pd.read_sql(text(query), connection, params=params or {})


@st.cache_data(ttl=30, show_spinner=False)
def table_exists(table_name: str) -> bool:
    """
    Vérifie si une table existe dans le schéma public.

    Parameters
    ----------
    table_name : str
        Nom de la table PostgreSQL.

    Returns
    -------
    bool
        True si la table existe, sinon False.
    """
    if not DATABASE_URL:
        return False

    query = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = :table_name
    ) AS exists_flag
    """

    df = run_query(query, {"table_name": table_name})
    if df.empty:
        return False

    return bool(df.iloc[0]["exists_flag"])


@st.cache_data(ttl=30, show_spinner=False)
def get_table_columns(table_name: str) -> list[str]:
    """
    Retourne la liste des colonnes d'une table du schéma public.

    Parameters
    ----------
    table_name : str
        Nom de la table.

    Returns
    -------
    list[str]
        Liste ordonnée des colonnes.
    """
    if not table_exists(table_name):
        return []

    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = :table_name
    ORDER BY ordinal_position
    """
    df = run_query(query, {"table_name": table_name})

    if df.empty:
        return []

    return df["column_name"].tolist()


@st.cache_data(ttl=30, show_spinner=False)
def load_client_features(client_id: int, table_name: str) -> pd.DataFrame:
    """
    Charge les features d'un client depuis une table SQL autorisée.

    Parameters
    ----------
    client_id : int
        Identifiant client `SK_ID_CURR`.
    table_name : str
        Nom de la table source.

    Returns
    -------
    pd.DataFrame
        DataFrame d'une ligne maximum.

    Notes
    -----
    Le nom de table n'est pas paramétrable via SQLAlchemy `:param`,
    donc on applique une liste blanche de tables autorisées.
    """
    if table_name not in ALLOWED_FEATURES_TABLES:
        return pd.DataFrame()

    if not table_exists(table_name):
        return pd.DataFrame()

    query = f'SELECT * FROM "{table_name}" WHERE "SK_ID_CURR" = :client_id LIMIT 1'
    return run_query(query, {"client_id": client_id})


# =============================================================================
# Helpers métier
# =============================================================================

def safe_json_loads(value: Any) -> Any:
    """
    Essaie de parser une valeur JSON stockée en texte.

    Parameters
    ----------
    value : Any
        Valeur potentiellement JSON.

    Returns
    -------
    Any
        Objet Python parsé si possible, sinon valeur brute.
    """
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value

    return value


def format_datetime_col(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convertit proprement certaines colonnes en datetime pandas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    columns : list[str]
        Colonnes à convertir si elles existent.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame avec conversions appliquées.
    """
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


def call_predict_api(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """
    Appelle l'endpoint /predict de l'API FastAPI.

    Parameters
    ----------
    payload : dict[str, Any]
        Payload JSON envoyé à l'API.

    Returns
    -------
    tuple[bool, dict[str, Any]]
        - bool : True si appel réussi, sinon False
        - dict : réponse JSON ou détails d'erreur
    """
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


def dataframe_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    """
    Transforme une ligne DataFrame en payload JSON compatible API.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant une seule ligne logique.

    Returns
    -------
    dict[str, Any]
        Payload JSON sérialisable.
    """
    if df.empty:
        return {}

    row = df.iloc[0].replace({pd.NA: None})
    payload: dict[str, Any] = {}

    for key, value in row.items():
        if pd.isna(value):
            payload[key] = None
        elif isinstance(value, pd.Timestamp):
            payload[key] = value.isoformat()
        else:
            payload[key] = value.item() if hasattr(value, "item") else value

    return payload


def make_download_json(data: dict[str, Any], filename: str, label: str) -> None:
    """
    Ajoute un bouton de téléchargement JSON.

    Parameters
    ----------
    data : dict[str, Any]
        Données à exporter.
    filename : str
        Nom du fichier téléchargé.
    label : str
        Libellé du bouton.
    """
    st.download_button(
        label=label,
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


def query_if_table_exists(
    table_name: str,
    query: str,
    params: dict[str, Any] | None = None,
    datetime_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Exécute une requête uniquement si la table existe.

    Parameters
    ----------
    table_name : str
        Table attendue.
    query : str
        Requête SQL à exécuter.
    params : dict[str, Any] | None, default=None
        Paramètres SQL.
    datetime_cols : list[str] | None, default=None
        Colonnes à convertir en datetime.

    Returns
    -------
    pd.DataFrame
        Résultat de la requête ou DataFrame vide si la table n'existe pas.
    """
    if not table_exists(table_name):
        return pd.DataFrame()

    df = run_query(query, params)
    if datetime_cols:
        df = format_datetime_col(df, datetime_cols)

    return df


# =============================================================================
# Construction de l'interface
# =============================================================================

inject_css()

st.markdown(
    '<div class="main-title">Dashboard scoring crédit et monitoring</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Prédiction en direct, historique des appels et suivi du modèle dans une seule interface.</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.header("Paramètres")

    st.write(f"API_URL : `{API_URL}`")
    st.write(f"FEATURES_TABLE : `{FEATURES_TABLE}`")

    if DATABASE_URL:
        st.success("Connexion DB configurée")
    else:
        st.warning("DATABASE_URL manquant")

    if API_KEY:
        st.success("API_KEY configurée")
    else:
        st.info("API_KEY non définie. L'appel API sera tenté sans clé.")

    history_limit = st.slider(
        "Nb max lignes historique",
        min_value=20,
        max_value=1000,
        value=DEFAULT_LIMIT,
        step=20,
    )


# =============================================================================
# Onglets principaux
# =============================================================================

prediction_tab, history_tab, monitoring_tab = st.tabs([
    "Prédiction",
    "Historique",
    "Monitoring",
])


# =============================================================================
# Onglet Prédiction
# =============================================================================

with prediction_tab:
    st.markdown('<div class="section-title">Lancer une prédiction</div>', unsafe_allow_html=True)

    subtab_client, subtab_json = st.tabs([
        "Depuis un client en base",
        "Depuis un JSON libre",
    ])

    with subtab_client:
        st.write(
            "Recharge une ligne depuis une table SQL de features prêtes pour le modèle, "
            "puis envoie ce payload à l'API."
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            client_id = st.number_input(
                "SK_ID_CURR",
                min_value=100000,
                max_value=999999,
                value=100001,
                step=1,
            )

        with col2:
            selected_table = st.selectbox(
                "Table source",
                options=sorted(ALLOWED_FEATURES_TABLES),
                index=0 if FEATURES_TABLE not in ALLOWED_FEATURES_TABLES else sorted(ALLOWED_FEATURES_TABLES).index(FEATURES_TABLE),
            )

        fetch_clicked = st.button("Charger le client", use_container_width=True)

        if fetch_clicked:
            if not table_exists(selected_table):
                st.error(f"La table `{selected_table}` n'existe pas dans la base.")
            else:
                client_df = load_client_features(int(client_id), selected_table)

                if client_df.empty:
                    st.error("Aucune ligne trouvée pour ce client dans la table sélectionnée.")
                else:
                    st.success("Client chargé avec succès.")
                    st.dataframe(client_df, use_container_width=True)

                    payload = dataframe_to_payload(client_df)

                    # Si TARGET existe dans une table intermédiaire, on l'enlève
                    # pour éviter d'envoyer une cible d'entraînement à l'API.
                    payload.pop("TARGET", None)

                    with st.expander("Voir le payload JSON envoyé à l'API"):
                        st.json(payload)

                    make_download_json(
                        payload,
                        filename=f"payload_client_{client_id}.json",
                        label="Télécharger le payload JSON",
                    )

                    if st.button("Envoyer à l'API /predict", key="predict_from_db", use_container_width=True):
                        ok, result = call_predict_api(payload)

                        if ok:
                            st.success("Prédiction reçue.")
                            st.json(result)
                        else:
                            st.error("Échec de l'appel API.")
                            st.json(result)

    with subtab_json:
        st.write(
            "Colle un JSON brut pour tester l'endpoint `/predict` sans passer par la base."
        )

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

        col_a, col_b = st.columns(2)

        with col_a:
            validate_json = st.button("Valider le JSON", use_container_width=True)

        with col_b:
            send_json = st.button("Envoyer à l'API", use_container_width=True)

        parsed_payload: dict[str, Any] | None = None

        if validate_json or send_json:
            try:
                parsed_payload = json.loads(json_text)
                st.success("JSON valide.")
                st.json(parsed_payload)
            except json.JSONDecodeError as exc:
                st.error(f"JSON invalide : {exc}")

        if send_json and parsed_payload is not None:
            ok, result = call_predict_api(parsed_payload)

            if ok:
                st.success("Prédiction reçue.")
                st.json(result)
            else:
                st.error("Échec de l'appel API.")
                st.json(result)


# =============================================================================
# Onglet Historique
# =============================================================================

with history_tab:
    st.markdown('<div class="section-title">Historique des prédictions</div>', unsafe_allow_html=True)

    logs_query = """
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

    logs_df = query_if_table_exists(
        "prediction_logs",
        logs_query,
        {"limit_rows": history_limit},
        ["prediction_timestamp"],
    )

    if not table_exists("prediction_logs"):
        st.info("La table `prediction_logs` n'existe pas encore.")
    elif logs_df.empty:
        st.info("Aucune donnée dans `prediction_logs` pour le moment.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Prédictions chargées", len(logs_df))
        c2.metric(
            "Score moyen",
            round(float(logs_df["score"].mean()), 4) if "score" in logs_df else 0,
        )
        c3.metric(
            "Latence moyenne (ms)",
            round(float(logs_df["latency_ms"].dropna().mean()), 2) if "latency_ms" in logs_df else 0,
        )
        c4.metric(
            "Taux classe 1",
            f"{100 * logs_df['prediction'].mean():.1f}%" if "prediction" in logs_df else "0%",
        )

        left, right = st.columns([1.4, 1])

        with left:
            view_df = logs_df.copy()
            if "prediction_timestamp" in view_df.columns:
                view_df = view_df.sort_values("prediction_timestamp")
                st.line_chart(view_df.set_index("prediction_timestamp")[["score"]])

        with right:
            pred_counts = logs_df["prediction"].value_counts(dropna=False).sort_index()
            pred_counts.index = pred_counts.index.astype(str)
            st.bar_chart(pred_counts)

        st.dataframe(logs_df, use_container_width=True)

        request_ids = logs_df["request_id"].dropna().astype(str).tolist()

        selected_request_id = (
            st.selectbox(
                "Voir le détail d'une requête",
                options=request_ids,
                index=0 if request_ids else None,
            )
            if request_ids
            else None
        )

        if selected_request_id:
            detail_query = """
            SELECT request_id, client_id, model_name, model_version,
                   feature_name, feature_value, feature_type, snapshot_timestamp
            FROM prediction_features_snapshot
            WHERE request_id = :request_id
            ORDER BY feature_name
            """

            detail_df = query_if_table_exists(
                "prediction_features_snapshot",
                detail_query,
                {"request_id": selected_request_id},
                ["snapshot_timestamp"],
            )

            st.markdown("#### Snapshot de features")

            if not table_exists("prediction_features_snapshot"):
                st.info("La table `prediction_features_snapshot` n'existe pas encore.")
            elif detail_df.empty:
                st.info("Aucun snapshot de features pour cette requête.")
            else:
                st.dataframe(detail_df, use_container_width=True)

    gt_query = """
    SELECT id, request_id, client_id, true_label, label_source, observed_at, notes
    FROM ground_truth_labels
    ORDER BY observed_at DESC
    LIMIT :limit_rows
    """

    gt_df = query_if_table_exists(
        "ground_truth_labels",
        gt_query,
        {"limit_rows": history_limit},
        ["observed_at"],
    )

    with st.expander("Vérités terrain disponibles"):
        if not table_exists("ground_truth_labels"):
            st.info("La table `ground_truth_labels` n'existe pas encore.")
        elif gt_df.empty:
            st.info("Aucune ligne dans `ground_truth_labels`.")
        else:
            st.dataframe(gt_df, use_container_width=True)


# =============================================================================
# Onglet Monitoring
# =============================================================================

with monitoring_tab:
    st.markdown('<div class="section-title">Monitoring du modèle</div>', unsafe_allow_html=True)

    registry_df = query_if_table_exists(
        "model_registry",
        """
        SELECT id, model_name, model_version, stage, run_id, deployed_at,
               is_active, created_at, metrics, hyperparameters
        FROM model_registry
        ORDER BY created_at DESC
        """,
        datetime_cols=["deployed_at", "created_at"],
    )

    eval_df = query_if_table_exists(
        "evaluation_metrics",
        """
        SELECT model_name, model_version, dataset_name, window_start, window_end,
               roc_auc, pr_auc, precision_score, recall_score, f1_score,
               fbeta_score, business_cost, sample_size, computed_at
        FROM evaluation_metrics
        ORDER BY computed_at DESC
        LIMIT :limit_rows
        """,
        {"limit_rows": history_limit},
        ["window_start", "window_end", "computed_at"],
    )

    drift_df = query_if_table_exists(
        "drift_metrics",
        """
        SELECT model_name, model_version, feature_name, metric_name,
               metric_value, threshold_value, drift_detected, computed_at
        FROM drift_metrics
        ORDER BY computed_at DESC
        LIMIT :limit_rows
        """,
        {"limit_rows": history_limit},
        ["computed_at"],
    )

    alerts_df = query_if_table_exists(
        "alerts",
        """
        SELECT id, alert_type, severity, model_name, model_version,
               feature_name, title, message, status, created_at,
               acknowledged_at, resolved_at
        FROM alerts
        ORDER BY created_at DESC
        LIMIT :limit_rows
        """,
        {"limit_rows": history_limit},
        ["created_at", "acknowledged_at", "resolved_at"],
    )

    feature_store_df = query_if_table_exists(
        "feature_store_monitoring",
        """
        SELECT request_id, client_id, model_name, model_version,
               feature_name, feature_value, feature_type, source_table,
               snapshot_timestamp
        FROM feature_store_monitoring
        ORDER BY snapshot_timestamp DESC
        LIMIT :limit_rows
        """,
        {"limit_rows": history_limit},
        ["snapshot_timestamp"],
    )

    top1, top2, top3, top4 = st.columns(4)

    top1.metric("Versions de modèle", 0 if registry_df.empty else len(registry_df))
    top2.metric("Évaluations", 0 if eval_df.empty else len(eval_df))
    top3.metric(
        "Drifts détectés",
        0 if drift_df.empty else int(
            drift_df.get("drift_detected", pd.Series(dtype=bool)).fillna(False).sum()
        ),
    )
    top4.metric(
        "Alertes ouvertes",
        0 if alerts_df.empty else int(
            (alerts_df.get("status", pd.Series(dtype=str)) == "open").sum()
        ),
    )

    reg_col, alert_col = st.columns([1.2, 1])

    with reg_col:
        st.markdown("#### Registre des modèles")

        if not table_exists("model_registry"):
            st.info("La table `model_registry` n'existe pas encore.")
        elif registry_df.empty:
            st.info("Aucune donnée dans `model_registry`.")
        else:
            st.dataframe(registry_df, use_container_width=True)

            active_df = registry_df[registry_df["is_active"] == True]  # noqa: E712
            if not active_df.empty:
                row = active_df.iloc[0]
                st.success(
                    f"Modèle actif : {row['model_name']} | version {row['model_version']} | stage {row['stage']}"
                )

    with alert_col:
        st.markdown("#### Alertes")

        if not table_exists("alerts"):
            st.info("La table `alerts` n'existe pas encore.")
        elif alerts_df.empty:
            st.info("Aucune alerte enregistrée.")
        else:
            severity_counts = alerts_df["severity"].fillna("unknown").value_counts()
            st.bar_chart(severity_counts)
            st.dataframe(alerts_df, use_container_width=True)

    st.markdown("#### Performance")

    if not table_exists("evaluation_metrics"):
        st.info("La table `evaluation_metrics` n'existe pas encore.")
    elif eval_df.empty:
        st.info("Aucune métrique dans `evaluation_metrics`.")
    else:
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
            if c in eval_df.columns
        ]

        if metric_options:
            selected_metric = st.selectbox("Métrique à tracer", options=metric_options, index=0)
            plot_df = eval_df.dropna(subset=[selected_metric]).sort_values("computed_at")

            if not plot_df.empty and "computed_at" in plot_df.columns:
                chart_df = plot_df.set_index("computed_at")[[selected_metric]]
                st.line_chart(chart_df)

        st.dataframe(eval_df, use_container_width=True)

    st.markdown("#### Drift")

    if not table_exists("drift_metrics"):
        st.info("La table `drift_metrics` n'existe pas encore.")
    elif drift_df.empty:
        st.info("Aucune ligne dans `drift_metrics`.")
    else:
        only_drift = st.toggle("Afficher uniquement les drifts détectés", value=True)

        drift_view = drift_df.copy()
        if only_drift and "drift_detected" in drift_view.columns:
            drift_view = drift_view[drift_view["drift_detected"] == True]  # noqa: E712

        if not drift_view.empty:
            drift_counts = drift_view["feature_name"].value_counts().head(15)
            st.bar_chart(drift_counts)

        st.dataframe(drift_view, use_container_width=True)

    with st.expander("Feature store monitoring"):
        if not table_exists("feature_store_monitoring"):
            st.info("La table `feature_store_monitoring` n'existe pas encore.")
        elif feature_store_df.empty:
            st.info("Aucune donnée dans `feature_store_monitoring`.")
        else:
            st.dataframe(feature_store_df, use_container_width=True)


# =============================================================================
# Pied de page diagnostic
# =============================================================================

with st.expander("Diagnostic technique"):
    st.write("Heure de rendu :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("API_URL :", API_URL)
    st.write("DATABASE_URL configurée :", bool(DATABASE_URL))
    st.write("Tables de features autorisées :", sorted(ALLOWED_FEATURES_TABLES))

    features_columns = get_table_columns(FEATURES_TABLE)
    if features_columns:
        st.write(f"Nb colonnes dans `{FEATURES_TABLE}` :", len(features_columns))
        st.write("Aperçu des colonnes :", features_columns[:50])
    else:
        st.write(f"Aucune colonne récupérée pour `{FEATURES_TABLE}`.")