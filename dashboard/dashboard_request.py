"""
Requêtes HTTP centralisées pour le dashboard Streamlit.

Ce module regroupe tous les appels à l'API FastAPI utilisés
par le dashboard de scoring crédit et de monitoring MLOps.

Objectif
--------
Centraliser les appels API pour :
- améliorer la lisibilité du dashboard
- éviter de dupliquer les appels requests
- faciliter la maintenance
- rendre les pages Streamlit plus courtes et plus claires

Principe d'architecture
-----------------------
Le dashboard ne parle jamais directement à PostgreSQL.
Toutes les données sont récupérées via l'API FastAPI.

Endpoints concernés
-------------------
Prédiction
- GET /predict/health
- POST /predict
- GET /predict/{client_id}
- POST /predict/batch
- POST /predict/simulate/real-sample
- POST /predict/simulate/random

Historique
- GET /history/predictions
- GET /history/predictions/{request_id}
- GET /history/ground-truth
- GET /history/features/{request_id}

Monitoring
- GET /monitoring/summary
- GET /monitoring/health
- GET /monitoring/models
- GET /monitoring/active-model
- GET /monitoring/drift
- GET /monitoring/evaluation
- GET /monitoring/alerts
- GET /monitoring/feature-store

Analyse
- POST /analyse/evidently/run
- POST /analyse/evaluation/run

Features
- GET /features/client/{client_id}
  ou équivalent si cet endpoint existe côté API
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import requests


# =============================================================================
# Utilitaires HTTP génériques
# =============================================================================

DEFAULT_TIMEOUT = 30
DEFAULT_BATCH_TIMEOUT = 60
DEFAULT_SIMULATION_TIMEOUT = 120
DEFAULT_ANALYSIS_TIMEOUT = 300


def build_headers(api_key: str | None = None) -> dict[str, str]:
    """
    Construit les headers HTTP à envoyer à l'API.
    """
    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["X-API-Key"] = api_key

    return headers


def _normalize_base_url(base_url: str | None) -> str:
    """
    Nettoie l'URL de base de l'API.
    """
    if not base_url:
        return ""
    return str(base_url).strip().rstrip("/")


def _build_url(base_url: str, endpoint: str) -> str:
    """
    Construit l'URL finale à partir de l'URL de base et de l'endpoint.
    """
    return f"{base_url}/{endpoint.lstrip('/')}"


def _safe_json_response(response: requests.Response) -> Any:
    """
    Essaie de parser la réponse HTTP en JSON.
    """
    try:
        return response.json()
    except ValueError:
        text = response.text.strip()
        if text:
            return {"detail": text}
        return {"detail": "Réponse sans JSON."}


def call_api(
    endpoint: str,
    *,
    base_url: str,
    api_key: str | None = None,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    json_data: Any | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Appelle un endpoint de l'API FastAPI.
    """
    normalized_base_url = _normalize_base_url(base_url)

    if not normalized_base_url:
        return False, {"detail": "API_URL non configurée côté dashboard."}

    url = _build_url(normalized_base_url, endpoint)
    response: requests.Response | None = None

    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=build_headers(api_key),
            params=params,
            json=json_data,
            timeout=timeout,
        )
        response.raise_for_status()
        return True, _safe_json_response(response)

    except requests.Timeout:
        return False, {
            "detail": f"Timeout HTTP sur {method.upper()} {url} après {timeout}s."
        }

    except requests.RequestException as exc:
        if response is not None:
            return False, _safe_json_response(response)

        return False, {"detail": str(exc)}


# =============================================================================
# Helpers de conversion
# =============================================================================

def items_payload_to_dataframe(payload: Any) -> pd.DataFrame:
    """
    Convertit une réponse API en DataFrame.
    """
    if isinstance(payload, pd.DataFrame):
        return payload.copy()

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if not isinstance(payload, dict):
        return pd.DataFrame()

    for key in ["items", "data", "results"]:
        value = payload.get(key)
        if isinstance(value, list):
            return pd.DataFrame(value)

    return pd.DataFrame()


def dict_payload_to_dataframe(payload: Any) -> pd.DataFrame:
    """
    Convertit un dictionnaire JSON simple en DataFrame à une ligne.
    """
    if isinstance(payload, pd.DataFrame):
        return payload.copy()

    if not isinstance(payload, dict) or not payload:
        return pd.DataFrame()

    return pd.DataFrame([payload])


def dataframe_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    """
    Convertit un DataFrame de features client en payload JSON pour l'API.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    row = df.iloc[0].to_dict()
    client_id = row.pop("SK_ID_CURR", None)

    payload = {
        "features": row,
    }

    if client_id is not None and not pd.isna(client_id):
        try:
            payload["SK_ID_CURR"] = int(client_id)
        except Exception:
            payload["SK_ID_CURR"] = client_id

    return payload


def _coerce_datetime_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Convertit une liste de colonnes en datetime si elles existent.
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return df.copy()

    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Convertit une liste de colonnes en numérique si elles existent.
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return df.copy()

    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _postprocess_prediction_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-traitement standard des logs de prédiction.
    """
    df = _coerce_datetime_columns(df, ["prediction_timestamp", "created_at"])
    df = _coerce_numeric_columns(
        df,
        ["client_id", "prediction", "score", "threshold", "threshold_used", "latency_ms"],
    )

    if "prediction_timestamp" in df.columns:
        df = df.sort_values("prediction_timestamp", ascending=False, na_position="last")
    elif "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False, na_position="last")

    return df


def _postprocess_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-traitement standard des vérités terrain.
    """
    df = _coerce_datetime_columns(df, ["observed_at", "created_at", "gt_created_at"])
    df = _coerce_numeric_columns(df, ["client_id", "ground_truth", "y_true"])

    if "observed_at" in df.columns:
        df = df.sort_values("observed_at", ascending=False, na_position="last")
    elif "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False, na_position="last")
    elif "gt_created_at" in df.columns:
        df = df.sort_values("gt_created_at", ascending=False, na_position="last")

    return df


def _postprocess_monitoring_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-traitement standard du registre des modèles.
    """
    df = _coerce_datetime_columns(df, ["deployed_at", "created_at"])

    if "deployed_at" in df.columns:
        df = df.sort_values("deployed_at", ascending=False, na_position="last")
    elif "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False, na_position="last")

    return df


def _postprocess_evaluation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-traitement standard des métriques d'évaluation.
    """
    df = _coerce_datetime_columns(df, ["computed_at", "window_start", "window_end"])
    df = _coerce_numeric_columns(
        df,
        [
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

    if "computed_at" in df.columns:
        df = df.sort_values("computed_at", ascending=False, na_position="last")

    return df


def _postprocess_drift_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-traitement standard des métriques de drift.
    """
    df = _coerce_datetime_columns(
        df,
        [
            "computed_at",
            "reference_window_start",
            "reference_window_end",
            "current_window_start",
            "current_window_end",
        ],
    )
    df = _coerce_numeric_columns(df, ["metric_value", "threshold_value"])

    if "computed_at" in df.columns:
        df = df.sort_values("computed_at", ascending=False, na_position="last")

    return df


def _postprocess_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-traitement standard des alertes.
    """
    df = _coerce_datetime_columns(df, ["created_at", "acknowledged_at", "resolved_at"])

    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False, na_position="last")

    return df


def _postprocess_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-traitement standard du feature store monitoring.
    """
    df = _coerce_datetime_columns(df, ["snapshot_timestamp"])
    df = _coerce_numeric_columns(df, ["client_id"])

    if "snapshot_timestamp" in df.columns:
        df = df.sort_values("snapshot_timestamp", ascending=False, na_position="last")

    return df


def metric_safe_number(
    df: pd.DataFrame,
    col: str,
    metric: str,
    default: float | int | None = 0,
) -> float | int | None:
    """
    Calcule proprement une métrique numérique sur une colonne.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return default

    series = pd.to_numeric(df[col], errors="coerce").dropna()

    if series.empty:
        return default

    try:
        if metric == "mean":
            return float(series.mean())
        if metric == "min":
            return float(series.min())
        if metric == "max":
            return float(series.max())
        if metric == "median":
            return float(series.median())
        if metric == "p95":
            return float(series.quantile(0.95))
        if metric == "p99":
            return float(series.quantile(0.99))
        if metric == "sum":
            return float(series.sum())
    except Exception:
        return default

    return default


def normalize_prediction_result(payload: Any) -> dict[str, Any]:
    """
    Normalise un résultat de prédiction unitaire en dictionnaire.
    """
    if isinstance(payload, dict):
        return payload
    return {"result": payload}


# =============================================================================
# Endpoints santé / système
# =============================================================================

def get_health(
    *,
    base_url: str,
) -> dict[str, Any]:
    """
    Récupère l'état de santé de l'API.
    """
    ok, result = call_api(
        "/predict/health",
        base_url=base_url,
        method="GET",
        api_key=None,
    )

    if ok and isinstance(result, dict):
        return result

    return {}


# =============================================================================
# Endpoints features
# =============================================================================

def load_client_features(
    client_id: int,
    source_table: str,
    *,
    base_url: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Charge les features d'un client via l'API.

    Notes
    -----
    Cet endpoint est optionnel et dépend de l'existence côté API
    d'une route de type `/features/client/{client_id}`.
    """
    ok, result = call_api(
        f"/features/client/{client_id}",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params={"source_table": source_table},
    )

    if not ok:
        return pd.DataFrame()

    if isinstance(result, dict):
        if "items" in result:
            return items_payload_to_dataframe(result)

        if "features" in result and isinstance(result["features"], dict):
            row = {"SK_ID_CURR": client_id, **result["features"]}
            return pd.DataFrame([row])

        if "data" in result and isinstance(result["data"], dict):
            row = {"SK_ID_CURR": client_id, **result["data"]}
            return pd.DataFrame([row])

    return pd.DataFrame()


# =============================================================================
# Endpoints prédiction
# =============================================================================

def call_predict_api(
    payload: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Appelle l'endpoint de prédiction unitaire via payload JSON.
    """
    return call_api(
        "/predict",
        base_url=base_url,
        api_key=api_key,
        method="POST",
        json_data=payload,
        timeout=timeout,
    )


def call_predict_client_api(
    client_id: int,
    *,
    base_url: str,
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Appelle l'endpoint de prédiction unitaire à partir d'un identifiant client.
    """
    return call_api(
        f"/predict/{int(client_id)}",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        timeout=timeout,
    )


def call_predict_batch_api(
    payloads: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    timeout: int = DEFAULT_BATCH_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Appelle l'endpoint de prédiction batch.
    """
    return call_api(
        "/predict/batch",
        base_url=base_url,
        api_key=api_key,
        method="POST",
        json_data=payloads,
        timeout=timeout,
    )


def call_predict_real_random_batch_api(
    *,
    batch_size: int,
    base_url: str,
    api_key: str,
    random_seed: int | None = None,
    timeout: int = DEFAULT_SIMULATION_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Appelle l'endpoint de simulation basé sur des clients réels tirés aléatoirement.
    """
    params: dict[str, Any] = {"limit": batch_size}

    if random_seed is not None:
        params["random_seed"] = random_seed

    return call_api(
        "/predict/simulate/real-sample",
        base_url=base_url,
        api_key=api_key,
        method="POST",
        params=params,
        timeout=timeout,
    )


def call_predict_fully_random_batch_api(
    *,
    batch_size: int,
    base_url: str,
    api_key: str,
    timeout: int = DEFAULT_SIMULATION_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Appelle l'endpoint de simulation basé sur des données totalement aléatoires.
    """
    return call_api(
        "/predict/simulate/random",
        base_url=base_url,
        api_key=api_key,
        method="POST",
        params={"limit": batch_size},
        timeout=timeout,
    )


def prediction_result_to_dataframe(payload: Any) -> pd.DataFrame:
    """
    Convertit un résultat de prédiction unitaire en DataFrame.
    """
    return dict_payload_to_dataframe(payload)


def simulation_result_to_dataframe(payload: Any) -> pd.DataFrame:
    """
    Convertit un résultat de simulation batch en DataFrame.
    """
    return items_payload_to_dataframe(payload)


# =============================================================================
# Endpoints historique
# =============================================================================

def get_prediction_history(
    *,
    base_url: str,
    api_key: str,
    limit: int = 200,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    only_errors: bool = False,
    decision: str | None = None,
) -> pd.DataFrame:
    """
    Récupère l'historique des prédictions.
    """
    params: dict[str, Any] = {"limit": limit}

    if client_id is not None:
        params["client_id"] = client_id

    if model_name is not None:
        params["model_name"] = model_name

    if model_version is not None:
        params["model_version"] = model_version

    if only_errors:
        params["only_errors"] = True

    if decision is not None:
        params["decision"] = decision

    ok, result = call_api(
        "/history/predictions",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok:
        return _postprocess_prediction_logs(items_payload_to_dataframe(result))

    return pd.DataFrame()


def get_prediction_detail(
    request_id: str,
    *,
    base_url: str,
    api_key: str,
) -> dict[str, Any] | None:
    """
    Récupère le détail d'une prédiction.
    """
    ok, result = call_api(
        f"/history/predictions/{request_id}",
        base_url=base_url,
        api_key=api_key,
        method="GET",
    )

    if ok and isinstance(result, dict):
        return result

    return None


def get_ground_truth_history(
    *,
    base_url: str,
    api_key: str,
    limit: int = 200,
    client_id: int | None = None,
    request_id: str | None = None,
) -> pd.DataFrame:
    """
    Récupère l'historique des vérités terrain.
    """
    params: dict[str, Any] = {"limit": limit}

    if client_id is not None:
        params["client_id"] = client_id

    if request_id is not None:
        params["request_id"] = request_id

    ok, result = call_api(
        "/history/ground-truth",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok:
        return _postprocess_ground_truth(items_payload_to_dataframe(result))

    return pd.DataFrame()


def get_prediction_features_snapshot(
    request_id: str,
    *,
    base_url: str,
    api_key: str,
) -> dict[str, Any] | None:
    """
    Récupère le snapshot de features d'une requête.
    """
    ok, result = call_api(
        f"/history/features/{request_id}",
        base_url=base_url,
        api_key=api_key,
        method="GET",
    )

    if ok and isinstance(result, dict):
        return result

    return None


def get_prediction_features_snapshot_df(
    request_id: str,
    *,
    base_url: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Récupère le snapshot de features d'une requête et le convertit en DataFrame.
    """
    payload = get_prediction_features_snapshot(
        request_id,
        base_url=base_url,
        api_key=api_key,
    )

    if not payload:
        return pd.DataFrame()

    items = payload.get("items")
    if isinstance(items, list):
        return pd.DataFrame(items)

    return pd.DataFrame()


def get_ground_truth_by_request_id(
    request_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    ground_truth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Retourne la vérité terrain pour une requête donnée.
    """
    if ground_truth_df is not None:
        if ground_truth_df.empty or "request_id" not in ground_truth_df.columns:
            return pd.DataFrame()

        return ground_truth_df[
            ground_truth_df["request_id"].astype(str) == str(request_id)
        ].copy()

    if base_url and api_key:
        return get_ground_truth_history(
            base_url=base_url,
            api_key=api_key,
            limit=50,
            request_id=request_id,
        )

    return pd.DataFrame()


# =============================================================================
# Endpoints monitoring
# =============================================================================

def get_monitoring_summary(
    *,
    base_url: str,
    api_key: str,
    model_name: str = "credit_scoring_model",
    model_version: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> dict[str, Any]:
    """
    Récupère le résumé global de monitoring.
    """
    params: dict[str, Any] = {"model_name": model_name}

    if model_version is not None:
        params["model_version"] = model_version

    if window_start is not None and window_end is not None:
        params["window_start"] = window_start
        params["window_end"] = window_end

    ok, result = call_api(
        "/monitoring/summary",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok and isinstance(result, dict):
        return result

    return {}


def get_monitoring_health(
    *,
    base_url: str,
    api_key: str,
    model_name: str = "credit_scoring_model",
    model_version: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> dict[str, Any]:
    """
    Récupère l'état lisible du monitoring.
    """
    params: dict[str, Any] = {"model_name": model_name}

    if model_version is not None:
        params["model_version"] = model_version

    if window_start is not None and window_end is not None:
        params["window_start"] = window_start
        params["window_end"] = window_end

    ok, result = call_api(
        "/monitoring/health",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok and isinstance(result, dict):
        return result

    return {}


def get_models(
    *,
    base_url: str,
    api_key: str,
    limit: int = 200,
    model_name: str | None = None,
    is_active: bool | None = None,
) -> pd.DataFrame:
    """
    Récupère le registre des modèles.
    """
    params: dict[str, Any] = {"limit": limit}

    if model_name is not None:
        params["model_name"] = model_name

    if is_active is not None:
        params["is_active"] = is_active

    ok, result = call_api(
        "/monitoring/models",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok:
        return _postprocess_monitoring_models(items_payload_to_dataframe(result))

    return pd.DataFrame()


def get_active_model(
    *,
    base_url: str,
    api_key: str,
    model_name: str | None = None,
) -> pd.DataFrame:
    """
    Récupère le modèle actif.
    """
    params: dict[str, Any] = {}

    if model_name is not None:
        params["model_name"] = model_name

    ok, result = call_api(
        "/monitoring/active-model",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok and isinstance(result, dict):
        return _postprocess_monitoring_models(pd.DataFrame([result]))

    return pd.DataFrame()


def get_evaluation_metrics(
    *,
    base_url: str,
    api_key: str,
    limit: int = 200,
    model_name: str | None = None,
    model_version: str | None = None,
    dataset_name: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> pd.DataFrame:
    """
    Récupère les métriques d'évaluation.
    """
    params: dict[str, Any] = {"limit": limit}

    if model_name is not None:
        params["model_name"] = model_name

    if model_version is not None:
        params["model_version"] = model_version

    if dataset_name is not None:
        params["dataset_name"] = dataset_name

    if window_start is not None and window_end is not None:
        params["window_start"] = window_start
        params["window_end"] = window_end

    ok, result = call_api(
        "/monitoring/evaluation",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok:
        return _postprocess_evaluation_metrics(items_payload_to_dataframe(result))

    return pd.DataFrame()


def get_drift_metrics(
    *,
    base_url: str,
    api_key: str,
    limit: int = 200,
    model_name: str | None = None,
    model_version: str | None = None,
    feature_name: str | None = None,
    metric_name: str | None = None,
    drift_detected: bool | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> pd.DataFrame:
    """
    Récupère les métriques de drift.
    """
    params: dict[str, Any] = {"limit": limit}

    if model_name is not None:
        params["model_name"] = model_name

    if model_version is not None:
        params["model_version"] = model_version

    if feature_name is not None:
        params["feature_name"] = feature_name

    if metric_name is not None:
        params["metric_name"] = metric_name

    if drift_detected is not None:
        params["drift_detected"] = drift_detected

    if window_start is not None and window_end is not None:
        params["window_start"] = window_start
        params["window_end"] = window_end

    ok, result = call_api(
        "/monitoring/drift",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok:
        return _postprocess_drift_metrics(items_payload_to_dataframe(result))

    return pd.DataFrame()


def get_alerts(
    *,
    base_url: str,
    api_key: str,
    limit: int = 200,
    status_filter: str | None = None,
    severity: str | None = None,
    alert_type: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    feature_name: str | None = None,
) -> pd.DataFrame:
    """
    Récupère les alertes de monitoring.
    """
    params: dict[str, Any] = {"limit": limit}

    if status_filter is not None:
        params["status"] = status_filter

    if severity is not None:
        params["severity"] = severity

    if alert_type is not None:
        params["alert_type"] = alert_type

    if model_name is not None:
        params["model_name"] = model_name

    if model_version is not None:
        params["model_version"] = model_version

    if feature_name is not None:
        params["feature_name"] = feature_name

    ok, result = call_api(
        "/monitoring/alerts",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok:
        return _postprocess_alerts(items_payload_to_dataframe(result))

    return pd.DataFrame()


def get_feature_store_monitoring(
    *,
    base_url: str,
    api_key: str,
    limit: int = 200,
    request_id: str | None = None,
    client_id: int | None = None,
    feature_name: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    source_table: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> pd.DataFrame:
    """
    Récupère les données du feature store de monitoring.
    """
    params: dict[str, Any] = {"limit": limit}

    if request_id is not None:
        params["request_id"] = request_id

    if client_id is not None:
        params["client_id"] = client_id

    if feature_name is not None:
        params["feature_name"] = feature_name

    if model_name is not None:
        params["model_name"] = model_name

    if model_version is not None:
        params["model_version"] = model_version

    if source_table is not None:
        params["source_table"] = source_table

    if window_start is not None and window_end is not None:
        params["window_start"] = window_start
        params["window_end"] = window_end

    ok, result = call_api(
        "/monitoring/feature-store",
        base_url=base_url,
        api_key=api_key,
        method="GET",
        params=params,
    )

    if ok:
        return _postprocess_feature_store(items_payload_to_dataframe(result))

    return pd.DataFrame()


# =============================================================================
# Helpers dashboard
# =============================================================================

def build_tables_status_dataframe(
    *,
    prediction_logs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    model_registry_df: pd.DataFrame,
    feature_store_monitoring_df: pd.DataFrame,
    drift_metrics_df: pd.DataFrame,
    evaluation_metrics_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit un DataFrame de synthèse sur les ressources visibles
    dans le dashboard.
    """
    rows = [
        {
            "resource_name": "prediction_logs",
            "is_available": not prediction_logs_df.empty,
            "row_count": len(prediction_logs_df),
            "comment": "Historique des prédictions",
        },
        {
            "resource_name": "ground_truth_labels",
            "is_available": not ground_truth_df.empty,
            "row_count": len(ground_truth_df),
            "comment": "Vérités terrain",
        },
        {
            "resource_name": "model_registry",
            "is_available": not model_registry_df.empty,
            "row_count": len(model_registry_df),
            "comment": "Registre des modèles",
        },
        {
            "resource_name": "feature_store_monitoring",
            "is_available": not feature_store_monitoring_df.empty,
            "row_count": len(feature_store_monitoring_df),
            "comment": "Features de monitoring",
        },
        {
            "resource_name": "drift_metrics",
            "is_available": not drift_metrics_df.empty,
            "row_count": len(drift_metrics_df),
            "comment": "Métriques de drift",
        },
        {
            "resource_name": "evaluation_metrics",
            "is_available": not evaluation_metrics_df.empty,
            "row_count": len(evaluation_metrics_df),
            "comment": "Métriques d'évaluation",
        },
        {
            "resource_name": "alerts",
            "is_available": not alerts_df.empty,
            "row_count": len(alerts_df),
            "comment": "Alertes de monitoring",
        },
    ]

    return pd.DataFrame(rows)


def build_preview_map(
    *,
    prediction_logs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    model_registry_df: pd.DataFrame,
    feature_store_monitoring_df: pd.DataFrame,
    drift_metrics_df: pd.DataFrame,
    evaluation_metrics_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    max_rows: int = 200,
) -> dict[str, pd.DataFrame]:
    """
    Construit le mapping des aperçus de ressources pour la page système.
    """
    return {
        "prediction_logs": prediction_logs_df.head(max_rows),
        "ground_truth_labels": ground_truth_df.head(max_rows),
        "model_registry": model_registry_df.head(max_rows),
        "feature_store_monitoring": feature_store_monitoring_df.head(max_rows),
        "drift_metrics": drift_metrics_df.head(max_rows),
        "evaluation_metrics": evaluation_metrics_df.head(max_rows),
        "alerts": alerts_df.head(max_rows),
    }


# =============================================================================
# Endpoints analyse
# =============================================================================

def run_evidently_analysis(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    model_version: str | None = None,
    reference_kind: str = "transformed",
    current_kind: str = "transformed",
    monitoring_dir: str | None = None,
    save_html_path: str | None = "artifacts/evidently/report.html",
    timeout: int = DEFAULT_ANALYSIS_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Lance une analyse Evidently via l'API FastAPI.

    Notes
    -----
    Cette fonction appelle la route :
    - POST /analyse/evidently/run

    Les paramètres sont passés en query params pour rester cohérents
    avec la signature actuelle de la route FastAPI.
    """
    params: dict[str, Any] = {
        "model_name": model_name,
        "reference_kind": reference_kind,
        "current_kind": current_kind,
    }

    if model_version is not None:
        params["model_version"] = model_version

    if monitoring_dir is not None:
        params["monitoring_dir"] = monitoring_dir

    if save_html_path is not None:
        params["save_html_path"] = save_html_path

    return call_api(
        "/analyse/evidently/run",
        base_url=base_url,
        api_key=api_key,
        method="POST",
        params=params,
        timeout=timeout,
    )


def run_monitoring_evaluation_analysis(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    model_version: str | None = None,
    dataset_name: str = "scoring_prod",
    window_start: str | None = None,
    window_end: str | None = None,
    beta: float = 2.0,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    timeout: int = DEFAULT_ANALYSIS_TIMEOUT,
) -> tuple[bool, Any]:
    """
    Lance une analyse d'évaluation monitoring via l'API FastAPI.

    Notes
    -----
    Cette fonction appelle la route :
    - POST /analyse/evaluation/run

    Les paramètres sont passés en query params pour rester cohérents
    avec la signature actuelle de la route FastAPI.
    """
    params: dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "beta": beta,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
    }

    if model_version is not None:
        params["model_version"] = model_version

    if window_start is not None:
        params["window_start"] = window_start

    if window_end is not None:
        params["window_end"] = window_end

    return call_api(
        "/analyse/evaluation/run",
        base_url=base_url,
        api_key=api_key,
        method="POST",
        params=params,
        timeout=timeout,
    )