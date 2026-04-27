"""
Service de prédiction du modèle de scoring crédit.

Ce module encapsule la logique métier liée à l'inférence du modèle.
Il ne recharge pas le modèle à chaque appel : le modèle et le seuil
de décision sont récupérés depuis `app.services.loader_services.model_loading_service`,
qui les charge une seule fois au démarrage de l'application.

Fonctionnalités
---------------
- prédiction unitaire à partir d'un dictionnaire de features
- prédiction batch à partir d'une liste de payloads
- simulation batch à partir de clients réels choisis aléatoirement
- simulation batch à partir de données artificielles construites
  à partir du profil observé dans la source CSV configurée
- journalisation des résultats en base PostgreSQL
- enregistrement d'une vérité terrain liée à une prédiction
- validation légère des entrées

Architecture actuelle
---------------------
- les données source proviennent du CSV configuré via APPLICATION_CSV
- les features sont construites en mémoire via `features_builder_service`
- PostgreSQL sert uniquement au logging, à l'historique et au monitoring

Notes
-----
- Le score retourné correspond à la probabilité de la classe positive.
- La décision finale est calculée avec le seuil chargé depuis
  `app.services.loader_services.model_loading_service`.
- Les simulations "réelles" utilisent les clients présents dans
  la source applicative configurée.
- Les commits et rollbacks sont gérés par l'appelant.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import MODEL_NAME, MODEL_VERSION
from app.crud import prediction as prediction_crud
from app.services.features_builder_service import build_features_from_loaded_data
from app.services.loader_services.data_loading_service import (
    get_data_cache,
    get_features_for_client_from_cache,
)
from app.services.loader_services.model_loading_service import (
    get_threshold,
    predict_proba_with_backend,
)
from app.services.prediction_logging_service import PredictionLoggingService


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

MAX_BATCH_SIZE = 200
CLIENT_ID_COLUMN = "SK_ID_CURR"
RANDOM_PROFILE_SAMPLE_SIZE = 1000


# =============================================================================
# Helpers de validation
# =============================================================================

def _ensure_dataframe_from_dict(features: dict[str, Any]) -> pd.DataFrame:
    """
    Convertit un dictionnaire de features en DataFrame à une seule ligne.
    """
    if not isinstance(features, dict):
        raise TypeError("`features` doit être un dictionnaire Python.")

    if not features:
        raise ValueError(
            "`features` est vide. Impossible de prédire sans variables d'entrée."
        )

    return pd.DataFrame([features])


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie qu'un objet est bien un DataFrame pandas exploitable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'entrée doit être un DataFrame pandas.")

    if df.empty:
        raise ValueError(
            "Le DataFrame fourni est vide. Impossible de lancer une prédiction."
        )

    return df


def _ensure_batch_size(size: int) -> None:
    """
    Vérifie qu'un batch respecte la limite maximale autorisée.
    """
    if size <= 0:
        raise ValueError("Le batch est vide.")

    if size > MAX_BATCH_SIZE:
        raise ValueError(
            f"Le batch contient {size} éléments. "
            f"La limite autorisée est {MAX_BATCH_SIZE}."
        )


def _ensure_non_string_sequence(values: Sequence[Any], name: str) -> None:
    """
    Vérifie qu'une entrée est bien une séquence non vide
    et qu'il ne s'agit pas d'une chaîne de caractères.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"`{name}` doit être une séquence et non une chaîne de caractères."
        )

    if not isinstance(values, Sequence):
        raise TypeError(f"`{name}` doit être une séquence Python.")

    if len(values) == 0:
        raise ValueError(f"`{name}` doit contenir au moins un élément.")


def _clean_feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    """
    Remplace les NaN pandas par None dans un dictionnaire de features.
    """
    clean_features: dict[str, Any] = {}

    for key, value in features.items():
        try:
            clean_features[key] = None if pd.isna(value) else value
        except Exception:
            clean_features[key] = value

    return clean_features


def _validate_ground_truth_inputs(
    *,
    request_id: str | None,
    client_id: int | None,
    true_label: int,
    observed_at: datetime | None,
) -> None:
    """
    Valide les paramètres métier d'un ground truth.
    """
    if request_id is None and client_id is None:
        raise ValueError(
            "Au moins un identifiant doit être fourni : `request_id` ou `client_id`."
        )

    if true_label not in {0, 1}:
        raise ValueError("`true_label` doit valoir 0 ou 1.")

    if observed_at is not None and not isinstance(observed_at, datetime):
        raise ValueError("`observed_at` doit être un datetime valide.")


# =============================================================================
# Helpers techniques
# =============================================================================

def _safe_request_id() -> str:
    """
    Génère un identifiant de requête robuste.

    Notes
    -----
    Utile notamment en tests quand `uuid.uuid4` est mocké avec un
    `side_effect` trop court.
    """
    try:
        return str(uuid.uuid4())
    except Exception:
        return (
            f"fallback-{int(time.time() * 1_000_000)}-"
            f"{random.randint(1000, 9999)}"
        )


# =============================================================================
# Helpers modèle
# =============================================================================

def _predict_scores(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    Calcule les scores de probabilité de la classe positive
    avec le backend configuré : sklearn ou ONNX.
    """
    df = _ensure_dataframe(df)

    inference_start = time.perf_counter()

    scores_list: list[float] = []

    for _, row in df.iterrows():
        features = _clean_feature_dict(row.to_dict())
        score = predict_proba_with_backend(pd.DataFrame([features]))
        scores_list.append(float(score))

    inference_latency_ms = (time.perf_counter() - inference_start) * 1000

    scores = pd.Series(scores_list, index=df.index, dtype="float64")
    return scores, inference_latency_ms


def _predict_raw(features: dict[str, Any]) -> tuple[int, float, float, float]:
    """
    Réalise la prédiction brute sans journalisation.
    """
    threshold = float(get_threshold())
    df = _ensure_dataframe_from_dict(features)

    scores, inference_latency_ms = _predict_scores(df)
    score = float(scores.iloc[0])
    prediction = int(score >= threshold)

    return prediction, score, threshold, inference_latency_ms


# =============================================================================
# Helpers clients / simulation
# =============================================================================

def _extract_application_df() -> pd.DataFrame:
    """
    Extrait le DataFrame applicatif brut depuis le cache.
    """
    cache = get_data_cache()

    if isinstance(cache, pd.DataFrame):
        return cache.copy()

    if isinstance(cache, dict):
        if "application" in cache:
            df = cache["application"]
        elif "application_test" in cache:
            df = cache["application_test"]
        elif "app" in cache:
            df = cache["app"]
        else:
            raise ValueError(
                "Le cache ne contient ni 'application', ni 'application_test', ni 'app'."
            )

        if not isinstance(df, pd.DataFrame):
            raise TypeError("La source extraite depuis le cache n'est pas un DataFrame.")

        return df.copy()

    raise TypeError(
        "Le cache de données doit être un DataFrame ou un dictionnaire de DataFrames."
    )


def extract_existing_client_ids(
    df: pd.DataFrame,
    client_id_column: str = CLIENT_ID_COLUMN,
) -> list[int]:
    """
    Extrait la liste des identifiants clients existants depuis un DataFrame.
    """
    df = _ensure_dataframe(df)

    if client_id_column not in df.columns:
        raise ValueError(
            f"La colonne `{client_id_column}` est absente du DataFrame source."
        )

    client_ids = (
        pd.to_numeric(df[client_id_column], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .tolist()
    )

    if not client_ids:
        raise ValueError(
            f"Aucun identifiant client exploitable trouvé dans `{client_id_column}`."
        )

    logger.info(
        "Existing client IDs extracted",
        extra={
            "extra_data": {
                "event": "prediction_service_extract_client_ids_success",
                "client_id_column": client_id_column,
                "count": len(client_ids),
            }
        },
    )

    return client_ids


def get_random_existing_client_ids(
    df: pd.DataFrame,
    limit: int = MAX_BATCH_SIZE,
    client_id_column: str = CLIENT_ID_COLUMN,
    random_seed: int | None = None,
) -> list[int]:
    """
    Sélectionne aléatoirement des identifiants clients existants.
    """
    _ensure_batch_size(limit)

    client_ids = extract_existing_client_ids(
        df=df,
        client_id_column=client_id_column,
    )

    if len(client_ids) < limit:
        raise ValueError(
            f"Seulement {len(client_ids)} clients disponibles, "
            f"impossible d'en tirer {limit} aléatoirement."
        )

    rng = random.Random(random_seed)
    selected = rng.sample(client_ids, limit)

    logger.info(
        "Random existing client IDs selected",
        extra={
            "extra_data": {
                "event": "prediction_service_select_random_client_ids_success",
                "requested_limit": limit,
                "available_clients": len(client_ids),
                "selected_count": len(selected),
                "random_seed": random_seed,
            }
        },
    )

    return selected


def _load_client_source_dataframe() -> pd.DataFrame:
    """
    Charge la source client utilisée pour sélectionner des SK_ID_CURR réels.
    """
    app_df = _extract_application_df()

    if CLIENT_ID_COLUMN not in app_df.columns:
        raise ValueError(
            f"La colonne `{CLIENT_ID_COLUMN}` est absente de la source applicative."
        )

    return app_df[[CLIENT_ID_COLUMN]].copy()


def _generate_random_value_from_series(series: pd.Series) -> Any:
    """
    Génère une valeur artificielle volontairement décalée pour simuler du drift.

    Objectif
    --------
    Cette fonction ne cherche pas à générer un client réaliste.
    Elle sert à produire une population volontairement différente
    de la population de référence afin de tester le monitoring de drift.
    """
    clean = series.dropna()

    if clean.empty:
        return None

    numeric = pd.to_numeric(clean, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    # Colonnes numériques
    if numeric_ratio >= 0.90:
        numeric_clean = numeric.dropna()

        if numeric_clean.empty:
            return None

        if pd.api.types.is_integer_dtype(clean):
            min_value = int(numeric_clean.min())
            max_value = int(numeric_clean.max())

            if min_value == max_value:
                return min_value

            return random.randint(min_value, max_value)

        q05 = numeric_clean.quantile(0.05)
        q25 = numeric_clean.quantile(0.25)
        q75 = numeric_clean.quantile(0.75)
        q95 = numeric_clean.quantile(0.95)

        col = str(series.name or "")

        # Drift métier ciblé : population plus risquée
        if col == "AMT_INCOME_TOTAL":
            value = random.uniform(float(q05), float(q25))

        elif col in {"AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}:
            value = random.uniform(float(q75), float(q95))

        elif col in {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}:
            value = random.uniform(float(q05), float(q25))

        elif col in {"DAYS_BIRTH"}:
            value = random.uniform(float(q05), float(q25))

        elif col in {"DAYS_EMPLOYED"}:
            value = random.uniform(float(q05), float(q25))

        else:
            # Drift générique : on tire surtout dans les queues de distribution
            if random.random() < 0.5:
                value = random.uniform(float(q05), float(q25))
            else:
                value = random.uniform(float(q75), float(q95))

        is_integer_like = (
            pd.api.types.is_integer_dtype(numeric_clean)
            or all(
                float(x).is_integer()
                for x in numeric_clean.head(min(len(numeric_clean), 20))
            )
        )

        if is_integer_like:
            return int(round(value))

        return float(value)

    # Colonnes catégorielles
    values = clean.astype(str).value_counts(normalize=True)

    if values.empty:
        return None

    # Drift catégoriel : on favorise les modalités rares
    rare_values = values.sort_values(ascending=True).head(
        min(5, len(values))
    ).index.tolist()

    return random.choice(rare_values)

def _build_random_feature_rows_from_application(
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Construit des lignes artificielles aléatoires à partir du profil
    observé dans la source applicative CSV.
    """
    app_df = _extract_application_df()
    app_df = _ensure_dataframe(app_df)

    sample_df = app_df.sample(
        n=min(len(app_df), RANDOM_PROFILE_SAMPLE_SIZE),
        random_state=None,
    ).copy()

    columns = sample_df.columns.tolist()
    generated_rows: list[dict[str, Any]] = []

    for _ in range(limit):
        row: dict[str, Any] = {}

        for col in columns:
            if col == CLIENT_ID_COLUMN:
                row[col] = None
                continue

            row[col] = _generate_random_value_from_series(sample_df[col])

        generated_rows.append(row)

    logger.info(
        "Random artificial feature rows built from application profile",
        extra={
            "extra_data": {
                "event": "prediction_service_build_random_rows_success",
                "requested_limit": limit,
                "generated_rows": len(generated_rows),
                "sample_size": len(sample_df),
                "columns": len(columns),
            }
        },
    )

    return generated_rows


def _sanitize_feature_row(row: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    """
    Sépare l'identifiant client éventuel des features modèle.
    """
    row_copy = dict(row)
    client_id = row_copy.pop(CLIENT_ID_COLUMN, None)

    clean_features = _clean_feature_dict(row_copy)

    if client_id is not None:
        try:
            client_id = int(client_id)
        except Exception:
            client_id = None

    return client_id, clean_features


def _get_single_row_feature_dict(df: pd.DataFrame, *, context: str) -> dict[str, Any]:
    """
    Convertit un DataFrame à une seule ligne en dictionnaire de features propre.
    """
    df = _ensure_dataframe(df)

    if len(df) != 1:
        raise ValueError(
            f"{context} doit contenir exactement une ligne. "
            f"Lignes reçues : {len(df)}."
        )

    return _clean_feature_dict(df.iloc[0].to_dict())


def _get_features_for_client(client_id: int) -> dict[str, Any]:
    """
    Retourne les features prêtes pour le modèle pour un client donné
    à partir du cache applicatif.
    """
    client_df = get_features_for_client_from_cache(
        int(client_id),
        keep_id=False,
    )

    features = _get_single_row_feature_dict(
        client_df,
        context=f"Les features du client {client_id}",
    )

    logger.info(
        "Features retrieved for client from cache",
        extra={
            "extra_data": {
                "event": "prediction_service_get_client_features_success",
                "client_id": int(client_id),
                "feature_count": len(features),
            }
        },
    )

    return features


# =============================================================================
# Helpers logging
# =============================================================================

def _log_success_prediction(
    prediction_logger: PredictionLoggingService,
    *,
    request_id: str,
    client_id: int | None,
    features: dict[str, Any],
    prediction: int,
    score: float,
    threshold_used: float,
    latency_ms: float | None,
    inference_latency_ms: float | None,
    source_table: str,
) -> None:
    """
    Journalise une prédiction réussie.
    """
    features_df = pd.DataFrame([features])

    prediction_logger.log_full_prediction_event(
        request_id=request_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        features_df=features_df,
        raw_input_data=features,
        prediction=prediction,
        score=score,
        threshold_used=threshold_used,
        latency_ms=latency_ms,
        inference_latency_ms=inference_latency_ms,
        client_id=client_id,
        write_feature_store_monitoring=True,
        source_table=source_table,
        output_data={
            "request_id": request_id,
            "prediction": prediction,
            "score": score,
            "threshold_used": threshold_used,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
            "inference_latency_ms": inference_latency_ms,
        },
        status_code=200,
    )


def _log_error_prediction(
    prediction_logger: PredictionLoggingService,
    *,
    request_id: str,
    client_id: int | None,
    input_data: dict[str, Any],
    error_message: str,
    latency_ms: float | None,
    inference_latency_ms: float | None,
    status_code: int = 500,
) -> None:
    """
    Journalise une erreur de prédiction.
    """
    prediction_logger.log_prediction_error(
        request_id=request_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        input_data=input_data,
        error_message=error_message,
        client_id=client_id,
        status_code=status_code,
        latency_ms=latency_ms,
        inference_latency_ms=inference_latency_ms,
    )


def _build_success_item(
    *,
    request_id: str,
    client_id: int | None,
    prediction: int,
    score: float,
    threshold_used: float,
    latency_ms: float | None,
    inference_latency_ms: float | None,
) -> dict[str, Any]:
    """
    Construit un item de succès standardisé pour les réponses batch.
    """
    return {
        "request_id": request_id,
        "client_id": client_id,
        "prediction": prediction,
        "score": score,
        "threshold_used": threshold_used,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
        "inference_latency_ms": inference_latency_ms,
        "status": "success",
    }


def _build_error_item(
    *,
    request_id: str,
    client_id: int | None,
    error_message: str,
    latency_ms: float | None,
    inference_latency_ms: float | None,
) -> dict[str, Any]:
    """
    Construit un item d'erreur standardisé pour les réponses batch.
    """
    return {
        "request_id": request_id,
        "client_id": client_id,
        "prediction": None,
        "score": None,
        "threshold_used": None,
        "error_message": error_message,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
        "inference_latency_ms": inference_latency_ms,
        "status": "error",
    }


# =============================================================================
# API publique - prédiction unitaire
# =============================================================================

def make_prediction(
    features: dict[str, Any],
    *,
    client_id: int | None = None,
    db: Session | None = None,
    source_table: str = "api_request",
) -> dict[str, Any] | tuple[int, float, float, float]:
    """
    Réalise une prédiction unitaire.
    """
    clean_features = _clean_feature_dict(features)

    logger.info(
        "Single prediction requested in prediction service",
        extra={
            "extra_data": {
                "event": "prediction_service_single_start",
                "client_id": client_id,
                "source_table": source_table,
                "has_db_session": db is not None,
                "feature_count": len(clean_features),
            }
        },
    )

    if db is None:
        result = _predict_raw(clean_features)
        logger.info(
            "Single prediction completed without DB logging",
            extra={
                "extra_data": {
                    "event": "prediction_service_single_no_db_success",
                    "client_id": client_id,
                    "source_table": source_table,
                }
            },
        )
        return result

    prediction_logger = PredictionLoggingService(db=db)
    request_id = _safe_request_id()
    start_time = time.perf_counter()
    inference_latency_ms: float | None = None

    try:
        prediction, score, threshold_used, inference_latency_ms = _predict_raw(clean_features)
        latency_ms = (time.perf_counter() - start_time) * 1000

        _log_success_prediction(
            prediction_logger,
            request_id=request_id,
            client_id=client_id,
            features=clean_features,
            prediction=prediction,
            score=score,
            threshold_used=threshold_used,
            latency_ms=latency_ms,
            inference_latency_ms=inference_latency_ms,
            source_table=source_table,
        )

        payload = {
            "request_id": request_id,
            "client_id": client_id,
            "prediction": prediction,
            "score": score,
            "threshold_used": threshold_used,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
            "inference_latency_ms" : inference_latency_ms,
            "status": "success",
        }

        logger.info(
            "Single prediction completed with DB logging",
            extra={
                "extra_data": {
                    "event": "prediction_service_single_success",
                    "request_id": request_id,
                    "client_id": client_id,
                    "prediction": prediction,
                    "score": score,
                    "threshold_used": threshold_used,
                    "latency_ms": latency_ms,
                    "inference_latency_ms" : inference_latency_ms,
                    "source_table": source_table,
                }
            },
        )

        return payload

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "Unexpected error during single prediction",
            extra={
                "extra_data": {
                    "event": "prediction_service_single_exception",
                    "request_id": request_id,
                    "client_id": client_id,
                    "latency_ms": latency_ms,
                    "inference_latency_ms" : inference_latency_ms,
                    "source_table": source_table,
                    "error": str(exc),
                }
            },
        )

        try:
            _log_error_prediction(
                prediction_logger,
                request_id=request_id,
                client_id=client_id,
                input_data=clean_features,
                error_message=str(exc),
                latency_ms=latency_ms,
                inference_latency_ms=inference_latency_ms,
                status_code=500,
            )
        except Exception as log_exc:
            logger.exception(
                "Failed to log single prediction error",
                extra={
                    "extra_data": {
                        "event": "prediction_service_single_error_log_exception",
                        "request_id": request_id,
                        "client_id": client_id,
                        "error": str(log_exc),
                    }
                },
            )

        raise


def make_prediction_from_client_id(
    client_id: int,
    *,
    db: Session | None = None,
    source_table: str = "features_ready_cache",
) -> dict[str, Any] | tuple[int, float, float, float]:
    """
    Réalise une prédiction unitaire à partir d'un identifiant client.
    """
    logger.info(
        "Prediction from client_id requested in prediction service",
        extra={
            "extra_data": {
                "event": "prediction_service_client_id_start",
                "client_id": int(client_id),
                "source_table": source_table,
                "has_db_session": db is not None,
            }
        },
    )

    features = _get_features_for_client(int(client_id))

    return make_prediction(
        features=features,
        client_id=int(client_id),
        db=db,
        source_table=source_table,
    )


def make_prediction_from_dataframe(df: pd.DataFrame) -> tuple[int, float, float, float]:
    """
    Réalise une prédiction unitaire à partir d'un DataFrame contenant une seule ligne.
    """
    df = _ensure_dataframe(df)

    if len(df) != 1:
        raise ValueError(
            "make_prediction_from_dataframe attend un DataFrame avec exactement une ligne."
        )

    features = _clean_feature_dict(df.iloc[0].to_dict())

    logger.info(
        "Single prediction from dataframe requested",
        extra={
            "extra_data": {
                "event": "prediction_service_dataframe_start",
                "rows": len(df),
                "columns": len(df.columns),
            }
        },
    )

    result = _predict_raw(features)

    logger.info(
        "Single prediction from dataframe completed",
        extra={
            "extra_data": {
                "event": "prediction_service_dataframe_success",
                "rows": len(df),
                "columns": len(df.columns),
            }
        },
    )

    return result


def predict_one_row(df: pd.DataFrame) -> dict[str, Any]:
    """
    Retourne le résultat d'une prédiction unitaire sous forme de dictionnaire.
    """
    prediction, score, threshold, inference_latency_ms = make_prediction_from_dataframe(df)

    return {
        "prediction": prediction,
        "score": score,
        "threshold_used": threshold,
        "inference_latency_ms": inference_latency_ms,
    }


# =============================================================================
# API publique - batch payloads
# =============================================================================

def make_batch_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réalise des prédictions sur un lot de données sans logging.
    """
    df = _ensure_dataframe(df)
    _ensure_batch_size(len(df))

    logger.info(
        "Batch prediction without DB logging requested",
        extra={
            "extra_data": {
                "event": "prediction_service_batch_no_db_start",
                "batch_size": len(df),
                "columns": len(df.columns),
            }
        },
    )

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        feature_dict = _clean_feature_dict(row.to_dict())
        prediction, score, threshold, inference_latency_ms = _predict_raw(feature_dict)

        enriched_row = dict(feature_dict)
        enriched_row["score"] = score
        enriched_row["prediction"] = prediction
        enriched_row["threshold_used"] = threshold

        rows.append(enriched_row)

    result_df = pd.DataFrame(rows)

    logger.info(
        "Batch prediction without DB logging completed",
        extra={
            "extra_data": {
                "event": "prediction_service_batch_no_db_success",
                "batch_size": len(df),
                "result_rows": len(result_df),
            }
        },
    )

    return result_df


def run_batch_prediction(
    payloads: Sequence[dict[str, Any]],
    *,
    db: Session,
    source_table: str,
) -> dict[str, Any]:
    """
    Réalise un batch de prédictions à partir d'une liste de payloads normalisés.
    """
    _ensure_non_string_sequence(payloads, "payloads")
    _ensure_batch_size(len(payloads))

    logger.info(
        "Batch prediction with payloads requested",
        extra={
            "extra_data": {
                "event": "prediction_service_batch_payloads_start",
                "batch_size": len(payloads),
                "source_table": source_table,
            }
        },
    )

    prediction_logger = PredictionLoggingService(db=db)
    batch_start_time = time.perf_counter()

    items: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0
    batch_inference_latency_ms = 0.0

    for payload in payloads:
        request_id = _safe_request_id()
        start_time = time.perf_counter()
        inference_latency_ms: float | None = None

        client_id = payload.get("client_id")
        features = payload.get("features", {})

        try:
            if not isinstance(features, dict):
                raise TypeError("Chaque payload doit contenir un dictionnaire `features`.")

            clean_features = _clean_feature_dict(features)

            prediction, score, threshold_used, inference_latency_ms = _predict_raw(
                clean_features
            )
            batch_inference_latency_ms += inference_latency_ms or 0.0

            latency_ms = (time.perf_counter() - start_time) * 1000

            _log_success_prediction(
                prediction_logger,
                request_id=request_id,
                client_id=client_id,
                features=clean_features,
                prediction=prediction,
                score=score,
                threshold_used=threshold_used,
                latency_ms=latency_ms,
                inference_latency_ms=inference_latency_ms,
                source_table=source_table,
            )

            items.append(
                _build_success_item(
                    request_id=request_id,
                    client_id=client_id,
                    prediction=prediction,
                    score=score,
                    threshold_used=threshold_used,
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                )
            )
            success_count += 1

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "Unexpected error during batch payload prediction item",
                extra={
                    "extra_data": {
                        "event": "prediction_service_batch_payload_item_exception",
                        "request_id": request_id,
                        "client_id": client_id,
                        "source_table": source_table,
                        "latency_ms": latency_ms,
                        "inference_latency_ms": inference_latency_ms,
                        "error": str(exc),
                    }
                },
            )

            try:
                _log_error_prediction(
                    prediction_logger,
                    request_id=request_id,
                    client_id=client_id,
                    input_data=(
                        features if isinstance(features, dict) else {"payload": payload}
                    ),
                    error_message=str(exc),
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                    status_code=500,
                )
            except Exception as log_exc:
                logger.exception(
                    "Failed to log batch payload prediction error",
                    extra={
                        "extra_data": {
                            "event": "prediction_service_batch_payload_item_error_log_exception",
                            "request_id": request_id,
                            "client_id": client_id,
                            "error": str(log_exc),
                        }
                    },
                )

            items.append(
                _build_error_item(
                    request_id=request_id,
                    client_id=client_id,
                    error_message=str(exc),
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                )
            )
            error_count += 1

    batch_latency_ms = (time.perf_counter() - batch_start_time) * 1000

    payload = {
        "batch_size": len(payloads),
        "success_count": success_count,
        "error_count": error_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "batch_latency_ms": batch_latency_ms,
        "batch_inference_latency_ms": batch_inference_latency_ms,
        "items": items,
    }

    logger.info(
        "Batch prediction with payloads completed",
        extra={
            "extra_data": {
                "event": "prediction_service_batch_payloads_success",
                "batch_size": len(payloads),
                "success_count": success_count,
                "error_count": error_count,
                "batch_latency_ms": batch_latency_ms,
                "batch_inference_latency_ms": batch_inference_latency_ms,
                "source_table": source_table,
            }
        },
    )

    return payload

# =============================================================================
# API publique - batch clients
# =============================================================================

def predict_batch_from_client_ids(
    client_ids: Sequence[int],
    *,
    db: Session,
    source_table: str,
) -> dict[str, Any]:
    """
    Réalise un batch de prédictions à partir d'une liste de clés clients.
    """
    _ensure_non_string_sequence(client_ids, "client_ids")
    _ensure_batch_size(len(client_ids))

    logger.info(
        "Batch prediction from client IDs requested",
        extra={
            "extra_data": {
                "event": "prediction_service_batch_client_ids_start",
                "batch_size": len(client_ids),
                "source_table": source_table,
            }
        },
    )

    prediction_logger = PredictionLoggingService(db=db)
    batch_start_time = time.perf_counter()
    inference_latency_ms: float | None = None
    
    items: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0
    batch_inference_latency_ms = 0.0
    
    for client_id in client_ids:
        request_id = _safe_request_id()
        start_time = time.perf_counter()

        try:
            features = _get_features_for_client(int(client_id))

            if not isinstance(features, dict):
                raise TypeError(
                    "La récupération des features client doit retourner un dictionnaire."
                )

            clean_features = _clean_feature_dict(features)
            prediction, score, threshold_used, inference_latency_ms = _predict_raw(clean_features)
            latency_ms = (time.perf_counter() - start_time) * 1000

            _log_success_prediction(
                prediction_logger,
                request_id=request_id,
                client_id=int(client_id),
                features=clean_features,
                prediction=prediction,
                score=score,
                threshold_used=threshold_used,
                latency_ms=latency_ms,
                inference_latency_ms=inference_latency_ms,
                source_table=source_table,
            )

            items.append(
                _build_success_item(
                    request_id=request_id,
                    client_id=int(client_id),
                    prediction=prediction,
                    score=score,
                    threshold_used=threshold_used,
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                )
            )
            success_count += 1
            batch_inference_latency_ms += inference_latency_ms or 0.0

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            safe_client_id = int(client_id) if pd.notna(client_id) else None

            logger.exception(
                "Unexpected error during batch client prediction item",
                extra={
                    "extra_data": {
                        "event": "prediction_service_batch_client_item_exception",
                        "request_id": request_id,
                        "client_id": safe_client_id,
                        "source_table": source_table,
                        "latency_ms": latency_ms,
                        "inference_latency_ms": inference_latency_ms,
                        "error": str(exc),
                    }
                },
            )

            try:
                _log_error_prediction(
                    prediction_logger,
                    request_id=request_id,
                    client_id=safe_client_id,
                    input_data={"client_id": client_id},
                    error_message=str(exc),
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                    status_code=500,
                )
            except Exception as log_exc:
                logger.exception(
                    "Failed to log batch client prediction error",
                    extra={
                        "extra_data": {
                            "event": "prediction_service_batch_client_item_error_log_exception",
                            "request_id": request_id,
                            "client_id": safe_client_id,
                            "error": str(log_exc),
                        }
                    },
                )

            items.append(
                _build_error_item(
                    request_id=request_id,
                    client_id=safe_client_id,
                    error_message=str(exc),
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                )
            )
            error_count += 1

    batch_latency_ms = (time.perf_counter() - batch_start_time) * 1000

    payload = {
        "batch_size": len(client_ids),
        "success_count": success_count,
        "error_count": error_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "batch_latency_ms": batch_latency_ms,
        "batch_inference_latency_ms": batch_inference_latency_ms,
        "items": items,
    }

    logger.info(
        "Batch prediction from client IDs completed",
        extra={
            "extra_data": {
                "event": "prediction_service_batch_client_ids_success",
                "batch_size": len(client_ids),
                "success_count": success_count,
                "error_count": error_count,
                "batch_latency_ms": batch_latency_ms,
                "batch_inference_latency_ms": batch_inference_latency_ms,
                "source_table": source_table,
            }
        },
    )

    return payload


# =============================================================================
# API publique - simulations
# =============================================================================

def run_real_client_simulation(
    *,
    limit: int,
    random_seed: int | None,
    db: Session,
    source_table: str,
) -> dict[str, Any]:
    """
    Sélectionne des clients réels aléatoires, récupère leurs features,
    lance les prédictions puis journalise les résultats.
    """
    _ensure_batch_size(limit)

    logger.info(
        "Real client simulation requested",
        extra={
            "extra_data": {
                "event": "prediction_service_real_simulation_start",
                "limit": limit,
                "random_seed": random_seed,
                "source_table": source_table,
            }
        },
    )

    source_df = _load_client_source_dataframe()

    if source_df.empty:
        raise ValueError("Aucune source client disponible pour la simulation.")

    selected_client_ids = get_random_existing_client_ids(
        df=source_df,
        limit=limit,
        client_id_column=CLIENT_ID_COLUMN,
        random_seed=random_seed,
    )

    result = predict_batch_from_client_ids(
        client_ids=selected_client_ids,
        db=db,
        source_table=source_table,
    )
    result["selected_client_ids"] = selected_client_ids

    logger.info(
        "Real client simulation completed",
        extra={
            "extra_data": {
                "event": "prediction_service_real_simulation_success",
                "limit": limit,
                "selected_client_ids_count": len(selected_client_ids),
                "success_count": result.get("success_count"),
                "error_count": result.get("error_count"),
                "source_table": source_table,
            }
        },
    )

    return result


def run_random_feature_simulation(
    *,
    limit: int,
    db: Session,
    source_table: str,
) -> dict[str, Any]:
    """
    Génère des données artificielles aléatoires à partir du profil
    observé dans la source applicative CSV, puis exécute les prédictions.
    """
    _ensure_batch_size(limit)

    logger.info(
        "Random feature simulation requested",
        extra={
            "extra_data": {
                "event": "prediction_service_random_simulation_start",
                "limit": limit,
                "source_table": source_table,
            }
        },
    )

    raw_rows = _build_random_feature_rows_from_application(limit=limit)

    if not raw_rows:
        raise ValueError("Impossible de générer des données artificielles.")

    generated_app_df = pd.DataFrame(raw_rows)

    model_ready_df = build_features_from_loaded_data(
        application_df=generated_app_df,
        client_ids=None,
        debug=False,
        keep_id=True,
    )

    model_ready_df = _ensure_dataframe(model_ready_df)

    prediction_logger = PredictionLoggingService(db=db)
    batch_start_time = time.perf_counter()

    items: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0
    batch_inference_latency_ms = 0.0
    inference_latency_ms: float | None = None

    for _, row in model_ready_df.iterrows():
        request_id = _safe_request_id()
        start_time = time.perf_counter()

        client_id, features = _sanitize_feature_row(row.to_dict())

        try:
            prediction, score, threshold_used, inference_latency_ms = _predict_raw(features)
            latency_ms = (time.perf_counter() - start_time) * 1000

            _log_success_prediction(
                prediction_logger,
                request_id=request_id,
                client_id=client_id,
                features=features,
                prediction=prediction,
                score=score,
                threshold_used=threshold_used,
                latency_ms=latency_ms,
                inference_latency_ms=inference_latency_ms,
                source_table=source_table,
            )

            items.append(
                _build_success_item(
                    request_id=request_id,
                    client_id=client_id,
                    prediction=prediction,
                    score=score,
                    threshold_used=threshold_used,
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                )
            )
            success_count += 1
            batch_inference_latency_ms += inference_latency_ms or 0.0

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "Unexpected error during random feature simulation item",
                extra={
                    "extra_data": {
                        "event": "prediction_service_random_simulation_item_exception",
                        "request_id": request_id,
                        "client_id": client_id,
                        "source_table": source_table,
                        "latency_ms": latency_ms,
                        "inference_latency_ms": inference_latency_ms,
                        "error": str(exc),
                    }
                },
            )

            try:
                _log_error_prediction(
                    prediction_logger,
                    request_id=request_id,
                    client_id=client_id,
                    input_data=features,
                    error_message=str(exc),
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                    status_code=500,
                )
            except Exception as log_exc:
                logger.exception(
                    "Failed to log random feature simulation error",
                    extra={
                        "extra_data": {
                            "event": "prediction_service_random_simulation_item_error_log_exception",
                            "request_id": request_id,
                            "client_id": client_id,
                            "error": str(log_exc),
                        }
                    },
                )

            items.append(
                _build_error_item(
                    request_id=request_id,
                    client_id=client_id,
                    error_message=str(exc),
                    latency_ms=latency_ms,
                    inference_latency_ms=inference_latency_ms,
                )
            )
            error_count += 1

    batch_latency_ms = (time.perf_counter() - batch_start_time) * 1000

    payload = {
        "batch_size": len(model_ready_df),
        "success_count": success_count,
        "error_count": error_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "batch_latency_ms": batch_latency_ms,
        "batch_inference_latency_ms": batch_inference_latency_ms,
        "items": items,
    }

    logger.info(
        "Random feature simulation completed",
        extra={
            "extra_data": {
                "event": "prediction_service_random_simulation_success",
                "batch_size": len(model_ready_df),
                "success_count": success_count,
                "error_count": error_count,
                "batch_latency_ms": batch_latency_ms,
                "batch_inference_latency_ms": batch_inference_latency_ms,
                "source_table": source_table,
            }
        },
    )

    return payload


# =============================================================================
# API publique - ground truth
# =============================================================================

def create_ground_truth_label(
    *,
    db: Session,
    request_id: str | None,
    client_id: int | None,
    true_label: int,
    label_source: str | None,
    observed_at: datetime | None,
    notes: str | None,
) -> dict[str, Any]:
    """
    Enregistre une vérité terrain dans `ground_truth_labels`.
    """
    logger.info(
        "Ground truth creation requested in prediction service",
        extra={
            "extra_data": {
                "event": "prediction_service_ground_truth_create_start",
                "request_id": request_id,
                "client_id": client_id,
                "true_label": true_label,
                "label_source": label_source,
            }
        },
    )

    _validate_ground_truth_inputs(
        request_id=request_id,
        client_id=client_id,
        true_label=true_label,
        observed_at=observed_at,
    )

    observed_at = observed_at or datetime.now(timezone.utc)

    entity = prediction_crud.create_ground_truth_label(
        db,
        request_id=request_id,
        client_id=client_id,
        true_label=int(true_label),
        label_source=label_source,
        observed_at=observed_at,
        notes=notes,
    )

    payload = {
        "id": entity.id,
        "request_id": entity.request_id,
        "client_id": entity.client_id,
        "true_label": entity.true_label,
        "label_source": entity.label_source,
        "observed_at": entity.observed_at,
        "notes": entity.notes,
        "message": "Ground truth enregistré avec succès.",
    }

    logger.info(
        "Ground truth created successfully in prediction service",
        extra={
            "extra_data": {
                "event": "prediction_service_ground_truth_create_success",
                "id": entity.id,
                "request_id": entity.request_id,
                "client_id": entity.client_id,
                "true_label": entity.true_label,
            }
        },
    )

    return payload


# =============================================================================
# Helpers de résumé
# =============================================================================

def summarize_batch_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Résume les résultats d'un batch de prédictions.
    """
    total = len(results)
    success_count = sum(1 for item in results if item.get("status") == "success")
    error_count = total - success_count

    payload = {
        "batch_size": total,
        "success_count": success_count,
        "error_count": error_count,
        "items": list(results),
    }

    logger.info(
        "Batch results summarized",
        extra={
            "extra_data": {
                "event": "prediction_service_summarize_batch_success",
                "batch_size": total,
                "success_count": success_count,
                "error_count": error_count,
            }
        },
    )

    return payload