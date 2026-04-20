"""
Service de prédiction du modèle de scoring crédit.

Ce module encapsule la logique métier liée à l'inférence du modèle.
Il ne recharge pas le modèle à chaque appel : le modèle et le seuil
de décision sont récupérés depuis `app.services.model_loader_service`,
qui les charge une seule fois au démarrage de l'application.

Fonctionnalités
---------------
- prédiction unitaire à partir d'un dictionnaire de features
- prédiction batch à partir d'une liste de payloads
- simulation batch à partir de clients réels choisis aléatoirement
- simulation batch à partir de données artificielles construites
  à partir du profil observé dans `application_test.csv`
- journalisation des résultats en base PostgreSQL
- validation légère des entrées

Architecture actuelle
---------------------
- les données source proviennent exclusivement de `application_test.csv`
- les features sont construites en mémoire via `feature_builder_service`
- PostgreSQL sert uniquement au logging et au monitoring

Notes
-----
- Le score retourné correspond à la probabilité de la classe positive.
- La décision finale est calculée avec le seuil chargé depuis
  `app.services.model_loader_service`.
- Les simulations "réelles" utilisent les clients présents dans
  `application_test.csv`.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import MODEL_NAME, MODEL_VERSION
from app.services.data_loader_service import get_data_cache
from app.services.feature_builder_service import (
    build_features_for_client,
    build_model_ready_features,
)
from app.services.model_loader_service import get_model, get_threshold
from app.services.prediction_logging_service import PredictionLoggingService


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

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire de features.

    Returns
    -------
    pd.DataFrame
        DataFrame pandas contenant une seule ligne.

    Raises
    ------
    TypeError
        Si `features` n'est pas un dictionnaire.
    ValueError
        Si `features` est vide.
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

    Parameters
    ----------
    df : pd.DataFrame
        Objet à contrôler.

    Returns
    -------
    pd.DataFrame
        DataFrame validé.

    Raises
    ------
    TypeError
        Si l'objet n'est pas un DataFrame.
    ValueError
        Si le DataFrame est vide.
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

    Parameters
    ----------
    size : int
        Taille du batch.

    Raises
    ------
    ValueError
        Si le batch est vide ou trop grand.
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

    Parameters
    ----------
    values : Sequence[Any]
        Valeur à contrôler.
    name : str
        Nom logique du paramètre.

    Raises
    ------
    TypeError
        Si l'entrée n'est pas une séquence valide.
    ValueError
        Si la séquence est vide.
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

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire source.

    Returns
    -------
    dict[str, Any]
        Dictionnaire nettoyé.
    """
    clean_features: dict[str, Any] = {}

    for key, value in features.items():
        if pd.isna(value):
            clean_features[key] = None
        else:
            clean_features[key] = value

    return clean_features


# =============================================================================
# Helpers modèle
# =============================================================================

def _predict_scores(df: pd.DataFrame) -> pd.Series:
    """
    Calcule les scores de probabilité de la classe positive.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de features déjà aligné sur le modèle.

    Returns
    -------
    pd.Series
        Série des probabilités de la classe positive.

    Raises
    ------
    AttributeError
        Si le modèle ne possède pas `predict_proba`.
    ValueError
        Si la sortie de `predict_proba` est invalide.
    """
    model = get_model()

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Le modèle chargé ne possède pas de méthode `predict_proba`."
        )

    proba = model.predict_proba(df)

    if len(proba.shape) != 2 or proba.shape[1] < 2:
        raise ValueError(
            "La sortie de `predict_proba` ne contient pas deux colonnes de probabilités."
        )

    return pd.Series(proba[:, 1], index=df.index, dtype="float64")


def _predict_raw(features: dict[str, Any]) -> tuple[int, float, float]:
    """
    Réalise la prédiction brute sans journalisation.

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire de features prêt pour le modèle.

    Returns
    -------
    tuple[int, float, float]
        (prediction, score, threshold_used)
    """
    threshold = get_threshold()
    df = _ensure_dataframe_from_dict(features)

    score = float(_predict_scores(df).iloc[0])
    prediction = int(score >= threshold)

    return prediction, score, threshold


# =============================================================================
# Helpers clients / simulation
# =============================================================================

def _extract_application_test_df() -> pd.DataFrame:
    """
    Extrait le DataFrame `application_test` depuis le cache applicatif.

    Returns
    -------
    pd.DataFrame
        DataFrame source application_test.

    Raises
    ------
    ValueError
        Si aucune source exploitable n'est trouvée.
    TypeError
        Si l'objet renvoyé n'est pas un DataFrame.
    """
    cache = get_data_cache()

    if isinstance(cache, pd.DataFrame):
        return cache.copy()

    if isinstance(cache, dict):
        if "application_test" in cache:
            df = cache["application_test"]
        elif "app" in cache:
            df = cache["app"]
        else:
            raise ValueError(
                "Le cache de données ne contient ni 'application_test' ni 'app'."
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

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    client_id_column : str, default=CLIENT_ID_COLUMN
        Nom de la colonne identifiant client.

    Returns
    -------
    list[int]
        Liste des identifiants clients distincts.

    Raises
    ------
    ValueError
        Si la colonne est absente ou si aucun identifiant n'est exploitable.
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

    return client_ids


def get_random_existing_client_ids(
    df: pd.DataFrame,
    limit: int = MAX_BATCH_SIZE,
    client_id_column: str = CLIENT_ID_COLUMN,
    random_seed: int | None = None,
) -> list[int]:
    """
    Sélectionne aléatoirement des identifiants clients existants
    à partir d'un DataFrame source.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    limit : int, default=MAX_BATCH_SIZE
        Nombre de clients à tirer.
    client_id_column : str, default=CLIENT_ID_COLUMN
        Nom de la colonne identifiant client.
    random_seed : int | None, default=None
        Graine optionnelle.

    Returns
    -------
    list[int]
        Liste des identifiants sélectionnés.
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
    return rng.sample(client_ids, limit)


def _load_client_source_dataframe() -> pd.DataFrame:
    """
    Charge la source client utilisée pour sélectionner des SK_ID_CURR réels.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant au minimum `SK_ID_CURR`.
    """
    app_df = _extract_application_test_df()

    if CLIENT_ID_COLUMN not in app_df.columns:
        raise ValueError(
            f"La colonne `{CLIENT_ID_COLUMN}` est absente de application_test.csv."
        )

    return app_df[[CLIENT_ID_COLUMN]].copy()


def _generate_random_value_from_series(series: pd.Series) -> Any:
    """
    Génère une valeur aléatoire à partir du profil observé d'une colonne.

    Parameters
    ----------
    series : pd.Series
        Série source.

    Returns
    -------
    Any
        Valeur artificielle générée.
    """
    clean = series.dropna()

    if clean.empty:
        return None

    numeric = pd.to_numeric(clean, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio >= 0.90:
        numeric_clean = numeric.dropna()

        if numeric_clean.empty:
            return None

        min_val = numeric_clean.min()
        max_val = numeric_clean.max()

        if min_val == max_val:
            return float(min_val) if not float(min_val).is_integer() else int(min_val)

        is_integer_like = (
            pd.api.types.is_integer_dtype(numeric_clean)
            or all(
                float(x).is_integer()
                for x in numeric_clean.head(min(len(numeric_clean), 20))
            )
        )

        if is_integer_like:
            return int(random.randint(int(min_val), int(max_val)))

        return float(random.uniform(float(min_val), float(max_val)))

    values = clean.astype(str).unique().tolist()
    if not values:
        return None

    return random.choice(values)


def _build_random_feature_rows_from_application_test(
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Construit des lignes artificielles aléatoires à partir du profil
    observé dans `application_test.csv`.

    Parameters
    ----------
    limit : int
        Nombre de lignes artificielles à générer.

    Returns
    -------
    list[dict[str, Any]]
        Lignes artificielles brutes inspirées du profil observé.
    """
    app_df = _extract_application_test_df()
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

    return generated_rows


def _sanitize_feature_row(row: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    """
    Sépare l'identifiant client éventuel des features modèle.

    Parameters
    ----------
    row : dict[str, Any]
        Ligne source.

    Returns
    -------
    tuple[int | None, dict[str, Any]]
        (client_id éventuel, dictionnaire de features)
    """
    row_copy = dict(row)
    client_id = row_copy.pop(CLIENT_ID_COLUMN, None)

    clean_features: dict[str, Any] = {}
    for key, value in row_copy.items():
        if pd.isna(value):
            clean_features[key] = None
        else:
            clean_features[key] = value

    return client_id, clean_features


def _get_features_for_client(client_id: int) -> dict[str, Any]:
    """
    Retourne les features prêtes pour le modèle pour un client donné.

    Parameters
    ----------
    client_id : int
        Identifiant client.

    Returns
    -------
    dict[str, Any]
        Features prêtes pour le modèle.
    """
    return _clean_feature_dict(build_features_for_client(int(client_id), keep_id=False))


# =============================================================================
# Helpers logging
# =============================================================================

def _log_success_prediction(
    logger: PredictionLoggingService,
    *,
    request_id: str,
    client_id: int | None,
    features: dict[str, Any],
    prediction: int,
    score: float,
    threshold_used: float,
    latency_ms: float | None,
    source_table: str,
) -> None:
    """
    Journalise une prédiction réussie.

    Parameters
    ----------
    logger : PredictionLoggingService
        Service de journalisation.
    request_id : str
        Identifiant de requête.
    client_id : int | None
        Identifiant client éventuel.
    features : dict[str, Any]
        Features utilisées.
    prediction : int
        Classe prédite.
    score : float
        Score prédit.
    threshold_used : float
        Seuil utilisé.
    latency_ms : float | None
        Temps d'inférence.
    source_table : str
        Source logique des données.
    """
    features_df = pd.DataFrame([features])

    logger.log_full_prediction_event(
        request_id=request_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        features_df=features_df,
        raw_input_data=features,
        prediction=prediction,
        score=score,
        threshold_used=threshold_used,
        latency_ms=latency_ms,
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
        },
        status_code=200,
    )


def _log_error_prediction(
    logger: PredictionLoggingService,
    *,
    request_id: str,
    client_id: int | None,
    input_data: dict[str, Any],
    error_message: str,
    latency_ms: float | None,
) -> None:
    """
    Journalise une erreur de prédiction.

    Parameters
    ----------
    logger : PredictionLoggingService
        Service de journalisation.
    request_id : str
        Identifiant de requête.
    client_id : int | None
        Identifiant client éventuel.
    input_data : dict[str, Any]
        Données d'entrée utiles au debug.
    error_message : str
        Message d'erreur.
    latency_ms : float | None
        Temps écoulé avant l'erreur.
    """
    logger.log_prediction_error(
        request_id=request_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        input_data=input_data,
        error_message=error_message,
        client_id=client_id,
        status_code=500,
        latency_ms=latency_ms,
    )


# =============================================================================
# API publique - prédiction unitaire
# =============================================================================

def make_prediction(
    features: dict[str, Any],
    *,
    client_id: int | None = None,
    db: Session | None = None,
    source_table: str = "api_request",
) -> dict[str, Any] | tuple[int, float, float]:
    """
    Réalise une prédiction unitaire.

    Deux modes :
    - sans `db` : retourne (prediction, score, threshold_used)
    - avec `db` : journalise et retourne un dictionnaire complet

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire de features prêtes pour le modèle.
    client_id : int | None, default=None
        Identifiant client éventuel.
    db : Session | None, default=None
        Session SQLAlchemy pour la journalisation.
    source_table : str, default="api_request"
        Source logique des données.

    Returns
    -------
    dict[str, Any] | tuple[int, float, float]
        Résultat complet si `db` est fourni, sinon tuple brut.
    """
    clean_features = _clean_feature_dict(features)

    if db is None:
        return _predict_raw(clean_features)

    logger = PredictionLoggingService(db=db)
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        prediction, score, threshold_used = _predict_raw(clean_features)
        latency_ms = (time.perf_counter() - start_time) * 1000

        _log_success_prediction(
            logger,
            request_id=request_id,
            client_id=client_id,
            features=clean_features,
            prediction=prediction,
            score=score,
            threshold_used=threshold_used,
            latency_ms=latency_ms,
            source_table=source_table,
        )

        logger.commit()

        return {
            "request_id": request_id,
            "client_id": client_id,
            "prediction": prediction,
            "score": score,
            "threshold_used": threshold_used,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
            "status": "success",
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.rollback()

        try:
            _log_error_prediction(
                logger,
                request_id=request_id,
                client_id=client_id,
                input_data=clean_features,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            logger.commit()
        except Exception:
            logger.rollback()

        raise


def make_prediction_from_dataframe(df: pd.DataFrame) -> tuple[int, float, float]:
    """
    Réalise une prédiction unitaire à partir d'un DataFrame contenant une seule ligne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à une ligne, déjà aligné sur le modèle.

    Returns
    -------
    tuple[int, float, float]
        (prediction, score, threshold_used)
    """
    df = _ensure_dataframe(df)

    if len(df) != 1:
        raise ValueError(
            "make_prediction_from_dataframe attend un DataFrame avec exactement une ligne."
        )

    features = _clean_feature_dict(df.iloc[0].to_dict())
    return _predict_raw(features)


def predict_one_row(df: pd.DataFrame) -> dict[str, Any]:
    """
    Retourne le résultat d'une prédiction unitaire sous forme de dictionnaire.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à une ligne.

    Returns
    -------
    dict[str, Any]
        Résultat de prédiction.
    """
    prediction, score, threshold = make_prediction_from_dataframe(df)

    return {
        "prediction": prediction,
        "score": score,
        "threshold_used": threshold,
    }


# =============================================================================
# API publique - batch payloads
# =============================================================================

def make_batch_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réalise des prédictions sur un lot de données sans logging.

    Parameters
    ----------
    df : pd.DataFrame
        Batch de données prêt pour le modèle.

    Returns
    -------
    pd.DataFrame
        Batch enrichi avec score, prediction, threshold_used.
    """
    df = _ensure_dataframe(df)
    _ensure_batch_size(len(df))

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        feature_dict = _clean_feature_dict(row.to_dict())
        prediction, score, threshold = _predict_raw(feature_dict)

        enriched_row = dict(feature_dict)
        enriched_row["score"] = score
        enriched_row["prediction"] = prediction
        enriched_row["threshold_used"] = threshold

        rows.append(enriched_row)

    return pd.DataFrame(rows)


def run_batch_prediction(
    payloads: Sequence[dict[str, Any]],
    *,
    db: Session,
    source_table: str,
) -> dict[str, Any]:
    """
    Réalise un batch de prédictions à partir d'une liste de payloads normalisés.

    Chaque payload doit contenir :
    - client_id
    - features

    Parameters
    ----------
    payloads : Sequence[dict[str, Any]]
        Liste des payloads normalisés.
    db : Session
        Session SQLAlchemy.
    source_table : str
        Source logique des données.

    Returns
    -------
    dict[str, Any]
        Résumé global du batch et résultats détaillés.
    """
    _ensure_non_string_sequence(payloads, "payloads")
    _ensure_batch_size(len(payloads))

    logger = PredictionLoggingService(db=db)
    batch_start_time = time.perf_counter()

    items: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0

    for payload in payloads:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        client_id = payload.get("client_id")
        features = payload.get("features", {})

        try:
            if not isinstance(features, dict):
                raise TypeError("Chaque payload doit contenir un dictionnaire `features`.")

            clean_features = _clean_feature_dict(features)
            prediction, score, threshold_used = _predict_raw(clean_features)
            latency_ms = (time.perf_counter() - start_time) * 1000

            _log_success_prediction(
                logger,
                request_id=request_id,
                client_id=client_id,
                features=clean_features,
                prediction=prediction,
                score=score,
                threshold_used=threshold_used,
                latency_ms=latency_ms,
                source_table=source_table,
            )

            items.append(
                {
                    "request_id": request_id,
                    "client_id": client_id,
                    "prediction": prediction,
                    "score": score,
                    "threshold_used": threshold_used,
                    "model_version": MODEL_VERSION,
                    "latency_ms": latency_ms,
                    "status": "success",
                }
            )
            success_count += 1

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            try:
                _log_error_prediction(
                    logger,
                    request_id=request_id,
                    client_id=client_id,
                    input_data=features if isinstance(features, dict) else {"payload": payload},
                    error_message=str(exc),
                    latency_ms=latency_ms,
                )
            except Exception:
                pass

            items.append(
                {
                    "request_id": request_id,
                    "client_id": client_id,
                    "prediction": None,
                    "score": None,
                    "threshold_used": None,
                    "error_message": str(exc),
                    "model_version": MODEL_VERSION,
                    "latency_ms": latency_ms,
                    "status": "error",
                }
            )
            error_count += 1

    try:
        logger.commit()
    except Exception as exc:
        logger.rollback()
        raise RuntimeError(f"Batch logging failed: {exc}") from exc

    batch_latency_ms = (time.perf_counter() - batch_start_time) * 1000

    return {
        "batch_size": len(payloads),
        "success_count": success_count,
        "error_count": error_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "batch_latency_ms": batch_latency_ms,
        "items": items,
    }


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
    Réalise un batch de prédictions à partir d'une liste de clés clients,
    avec récupération des features, prédiction et logging.

    Parameters
    ----------
    client_ids : Sequence[int]
        Liste des identifiants clients.
    db : Session
        Session SQLAlchemy.
    source_table : str
        Source logique des données.

    Returns
    -------
    dict[str, Any]
        Résumé global du batch et résultats détaillés.
    """
    _ensure_non_string_sequence(client_ids, "client_ids")
    _ensure_batch_size(len(client_ids))

    logger = PredictionLoggingService(db=db)
    batch_start_time = time.perf_counter()

    items: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0

    for client_id in client_ids:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            features = _get_features_for_client(int(client_id))

            if not isinstance(features, dict):
                raise TypeError(
                    "La récupération des features client doit retourner un dictionnaire."
                )

            clean_features = _clean_feature_dict(features)
            prediction, score, threshold_used = _predict_raw(clean_features)
            latency_ms = (time.perf_counter() - start_time) * 1000

            _log_success_prediction(
                logger,
                request_id=request_id,
                client_id=int(client_id),
                features=clean_features,
                prediction=prediction,
                score=score,
                threshold_used=threshold_used,
                latency_ms=latency_ms,
                source_table=source_table,
            )

            items.append(
                {
                    "request_id": request_id,
                    "client_id": int(client_id),
                    "prediction": prediction,
                    "score": score,
                    "threshold_used": threshold_used,
                    "model_version": MODEL_VERSION,
                    "latency_ms": latency_ms,
                    "status": "success",
                }
            )
            success_count += 1

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            try:
                _log_error_prediction(
                    logger,
                    request_id=request_id,
                    client_id=int(client_id) if pd.notna(client_id) else None,
                    input_data={"client_id": client_id},
                    error_message=str(exc),
                    latency_ms=latency_ms,
                )
            except Exception:
                pass

            items.append(
                {
                    "request_id": request_id,
                    "client_id": int(client_id) if pd.notna(client_id) else None,
                    "prediction": None,
                    "score": None,
                    "threshold_used": None,
                    "error_message": str(exc),
                    "model_version": MODEL_VERSION,
                    "latency_ms": latency_ms,
                    "status": "error",
                }
            )
            error_count += 1

    try:
        logger.commit()
    except Exception as exc:
        logger.rollback()
        raise RuntimeError(f"Batch logging failed: {exc}") from exc

    batch_latency_ms = (time.perf_counter() - batch_start_time) * 1000

    return {
        "batch_size": len(client_ids),
        "success_count": success_count,
        "error_count": error_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "batch_latency_ms": batch_latency_ms,
        "items": items,
    }


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

    Parameters
    ----------
    limit : int
        Nombre de clients à tirer.
    random_seed : int | None
        Graine optionnelle.
    db : Session
        Session SQLAlchemy.
    source_table : str
        Source logique des données.

    Returns
    -------
    dict[str, Any]
        Résumé global de la simulation.
    """
    _ensure_batch_size(limit)

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

    return result


def run_random_feature_simulation(
    *,
    limit: int,
    db: Session,
    source_table: str,
) -> dict[str, Any]:
    """
    Génère des données artificielles aléatoires à partir du profil
    observé dans `application_test.csv`, puis exécute les prédictions.

    Parameters
    ----------
    limit : int
        Nombre de lignes artificielles à générer.
    db : Session
        Session SQLAlchemy.
    source_table : str
        Source logique des données.

    Returns
    -------
    dict[str, Any]
        Résumé global de la simulation.
    """
    _ensure_batch_size(limit)

    raw_rows = _build_random_feature_rows_from_application_test(limit=limit)

    if not raw_rows:
        raise ValueError(
            "Impossible de générer des données artificielles depuis application_test.csv."
        )

    # On reconstruit les features modèle finales à partir des lignes brutes.
    generated_app_df = pd.DataFrame(raw_rows)
    model_ready_df = build_model_ready_features(
        client_ids=None,
        debug=False,
        keep_id=True,
    )

    # Le builder métier principal travaille depuis le cache applicatif.
    # Ici, pour une simulation artificielle, on reproduit localement la même
    # logique en s'appuyant sur la version "loaded_data" du feature builder.
    from app.services.feature_builder_service import build_features_from_loaded_data

    model_ready_df = build_features_from_loaded_data(
        application_test_df=generated_app_df,
        client_ids=None,
        debug=False,
        keep_id=True,
    )

    logger = PredictionLoggingService(db=db)
    batch_start_time = time.perf_counter()

    items: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0

    for _, row in model_ready_df.iterrows():
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        client_id, features = _sanitize_feature_row(row.to_dict())

        try:
            prediction, score, threshold_used = _predict_raw(features)
            latency_ms = (time.perf_counter() - start_time) * 1000

            _log_success_prediction(
                logger,
                request_id=request_id,
                client_id=client_id,
                features=features,
                prediction=prediction,
                score=score,
                threshold_used=threshold_used,
                latency_ms=latency_ms,
                source_table=source_table,
            )

            items.append(
                {
                    "request_id": request_id,
                    "client_id": client_id,
                    "prediction": prediction,
                    "score": score,
                    "threshold_used": threshold_used,
                    "model_version": MODEL_VERSION,
                    "latency_ms": latency_ms,
                    "status": "success",
                }
            )
            success_count += 1

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            try:
                _log_error_prediction(
                    logger,
                    request_id=request_id,
                    client_id=client_id,
                    input_data=features,
                    error_message=str(exc),
                    latency_ms=latency_ms,
                )
            except Exception:
                pass

            items.append(
                {
                    "request_id": request_id,
                    "client_id": client_id,
                    "prediction": None,
                    "score": None,
                    "threshold_used": None,
                    "error_message": str(exc),
                    "model_version": MODEL_VERSION,
                    "latency_ms": latency_ms,
                    "status": "error",
                }
            )
            error_count += 1

    try:
        logger.commit()
    except Exception as exc:
        logger.rollback()
        raise RuntimeError(f"Simulation logging failed: {exc}") from exc

    batch_latency_ms = (time.perf_counter() - batch_start_time) * 1000

    return {
        "batch_size": len(model_ready_df),
        "success_count": success_count,
        "error_count": error_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "batch_latency_ms": batch_latency_ms,
        "items": items,
    }


# =============================================================================
# Helpers de résumé
# =============================================================================

def summarize_batch_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Résume les résultats d'un batch de prédictions.

    Parameters
    ----------
    results : Sequence[dict[str, Any]]
        Résultats détaillés.

    Returns
    -------
    dict[str, Any]
        Résumé du batch.
    """
    total = len(results)
    success_count = sum(1 for item in results if item.get("status") == "success")
    error_count = total - success_count

    return {
        "batch_size": total,
        "success_count": success_count,
        "error_count": error_count,
        "items": list(results),
    }