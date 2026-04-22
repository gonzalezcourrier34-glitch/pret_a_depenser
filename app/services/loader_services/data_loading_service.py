"""
Service de chargement et de mise en cache des données source et des features prêtes.

Objectif
--------
- charger une seule fois le CSV brut source configuré
- construire une seule fois les features enrichies
- servir les données en mémoire pour les prédictions unitaires ou batch
- charger les références de monitoring depuis le dossier configuré

Architecture
------------
CSV source unique -> RAW_DATA_CACHE -> FEATURES_READY_CACHE -> API
Références monitoring -> MONITORING_REFERENCE_CACHE -> analyse / Evidently

Notes
-----
- ce service est utile quand l'API fonctionne en mode CSV / mémoire
- il évite de relire les fichiers et de recalculer les features à chaque requête
- il repose sur des caches globaux simples, suffisants pour ce projet
- le fichier chargé est piloté par `APPLICATION_CSV`
- le dossier de monitoring est piloté par `MONITORING_DIR`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import APPLICATION_CSV, MONITORING_DIR
from app.services.features_builder_service import build_features_from_loaded_data


logger = logging.getLogger(__name__)


# =============================================================================
# Caches globaux
# =============================================================================

RAW_DATA_CACHE: dict[str, pd.DataFrame] = {}
FEATURES_READY_CACHE: pd.DataFrame | None = None
MONITORING_REFERENCE_CACHE: dict[str, Any] = {}


# =============================================================================
# Helpers internes
# =============================================================================

def _validate_dataframe(df: pd.DataFrame | None, name: str) -> None:
    """
    Vérifie qu'un DataFrame est bien exploitable.

    Parameters
    ----------
    df : pd.DataFrame | None
        Objet à valider.
    name : str
        Nom logique du DataFrame pour les messages d'erreur.

    Raises
    ------
    RuntimeError
        Si l'objet est None, n'est pas un DataFrame ou est vide.
    """
    if df is None:
        raise RuntimeError(f"Le DataFrame `{name}` est None.")

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"`{name}` n'est pas un DataFrame pandas.")

    if df.empty:
        raise RuntimeError(f"Le DataFrame `{name}` est vide.")


def _ensure_sk_id_curr(df: pd.DataFrame) -> None:
    """
    Vérifie la présence de la colonne SK_ID_CURR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à vérifier.

    Raises
    ------
    ValueError
        Si la colonne `SK_ID_CURR` est absente.
    """
    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Colonne `SK_ID_CURR` absente du DataFrame.")


def _resolve_application_csv(csv_path: Path | str | None = None) -> Path:
    """
    Résout le chemin du CSV source principal.

    Parameters
    ----------
    csv_path : Path | str | None, optional
        Chemin explicite éventuel.
        Si None, utilise `APPLICATION_CSV`.

    Returns
    -------
    Path
        Chemin résolu du CSV applicatif.
    """
    return Path(csv_path) if csv_path is not None else Path(APPLICATION_CSV)


def _resolve_monitoring_dir(monitoring_dir: Path | str | None = None) -> Path:
    """
    Résout le répertoire de monitoring.

    Parameters
    ----------
    monitoring_dir : Path | str | None, optional
        Dossier explicite éventuel.
        Si None, utilise la configuration `MONITORING_DIR`.

    Returns
    -------
    Path
        Chemin résolu du dossier de monitoring.
    """
    return Path(monitoring_dir) if monitoring_dir is not None else Path(MONITORING_DIR)


def _load_parquet_file(file_path: Path, logical_name: str) -> pd.DataFrame:
    """
    Charge un fichier parquet et vérifie qu'il est exploitable.

    Parameters
    ----------
    file_path : Path
        Chemin du fichier parquet.
    logical_name : str
        Nom logique utilisé dans les logs et erreurs.

    Returns
    -------
    pd.DataFrame
        DataFrame chargé depuis le fichier parquet.

    Raises
    ------
    FileNotFoundError
        Si le fichier est introuvable.
    RuntimeError
        Si le fichier chargé ne produit pas un DataFrame exploitable.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable pour `{logical_name}` : {file_path}"
        )

    df = pd.read_parquet(file_path)
    _validate_dataframe(df, logical_name)
    return df


def _load_json_file(file_path: Path, logical_name: str) -> Any:
    """
    Charge un fichier JSON.

    Parameters
    ----------
    file_path : Path
        Chemin du fichier JSON.
    logical_name : str
        Nom logique utilisé dans les logs et erreurs.

    Returns
    -------
    Any
        Contenu JSON désérialisé.

    Raises
    ------
    FileNotFoundError
        Si le fichier est introuvable.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable pour `{logical_name}` : {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Chargement des données brutes
# =============================================================================

def load_all_csv(csv_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Charge uniquement le CSV source principal en mémoire.

    Parameters
    ----------
    csv_path : Path | str | None, optional
        Chemin du fichier CSV source à charger.
        Si None, utilise la configuration `APPLICATION_CSV`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping nom logique -> DataFrame brut.

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV est introuvable.
    RuntimeError
        Si le CSV chargé est vide ou invalide.
    """
    resolved_csv_path = _resolve_application_csv(csv_path)

    if not resolved_csv_path.exists():
        raise FileNotFoundError(
            f"Le fichier CSV source est introuvable : {resolved_csv_path}"
        )

    if not resolved_csv_path.is_file():
        raise FileNotFoundError(
            f"Le chemin fourni n'est pas un fichier CSV : {resolved_csv_path}"
        )

    logger.info(
        "Loading source CSV into memory",
        extra={
            "extra_data": {
                "event": "data_load_csv_start",
                "csv_path": str(resolved_csv_path),
            }
        },
    )

    df = pd.read_csv(resolved_csv_path)
    _validate_dataframe(df, "application_source")

    logger.info(
        "Source CSV loaded successfully",
        extra={
            "extra_data": {
                "event": "data_load_csv_success",
                "csv_path": str(resolved_csv_path),
                "rows": len(df),
                "columns": len(df.columns),
            }
        },
    )

    return {"application": df}


def init_raw_data_cache(csv_path: Path | str | None = None) -> None:
    """
    Initialise le cache du CSV brut source.

    Parameters
    ----------
    csv_path : Path | str | None, optional
        Chemin du CSV à charger.
        Si None, utilise `APPLICATION_CSV`.
    """
    global RAW_DATA_CACHE

    if RAW_DATA_CACHE:
        logger.info(
            "RAW_DATA_CACHE already initialized, skipping",
            extra={
                "extra_data": {
                    "event": "data_raw_cache_skip",
                    "cached_sources": list(RAW_DATA_CACHE.keys()),
                }
            },
        )
        return

    resolved_csv_path = _resolve_application_csv(csv_path)

    logger.info(
        "Initializing RAW_DATA_CACHE",
        extra={
            "extra_data": {
                "event": "data_raw_cache_init_start",
                "csv_path": str(resolved_csv_path),
            }
        },
    )

    RAW_DATA_CACHE = load_all_csv(resolved_csv_path)

    if not RAW_DATA_CACHE:
        raise RuntimeError("Aucune source brute n'a été chargée.")

    cache_shapes = {
        name: list(df.shape)
        for name, df in RAW_DATA_CACHE.items()
        if isinstance(df, pd.DataFrame)
    }

    logger.info(
        "RAW_DATA_CACHE initialized successfully",
        extra={
            "extra_data": {
                "event": "data_raw_cache_init_success",
                "sources": list(RAW_DATA_CACHE.keys()),
                "shapes": cache_shapes,
            }
        },
    )


def get_raw_data_cache() -> dict[str, pd.DataFrame]:
    """
    Retourne le cache des données brutes.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cache des DataFrames bruts.

    Raises
    ------
    RuntimeError
        Si le cache n'a pas été initialisé.
    """
    if not RAW_DATA_CACHE:
        raise RuntimeError(
            "RAW_DATA_CACHE non initialisé. "
            "Appelle `init_raw_data_cache()` au démarrage."
        )

    return RAW_DATA_CACHE


def get_data_cache() -> dict[str, pd.DataFrame]:
    """
    Alias de compatibilité pour l'ancien nom `get_data_cache`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cache des DataFrames bruts.
    """
    return get_raw_data_cache()


# =============================================================================
# Features prêtes
# =============================================================================

def init_features_ready_cache(
    *,
    keep_id: bool = True,
    debug: bool = False,
) -> None:
    """
    Construit les features enrichies une seule fois et les garde en mémoire.

    Parameters
    ----------
    keep_id : bool, optional
        Si True, conserve la colonne `SK_ID_CURR`.
    debug : bool, optional
        Active le mode debug pour le builder de features.
    """
    global FEATURES_READY_CACHE

    if FEATURES_READY_CACHE is not None:
        logger.info(
            "FEATURES_READY_CACHE already initialized, skipping",
            extra={
                "extra_data": {
                    "event": "data_features_cache_skip",
                    "shape": list(FEATURES_READY_CACHE.shape),
                }
            },
        )
        return

    raw_sources = get_raw_data_cache()

    logger.info(
        "Building enriched features dataframe",
        extra={
            "extra_data": {
                "event": "data_features_cache_build_start",
                "keep_id": keep_id,
                "debug": debug,
                "raw_sources": list(raw_sources.keys()),
            }
        },
    )

    FEATURES_READY_CACHE = build_features_from_loaded_data(
        raw_sources=raw_sources,
        client_ids=None,
        debug=debug,
        keep_id=keep_id,
    )

    _validate_dataframe(FEATURES_READY_CACHE, "FEATURES_READY_CACHE")

    if keep_id:
        _ensure_sk_id_curr(FEATURES_READY_CACHE)

    logger.info(
        "FEATURES_READY_CACHE initialized successfully",
        extra={
            "extra_data": {
                "event": "data_features_cache_build_success",
                "shape": list(FEATURES_READY_CACHE.shape),
                "keep_id": keep_id,
            }
        },
    )


def get_features_ready_cache() -> pd.DataFrame:
    """
    Retourne le DataFrame enrichi mis en cache.

    Returns
    -------
    pd.DataFrame
        DataFrame des features prêtes.

    Raises
    ------
    RuntimeError
        Si le cache n'a pas été initialisé.
    """
    if FEATURES_READY_CACHE is None:
        raise RuntimeError(
            "FEATURES_READY_CACHE non initialisé. "
            "Appelle `init_features_ready_cache()`."
        )

    return FEATURES_READY_CACHE


# =============================================================================
# Accès client
# =============================================================================

def get_features_for_client_from_cache(
    client_id: int,
    *,
    keep_id: bool = False,
) -> pd.DataFrame:
    """
    Retourne les features d'un seul client depuis le cache.

    Parameters
    ----------
    client_id : int
        Identifiant client recherché.
    keep_id : bool, optional
        Si True, conserve `SK_ID_CURR` dans le résultat.

    Returns
    -------
    pd.DataFrame
        DataFrame d'une seule ligne correspondant au client.

    Raises
    ------
    ValueError
        Si le client est introuvable ou dupliqué.
    """
    df = get_features_ready_cache()
    _ensure_sk_id_curr(df)

    client_id = int(client_id)
    client_df = df.loc[df["SK_ID_CURR"] == client_id].copy()

    if client_df.empty:
        raise ValueError(f"Client {client_id} introuvable.")

    if len(client_df) > 1:
        raise ValueError(f"Doublon détecté pour le client {client_id}.")

    if not keep_id and "SK_ID_CURR" in client_df.columns:
        client_df.drop(columns=["SK_ID_CURR"], inplace=True)

    logger.info(
        "Single client features retrieved from cache",
        extra={
            "extra_data": {
                "event": "data_get_client_features_success",
                "client_id": client_id,
                "keep_id": keep_id,
                "shape": list(client_df.shape),
            }
        },
    )

    return client_df


def get_features_for_clients_from_cache(
    client_ids: list[int],
    *,
    keep_id: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Retourne les features de plusieurs clients depuis le cache.

    Parameters
    ----------
    client_ids : list[int]
        Liste des identifiants clients à rechercher.
    keep_id : bool, optional
        Si True, conserve `SK_ID_CURR`.
    strict : bool, optional
        Si True, lève une erreur si certains clients sont absents.

    Returns
    -------
    pd.DataFrame
        Sous-ensemble des features pour les clients trouvés.

    Raises
    ------
    ValueError
        Si aucun client n'est trouvé, ou si `strict=True` et qu'il manque des IDs.
    """
    df = get_features_ready_cache()
    _ensure_sk_id_curr(df)

    ids = [int(x) for x in client_ids]
    subset = df.loc[df["SK_ID_CURR"].isin(ids)].copy()

    if subset.empty:
        raise ValueError(f"Aucun client trouvé pour {ids}.")

    found_ids = set(
        pd.to_numeric(subset["SK_ID_CURR"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    missing_ids = sorted(set(ids) - found_ids)

    if strict and missing_ids:
        raise ValueError(f"Clients absents du cache : {missing_ids}")

    if not keep_id and "SK_ID_CURR" in subset.columns:
        subset.drop(columns=["SK_ID_CURR"], inplace=True)

    logger.info(
        "Multiple client features retrieved from cache",
        extra={
            "extra_data": {
                "event": "data_get_clients_features_success",
                "requested_client_ids": ids,
                "missing_client_ids": missing_ids,
                "keep_id": keep_id,
                "strict": strict,
                "shape": list(subset.shape),
            }
        },
    )

    return subset


# =============================================================================
# Initialisation globale
# =============================================================================

def init_full_data_cache(
    csv_path: Path | str | None = None,
    *,
    debug: bool = False,
) -> None:
    """
    Initialise l'ensemble du cache nécessaire au mode CSV optimisé.

    Parameters
    ----------
    csv_path : Path | str | None, optional
        Chemin explicite vers le CSV source.
    debug : bool, optional
        Active le mode debug pour la construction des features.
    """
    resolved_csv_path = _resolve_application_csv(csv_path)

    logger.info(
        "Full data cache initialization started",
        extra={
            "extra_data": {
                "event": "data_full_cache_init_start",
                "csv_path": str(resolved_csv_path),
                "debug": debug,
            }
        },
    )

    init_raw_data_cache(resolved_csv_path)
    init_features_ready_cache(keep_id=True, debug=debug)

    logger.info(
        "Full data cache initialization completed",
        extra={
            "extra_data": {
                "event": "data_full_cache_init_success",
                "csv_path": str(resolved_csv_path),
            }
        },
    )


def reset_data_cache() -> None:
    """
    Réinitialise complètement les caches globaux de données applicatives.
    """
    global RAW_DATA_CACHE, FEATURES_READY_CACHE

    RAW_DATA_CACHE = {}
    FEATURES_READY_CACHE = None

    logger.info(
        "Application data caches reset",
        extra={
            "extra_data": {
                "event": "data_cache_reset",
            }
        },
    )


# =============================================================================
# Monitoring Evidently - références parquet / json
# =============================================================================

def init_monitoring_reference_cache(
    monitoring_dir: Path | str | None = None,
) -> None:
    """
    Initialise le cache des fichiers de référence du monitoring.

    Parameters
    ----------
    monitoring_dir : Path | str | None, optional
        Dossier explicite contenant les fichiers de monitoring.
        Si None, utilise `MONITORING_DIR`.

    Raises
    ------
    FileNotFoundError
        Si le dossier monitoring n'existe pas.
    """
    global MONITORING_REFERENCE_CACHE

    if MONITORING_REFERENCE_CACHE:
        logger.info(
            "MONITORING_REFERENCE_CACHE already initialized, skipping",
            extra={
                "extra_data": {
                    "event": "monitoring_reference_cache_skip",
                    "keys": list(MONITORING_REFERENCE_CACHE.keys()),
                }
            },
        )
        return

    base_dir = _resolve_monitoring_dir(monitoring_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Dossier monitoring introuvable : {base_dir}")

    logger.info(
        "Monitoring reference cache initialization started",
        extra={
            "extra_data": {
                "event": "monitoring_reference_cache_init_start",
                "monitoring_dir": str(base_dir),
            }
        },
    )

    cache: dict[str, Any] = {}

    raw_path = base_dir / "reference_features_raw.parquet"
    transformed_path = base_dir / "reference_features_transformed.parquet"

    cache["reference_features_raw"] = _load_parquet_file(
        raw_path,
        "reference_features_raw",
    )
    cache["reference_features_transformed"] = _load_parquet_file(
        transformed_path,
        "reference_features_transformed",
    )

    optional_parquets = {
        "reference_target": base_dir / "reference_target.parquet",
    }

    for key, path in optional_parquets.items():
        if path.exists():
            cache[key] = pd.read_parquet(path)

    optional_jsons = {
        "input_feature_names": base_dir / "input_feature_names.json",
        "transformed_feature_names": base_dir / "transformed_feature_names.json",
        "reference_metadata": base_dir / "reference_metadata.json",
        "reference_stats_raw": base_dir / "reference_stats_raw.json",
        "reference_stats_transformed": base_dir / "reference_stats_transformed.json",
    }

    for key, path in optional_jsons.items():
        if path.exists():
            cache[key] = _load_json_file(path, key)

    MONITORING_REFERENCE_CACHE = cache

    summary: dict[str, Any] = {}
    for key, value in MONITORING_REFERENCE_CACHE.items():
        if isinstance(value, pd.DataFrame):
            summary[key] = {"type": "DataFrame", "shape": list(value.shape)}
        else:
            summary[key] = {"type": type(value).__name__}

    logger.info(
        "Monitoring reference cache initialized successfully",
        extra={
            "extra_data": {
                "event": "monitoring_reference_cache_init_success",
                "monitoring_dir": str(base_dir),
                "summary": summary,
            }
        },
    )


def get_monitoring_reference_cache() -> dict[str, Any]:
    """
    Retourne le cache des références monitoring.

    Returns
    -------
    dict[str, Any]
        Cache contenant DataFrames et métadonnées de référence.

    Raises
    ------
    RuntimeError
        Si le cache n'a pas été initialisé.
    """
    if not MONITORING_REFERENCE_CACHE:
        raise RuntimeError(
            "MONITORING_REFERENCE_CACHE non initialisé. "
            "Appelle `init_monitoring_reference_cache()`."
        )

    return MONITORING_REFERENCE_CACHE


def get_reference_features_raw_df() -> pd.DataFrame:
    """
    Retourne le DataFrame de référence brut pour Evidently.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame de référence brut.
    """
    cache = get_monitoring_reference_cache()
    df = cache.get("reference_features_raw")
    _validate_dataframe(df, "reference_features_raw")
    return df.copy()


def get_reference_features_transformed_df() -> pd.DataFrame:
    """
    Retourne le DataFrame de référence transformé pour Evidently.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame de référence transformé.
    """
    cache = get_monitoring_reference_cache()
    df = cache.get("reference_features_transformed")
    _validate_dataframe(df, "reference_features_transformed")
    return df.copy()


def get_reference_target_df() -> pd.DataFrame | None:
    """
    Retourne le DataFrame target de référence si disponible.

    Returns
    -------
    pd.DataFrame | None
        Copie du DataFrame target, ou None s'il n'existe pas.
    """
    cache = get_monitoring_reference_cache()
    df = cache.get("reference_target")

    if df is None:
        return None

    if not isinstance(df, pd.DataFrame):
        return None

    return df.copy()


def get_input_feature_names() -> list[str]:
    """
    Retourne la liste des features d'entrée si disponible.

    Returns
    -------
    list[str]
        Liste des noms de features brutes.
    """
    cache = get_monitoring_reference_cache()
    value = cache.get("input_feature_names", [])

    if isinstance(value, list):
        return [str(x) for x in value]

    return []


def get_transformed_feature_names() -> list[str]:
    """
    Retourne la liste des features transformées si disponible.

    Returns
    -------
    list[str]
        Liste des noms de features transformées.
    """
    cache = get_monitoring_reference_cache()
    value = cache.get("transformed_feature_names", [])

    if isinstance(value, list):
        return [str(x) for x in value]

    return []


def reset_monitoring_reference_cache() -> None:
    """
    Réinitialise le cache des références monitoring.
    """
    global MONITORING_REFERENCE_CACHE
    MONITORING_REFERENCE_CACHE = {}

    logger.info(
        "Monitoring reference cache reset",
        extra={
            "extra_data": {
                "event": "monitoring_reference_cache_reset",
            }
        },
    )