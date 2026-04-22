"""
Service de chargement et de mise en cache des données source et des features prêtes.

Objectif
--------
- charger une seule fois le CSV brut source configuré
- construire une seule fois les features enrichies
- servir les données en mémoire pour les prédictions unitaires ou batch

Architecture
------------
CSV source unique -> RAW_DATA_CACHE -> FEATURES_READY_CACHE -> API

Notes
-----
- ce service est utile quand l'API fonctionne en mode CSV / mémoire
- il évite de relire les fichiers et de recalculer les features à chaque requête
- il repose sur un cache global simple, suffisant pour ce projet
- le fichier chargé est piloté par la configuration `APPLICATION_CSV`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import APPLICATION_CSV
from app.services.features_builder_service import build_features_from_loaded_data


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
    """
    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Colonne `SK_ID_CURR` absente du DataFrame.")


def _resolve_application_csv(csv_path: Path | str | None = None) -> Path:
    """
    Résout le chemin du CSV source principal.
    """
    return Path(csv_path) if csv_path is not None else Path(APPLICATION_CSV)


# =============================================================================
# Chargement des données brutes
# =============================================================================

def load_all_csv(csv_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Charge uniquement le CSV source principal en mémoire.

    Parameters
    ----------
    csv_path : Path | str | None
        Chemin du fichier CSV source à charger.
        Si None, utilise la configuration APPLICATION_CSV.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping nom logique -> DataFrame brut.

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV est introuvable.
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

    df = pd.read_csv(resolved_csv_path)
    _validate_dataframe(df, "application_source")

    return {"application": df}


def init_raw_data_cache(csv_path: Path | str | None = None) -> None:
    """
    Initialise le cache du CSV brut source.
    """
    global RAW_DATA_CACHE

    if RAW_DATA_CACHE:
        print("[DATA] RAW_DATA_CACHE déjà initialisé -> skip")
        return

    resolved_csv_path = _resolve_application_csv(csv_path)

    print(f"[DATA] Chargement du CSV brut source en mémoire : {resolved_csv_path}")
    RAW_DATA_CACHE = load_all_csv(resolved_csv_path)

    if not RAW_DATA_CACHE:
        raise RuntimeError("Aucune source brute n'a été chargée.")

    print("[DATA] CSV brut chargé :")
    for name, df in RAW_DATA_CACHE.items():
        shape = df.shape if isinstance(df, pd.DataFrame) else ("?", "?")
        print(f"   - {name:<25} shape={shape}")


def get_raw_data_cache() -> dict[str, pd.DataFrame]:
    """
    Retourne le cache des données brutes.
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
    """
    global FEATURES_READY_CACHE

    if FEATURES_READY_CACHE is not None:
        print("[DATA] FEATURES_READY_CACHE déjà initialisé -> skip")
        return

    raw_sources = get_raw_data_cache()

    print("[DATA] Construction du DataFrame enrichi...")
    FEATURES_READY_CACHE = build_features_from_loaded_data(
        raw_sources=raw_sources,
        client_ids=None,
        debug=debug,
        keep_id=keep_id,
    )

    _validate_dataframe(FEATURES_READY_CACHE, "FEATURES_READY_CACHE")

    print(f"[DATA] Features prêtes OK (shape={FEATURES_READY_CACHE.shape})")

    if keep_id:
        _ensure_sk_id_curr(FEATURES_READY_CACHE)


def get_features_ready_cache() -> pd.DataFrame:
    """
    Retourne le DataFrame enrichi mis en cache.
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

    return client_df


def get_features_for_clients_from_cache(
    client_ids: list[int],
    *,
    keep_id: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Retourne les features de plusieurs clients depuis le cache.
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
    """
    resolved_csv_path = _resolve_application_csv(csv_path)

    print(f"[DATA] Initialisation complète du cache depuis : {resolved_csv_path}")

    init_raw_data_cache(resolved_csv_path)
    init_features_ready_cache(keep_id=True, debug=debug)

    print("[DATA] Initialisation complète terminée.")


def reset_data_cache() -> None:
    """
    Réinitialise complètement les caches globaux.
    """
    global RAW_DATA_CACHE, FEATURES_READY_CACHE

    RAW_DATA_CACHE = {}
    FEATURES_READY_CACHE = None

    print("[DATA] Cache reset effectué.")


# =============================================================================
# Monitoring Evidently - références parquet / json
# =============================================================================

def _resolve_monitoring_dir(monitoring_dir: Path | str | None = None) -> Path:
    """
    Résout le répertoire de monitoring.
    """
    if monitoring_dir is not None:
        return Path(monitoring_dir)

    return Path("monitoring")


def _load_parquet_file(file_path: Path, logical_name: str) -> pd.DataFrame:
    """
    Charge un fichier parquet et vérifie qu'il est exploitable.
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
    """
    import json

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable pour `{logical_name}` : {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_monitoring_reference_cache(
    monitoring_dir: Path | str | None = None,
) -> None:
    """
    Initialise le cache des fichiers de référence du monitoring.
    """
    global MONITORING_REFERENCE_CACHE

    if MONITORING_REFERENCE_CACHE:
        print("[DATA][MONITORING] MONITORING_REFERENCE_CACHE déjà initialisé -> skip")
        return

    base_dir = _resolve_monitoring_dir(monitoring_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Dossier monitoring introuvable : {base_dir}")

    print(f"[DATA][MONITORING] Chargement des références monitoring depuis : {base_dir}")

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
            print(f"[DATA][MONITORING] {key:<35} loaded shape={cache[key].shape}")

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
            print(f"[DATA][MONITORING] {key:<35} loaded")

    MONITORING_REFERENCE_CACHE = cache

    print("[DATA][MONITORING] Références monitoring chargées :")
    for key, value in MONITORING_REFERENCE_CACHE.items():
        if isinstance(value, pd.DataFrame):
            print(f"   - {key:<35} shape={value.shape}")
        else:
            print(f"   - {key:<35} type={type(value).__name__}")


def get_monitoring_reference_cache() -> dict[str, Any]:
    """
    Retourne le cache des références monitoring.
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
    """
    cache = get_monitoring_reference_cache()
    df = cache.get("reference_features_raw")
    _validate_dataframe(df, "reference_features_raw")
    return df.copy()


def get_reference_features_transformed_df() -> pd.DataFrame:
    """
    Retourne le DataFrame de référence transformé pour Evidently.
    """
    cache = get_monitoring_reference_cache()
    df = cache.get("reference_features_transformed")
    _validate_dataframe(df, "reference_features_transformed")
    return df.copy()


def get_reference_target_df() -> pd.DataFrame | None:
    """
    Retourne le DataFrame target de référence si disponible.
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
    """
    cache = get_monitoring_reference_cache()
    value = cache.get("input_feature_names", [])

    if isinstance(value, list):
        return [str(x) for x in value]

    return []


def get_transformed_feature_names() -> list[str]:
    """
    Retourne la liste des features transformées si disponible.
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
    print("[DATA][MONITORING] Cache monitoring reset effectué.")