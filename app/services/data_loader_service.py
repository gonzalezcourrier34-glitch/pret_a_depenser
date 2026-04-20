"""
Service de chargement et de mise en cache des données source et des features prêtes.

Objectif
--------
- charger une seule fois les CSV bruts
- construire une seule fois les features enrichies
- servir les données en mémoire pour les prédictions unitaires ou batch

Architecture
------------
CSV -> RAW_DATA_CACHE -> FEATURES_READY_CACHE -> API

Notes
-----
- ce service est utile quand l'API fonctionne en mode CSV / mémoire
- il évite de relire les fichiers et de recalculer les features à chaque requête
- il repose sur un cache global simple, suffisant pour ce projet
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from app.core.config import DATA_DIR
from app.services.feature_builder_service import (
    build_features_from_loaded_data,
    load_raw_csv_sources,
)


# =============================================================================
# Caches globaux
# =============================================================================

RAW_DATA_CACHE: Dict[str, pd.DataFrame] = {}
FEATURES_READY_CACHE: pd.DataFrame | None = None


# =============================================================================
# Helpers internes
# =============================================================================

def _validate_dataframe(df: pd.DataFrame | None, name: str) -> None:
    """
    Vérifie qu'un DataFrame est bien exploitable.

    Parameters
    ----------
    df : pd.DataFrame | None
        DataFrame à valider.
    name : str
        Nom logique du DataFrame pour le message d'erreur.

    Raises
    ------
    RuntimeError
        Si le DataFrame est absent, vide ou invalide.
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
        DataFrame à contrôler.

    Raises
    ------
    ValueError
        Si la colonne SK_ID_CURR est absente.
    """
    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Colonne `SK_ID_CURR` absente du DataFrame.")


# =============================================================================
# Chargement des données brutes
# =============================================================================

def load_all_csv(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """
    Charge tous les CSV bruts du projet en mémoire.

    Parameters
    ----------
    data_dir : Path | str
        Dossier contenant les fichiers CSV source.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping nom logique -> DataFrame brut.

    Raises
    ------
    FileNotFoundError
        Si le dossier de données est introuvable.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Le dossier data est introuvable : {data_dir}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Le chemin fourni n'est pas un dossier : {data_dir}")

    return load_raw_csv_sources(data_dir)


def init_raw_data_cache(data_dir: Path | str) -> None:
    """
    Initialise le cache des CSV bruts.

    Parameters
    ----------
    data_dir : Path | str
        Dossier contenant les fichiers CSV source.
    """
    global RAW_DATA_CACHE

    if RAW_DATA_CACHE:
        print("[DATA] RAW_DATA_CACHE déjà initialisé -> skip")
        return

    print("[DATA] Chargement des CSV bruts en mémoire...")
    RAW_DATA_CACHE = load_all_csv(data_dir)

    if not RAW_DATA_CACHE:
        raise RuntimeError("Aucune source brute n'a été chargée.")

    print("[DATA] CSV bruts chargés :")
    for name, df in RAW_DATA_CACHE.items():
        shape = df.shape if isinstance(df, pd.DataFrame) else ("?", "?")
        print(f"   - {name:<25} shape={shape}")


def get_raw_data_cache() -> Dict[str, pd.DataFrame]:
    """
    Retourne le cache des données brutes.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionnaire des tables brutes chargées en mémoire.

    Raises
    ------
    RuntimeError
        Si le cache n'est pas encore initialisé.
    """
    if not RAW_DATA_CACHE:
        raise RuntimeError(
            "RAW_DATA_CACHE non initialisé. "
            "Appelle `init_raw_data_cache()` au démarrage."
        )

    return RAW_DATA_CACHE


def get_data_cache() -> Dict[str, pd.DataFrame]:
    """
    Alias de compatibilité pour l'ancien nom `get_data_cache`.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Cache des données brutes.
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
    keep_id : bool, default=True
        Conserve ou non la colonne SK_ID_CURR dans le DataFrame final.
    debug : bool, default=False
        Active éventuellement les logs détaillés côté feature builder.

    Raises
    ------
    RuntimeError
        Si le DataFrame enrichi produit est vide ou invalide.
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

    print(
        "[DATA] Features prêtes OK "
        f"(shape={FEATURES_READY_CACHE.shape})"
    )

    if keep_id:
        _ensure_sk_id_curr(FEATURES_READY_CACHE)


def get_features_ready_cache() -> pd.DataFrame:
    """
    Retourne le DataFrame enrichi mis en cache.

    Returns
    -------
    pd.DataFrame
        DataFrame complet des features prêtes.

    Raises
    ------
    RuntimeError
        Si le cache des features n'est pas initialisé.
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
        Identifiant client SK_ID_CURR.
    keep_id : bool, default=False
        Conserve ou non la colonne SK_ID_CURR dans le résultat.

    Returns
    -------
    pd.DataFrame
        DataFrame d'une seule ligne correspondant au client demandé.

    Raises
    ------
    ValueError
        Si le client est introuvable, dupliqué, ou si la clé est absente.
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

    Parameters
    ----------
    client_ids : list[int]
        Liste des identifiants clients à récupérer.
    keep_id : bool, default=True
        Conserve ou non la colonne SK_ID_CURR.
    strict : bool, default=False
        Si True, lève une erreur si un ou plusieurs clients sont absents.

    Returns
    -------
    pd.DataFrame
        Sous-ensemble du cache contenant les clients trouvés.

    Raises
    ------
    ValueError
        Si aucun client n'est trouvé, ou si strict=True et qu'il manque des IDs.
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
    data_dir: Path | str | None = None,
    *,
    debug: bool = False,
) -> None:
    """
    Initialise l'ensemble du cache nécessaire au mode CSV optimisé.
    """
    resolved_data_dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)

    print(f"[DATA] Initialisation complète du cache depuis : {resolved_data_dir}")

    init_raw_data_cache(resolved_data_dir)
    init_features_ready_cache(keep_id=True, debug=debug)

    print("[DATA] Initialisation complète terminée.")


def reset_data_cache() -> None:
    """
    Réinitialise complètement les caches globaux.

    Notes
    -----
    Utiliser uniquement pour les tests ou pour forcer un rechargement.
    """
    global RAW_DATA_CACHE, FEATURES_READY_CACHE

    RAW_DATA_CACHE = {}
    FEATURES_READY_CACHE = None

    print("[DATA] Cache reset effectué.")