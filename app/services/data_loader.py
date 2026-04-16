"""
Chargement des données pour la prédiction avec mode debug.

Ce module sélectionne la source des données en fonction de
TYPE_ENTREE_DONNEES :

- CSV : lecture depuis un fichier local
- DB  : lecture depuis PostgreSQL

Objectif
--------
Fournir un DataFrame prêt à être utilisé par le modèle tout en
facilitant le débogage grâce à un affichage lisible des informations
essentielles sur les données chargées.

Fonctionnalités
---------------
- chargement depuis CSV ou PostgreSQL
- affichage de la source utilisée
- aperçu des dimensions du DataFrame
- affichage des types de colonnes
- résumé des valeurs manquantes
- aperçu des premières lignes
- option de vérification des colonnes attendues

Notes
-----
- Le mode debug peut être activé via la variable d'environnement DEBUG_DATA
  ou via le paramètre debug=True.
- Ce service ne modifie pas les données, il les charge et les inspecte.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine


# =============================================================================
# Configuration
# =============================================================================

TYPE_ENTREE_DONNEES = os.getenv("TYPE_ENTREE_DONNEES", "DB").upper()
DATABASE_URL = os.getenv("DATABASE_URL")

# Attention au nom du fichier : vérifie bien qu'il s'appelle exactement ainsi
FEATURES_CSV_PATH = Path("artifacts/ARTIFeatures_final.csv")

# Mode debug activable par variable d'environnement
DEBUG_DATA = os.getenv("DEBUG_DATA", "False").lower() == "true"

# Table SQL source
FEATURES_TABLE_NAME = "features_client_test_enriched"


# =============================================================================
# Fonctions utilitaires d'affichage / debug
# =============================================================================

def _print_separator(title: str) -> None:
    """
    Affiche un séparateur lisible dans la console.

    Parameters
    ----------
    title : str
        Titre de la section affichée.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _safe_memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Calcule la mémoire utilisée par le DataFrame en Mo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à mesurer.

    Returns
    -------
    float
        Taille mémoire estimée en Mo.
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def debug_dataframe(
    df: pd.DataFrame,
    *,
    source: str,
    preview_rows: int = 5,
    max_missing_display: int = 20,
) -> None:
    """
    Affiche un résumé visuel du DataFrame pour le debug.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chargé.
    source : str
        Source des données affichée dans le résumé.
    preview_rows : int, default=5
        Nombre de lignes à afficher en aperçu.
    max_missing_display : int, default=20
        Nombre maximum de colonnes avec NA à afficher.
    """
    _print_separator("DEBUG DATAFRAME")

    print(f"Source                : {source}")
    print(f"Shape                 : {df.shape}")
    print(f"Nb colonnes           : {len(df.columns)}")
    print(f"Mémoire estimée       : {_safe_memory_usage_mb(df):.2f} Mo")

    print("\nColonnes :")
    print(list(df.columns))

    print("\nTypes de colonnes :")
    dtype_df = df.dtypes.astype(str).reset_index()
    dtype_df.columns = ["colonne", "dtype"]
    print(dtype_df.to_string(index=False))

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\nValeurs manquantes :")
    if missing.empty:
        print("Aucune valeur manquante détectée.")
    else:
        missing_df = pd.DataFrame({
            "colonne": missing.index,
            "nb_manquants": missing.values,
            "pct_manquants": (missing.values / len(df) * 100).round(2),
        })
        print(missing_df.head(max_missing_display).to_string(index=False))

        if len(missing_df) > max_missing_display:
            print(
                f"... {len(missing_df) - max_missing_display} colonnes supplémentaires "
                "avec valeurs manquantes non affichées."
            )

    print("\nAperçu des premières lignes :")
    print(df.head(preview_rows).to_string())


def validate_expected_columns(
    df: pd.DataFrame,
    expected_columns: Iterable[str],
) -> None:
    """
    Vérifie la présence des colonnes attendues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chargé.
    expected_columns : Iterable[str]
        Colonnes attendues par le modèle ou le pipeline.
    """
    expected_set = set(expected_columns)
    actual_set = set(df.columns)

    missing_cols = sorted(expected_set - actual_set)
    extra_cols = sorted(actual_set - expected_set)

    _print_separator("VALIDATION DES COLONNES")

    print(f"Colonnes attendues : {len(expected_set)}")
    print(f"Colonnes présentes : {len(actual_set)}")
    print(f"Colonnes manquantes : {len(missing_cols)}")
    print(f"Colonnes en trop    : {len(extra_cols)}")

    if missing_cols:
        print("\nColonnes manquantes :")
        for col in missing_cols:
            print(f"- {col}")

    if extra_cols:
        print("\nColonnes en trop :")
        for col in extra_cols:
            print(f"- {col}")


# =============================================================================
# Fonctions de chargement
# =============================================================================

def _load_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.

    Parameters
    ----------
    csv_path : Path
        Chemin vers le fichier CSV.

    Returns
    -------
    pd.DataFrame
        Données chargées.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[DATA] Chargé depuis CSV : {df.shape}")
    return df


def _load_from_db(database_url: str, table_name: str) -> pd.DataFrame:
    """
    Charge les données depuis PostgreSQL.

    Parameters
    ----------
    database_url : str
        URL SQLAlchemy de connexion à PostgreSQL.
    table_name : str
        Nom de la table source.

    Returns
    -------
    pd.DataFrame
        Données chargées.
    """
    if not database_url:
        raise ValueError("DATABASE_URL manquant")

    engine = create_engine(database_url, echo=False)

    query = f'SELECT * FROM "{table_name}"'
    df = pd.read_sql(query, engine)

    print(f"[DATA] Chargé depuis DB ({table_name}) : {df.shape}")
    return df


# =============================================================================
# Fonction principale
# =============================================================================

def load_prediction_data(
    *,
    debug: bool | None = None,
    preview_rows: int = 5,
    expected_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Charge les données utilisées pour la prédiction.

    Parameters
    ----------
    debug : bool | None, default=None
        Active l'affichage détaillé du debug.
        Si None, utilise la variable d'environnement DEBUG_DATA.
    preview_rows : int, default=5
        Nombre de lignes affichées dans l'aperçu debug.
    expected_columns : Iterable[str] | None, default=None
        Liste optionnelle de colonnes attendues pour validation.

    Returns
    -------
    pd.DataFrame
        DataFrame chargé et prêt à être utilisé.
    """
    debug = DEBUG_DATA if debug is None else debug

    if TYPE_ENTREE_DONNEES == "CSV":
        df = _load_from_csv(FEATURES_CSV_PATH)
        source = f"CSV -> {FEATURES_CSV_PATH}"

    elif TYPE_ENTREE_DONNEES == "DB":
        df = _load_from_db(DATABASE_URL, FEATURES_TABLE_NAME)
        source = f"DB -> {FEATURES_TABLE_NAME}"

    else:
        raise ValueError(
            f"TYPE_ENTREE_DONNEES invalide : {TYPE_ENTREE_DONNEES}. "
            "Valeurs autorisées : CSV ou DB."
        )

    if debug:
        debug_dataframe(
            df,
            source=source,
            preview_rows=preview_rows,
        )

        if expected_columns is not None:
            validate_expected_columns(df, expected_columns)

    return df