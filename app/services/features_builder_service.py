"""
Service unique de construction des features prêtes pour le modèle.

Ce module centralise toute la logique utile pour produire les features
attendues par le modèle de scoring crédit, exclusivement à partir de la
source CSV configurée dans l'application.

Objectif
--------
Fournir un point d'entrée métier unique au reste de l'application afin de :
- charger les données source déjà présentes en mémoire
- sélectionner les colonnes de base utiles
- construire les variables dérivées
- aligner strictement le résultat sur `MODEL_FEATURES`
- retourner soit un DataFrame complet, soit les features d'un seul client

Architecture actuelle
---------------------
- les données source proviennent exclusivement du CSV configuré
  côté application
- aucune lecture de features n'est réalisée depuis PostgreSQL
- PostgreSQL sert uniquement au logging et au monitoring

Fonctions principales
---------------------
- load_raw_csv_sources(...)
    Charge les sources CSV utiles au builder
- build_features_from_loaded_data(...)
    Construit un DataFrame final prêt pour le modèle
- build_model_ready_features(...)
    Point d'entrée métier pour l'application
- build_features_for_client(...)
    Construit un dictionnaire de features pour un seul client

Notes
-----
- Le DataFrame final est strictement aligné sur `MODEL_FEATURES`
- Les colonnes manquantes sont créées avec NaN
- Les colonnes supplémentaires sont ignorées à l'étape finale
- `SK_ID_CURR` peut être conservée optionnellement
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.core.config import APPLICATION_CSV
from app.core.model_features import MODEL_FEATURES


logger = logging.getLogger(__name__)


# =============================================================================
# Colonnes de base attendues depuis la source CSV applicative
# =============================================================================

APPLICATION_BASE_COLUMNS = [
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "LANDAREA_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAREA_AVG",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
]


# =============================================================================
# Utilitaires debug
# =============================================================================

def _debug_title(title: str) -> None:
    """
    Écrit un séparateur lisible dans les logs de debug.
    """
    logger.debug("=" * 90)
    logger.debug("[FEATURE_BUILDER] %s", title)
    logger.debug("=" * 90)


def _debug_df(
    df: pd.DataFrame,
    name: str,
    *,
    preview_rows: int = 3,
    show_columns: bool = False,
    show_missing: bool = True,
) -> None:
    """
    Écrit un résumé lisible d'un DataFrame dans les logs de debug.
    """
    logger.debug("[DEBUG] %s", name)
    logger.debug("Shape: %s", df.shape)
    logger.debug("Nb colonnes: %s", len(df.columns))
    logger.debug("Doublons index: %s", int(df.index.duplicated().sum()))

    if show_missing:
        total_na = int(df.isna().sum().sum())
        logger.debug("Total valeurs NA: %s", total_na)

        na_cols = df.isna().sum()
        na_cols = na_cols[na_cols > 0].sort_values(ascending=False)

        if len(na_cols) == 0:
            logger.debug("Colonnes avec NA: aucune")
        else:
            logger.debug("Colonnes avec NA: %s", len(na_cols))
            for col, nb in na_cols.head(10).items():
                pct = (nb / len(df) * 100) if len(df) > 0 else 0
                logger.debug("  - %s: %s (%.2f %%)", col, int(nb), pct)

    if show_columns:
        logger.debug("Colonnes: %s", list(df.columns))

    if preview_rows > 0 and len(df) > 0:
        logger.debug("Aperçu:\n%s", df.head(preview_rows).to_string())


# =============================================================================
# Chargement des sources CSV
# =============================================================================

def load_raw_csv_sources(csv_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Charge les sources CSV utiles au feature builder.

    Dans la version actuelle, une seule source applicative brute est nécessaire.
    """
    resolved_csv_path = Path(csv_path) if csv_path is not None else Path(APPLICATION_CSV)

    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {resolved_csv_path}")

    if not resolved_csv_path.is_file():
        raise FileNotFoundError(
            f"Le chemin fourni n'est pas un fichier : {resolved_csv_path}"
        )

    logger.info(
        "Loading raw CSV sources for feature builder",
        extra={
            "extra_data": {
                "event": "feature_builder_load_raw_sources_start",
                "csv_path": str(resolved_csv_path),
            }
        },
    )

    application_df = pd.read_csv(resolved_csv_path)

    logger.info(
        "Raw CSV sources loaded successfully for feature builder",
        extra={
            "extra_data": {
                "event": "feature_builder_load_raw_sources_success",
                "csv_path": str(resolved_csv_path),
                "rows": len(application_df),
                "columns": len(application_df.columns),
            }
        },
    )

    return {
        "application": application_df,
    }


# =============================================================================
# Helpers généraux
# =============================================================================

def _normalize_client_ids(client_ids: Sequence[int] | None) -> list[int] | None:
    """
    Normalise une séquence d'identifiants clients.
    """
    if client_ids is None:
        return None

    return [int(x) for x in client_ids]


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Ajoute les colonnes manquantes dans un DataFrame avec NaN comme valeur.
    """
    result = df.copy()

    for col in columns:
        if col not in result.columns:
            result[col] = np.nan

    return result


def _coalesce_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Réalise une division protégée contre les divisions par zéro.
    """
    safe_denominator = denominator.replace(0, np.nan)
    result = numerator / safe_denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def _extract_application_df(source: Any) -> pd.DataFrame:
    """
    Extrait le DataFrame applicatif brut depuis une source souple.

    Formes supportées
    -----------------
    - un DataFrame directement
    - un dictionnaire contenant `application`
    - compatibilité : `application_test`
    - compatibilité : `app`
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, dict):
        if "application" in source:
            df = source["application"]
        elif "application_test" in source:
            df = source["application_test"]
        elif "app" in source:
            df = source["app"]
        else:
            raise ValueError(
                "La source ne contient ni 'application', ni 'application_test', ni 'app'."
            )

        if not isinstance(df, pd.DataFrame):
            raise TypeError("La source extraite n'est pas un DataFrame pandas.")

        return df.copy()

    raise TypeError(
        "La source doit être un DataFrame ou un dictionnaire de DataFrames."
    )


def _resolve_application_source(
    *,
    raw_sources: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    application_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Résout la source applicative brute à utiliser.

    Règles
    ------
    - si `application_df` est fourni, il est prioritaire
    - sinon, on tente d'extraire depuis `raw_sources`
    """
    if application_df is not None:
        if not isinstance(application_df, pd.DataFrame):
            raise TypeError("`application_df` doit être un DataFrame pandas.")
        return application_df.copy()

    if raw_sources is None:
        raise ValueError(
            "Aucune source fournie. Passe `raw_sources` ou `application_df`."
        )

    return _extract_application_df(raw_sources)


def _validate_feature_alignment(df: pd.DataFrame, *, debug: bool = False) -> None:
    """
    Valide l'alignement des colonnes par rapport aux features du modèle.
    """
    missing = [col for col in MODEL_FEATURES if col not in df.columns]
    extra = [col for col in df.columns if col not in MODEL_FEATURES and col != "SK_ID_CURR"]

    logger.info(
        "Feature alignment checked",
        extra={
            "extra_data": {
                "event": "feature_builder_alignment_checked",
                "expected_columns": len(MODEL_FEATURES),
                "present_columns": len(df.columns),
                "missing_columns_count": len(missing),
                "extra_columns_count": len(extra),
                "missing_columns_preview": missing[:10],
                "extra_columns_preview": extra[:10],
            }
        },
    )

    if debug and missing:
        logger.debug("Liste complète des colonnes manquantes :")
        for col in missing:
            logger.debug("  - %s", col)

    if debug and extra:
        logger.debug("Liste complète des colonnes en trop :")
        for col in extra:
            logger.debug("  - %s", col)


def _align_model_features(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    keep_id: bool = False,
) -> pd.DataFrame:
    """
    Aligne le DataFrame final sur la liste exacte des features du modèle.
    """
    _validate_feature_alignment(df, debug=debug)

    aligned_source = _ensure_columns(df, MODEL_FEATURES)

    selected_columns = MODEL_FEATURES.copy()
    if keep_id and "SK_ID_CURR" in aligned_source.columns:
        selected_columns = ["SK_ID_CURR"] + selected_columns

    aligned = aligned_source[selected_columns].copy()

    if debug:
        _debug_df(aligned, "features_aligned", preview_rows=3, show_missing=True)

    return aligned


# =============================================================================
# Enrichissement final
# =============================================================================

def _enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les variables dérivées finales attendues par le modèle.
    """
    f = df.copy()

    required_for_enrichment = [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
        "OWN_CAR_AGE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    f = _ensure_columns(f, required_for_enrichment)

    f["AGE_YEARS"] = -1.0 * f["DAYS_BIRTH"] / 365.25

    f["EMPLOYED_YEARS"] = np.where(
        f["DAYS_EMPLOYED"] == 365243,
        np.nan,
        -1.0 * f["DAYS_EMPLOYED"] / 365.25,
    )

    f["REGISTRATION_YEARS"] = -1.0 * f["DAYS_REGISTRATION"] / 365.25
    f["ID_PUBLISH_YEARS"] = -1.0 * f["DAYS_ID_PUBLISH"] / 365.25

    f["LAST_PHONE_CHANGE_YEARS"] = np.where(
        f["DAYS_LAST_PHONE_CHANGE"] == 0,
        np.nan,
        -1.0 * f["DAYS_LAST_PHONE_CHANGE"] / 365.25,
    )

    f["DAYS_EMPLOYED__isna"] = (
        (f["DAYS_EMPLOYED"].isna()) | (f["DAYS_EMPLOYED"] == 365243)
    ).astype(int)
    f["OWN_CAR_AGE__isna"] = f["OWN_CAR_AGE"].isna().astype(int)
    f["EXT_SOURCE_1__isna"] = f["EXT_SOURCE_1"].isna().astype(int)
    f["EXT_SOURCE_3__isna"] = f["EXT_SOURCE_3"].isna().astype(int)
    f["DAYS_LAST_PHONE_CHANGE__isna"] = (
        (f["DAYS_LAST_PHONE_CHANGE"].isna()) | (f["DAYS_LAST_PHONE_CHANGE"] == 0)
    ).astype(int)
    f["AMT_REQ_CREDIT_BUREAU_HOUR__isna"] = f["AMT_REQ_CREDIT_BUREAU_HOUR"].isna().astype(int)
    f["AMT_REQ_CREDIT_BUREAU_WEEK__isna"] = f["AMT_REQ_CREDIT_BUREAU_WEEK"].isna().astype(int)
    f["AMT_REQ_CREDIT_BUREAU_MON__isna"] = f["AMT_REQ_CREDIT_BUREAU_MON"].isna().astype(int)
    f["AMT_REQ_CREDIT_BUREAU_QRT__isna"] = f["AMT_REQ_CREDIT_BUREAU_QRT"].isna().astype(int)
    f["AMT_REQ_CREDIT_BUREAU_YEAR__isna"] = f["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int)

    f["CREDIT_INCOME_RATIO"] = _coalesce_divide(f["AMT_CREDIT"], f["AMT_INCOME_TOTAL"])
    f["ANNUITY_INCOME_RATIO"] = _coalesce_divide(f["AMT_ANNUITY"], f["AMT_INCOME_TOTAL"])
    f["ANNUITY_CREDIT_RATIO"] = _coalesce_divide(f["AMT_ANNUITY"], f["AMT_CREDIT"])
    f["CREDIT_GOODS_RATIO"] = _coalesce_divide(f["AMT_CREDIT"], f["AMT_GOODS_PRICE"])

    f["OVER_INDEBTED_40"] = (f["ANNUITY_INCOME_RATIO"] > 0.40).astype(int)

    f["LOG_INCOME"] = np.log(np.maximum(f["AMT_INCOME_TOTAL"].fillna(0), 0) + 1)
    f["LOG_CREDIT"] = np.log(np.maximum(f["AMT_CREDIT"].fillna(0), 0) + 1)
    f["LOG_ANNUITY"] = np.log(np.maximum(f["AMT_ANNUITY"].fillna(0), 0) + 1)
    f["LOG_GOODS"] = np.log(np.maximum(f["AMT_GOODS_PRICE"].fillna(0), 0) + 1)

    f["SOCIAL_DEFAULT_RATIO_30"] = _coalesce_divide(
        f["DEF_30_CNT_SOCIAL_CIRCLE"],
        f["OBS_30_CNT_SOCIAL_CIRCLE"] + 1,
    )
    f["SOCIAL_DEFAULT_RATIO_60"] = _coalesce_divide(
        f["DEF_60_CNT_SOCIAL_CIRCLE"],
        f["OBS_60_CNT_SOCIAL_CIRCLE"] + 1,
    )

    doc_cols = [
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
    ]
    f = _ensure_columns(f, doc_cols)

    f["DOC_COUNT"] = f[doc_cols].fillna(0).sum(axis=1)

    f["CONTACT_COUNT"] = f[
        ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"]
    ].fillna(0).sum(axis=1)

    f["ADDRESS_MISMATCH_COUNT"] = f[
        [
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
        ]
    ].fillna(0).sum(axis=1)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    f = _ensure_columns(f, ext_cols)
    ext_df = f[ext_cols]

    f["EXT_SOURCES_MEAN"] = ext_df.mean(axis=1)
    f["EXT_SOURCES_MIN"] = ext_df.min(axis=1)
    f["EXT_SOURCES_MAX"] = ext_df.max(axis=1)
    f["EXT_SOURCES_STD"] = ext_df.std(axis=1, ddof=0)
    f["EXT_SOURCES_RANGE"] = f["EXT_SOURCES_MAX"] - f["EXT_SOURCES_MIN"]

    f["EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_2"] = f["EXT_SOURCE_1"] * f["EXT_SOURCE_2"]
    f["EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_3"] = f["EXT_SOURCE_1"] * f["EXT_SOURCE_3"]
    f["EXT_INT__EXT_SOURCE_2_x_EXT_SOURCE_3"] = f["EXT_SOURCE_2"] * f["EXT_SOURCE_3"]

    f["EXT_POW2__EXT_SOURCE_1"] = f["EXT_SOURCE_1"] ** 2
    f["EXT_POW2__EXT_SOURCE_2"] = f["EXT_SOURCE_2"] ** 2
    f["EXT_POW2__EXT_SOURCE_3"] = f["EXT_SOURCE_3"] ** 2

    return f


# =============================================================================
# Construction principale
# =============================================================================

def build_features_from_loaded_data(
    raw_sources: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    *,
    application_df: pd.DataFrame | None = None,
    client_ids: Sequence[int] | None = None,
    debug: bool = False,
    keep_id: bool = False,
) -> pd.DataFrame:
    """
    Construit les features modèle à partir de sources déjà chargées.

    Parameters
    ----------
    raw_sources : pd.DataFrame | dict[str, pd.DataFrame] | None
        Soit directement la source applicative brute, soit un dictionnaire
        contenant `application`.
    application_df : pd.DataFrame | None, default=None
        DataFrame source fourni explicitement.
    client_ids : Sequence[int] | None, default=None
        Liste optionnelle de clients à reconstruire.
    debug : bool, default=False
        Active l'affichage détaillé des étapes.
    keep_id : bool, default=False
        Si True, conserve `SK_ID_CURR` dans la sortie.
    """
    normalized_client_ids = _normalize_client_ids(client_ids)

    logger.info(
        "Feature building from loaded data started",
        extra={
            "extra_data": {
                "event": "feature_builder_build_start",
                "keep_id": keep_id,
                "debug": debug,
                "client_ids_count": len(normalized_client_ids) if normalized_client_ids is not None else None,
            }
        },
    )

    if debug:
        _debug_title("CONSTRUCTION DES FEATURES DEPUIS LES SOURCES CHARGÉES")

    app = _resolve_application_source(
        raw_sources=raw_sources,
        application_df=application_df,
    )

    if normalized_client_ids is not None:
        if "SK_ID_CURR" not in app.columns:
            raise ValueError("Colonne SK_ID_CURR absente de la source applicative.")

        app = app[app["SK_ID_CURR"].isin(normalized_client_ids)].copy()

        if app.empty:
            raise ValueError(
                f"Aucun client trouvé dans la source applicative pour client_ids={normalized_client_ids}"
            )

    if debug:
        _debug_df(app, "application_source", preview_rows=3)

    app = _ensure_columns(app, APPLICATION_BASE_COLUMNS)
    features = app[APPLICATION_BASE_COLUMNS].copy()

    if debug:
        _debug_title("BASE CLIENT")
        _debug_df(features, "features_base", preview_rows=3)

    features = _enrich_features(features)
    features = _align_model_features(
        features,
        debug=debug,
        keep_id=keep_id,
    )

    if normalized_client_ids is not None and keep_id and "SK_ID_CURR" in features.columns:
        features = features[features["SK_ID_CURR"].isin(normalized_client_ids)].copy()

    if debug:
        _debug_title("FEATURES FINALES")
        _debug_df(features, "features_finales", preview_rows=5)

    logger.info(
        "Feature building from loaded data completed",
        extra={
            "extra_data": {
                "event": "feature_builder_build_success",
                "keep_id": keep_id,
                "rows": len(features),
                "columns": len(features.columns),
            }
        },
    )

    return features


# =============================================================================
# Point d'entrée métier pour l'application
# =============================================================================

def build_model_ready_features(
    *,
    client_ids: Sequence[int] | None = None,
    debug: bool = False,
    keep_id: bool = False,
) -> pd.DataFrame:
    """
    Point d'entrée métier pour obtenir les features prêtes pour le modèle.

    Notes
    -----
    L'import du cache loader est volontairement local pour éviter
    les imports circulaires au chargement des modules.
    """
    from app.services.loader_services.data_loading_service import get_raw_data_cache

    normalized_client_ids = _normalize_client_ids(client_ids)

    logger.info(
        "Model-ready feature build requested",
        extra={
            "extra_data": {
                "event": "feature_builder_model_ready_start",
                "application_csv": str(APPLICATION_CSV),
                "keep_id": keep_id,
                "debug": debug,
                "client_ids_count": len(normalized_client_ids) if normalized_client_ids is not None else None,
            }
        },
    )

    if debug:
        _debug_title("DÉMARRAGE FEATURE BUILDER")
        logger.debug("source de données configurée : %s", APPLICATION_CSV)
        logger.debug("keep_id : %s", keep_id)
        logger.debug(
            "client_ids fournis : %s",
            None if normalized_client_ids is None else len(normalized_client_ids),
        )

    raw_sources = get_raw_data_cache()

    result = build_features_from_loaded_data(
        raw_sources=raw_sources,
        client_ids=normalized_client_ids,
        debug=debug,
        keep_id=keep_id,
    )

    logger.info(
        "Model-ready feature build completed",
        extra={
            "extra_data": {
                "event": "feature_builder_model_ready_success",
                "rows": len(result),
                "columns": len(result.columns),
                "keep_id": keep_id,
            }
        },
    )

    return result


def build_features_for_client(
    client_id: int,
    *,
    debug: bool = False,
    keep_id: bool = False,
) -> dict[str, Any]:
    """
    Construit les features prêtes pour le modèle pour un seul client.
    """
    client_id = int(client_id)

    logger.info(
        "Single-client feature build requested",
        extra={
            "extra_data": {
                "event": "feature_builder_single_client_start",
                "client_id": client_id,
                "keep_id": keep_id,
                "debug": debug,
            }
        },
    )

    df = build_model_ready_features(
        client_ids=[client_id],
        debug=debug,
        keep_id=True,
    )

    if df.empty:
        raise ValueError(f"Aucune feature construite pour le client {client_id}.")

    if len(df) != 1:
        raise ValueError(
            f"Le feature builder a retourné {len(df)} lignes pour le client {client_id}, "
            "alors qu'une seule était attendue."
        )

    row = df.iloc[0].to_dict()

    if not keep_id:
        row.pop("SK_ID_CURR", None)

    logger.info(
        "Single-client feature build completed",
        extra={
            "extra_data": {
                "event": "feature_builder_single_client_success",
                "client_id": client_id,
                "feature_count": len(row),
                "keep_id": keep_id,
            }
        },
    )

    return row


# =============================================================================
# Transfrome de feature enrichies à features enrichies trabnsformées
# =============================================================================

def build_transformed_features_from_loaded_data(
    raw_sources: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    *,
    application_df: pd.DataFrame | None = None,
    client_ids: Sequence[int] | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Construit les features transformées réellement vues par le modèle.

    Notes
    -----
    - construit d'abord les features modèle-ready
    - applique ensuite toute la partie transformation du pipeline
      (toutes les étapes sauf la dernière, supposée être l'estimateur)
    - retourne un DataFrame pandas avec les noms de colonnes transformées
      si disponibles
    """
    from sklearn.pipeline import Pipeline

    from app.services.loader_services.model_loading_service import get_model

    raw_df = build_features_from_loaded_data(
        raw_sources=raw_sources,
        application_df=application_df,
        client_ids=client_ids,
        debug=debug,
        keep_id=False,
    )

    model = get_model()

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Le modèle chargé n'est pas un Pipeline sklearn. "
            "Impossible de reconstruire les features transformées."
        )

    if len(model.steps) < 2:
        raise ValueError(
            "Le pipeline chargé ne contient pas assez d'étapes pour "
            "distinguer transformation et estimateur final."
        )

    transformer_steps = model.steps[:-1]

    if not transformer_steps:
        raise ValueError(
            "Aucune étape de transformation détectée dans le pipeline chargé."
        )

    transformer_pipeline = Pipeline(transformer_steps)
    transformed = transformer_pipeline.transform(raw_df)

    feature_names: list[str] | None = None

    if hasattr(transformer_pipeline, "get_feature_names_out"):
        try:
            feature_names = list(transformer_pipeline.get_feature_names_out())
        except Exception:
            feature_names = None

    if feature_names is None:
        last_transformer = transformer_steps[-1][1]

        if hasattr(last_transformer, "get_feature_names_out"):
            try:
                feature_names = list(last_transformer.get_feature_names_out())
            except Exception:
                feature_names = None

    if feature_names is None:
        if hasattr(transformed, "shape") and len(transformed.shape) == 2:
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
        else:
            raise ValueError(
                "Impossible de déterminer les noms des features transformées."
            )

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=raw_df.index,
    )

    logger.info(
        "Transformed features built successfully from pipeline",
        extra={
            "extra_data": {
                "event": "feature_builder_transformed_build_success",
                "rows": len(transformed_df),
                "columns": len(transformed_df.columns),
                "transformer_steps": [name for name, _ in transformer_steps],
            }
        },
    )

    return transformed_df