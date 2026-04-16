"""
Service de construction des features prêtes pour la prédiction.

Ce module permet de produire un DataFrame final aligné sur les features
attendues par le modèle, à partir de deux sources possibles :

- TYPE_ENTREE_DONNEES = "DB"
    Lecture directe depuis PostgreSQL d'une table finale déjà préparée

- TYPE_ENTREE_DONNEES = "CSV"
    Lecture des fichiers CSV bruts et reconstruction des features
    par agrégations pandas, cleaning, enrichissement puis alignement final

Objectif
--------
Retourner un DataFrame prêt à être passé au pipeline de prédiction,
avec exactement les colonnes de `MODEL_FEATURES`, dans le bon ordre.

Notes
-----
- Ce service ne fait pas la prédiction lui-même.
- En mode CSV, il reproduit en pandas la logique métier principale :
  - agrégations bureau / bureau_balance
  - agrégations previous_application / POS / CC / installments
  - enrichissement des variables dérivées
  - alignement exact sur le schéma du modèle
- Les colonnes manquantes sont créées avec NaN.
- Un mode debug permet de visualiser les étapes intermédiaires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from app.core.config import DATABASE_URL, TYPE_ENTREE_DONNEES
from app.core.schemas import MODEL_FEATURES


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_DIR = Path("data")
DEFAULT_FINAL_DB_TABLE = "features_client_test_enriched"


# =============================================================================
# Colonnes de base attendues depuis application_test
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
    "FLAG_MOBIL",
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
# Spécifications d'agrégation
# =============================================================================

BUREAU_BASE_SPECS = {
    "DAYS_CREDIT": ["mean", "std"],
    "CREDIT_DAY_OVERDUE": ["mean", "std"],
    "DAYS_CREDIT_ENDDATE": ["mean", "std"],
    "DAYS_ENDDATE_FACT": ["mean", "std"],
    "AMT_CREDIT_MAX_OVERDUE": ["mean", "std"],
    "CNT_CREDIT_PROLONG": ["mean", "std"],
    "AMT_CREDIT_SUM": ["mean", "std"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "std"],
    "AMT_CREDIT_SUM_LIMIT": ["mean", "std"],
    "AMT_CREDIT_SUM_OVERDUE": ["mean", "std"],
    "DAYS_CREDIT_UPDATE": ["mean", "std"],
    "AMT_ANNUITY": ["mean", "std"],
    "bb__MONTHS_BALANCE__mean": ["mean", "std"],
    "bb__MONTHS_BALANCE__std": ["mean", "std"],
    "bb__count_rows": ["mean", "std"],
    "bb__recent_max_dpd": ["mean", "std"],
    "bb__months_late_ratio": ["mean", "std"],
    "bb__late_severity_sum": ["mean", "std"],
    "bureau__DEBT_RATIO": ["mean", "std"],
    "bureau__OVERDUE_RATIO": ["mean", "std"],
    "bureau__IS_ACTIVE": ["mean", "std"],
    "bureau__HAS_OVERDUE": ["mean", "std"],
    "bureau__CREDIT_AGE": ["mean", "std"],
}

POS_PREV_SPECS = {
    "pos__MONTHS_BALANCE__mean": ["mean", "std"],
    "pos__MONTHS_BALANCE__std": ["mean", "std"],
    "pos__CNT_INSTALMENT__mean": ["mean", "std"],
    "pos__CNT_INSTALMENT__std": ["mean", "std"],
    "pos__CNT_INSTALMENT_FUTURE__mean": ["mean", "std"],
    "pos__CNT_INSTALMENT_FUTURE__std": ["mean", "std"],
    "pos__SK_DPD__mean": ["mean", "std"],
    "pos__SK_DPD__max": ["mean", "std"],
    "pos__SK_DPD__std": ["mean", "std"],
    "pos__SK_DPD_DEF__mean": ["mean", "std"],
    "pos__SK_DPD_DEF__max": ["mean", "std"],
    "pos__SK_DPD_DEF__std": ["mean", "std"],
    "pos__POS_REMAIN_RATIO__mean": ["mean", "std"],
    "pos__POS_REMAIN_RATIO__std": ["mean", "std"],
    "pos__POS_DPD_POS__mean": ["mean", "std"],
    "pos__POS_DPD_POS__max": ["mean", "std"],
    "pos__POS_DPD_POS__std": ["mean", "std"],
    "pos__POS_IS_ACTIVE__mean": ["mean", "std"],
    "pos__POS_IS_ACTIVE__std": ["mean", "std"],
}

CC_PREV_SPECS = {
    "cc__MONTHS_BALANCE__mean": ["mean"],
    "cc__MONTHS_BALANCE__std": ["mean"],
    "cc__AMT_BALANCE__mean": ["mean"],
    "cc__AMT_BALANCE__std": ["mean"],
    "cc__AMT_CREDIT_LIMIT_ACTUAL__mean": ["mean"],
    "cc__AMT_CREDIT_LIMIT_ACTUAL__std": ["mean"],
    "cc__AMT_DRAWINGS_ATM_CURRENT__mean": ["mean"],
    "cc__AMT_DRAWINGS_ATM_CURRENT__std": ["mean"],
    "cc__AMT_DRAWINGS_CURRENT__mean": ["mean"],
    "cc__AMT_DRAWINGS_CURRENT__std": ["mean"],
    "cc__AMT_DRAWINGS_OTHER_CURRENT__mean": ["mean"],
    "cc__AMT_DRAWINGS_OTHER_CURRENT__std": ["mean"],
    "cc__AMT_DRAWINGS_POS_CURRENT__mean": ["mean"],
    "cc__AMT_DRAWINGS_POS_CURRENT__std": ["mean"],
    "cc__AMT_INST_MIN_REGULARITY__mean": ["mean"],
    "cc__AMT_INST_MIN_REGULARITY__std": ["mean"],
    "cc__AMT_PAYMENT_CURRENT__mean": ["mean"],
    "cc__AMT_PAYMENT_CURRENT__std": ["mean"],
    "cc__AMT_PAYMENT_TOTAL_CURRENT__mean": ["mean"],
    "cc__AMT_PAYMENT_TOTAL_CURRENT__std": ["mean"],
    "cc__AMT_RECEIVABLE_PRINCIPAL__mean": ["mean"],
    "cc__AMT_RECEIVABLE_PRINCIPAL__std": ["mean"],
    "cc__AMT_RECIVABLE__mean": ["mean"],
    "cc__AMT_RECIVABLE__std": ["mean"],
    "cc__AMT_TOTAL_RECEIVABLE__mean": ["mean"],
    "cc__AMT_TOTAL_RECEIVABLE__std": ["mean"],
    "cc__CNT_DRAWINGS_ATM_CURRENT__mean": ["mean"],
    "cc__CNT_DRAWINGS_ATM_CURRENT__std": ["mean"],
    "cc__CNT_DRAWINGS_CURRENT__mean": ["mean"],
    "cc__CNT_DRAWINGS_CURRENT__std": ["mean"],
    "cc__CNT_DRAWINGS_OTHER_CURRENT__mean": ["mean"],
    "cc__CNT_DRAWINGS_OTHER_CURRENT__std": ["mean"],
    "cc__CNT_DRAWINGS_POS_CURRENT__mean": ["mean"],
    "cc__CNT_DRAWINGS_POS_CURRENT__std": ["mean"],
    "cc__CNT_INSTALMENT_MATURE_CUM__mean": ["mean"],
    "cc__CNT_INSTALMENT_MATURE_CUM__std": ["mean"],
    "cc__SK_DPD__mean": ["mean"],
    "cc__SK_DPD__max": ["mean"],
    "cc__SK_DPD__std": ["mean"],
    "cc__SK_DPD_DEF__mean": ["mean"],
    "cc__SK_DPD_DEF__max": ["mean"],
    "cc__SK_DPD_DEF__std": ["mean"],
    "cc__CC_UTILIZATION_RATIO__mean": ["mean"],
    "cc__CC_UTILIZATION_RATIO__max": ["mean"],
    "cc__CC_UTILIZATION_RATIO__std": ["mean"],
    "cc__CC_PAYMENT_MIN_RATIO__mean": ["mean"],
    "cc__CC_PAYMENT_MIN_RATIO__max": ["mean"],
    "cc__CC_PAYMENT_MIN_RATIO__std": ["mean"],
    "cc__CC_PAYMENT_BALANCE_RATIO__mean": ["mean"],
    "cc__CC_PAYMENT_BALANCE_RATIO__max": ["mean"],
    "cc__CC_PAYMENT_BALANCE_RATIO__std": ["mean"],
    "cc__CC_DPD_POS__mean": ["mean"],
    "cc__CC_DPD_POS__max": ["mean"],
    "cc__CC_DPD_POS__std": ["mean"],
    "cc__CC_RECEIVABLE_RATIO__mean": ["mean"],
    "cc__CC_RECEIVABLE_RATIO__std": ["mean"],
}

INST_PREV_SPECS = {
    "inst__NUM_INSTALMENT_VERSION__mean": ["mean", "std"],
    "inst__NUM_INSTALMENT_VERSION__std": ["mean", "std"],
    "inst__NUM_INSTALMENT_NUMBER__mean": ["mean", "std"],
    "inst__NUM_INSTALMENT_NUMBER__std": ["mean", "std"],
    "inst__DAYS_INSTALMENT__mean": ["mean", "std"],
    "inst__DAYS_INSTALMENT__std": ["mean", "std"],
    "inst__DAYS_ENTRY_PAYMENT__mean": ["mean", "std"],
    "inst__DAYS_ENTRY_PAYMENT__std": ["mean", "std"],
    "inst__AMT_INSTALMENT__mean": ["mean", "std"],
    "inst__AMT_INSTALMENT__std": ["mean", "std"],
    "inst__AMT_PAYMENT__mean": ["mean", "std"],
    "inst__AMT_PAYMENT__std": ["mean", "std"],
    "inst__DPD_POS__mean": ["mean", "std"],
    "inst__DPD_POS__max": ["mean", "std"],
    "inst__DPD_POS__std": ["mean", "std"],
    "inst__SEVERE_LATE_30__mean": ["mean", "std"],
    "inst__SEVERE_LATE_30__max": ["mean", "std"],
    "inst__SEVERE_LATE_30__std": ["mean", "std"],
    "inst__PAY_RATIO__mean": ["mean", "std"],
    "inst__PAY_RATIO__std": ["mean", "std"],
}

PREV_BASE_SPECS = {
    "AMT_ANNUITY": ["mean", "std"],
    "AMT_APPLICATION": ["mean", "std"],
    "AMT_CREDIT": ["mean", "std"],
    "AMT_DOWN_PAYMENT": ["mean", "std"],
    "AMT_GOODS_PRICE": ["mean", "std"],
    "HOUR_APPR_PROCESS_START": ["mean", "std"],
    "NFLAG_LAST_APPL_IN_DAY": ["mean", "std"],
    "RATE_DOWN_PAYMENT": ["mean", "std"],
    "DAYS_DECISION": ["mean", "std"],
    "SELLERPLACE_AREA": ["mean", "std"],
    "CNT_PAYMENT": ["mean", "std"],
    "DAYS_FIRST_DRAWING": ["mean"],
    "DAYS_FIRST_DUE": ["mean", "std"],
    "DAYS_LAST_DUE_1ST_VERSION": ["mean", "std"],
    "DAYS_LAST_DUE": ["mean", "std"],
    "DAYS_TERMINATION": ["mean", "std"],
    "NFLAG_INSURED_ON_APPROVAL": ["mean", "std"],
    "prev__PREV_CREDIT_APPLICATION_RATIO": ["mean", "std"],
    "prev__PREV_IS_APPROVED": ["mean", "std"],
    "prev__PREV_IS_REFUSED": ["mean", "std"],
    "prev__PREV_DAYS_DECISION_AGE": ["mean", "std"],
    "prev__PREV_CREDIT_DURATION": ["mean", "std"],
}


# =============================================================================
# Utilitaires debug
# =============================================================================

def _debug_title(title: str) -> None:
    print("\n" + "=" * 90)
    print(f"[FEATURE_BUILDER] {title}")
    print("=" * 90)


def _debug_df(
    df: pd.DataFrame,
    name: str,
    *,
    preview_rows: int = 3,
    show_columns: bool = False,
    show_missing: bool = True,
) -> None:
    """
    Affiche un résumé lisible d'un DataFrame pour le debug.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à inspecter.
    name : str
        Nom logique affiché dans le log.
    preview_rows : int, default=3
        Nombre de lignes affichées en aperçu.
    show_columns : bool, default=False
        Affiche la liste complète des colonnes.
    show_missing : bool, default=True
        Affiche un résumé des valeurs manquantes.
    """
    print(f"\n[DEBUG] {name}")
    print(f"Shape               : {df.shape}")
    print(f"Nb colonnes         : {len(df.columns)}")
    print(f"Doublons index      : {df.index.duplicated().sum() if df.index is not None else 0}")

    if show_missing:
        total_na = int(df.isna().sum().sum())
        print(f"Total valeurs NA    : {total_na}")

        na_cols = df.isna().sum()
        na_cols = na_cols[na_cols > 0].sort_values(ascending=False)

        if len(na_cols) == 0:
            print("Colonnes avec NA    : aucune")
        else:
            print(f"Colonnes avec NA    : {len(na_cols)}")
            preview = na_cols.head(10)
            for col, nb in preview.items():
                pct = (nb / len(df) * 100) if len(df) > 0 else 0
                print(f"  - {col}: {nb} ({pct:.2f} %)")

    if show_columns:
        print("Colonnes :")
        print(list(df.columns))

    if preview_rows > 0 and len(df) > 0:
        print("Aperçu :")
        print(df.head(preview_rows).to_string())


# =============================================================================
# Utilitaires généraux
# =============================================================================

def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    """
    Lit un fichier CSV dans le dossier source.

    Parameters
    ----------
    data_dir : Path
        Dossier racine des fichiers CSV.
    name : str
        Nom du fichier CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame chargé depuis le disque.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    """
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path)


def _std_pop(series: pd.Series) -> float:
    """
    Calcule un écart-type population, équivalent à PostgreSQL `STDDEV_POP`.
    """
    return float(series.std(ddof=0)) if len(series.dropna()) > 0 else np.nan


def _agg_named(series: pd.Series, stat: str) -> float:
    """
    Applique une statistique nommée à une série.
    """
    clean = series.dropna()

    if stat == "mean":
        return float(clean.mean()) if len(clean) > 0 else np.nan
    if stat == "std":
        return _std_pop(series)
    if stat == "max":
        return float(clean.max()) if len(clean) > 0 else np.nan
    if stat == "min":
        return float(clean.min()) if len(clean) > 0 else np.nan
    if stat == "count":
        return float(series.count())
    if stat == "nunique":
        return float(series.nunique())

    raise ValueError(f"Statistique non supportée : {stat}")


def _aggregate_group(
    df: pd.DataFrame,
    group_col: str,
    specs: dict[str, list[str]],
    prefix: str,
) -> pd.DataFrame:
    """
    Agrège un DataFrame par groupe selon un mapping colonne -> statistiques.
    """
    rows = []

    for group_value, group_df in df.groupby(group_col, dropna=False):
        row = {group_col: group_value}

        for col, stats in specs.items():
            if col not in group_df.columns:
                continue

            for stat in stats:
                out_col = f"{prefix}{col}__{stat}"
                row[out_col] = _agg_named(group_df[col], stat)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[group_col])

    return pd.DataFrame(rows)


def _coalesce_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Réalise une division protégée contre les divisions par zéro.
    """
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Ajoute les colonnes manquantes dans un DataFrame avec NaN comme valeur.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _validate_feature_alignment(df: pd.DataFrame, *, debug: bool = False) -> None:
    """
    Valide l'alignement des colonnes par rapport aux features du modèle.
    """
    missing = [col for col in MODEL_FEATURES if col not in df.columns]
    extra = [col for col in df.columns if col not in MODEL_FEATURES and col != "SK_ID_CURR"]

    print(f"[FEATURE_BUILDER] Nb colonnes attendues        : {len(MODEL_FEATURES)}")
    print(f"[FEATURE_BUILDER] Nb colonnes présentes         : {len(df.columns)}")
    print(f"[FEATURE_BUILDER] Nb colonnes manquantes        : {len(missing)}")
    print(f"[FEATURE_BUILDER] Nb colonnes supplémentaires   : {len(extra)}")

    if missing:
        print(f"[FEATURE_BUILDER] Exemples colonnes manquantes : {missing[:10]}")
    if extra:
        print(f"[FEATURE_BUILDER] Exemples colonnes en trop    : {extra[:10]}")

    if debug and missing:
        print("[DEBUG] Liste complète des colonnes manquantes :")
        for col in missing:
            print(f"  - {col}")

    if debug and extra:
        print("[DEBUG] Liste complète des colonnes en trop :")
        for col in extra:
            print(f"  - {col}")


def _align_model_features(df: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    Aligne le DataFrame final sur la liste exacte des features du modèle.
    """
    _validate_feature_alignment(df, debug=debug)
    df = _ensure_columns(df, MODEL_FEATURES)
    aligned = df[MODEL_FEATURES].copy()

    if debug:
        _debug_df(aligned, "features_aligned", preview_rows=3, show_missing=True)

    return aligned


# =============================================================================
# Agrégations bureau / bureau_balance
# =============================================================================

def _build_bb_agg_csv(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bb = bureau_balance.copy()

    status_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

    bb["bb_status_num"] = bb["STATUS"].map(status_map).fillna(0)
    bb["bb_is_late"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    bb["bb_recent_dpd"] = np.where(bb["MONTHS_BALANCE"] >= -6, bb["bb_status_num"], np.nan)

    grouped = bb.groupby("SK_ID_BUREAU", dropna=False)

    bb_agg = grouped.agg(
        bb__count_rows=("MONTHS_BALANCE", "count"),
        bb__MONTHS_BALANCE__mean=("MONTHS_BALANCE", "mean"),
    ).reset_index()

    bb_std = grouped["MONTHS_BALANCE"].apply(lambda s: s.std(ddof=0)).reset_index(name="bb__MONTHS_BALANCE__std")
    bb_recent = grouped["bb_recent_dpd"].max().reset_index(name="bb__recent_max_dpd")
    bb_late_ratio = grouped["bb_is_late"].mean().reset_index(name="bb__months_late_ratio")
    bb_late_sum = grouped["bb_status_num"].sum().reset_index(name="bb__late_severity_sum")

    bb_agg = bb_agg.merge(bb_std, on="SK_ID_BUREAU", how="left")
    bb_agg = bb_agg.merge(bb_recent, on="SK_ID_BUREAU", how="left")
    bb_agg = bb_agg.merge(bb_late_ratio, on="SK_ID_BUREAU", how="left")
    bb_agg = bb_agg.merge(bb_late_sum, on="SK_ID_BUREAU", how="left")

    return bb_agg


def _build_bureau_agg_csv(bureau: pd.DataFrame, bb_agg: pd.DataFrame) -> pd.DataFrame:
    b = bureau.copy()
    b = b.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    b["bureau__DEBT_RATIO"] = _coalesce_divide(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])
    b["bureau__OVERDUE_RATIO"] = _coalesce_divide(b["AMT_CREDIT_SUM_OVERDUE"], b["AMT_CREDIT_SUM"])
    b["bureau__IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(float)
    b["bureau__HAS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(float)
    b["bureau__CREDIT_AGE"] = -1.0 * b["DAYS_CREDIT"]

    agg_df = _aggregate_group(
        df=b,
        group_col="SK_ID_CURR",
        specs=BUREAU_BASE_SPECS,
        prefix="bureau__",
    )

    counts = b.groupby("SK_ID_CURR", dropna=False).agg(
        bureau__count_rows=("SK_ID_BUREAU", "count"),
        bureau__nunique_SK_ID_BUREAU=("SK_ID_BUREAU", "nunique"),
    ).reset_index()

    agg_df = counts.merge(agg_df, on="SK_ID_CURR", how="left")
    return agg_df


# =============================================================================
# Agrégations POS / CC / installments au niveau SK_ID_PREV
# =============================================================================

def _build_pos_agg_csv(pos: pd.DataFrame) -> pd.DataFrame:
    p = pos.copy()

    p["pos__POS_REMAIN_RATIO"] = _coalesce_divide(p["CNT_INSTALMENT_FUTURE"], p["CNT_INSTALMENT"])
    p["pos__POS_DPD_POS"] = p["SK_DPD"].clip(lower=0)
    p["pos__POS_IS_ACTIVE"] = (p["NAME_CONTRACT_STATUS"] == "Active").astype(float)

    grouped = p.groupby("SK_ID_PREV", dropna=False)

    result = grouped.agg(
        pos__count_rows=("MONTHS_BALANCE", "count"),
        pos__MONTHS_BALANCE__mean=("MONTHS_BALANCE", "mean"),
        pos__CNT_INSTALMENT__mean=("CNT_INSTALMENT", "mean"),
        pos__CNT_INSTALMENT_FUTURE__mean=("CNT_INSTALMENT_FUTURE", "mean"),
        pos__SK_DPD__mean=("SK_DPD", "mean"),
        pos__SK_DPD__max=("SK_DPD", "max"),
        pos__SK_DPD_DEF__mean=("SK_DPD_DEF", "mean"),
        pos__SK_DPD_DEF__max=("SK_DPD_DEF", "max"),
        pos__POS_REMAIN_RATIO__mean=("pos__POS_REMAIN_RATIO", "mean"),
        pos__POS_DPD_POS__mean=("pos__POS_DPD_POS", "mean"),
        pos__POS_DPD_POS__max=("pos__POS_DPD_POS", "max"),
        pos__POS_IS_ACTIVE__mean=("pos__POS_IS_ACTIVE", "mean"),
    ).reset_index()

    std_specs = {
        "MONTHS_BALANCE": "pos__MONTHS_BALANCE__std",
        "CNT_INSTALMENT": "pos__CNT_INSTALMENT__std",
        "CNT_INSTALMENT_FUTURE": "pos__CNT_INSTALMENT_FUTURE__std",
        "SK_DPD": "pos__SK_DPD__std",
        "SK_DPD_DEF": "pos__SK_DPD_DEF__std",
        "pos__POS_REMAIN_RATIO": "pos__POS_REMAIN_RATIO__std",
        "pos__POS_DPD_POS": "pos__POS_DPD_POS__std",
        "pos__POS_IS_ACTIVE": "pos__POS_IS_ACTIVE__std",
    }

    for source_col, target_col in std_specs.items():
        tmp = grouped[source_col].apply(lambda s: s.std(ddof=0)).reset_index(name=target_col)
        result = result.merge(tmp, on="SK_ID_PREV", how="left")

    return result


def _build_cc_agg_csv(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc.copy()

    c["cc__CC_UTILIZATION_RATIO"] = _coalesce_divide(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    c["cc__CC_PAYMENT_MIN_RATIO"] = _coalesce_divide(c["AMT_PAYMENT_CURRENT"], c["AMT_INST_MIN_REGULARITY"])
    c["cc__CC_PAYMENT_BALANCE_RATIO"] = _coalesce_divide(c["AMT_PAYMENT_CURRENT"], c["AMT_BALANCE"])
    c["cc__CC_DPD_POS"] = c["SK_DPD"].clip(lower=0)
    c["cc__CC_RECEIVABLE_RATIO"] = _coalesce_divide(c["AMT_RECEIVABLE_PRINCIPAL"], c["AMT_TOTAL_RECEIVABLE"])

    grouped = c.groupby("SK_ID_PREV", dropna=False)

    result = grouped.agg(
        cc__count_rows=("MONTHS_BALANCE", "count"),
        cc__MONTHS_BALANCE__mean=("MONTHS_BALANCE", "mean"),
        cc__AMT_BALANCE__mean=("AMT_BALANCE", "mean"),
        cc__AMT_CREDIT_LIMIT_ACTUAL__mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        cc__AMT_DRAWINGS_ATM_CURRENT__mean=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
        cc__AMT_DRAWINGS_CURRENT__mean=("AMT_DRAWINGS_CURRENT", "mean"),
        cc__AMT_DRAWINGS_OTHER_CURRENT__mean=("AMT_DRAWINGS_OTHER_CURRENT", "mean"),
        cc__AMT_DRAWINGS_POS_CURRENT__mean=("AMT_DRAWINGS_POS_CURRENT", "mean"),
        cc__AMT_INST_MIN_REGULARITY__mean=("AMT_INST_MIN_REGULARITY", "mean"),
        cc__AMT_PAYMENT_CURRENT__mean=("AMT_PAYMENT_CURRENT", "mean"),
        cc__AMT_PAYMENT_TOTAL_CURRENT__mean=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
        cc__AMT_RECEIVABLE_PRINCIPAL__mean=("AMT_RECEIVABLE_PRINCIPAL", "mean"),
        cc__AMT_RECIVABLE__mean=("AMT_RECIVABLE", "mean"),
        cc__AMT_TOTAL_RECEIVABLE__mean=("AMT_TOTAL_RECEIVABLE", "mean"),
        cc__CNT_DRAWINGS_ATM_CURRENT__mean=("CNT_DRAWINGS_ATM_CURRENT", "mean"),
        cc__CNT_DRAWINGS_CURRENT__mean=("CNT_DRAWINGS_CURRENT", "mean"),
        cc__CNT_DRAWINGS_OTHER_CURRENT__mean=("CNT_DRAWINGS_OTHER_CURRENT", "mean"),
        cc__CNT_DRAWINGS_POS_CURRENT__mean=("CNT_DRAWINGS_POS_CURRENT", "mean"),
        cc__CNT_INSTALMENT_MATURE_CUM__mean=("CNT_INSTALMENT_MATURE_CUM", "mean"),
        cc__SK_DPD__mean=("SK_DPD", "mean"),
        cc__SK_DPD__max=("SK_DPD", "max"),
        cc__SK_DPD_DEF__mean=("SK_DPD_DEF", "mean"),
        cc__SK_DPD_DEF__max=("SK_DPD_DEF", "max"),
        cc__CC_UTILIZATION_RATIO__mean=("cc__CC_UTILIZATION_RATIO", "mean"),
        cc__CC_UTILIZATION_RATIO__max=("cc__CC_UTILIZATION_RATIO", "max"),
        cc__CC_PAYMENT_MIN_RATIO__mean=("cc__CC_PAYMENT_MIN_RATIO", "mean"),
        cc__CC_PAYMENT_MIN_RATIO__max=("cc__CC_PAYMENT_MIN_RATIO", "max"),
        cc__CC_PAYMENT_BALANCE_RATIO__mean=("cc__CC_PAYMENT_BALANCE_RATIO", "mean"),
        cc__CC_PAYMENT_BALANCE_RATIO__max=("cc__CC_PAYMENT_BALANCE_RATIO", "max"),
        cc__CC_DPD_POS__mean=("cc__CC_DPD_POS", "mean"),
        cc__CC_DPD_POS__max=("cc__CC_DPD_POS", "max"),
        cc__CC_RECEIVABLE_RATIO__mean=("cc__CC_RECEIVABLE_RATIO", "mean"),
    ).reset_index()

    std_cols = {
        "MONTHS_BALANCE": "cc__MONTHS_BALANCE__std",
        "AMT_BALANCE": "cc__AMT_BALANCE__std",
        "AMT_CREDIT_LIMIT_ACTUAL": "cc__AMT_CREDIT_LIMIT_ACTUAL__std",
        "AMT_DRAWINGS_ATM_CURRENT": "cc__AMT_DRAWINGS_ATM_CURRENT__std",
        "AMT_DRAWINGS_CURRENT": "cc__AMT_DRAWINGS_CURRENT__std",
        "AMT_DRAWINGS_OTHER_CURRENT": "cc__AMT_DRAWINGS_OTHER_CURRENT__std",
        "AMT_DRAWINGS_POS_CURRENT": "cc__AMT_DRAWINGS_POS_CURRENT__std",
        "AMT_INST_MIN_REGULARITY": "cc__AMT_INST_MIN_REGULARITY__std",
        "AMT_PAYMENT_CURRENT": "cc__AMT_PAYMENT_CURRENT__std",
        "AMT_PAYMENT_TOTAL_CURRENT": "cc__AMT_PAYMENT_TOTAL_CURRENT__std",
        "AMT_RECEIVABLE_PRINCIPAL": "cc__AMT_RECEIVABLE_PRINCIPAL__std",
        "AMT_RECIVABLE": "cc__AMT_RECIVABLE__std",
        "AMT_TOTAL_RECEIVABLE": "cc__AMT_TOTAL_RECEIVABLE__std",
        "CNT_DRAWINGS_ATM_CURRENT": "cc__CNT_DRAWINGS_ATM_CURRENT__std",
        "CNT_DRAWINGS_CURRENT": "cc__CNT_DRAWINGS_CURRENT__std",
        "CNT_DRAWINGS_OTHER_CURRENT": "cc__CNT_DRAWINGS_OTHER_CURRENT__std",
        "CNT_DRAWINGS_POS_CURRENT": "cc__CNT_DRAWINGS_POS_CURRENT__std",
        "CNT_INSTALMENT_MATURE_CUM": "cc__CNT_INSTALMENT_MATURE_CUM__std",
        "SK_DPD": "cc__SK_DPD__std",
        "SK_DPD_DEF": "cc__SK_DPD_DEF__std",
        "cc__CC_UTILIZATION_RATIO": "cc__CC_UTILIZATION_RATIO__std",
        "cc__CC_PAYMENT_MIN_RATIO": "cc__CC_PAYMENT_MIN_RATIO__std",
        "cc__CC_PAYMENT_BALANCE_RATIO": "cc__CC_PAYMENT_BALANCE_RATIO__std",
        "cc__CC_DPD_POS": "cc__CC_DPD_POS__std",
        "cc__CC_RECEIVABLE_RATIO": "cc__CC_RECEIVABLE_RATIO__std",
    }

    for source_col, target_col in std_cols.items():
        tmp = grouped[source_col].apply(lambda s: s.std(ddof=0)).reset_index(name=target_col)
        result = result.merge(tmp, on="SK_ID_PREV", how="left")

    return result


def _build_inst_agg_csv(inst: pd.DataFrame) -> pd.DataFrame:
    i = inst.copy()

    i["inst__DPD_POS"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["inst__SEVERE_LATE_30"] = (i["inst__DPD_POS"] >= 30).astype(float)
    i["inst__PAY_RATIO"] = _coalesce_divide(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])

    grouped = i.groupby("SK_ID_PREV", dropna=False)

    result = grouped.agg(
        inst__count_rows=("NUM_INSTALMENT_NUMBER", "count"),
        inst__NUM_INSTALMENT_VERSION__mean=("NUM_INSTALMENT_VERSION", "mean"),
        inst__NUM_INSTALMENT_NUMBER__mean=("NUM_INSTALMENT_NUMBER", "mean"),
        inst__DAYS_INSTALMENT__mean=("DAYS_INSTALMENT", "mean"),
        inst__DAYS_ENTRY_PAYMENT__mean=("DAYS_ENTRY_PAYMENT", "mean"),
        inst__AMT_INSTALMENT__mean=("AMT_INSTALMENT", "mean"),
        inst__AMT_PAYMENT__mean=("AMT_PAYMENT", "mean"),
        inst__DPD_POS__mean=("inst__DPD_POS", "mean"),
        inst__DPD_POS__max=("inst__DPD_POS", "max"),
        inst__SEVERE_LATE_30__mean=("inst__SEVERE_LATE_30", "mean"),
        inst__SEVERE_LATE_30__max=("inst__SEVERE_LATE_30", "max"),
        inst__PAY_RATIO__mean=("inst__PAY_RATIO", "mean"),
    ).reset_index()

    std_cols = {
        "NUM_INSTALMENT_VERSION": "inst__NUM_INSTALMENT_VERSION__std",
        "NUM_INSTALMENT_NUMBER": "inst__NUM_INSTALMENT_NUMBER__std",
        "DAYS_INSTALMENT": "inst__DAYS_INSTALMENT__std",
        "DAYS_ENTRY_PAYMENT": "inst__DAYS_ENTRY_PAYMENT__std",
        "AMT_INSTALMENT": "inst__AMT_INSTALMENT__std",
        "AMT_PAYMENT": "inst__AMT_PAYMENT__std",
        "inst__DPD_POS": "inst__DPD_POS__std",
        "inst__SEVERE_LATE_30": "inst__SEVERE_LATE_30__std",
        "inst__PAY_RATIO": "inst__PAY_RATIO__std",
    }

    for source_col, target_col in std_cols.items():
        tmp = grouped[source_col].apply(lambda s: s.std(ddof=0)).reset_index(name=target_col)
        result = result.merge(tmp, on="SK_ID_PREV", how="left")

    return result


# =============================================================================
# Agrégations previous_application au niveau SK_ID_CURR
# =============================================================================

def _build_prev_agg_csv(
    prev: pd.DataFrame,
    pos_agg: pd.DataFrame,
    cc_agg: pd.DataFrame,
    inst_agg: pd.DataFrame,
) -> pd.DataFrame:
    p = prev.copy()
    p = p.merge(pos_agg, on="SK_ID_PREV", how="left")
    p = p.merge(cc_agg, on="SK_ID_PREV", how="left")
    p = p.merge(inst_agg, on="SK_ID_PREV", how="left")

    p["prev__PREV_CREDIT_APPLICATION_RATIO"] = _coalesce_divide(p["AMT_CREDIT"], p["AMT_APPLICATION"])
    p["prev__PREV_IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(float)
    p["prev__PREV_IS_REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(float)
    p["prev__PREV_DAYS_DECISION_AGE"] = -1.0 * p["DAYS_DECISION"]
    p["prev__PREV_CREDIT_DURATION"] = p["DAYS_LAST_DUE"] - p["DAYS_FIRST_DUE"]

    base_agg = _aggregate_group(
        df=p,
        group_col="SK_ID_CURR",
        specs=PREV_BASE_SPECS,
        prefix="prev__",
    )

    pos_nested = _aggregate_group(
        df=p,
        group_col="SK_ID_CURR",
        specs=POS_PREV_SPECS,
        prefix="prev__",
    )

    cc_nested = _aggregate_group(
        df=p,
        group_col="SK_ID_CURR",
        specs=CC_PREV_SPECS,
        prefix="prev__",
    )

    inst_nested = _aggregate_group(
        df=p,
        group_col="SK_ID_CURR",
        specs=INST_PREV_SPECS,
        prefix="prev__",
    )

    counts = p.groupby("SK_ID_CURR", dropna=False).agg(
        prev__count_rows=("SK_ID_PREV", "count"),
        prev__nunique_SK_ID_PREV=("SK_ID_PREV", "nunique"),
        prev__pos__count_rows__mean=("pos__count_rows", "mean"),
        prev__pos__count_rows__std=("pos__count_rows", lambda s: s.std(ddof=0)),
        prev__cc__count_rows__mean=("cc__count_rows", "mean"),
        prev__inst__count_rows__mean=("inst__count_rows", "mean"),
        prev__inst__count_rows__std=("inst__count_rows", lambda s: s.std(ddof=0)),
    ).reset_index()

    result = counts.merge(base_agg, on="SK_ID_CURR", how="left")
    result = result.merge(pos_nested, on="SK_ID_CURR", how="left")
    result = result.merge(cc_nested, on="SK_ID_CURR", how="left")
    result = result.merge(inst_nested, on="SK_ID_CURR", how="left")

    return result


# =============================================================================
# Enrichissement final
# =============================================================================

def _enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

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

    f["DAYS_EMPLOYED__isna"] = ((f["DAYS_EMPLOYED"].isna()) | (f["DAYS_EMPLOYED"] == 365243)).astype(int)
    f["OWN_CAR_AGE__isna"] = f["OWN_CAR_AGE"].isna().astype(int)
    f["EXT_SOURCE_1__isna"] = f["EXT_SOURCE_1"].isna().astype(int)
    f["EXT_SOURCE_3__isna"] = f["EXT_SOURCE_3"].isna().astype(int)
    f["DAYS_LAST_PHONE_CHANGE__isna"] = ((f["DAYS_LAST_PHONE_CHANGE"].isna()) | (f["DAYS_LAST_PHONE_CHANGE"] == 0)).astype(int)
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

    f["SOCIAL_DEFAULT_RATIO_30"] = f["DEF_30_CNT_SOCIAL_CIRCLE"] / (f["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    f["SOCIAL_DEFAULT_RATIO_60"] = f["DEF_60_CNT_SOCIAL_CIRCLE"] / (f["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)

    doc_cols = [
        "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
    ]
    f = _ensure_columns(f, doc_cols + ["FLAG_MOBIL"])

    f["DOC_COUNT"] = f[doc_cols].fillna(0).sum(axis=1)
    f["CONTACT_COUNT"] = f[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"]].fillna(0).sum(axis=1)

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
# Construction depuis CSV
# =============================================================================

def build_features_from_csv(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Construit les features prêtes pour le modèle à partir des CSV bruts.
    """
    if debug:
        _debug_title("LECTURE DES CSV")

    app = _read_csv(data_dir, "application_test.csv")
    bureau = _read_csv(data_dir, "bureau.csv")
    bureau_balance = _read_csv(data_dir, "bureau_balance.csv")
    prev = _read_csv(data_dir, "previous_application.csv")
    pos = _read_csv(data_dir, "POS_CASH_balance.csv")
    cc = _read_csv(data_dir, "credit_card_balance.csv")
    inst = _read_csv(data_dir, "installments_payments.csv")

    if debug:
        _debug_df(app, "application_test", preview_rows=2)
        _debug_df(bureau, "bureau", preview_rows=2)
        _debug_df(bureau_balance, "bureau_balance", preview_rows=2)
        _debug_df(prev, "previous_application", preview_rows=2)
        _debug_df(pos, "POS_CASH_balance", preview_rows=2)
        _debug_df(cc, "credit_card_balance", preview_rows=2)
        _debug_df(inst, "installments_payments", preview_rows=2)

    if debug:
        _debug_title("CONSTRUCTION BASE CLIENT")

    app = _ensure_columns(app, APPLICATION_BASE_COLUMNS)
    base = app[APPLICATION_BASE_COLUMNS].copy()

    if debug:
        _debug_df(base, "base_client", preview_rows=3)

    if debug:
        _debug_title("AGRÉGATIONS")

    bb_agg = _build_bb_agg_csv(bureau_balance)
    bureau_agg = _build_bureau_agg_csv(bureau, bb_agg)

    pos_agg = _build_pos_agg_csv(pos)
    cc_agg = _build_cc_agg_csv(cc)
    inst_agg = _build_inst_agg_csv(inst)

    prev_agg = _build_prev_agg_csv(prev, pos_agg, cc_agg, inst_agg)

    if debug:
        _debug_df(bb_agg, "bb_agg", preview_rows=3)
        _debug_df(bureau_agg, "bureau_agg", preview_rows=3)
        _debug_df(pos_agg, "pos_agg", preview_rows=3)
        _debug_df(cc_agg, "cc_agg", preview_rows=3)
        _debug_df(inst_agg, "inst_agg", preview_rows=3)
        _debug_df(prev_agg, "prev_agg", preview_rows=3)

    if debug:
        _debug_title("JOINTURE FINALE")

    features = base.merge(bureau_agg, on="SK_ID_CURR", how="left")
    features = features.merge(prev_agg, on="SK_ID_CURR", how="left")

    if debug:
        _debug_df(features, "features_apres_merge", preview_rows=3)

    if debug:
        _debug_title("ENRICHISSEMENT FINAL")

    features = _enrich_features(features)

    if debug:
        _debug_df(features, "features_apres_enrichissement", preview_rows=3)

    if debug:
        _debug_title("ALIGNEMENT MODELE")

    features = _align_model_features(features, debug=debug)

    if debug:
        _debug_title("TERMINÉ")
        _debug_df(features, "features_finales", preview_rows=5)

    return features


# =============================================================================
# Construction depuis DB
# =============================================================================

def build_features_from_db(
    table_name: str = DEFAULT_FINAL_DB_TABLE,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Charge les features prêtes pour le modèle depuis PostgreSQL.
    """
    if debug:
        _debug_title("LECTURE DEPUIS POSTGRESQL")
        print(f"[DEBUG] Table source : {table_name}")

    engine = create_engine(DATABASE_URL, echo=False)
    query = f'SELECT * FROM "{table_name}"'
    df = pd.read_sql(query, engine)

    if debug:
        _debug_df(df, f"table_db_{table_name}", preview_rows=3)

    df = _align_model_features(df, debug=debug)

    return df


# =============================================================================
# Point d'entrée métier unique
# =============================================================================

def build_model_ready_features(*, debug: bool = False) -> pd.DataFrame:
    """
    Point d'entrée unique pour obtenir les features prêtes pour le modèle.

    Parameters
    ----------
    debug : bool, default=False
        Active l'affichage détaillé des étapes de construction.

    Returns
    -------
    pd.DataFrame
        DataFrame final aligné sur MODEL_FEATURES.

    Raises
    ------
    ValueError
        Si TYPE_ENTREE_DONNEES n'est ni CSV ni DB.
    """
    if debug:
        _debug_title("DÉMARRAGE FEATURE BUILDER")
        print(f"[DEBUG] TYPE_ENTREE_DONNEES : {TYPE_ENTREE_DONNEES}")

    if TYPE_ENTREE_DONNEES == "CSV":
        return build_features_from_csv(debug=debug)

    if TYPE_ENTREE_DONNEES == "DB":
        return build_features_from_db(debug=debug)

    raise ValueError(
        f"TYPE_ENTREE_DONNEES invalide : {TYPE_ENTREE_DONNEES}. "
        "Valeurs attendues : CSV ou DB."
    )