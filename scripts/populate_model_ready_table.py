"""
Script de remplissage de la table finale compatible modèle.

Ce module alimente `features_client_test_model` à partir de
`features_client_test_enriched`, en sélectionnant exactement
les colonnes attendues par le modèle, dans le bon ordre.

Objectif
--------
Séparer la création de la structure du remplissage des données afin de :
- faciliter le débogage
- isoler les erreurs de colonnes manquantes
- rendre le pipeline plus lisible

Notes
-----
- La table source est `features_client_test_enriched`.
- La table cible `features_client_test_model` doit déjà exister.
- Ce script vérifie que toutes les colonnes attendues sont présentes.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL n'est pas défini dans les variables d'environnement")


# =============================================================================
# Liste exacte des features du modèle
# =============================================================================

MODEL_FEATURES = """
SK_ID_CURR,
NAME_CONTRACT_TYPE,
CODE_GENDER,
FLAG_OWN_CAR,
FLAG_OWN_REALTY,
CNT_CHILDREN,
AMT_INCOME_TOTAL,
AMT_CREDIT,
AMT_ANNUITY,
AMT_GOODS_PRICE,
NAME_TYPE_SUITE,
NAME_INCOME_TYPE,
NAME_EDUCATION_TYPE,
NAME_FAMILY_STATUS,
NAME_HOUSING_TYPE,
REGION_POPULATION_RELATIVE,
DAYS_BIRTH,
DAYS_EMPLOYED,
DAYS_REGISTRATION,
DAYS_ID_PUBLISH,
OWN_CAR_AGE,
FLAG_EMP_PHONE,
FLAG_WORK_PHONE,
FLAG_PHONE,
FLAG_EMAIL,
OCCUPATION_TYPE,
CNT_FAM_MEMBERS,
REGION_RATING_CLIENT,
REGION_RATING_CLIENT_W_CITY,
WEEKDAY_APPR_PROCESS_START,
HOUR_APPR_PROCESS_START,
REG_REGION_NOT_LIVE_REGION,
REG_REGION_NOT_WORK_REGION,
LIVE_REGION_NOT_WORK_REGION,
REG_CITY_NOT_LIVE_CITY,
REG_CITY_NOT_WORK_CITY,
LIVE_CITY_NOT_WORK_CITY,
EXT_SOURCE_1,
EXT_SOURCE_2,
EXT_SOURCE_3,
APARTMENTS_AVG,
BASEMENTAREA_AVG,
YEARS_BEGINEXPLUATATION_AVG,
ELEVATORS_AVG,
ENTRANCES_AVG,
FLOORSMAX_AVG,
LANDAREA_AVG,
LIVINGAREA_AVG,
NONLIVINGAREA_AVG,
OBS_30_CNT_SOCIAL_CIRCLE,
DEF_30_CNT_SOCIAL_CIRCLE,
OBS_60_CNT_SOCIAL_CIRCLE,
DEF_60_CNT_SOCIAL_CIRCLE,
DAYS_LAST_PHONE_CHANGE,
AMT_REQ_CREDIT_BUREAU_HOUR,
AMT_REQ_CREDIT_BUREAU_WEEK,
AMT_REQ_CREDIT_BUREAU_MON,
AMT_REQ_CREDIT_BUREAU_QRT,
AMT_REQ_CREDIT_BUREAU_YEAR,
bureau__DAYS_CREDIT__mean,
bureau__DAYS_CREDIT__std,
bureau__CREDIT_DAY_OVERDUE__mean,
bureau__CREDIT_DAY_OVERDUE__std,
bureau__DAYS_CREDIT_ENDDATE__mean,
bureau__DAYS_CREDIT_ENDDATE__std,
bureau__DAYS_ENDDATE_FACT__mean,
bureau__DAYS_ENDDATE_FACT__std,
bureau__AMT_CREDIT_MAX_OVERDUE__mean,
bureau__AMT_CREDIT_MAX_OVERDUE__std,
bureau__CNT_CREDIT_PROLONG__mean,
bureau__CNT_CREDIT_PROLONG__std,
bureau__AMT_CREDIT_SUM__mean,
bureau__AMT_CREDIT_SUM__std,
bureau__AMT_CREDIT_SUM_DEBT__mean,
bureau__AMT_CREDIT_SUM_DEBT__std,
bureau__AMT_CREDIT_SUM_LIMIT__mean,
bureau__AMT_CREDIT_SUM_LIMIT__std,
bureau__AMT_CREDIT_SUM_OVERDUE__mean,
bureau__AMT_CREDIT_SUM_OVERDUE__std,
bureau__DAYS_CREDIT_UPDATE__mean,
bureau__DAYS_CREDIT_UPDATE__std,
bureau__AMT_ANNUITY__mean,
bureau__AMT_ANNUITY__std,
bureau__DEBT_RATIO__mean,
bureau__DEBT_RATIO__std,
bureau__OVERDUE_RATIO__mean,
bureau__OVERDUE_RATIO__std,
bureau__IS_ACTIVE__mean,
bureau__IS_ACTIVE__std,
bureau__HAS_OVERDUE__mean,
bureau__HAS_OVERDUE__std,
bureau__CREDIT_AGE__mean,
bureau__CREDIT_AGE__std,
bureau__bb__MONTHS_BALANCE__mean__mean,
bureau__bb__MONTHS_BALANCE__mean__std,
bureau__bb__MONTHS_BALANCE__std__mean,
bureau__bb__MONTHS_BALANCE__std__std,
bureau__bb__count_rows__mean,
bureau__bb__count_rows__std,
bureau__bb__recent_max_dpd__mean,
bureau__bb__recent_max_dpd__std,
bureau__bb__months_late_ratio__mean,
bureau__bb__months_late_ratio__std,
bureau__bb__late_severity_sum__mean,
bureau__bb__late_severity_sum__std,
bureau__count_rows,
bureau__nunique_SK_ID_BUREAU,
prev__AMT_ANNUITY__mean,
prev__AMT_ANNUITY__std,
prev__AMT_APPLICATION__mean,
prev__AMT_APPLICATION__std,
prev__AMT_CREDIT__mean,
prev__AMT_CREDIT__std,
prev__AMT_DOWN_PAYMENT__mean,
prev__AMT_DOWN_PAYMENT__std,
prev__AMT_GOODS_PRICE__mean,
prev__AMT_GOODS_PRICE__std,
prev__HOUR_APPR_PROCESS_START__mean,
prev__HOUR_APPR_PROCESS_START__std,
prev__NFLAG_LAST_APPL_IN_DAY__mean,
prev__NFLAG_LAST_APPL_IN_DAY__std,
prev__RATE_DOWN_PAYMENT__mean,
prev__RATE_DOWN_PAYMENT__std,
prev__DAYS_DECISION__mean,
prev__DAYS_DECISION__std,
prev__SELLERPLACE_AREA__mean,
prev__SELLERPLACE_AREA__std,
prev__CNT_PAYMENT__mean,
prev__CNT_PAYMENT__std,
prev__DAYS_FIRST_DRAWING__mean,
prev__DAYS_FIRST_DUE__mean,
prev__DAYS_FIRST_DUE__std,
prev__DAYS_LAST_DUE_1ST_VERSION__mean,
prev__DAYS_LAST_DUE_1ST_VERSION__std,
prev__DAYS_LAST_DUE__mean,
prev__DAYS_LAST_DUE__std,
prev__DAYS_TERMINATION__mean,
prev__DAYS_TERMINATION__std,
prev__NFLAG_INSURED_ON_APPROVAL__mean,
prev__NFLAG_INSURED_ON_APPROVAL__std,
prev__PREV_CREDIT_APPLICATION_RATIO__mean,
prev__PREV_CREDIT_APPLICATION_RATIO__std,
prev__PREV_IS_APPROVED__mean,
prev__PREV_IS_APPROVED__std,
prev__PREV_IS_REFUSED__mean,
prev__PREV_IS_REFUSED__std,
prev__PREV_DAYS_DECISION_AGE__mean,
prev__PREV_DAYS_DECISION_AGE__std,
prev__PREV_CREDIT_DURATION__mean,
prev__PREV_CREDIT_DURATION__std,
prev__pos__POS_REMAIN_RATIO__mean,
prev__pos__POS_DPD_POS__max,
prev__cc__CC_UTILIZATION_RATIO__mean,
prev__cc__CC_UTILIZATION_RATIO__max,
prev__cc__CC_PAYMENT_MIN_RATIO__mean,
prev__inst__DPD_POS__mean,
prev__inst__DPD_POS__max,
prev__inst__SEVERE_LATE_30__mean,
prev__inst__PAY_RATIO__mean,
prev__count_rows,
prev__nunique_SK_ID_PREV,
AGE_YEARS,
EMPLOYED_YEARS,
REGISTRATION_YEARS,
ID_PUBLISH_YEARS,
LAST_PHONE_CHANGE_YEARS,
DAYS_EMPLOYED__isna,
OWN_CAR_AGE__isna,
EXT_SOURCE_1__isna,
EXT_SOURCE_3__isna,
DAYS_LAST_PHONE_CHANGE__isna,
AMT_REQ_CREDIT_BUREAU_HOUR__isna,
AMT_REQ_CREDIT_BUREAU_WEEK__isna,
AMT_REQ_CREDIT_BUREAU_MON__isna,
AMT_REQ_CREDIT_BUREAU_QRT__isna,
AMT_REQ_CREDIT_BUREAU_YEAR__isna,
CREDIT_INCOME_RATIO,
ANNUITY_INCOME_RATIO,
ANNUITY_CREDIT_RATIO,
CREDIT_GOODS_RATIO,
OVER_INDEBTED_40,
LOG_INCOME,
LOG_CREDIT,
LOG_ANNUITY,
LOG_GOODS,
SOCIAL_DEFAULT_RATIO_30,
SOCIAL_DEFAULT_RATIO_60,
DOC_COUNT,
CONTACT_COUNT,
ADDRESS_MISMATCH_COUNT,
EXT_SOURCES_MEAN,
EXT_SOURCES_MIN,
EXT_SOURCES_MAX,
EXT_SOURCES_STD,
EXT_SOURCES_RANGE,
EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_2,
EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_3,
EXT_INT__EXT_SOURCE_2_x_EXT_SOURCE_3,
EXT_POW2__EXT_SOURCE_1,
EXT_POW2__EXT_SOURCE_2,
EXT_POW2__EXT_SOURCE_3
"""


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def parse_features(features_str: str) -> list[str]:
    """
    Convertit la chaîne de features en liste Python propre.

    Parameters
    ----------
    features_str : str
        Chaîne contenant les noms de colonnes séparés par des virgules.

    Returns
    -------
    list[str]
        Liste nettoyée des colonnes.
    """
    return [feature.strip() for feature in features_str.split(",") if feature.strip()]


def table_exists(engine, table_name: str) -> bool:
    """
    Vérifie si une table existe dans le schéma public.
    """
    sql = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = :table_name
        )
        """
    )

    with engine.begin() as connection:
        return bool(connection.execute(sql, {"table_name": table_name}).scalar())


def get_missing_columns(engine, table_name: str, expected_columns: list[str]) -> list[str]:
    """
    Retourne les colonnes attendues mais absentes dans une table.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    table_name : str
        Nom de la table source.
    expected_columns : list[str]
        Colonnes attendues.

    Returns
    -------
    list[str]
        Colonnes manquantes.
    """
    sql = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )

    with engine.begin() as connection:
        rows = connection.execute(sql, {"table_name": table_name}).fetchall()

    existing_columns = {row[0] for row in rows}
    return [col for col in expected_columns if col not in existing_columns]


# =============================================================================
# SQL de remplissage
# =============================================================================

TRUNCATE_MODEL_READY_TABLE_SQL = """
TRUNCATE TABLE features_client_test_model;
"""

COUNT_ROWS_SQL = """
SELECT COUNT(*) FROM features_client_test_model;
"""


# =============================================================================
# Fonction principale de remplissage
# =============================================================================

def populate_model_ready_table(engine) -> None:
    """
    Remplit `features_client_test_model` à partir de
    `features_client_test_enriched`.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.

    Raises
    ------
    ValueError
        Si la table source ou la table cible n'existent pas,
        ou si des colonnes attendues sont absentes.
    """
    source_table = "features_client_test_enriched"
    target_table = "features_client_test_model"

    if not table_exists(engine, source_table):
        raise ValueError(
            f"La table source '{source_table}' n'existe pas."
        )

    if not table_exists(engine, target_table):
        raise ValueError(
            f"La table cible '{target_table}' n'existe pas. "
            "Exécute d'abord le script de création."
        )

    features = parse_features(MODEL_FEATURES)

    missing_columns = get_missing_columns(engine, source_table, features)
    if missing_columns:
        raise ValueError(
            "Certaines colonnes attendues par le modèle sont absentes de la table source : "
            f"{missing_columns[:20]}"
            + (f" ... et {len(missing_columns) - 20} autres." if len(missing_columns) > 20 else "")
        )

    select_sql = ",\n    ".join(f'"{col}"' for col in features)

    insert_sql = f"""
    INSERT INTO {target_table}
    SELECT
        {select_sql}
    FROM {source_table};
    """

    with engine.begin() as connection:
        connection.execute(text(TRUNCATE_MODEL_READY_TABLE_SQL))
        connection.execute(text(insert_sql))
        row_count = connection.execute(text(COUNT_ROWS_SQL)).scalar()

    print(f"Table '{target_table}' remplie avec succès.")
    print(f"Lignes insérées : {row_count}")
    print(f"Colonnes sélectionnées : {len(features)}")


# =============================================================================
# Point d'entrée
# =============================================================================

def main() -> None:
    """
    Point d'entrée du script.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    populate_model_ready_table(engine)

    print("Remplissage de la table model ready terminé.")


if __name__ == "__main__":
    main()