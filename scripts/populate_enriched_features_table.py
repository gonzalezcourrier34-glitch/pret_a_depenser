"""
Script de remplissage de la table enrichie de features client.

Ce module alimente `features_client_test_enriched` à partir de
`features_client_test` en ajoutant les features dérivées nécessaires
à l'alignement avec le modèle entraîné.

Objectif
--------
Séparer la création de la table cible de la logique d'enrichissement afin de :
- faciliter le débogage
- isoler les erreurs SQL
- rendre le pipeline plus lisible

Notes
-----
- La table source est `features_client_test`.
- La table cible `features_client_test_enriched` doit déjà exister.
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
# SQL de remplissage de la table enrichie
# =============================================================================

TRUNCATE_ENRICHED_TABLE_SQL = """
TRUNCATE TABLE features_client_test_enriched;
"""

INSERT_FEATURES_CLIENT_TEST_ENRICHED_SQL = """
INSERT INTO features_client_test_enriched
SELECT
    f.*,

    -- =========================================================
    -- Conversions temporelles
    -- =========================================================
    (-1.0 * f."DAYS_BIRTH" / 365.25) AS "AGE_YEARS",

    CASE
        WHEN f."DAYS_EMPLOYED" = 365243 THEN NULL
        ELSE (-1.0 * f."DAYS_EMPLOYED" / 365.25)
    END AS "EMPLOYED_YEARS",

    (-1.0 * f."DAYS_REGISTRATION" / 365.25) AS "REGISTRATION_YEARS",
    (-1.0 * f."DAYS_ID_PUBLISH" / 365.25) AS "ID_PUBLISH_YEARS",

    CASE
        WHEN f."DAYS_LAST_PHONE_CHANGE" = 0 THEN NULL
        ELSE (-1.0 * f."DAYS_LAST_PHONE_CHANGE" / 365.25)
    END AS "LAST_PHONE_CHANGE_YEARS",

    -- =========================================================
    -- Flags de valeurs manquantes
    -- =========================================================
    CASE WHEN f."DAYS_EMPLOYED" IS NULL OR f."DAYS_EMPLOYED" = 365243 THEN 1 ELSE 0 END AS "DAYS_EMPLOYED__isna",
    CASE WHEN f."OWN_CAR_AGE" IS NULL THEN 1 ELSE 0 END AS "OWN_CAR_AGE__isna",
    CASE WHEN f."EXT_SOURCE_1" IS NULL THEN 1 ELSE 0 END AS "EXT_SOURCE_1__isna",
    CASE WHEN f."EXT_SOURCE_3" IS NULL THEN 1 ELSE 0 END AS "EXT_SOURCE_3__isna",
    CASE WHEN f."DAYS_LAST_PHONE_CHANGE" IS NULL OR f."DAYS_LAST_PHONE_CHANGE" = 0 THEN 1 ELSE 0 END AS "DAYS_LAST_PHONE_CHANGE__isna",
    CASE WHEN f."AMT_REQ_CREDIT_BUREAU_HOUR" IS NULL THEN 1 ELSE 0 END AS "AMT_REQ_CREDIT_BUREAU_HOUR__isna",
    CASE WHEN f."AMT_REQ_CREDIT_BUREAU_WEEK" IS NULL THEN 1 ELSE 0 END AS "AMT_REQ_CREDIT_BUREAU_WEEK__isna",
    CASE WHEN f."AMT_REQ_CREDIT_BUREAU_MON" IS NULL THEN 1 ELSE 0 END AS "AMT_REQ_CREDIT_BUREAU_MON__isna",
    CASE WHEN f."AMT_REQ_CREDIT_BUREAU_QRT" IS NULL THEN 1 ELSE 0 END AS "AMT_REQ_CREDIT_BUREAU_QRT__isna",
    CASE WHEN f."AMT_REQ_CREDIT_BUREAU_YEAR" IS NULL THEN 1 ELSE 0 END AS "AMT_REQ_CREDIT_BUREAU_YEAR__isna",

    -- =========================================================
    -- Ratios financiers
    -- =========================================================
    f."AMT_CREDIT" / NULLIF(f."AMT_INCOME_TOTAL", 0) AS "CREDIT_INCOME_RATIO",
    f."AMT_ANNUITY" / NULLIF(f."AMT_INCOME_TOTAL", 0) AS "ANNUITY_INCOME_RATIO",
    f."AMT_ANNUITY" / NULLIF(f."AMT_CREDIT", 0) AS "ANNUITY_CREDIT_RATIO",
    f."AMT_CREDIT" / NULLIF(f."AMT_GOODS_PRICE", 0) AS "CREDIT_GOODS_RATIO",

    CASE
        WHEN f."AMT_ANNUITY" / NULLIF(f."AMT_INCOME_TOTAL", 0) > 0.40 THEN 1
        ELSE 0
    END AS "OVER_INDEBTED_40",

    -- =========================================================
    -- Logs
    -- =========================================================
    LN(GREATEST(f."AMT_INCOME_TOTAL", 0) + 1) AS "LOG_INCOME",
    LN(GREATEST(f."AMT_CREDIT", 0) + 1) AS "LOG_CREDIT",
    LN(GREATEST(f."AMT_ANNUITY", 0) + 1) AS "LOG_ANNUITY",
    LN(GREATEST(f."AMT_GOODS_PRICE", 0) + 1) AS "LOG_GOODS",

    -- =========================================================
    -- Ratios sociaux
    -- =========================================================
    f."DEF_30_CNT_SOCIAL_CIRCLE" / NULLIF(f."OBS_30_CNT_SOCIAL_CIRCLE" + 1, 0) AS "SOCIAL_DEFAULT_RATIO_30",
    f."DEF_60_CNT_SOCIAL_CIRCLE" / NULLIF(f."OBS_60_CNT_SOCIAL_CIRCLE" + 1, 0) AS "SOCIAL_DEFAULT_RATIO_60",

    -- =========================================================
    -- Compteurs
    -- =========================================================
    (
        COALESCE(f."FLAG_DOCUMENT_2", 0)
        + COALESCE(f."FLAG_DOCUMENT_3", 0)
        + COALESCE(f."FLAG_DOCUMENT_4", 0)
        + COALESCE(f."FLAG_DOCUMENT_5", 0)
        + COALESCE(f."FLAG_DOCUMENT_6", 0)
        + COALESCE(f."FLAG_DOCUMENT_7", 0)
        + COALESCE(f."FLAG_DOCUMENT_8", 0)
        + COALESCE(f."FLAG_DOCUMENT_9", 0)
        + COALESCE(f."FLAG_DOCUMENT_10", 0)
        + COALESCE(f."FLAG_DOCUMENT_11", 0)
        + COALESCE(f."FLAG_DOCUMENT_12", 0)
        + COALESCE(f."FLAG_DOCUMENT_13", 0)
        + COALESCE(f."FLAG_DOCUMENT_14", 0)
        + COALESCE(f."FLAG_DOCUMENT_15", 0)
        + COALESCE(f."FLAG_DOCUMENT_16", 0)
        + COALESCE(f."FLAG_DOCUMENT_17", 0)
        + COALESCE(f."FLAG_DOCUMENT_18", 0)
        + COALESCE(f."FLAG_DOCUMENT_19", 0)
        + COALESCE(f."FLAG_DOCUMENT_20", 0)
        + COALESCE(f."FLAG_DOCUMENT_21", 0)
    ) AS "DOC_COUNT",

    (
        COALESCE(f."FLAG_MOBIL", 0)
        + COALESCE(f."FLAG_EMP_PHONE", 0)
        + COALESCE(f."FLAG_WORK_PHONE", 0)
        + COALESCE(f."FLAG_PHONE", 0)
        + COALESCE(f."FLAG_EMAIL", 0)
    ) AS "CONTACT_COUNT",

    (
        COALESCE(f."REG_REGION_NOT_LIVE_REGION", 0)
        + COALESCE(f."REG_REGION_NOT_WORK_REGION", 0)
        + COALESCE(f."LIVE_REGION_NOT_WORK_REGION", 0)
        + COALESCE(f."REG_CITY_NOT_LIVE_CITY", 0)
        + COALESCE(f."REG_CITY_NOT_WORK_CITY", 0)
        + COALESCE(f."LIVE_CITY_NOT_WORK_CITY", 0)
    ) AS "ADDRESS_MISMATCH_COUNT",

    -- =========================================================
    -- EXT_SOURCE stats
    -- =========================================================
    (
        COALESCE(f."EXT_SOURCE_1", 0)
        + COALESCE(f."EXT_SOURCE_2", 0)
        + COALESCE(f."EXT_SOURCE_3", 0)
    ) /
    NULLIF(
        (CASE WHEN f."EXT_SOURCE_1" IS NOT NULL THEN 1 ELSE 0 END)
        + (CASE WHEN f."EXT_SOURCE_2" IS NOT NULL THEN 1 ELSE 0 END)
        + (CASE WHEN f."EXT_SOURCE_3" IS NOT NULL THEN 1 ELSE 0 END),
        0
    ) AS "EXT_SOURCES_MEAN",

    LEAST(
        COALESCE(f."EXT_SOURCE_1", 9999.0),
        COALESCE(f."EXT_SOURCE_2", 9999.0),
        COALESCE(f."EXT_SOURCE_3", 9999.0)
    ) AS "EXT_SOURCES_MIN",

    GREATEST(
        COALESCE(f."EXT_SOURCE_1", -9999.0),
        COALESCE(f."EXT_SOURCE_2", -9999.0),
        COALESCE(f."EXT_SOURCE_3", -9999.0)
    ) AS "EXT_SOURCES_MAX",

    SQRT(
        (
            POWER(COALESCE(f."EXT_SOURCE_1", 0), 2)
            + POWER(COALESCE(f."EXT_SOURCE_2", 0), 2)
            + POWER(COALESCE(f."EXT_SOURCE_3", 0), 2)
        ) / 3.0
    ) AS "EXT_SOURCES_STD",

    (
        GREATEST(
            COALESCE(f."EXT_SOURCE_1", -9999.0),
            COALESCE(f."EXT_SOURCE_2", -9999.0),
            COALESCE(f."EXT_SOURCE_3", -9999.0)
        )
        -
        LEAST(
            COALESCE(f."EXT_SOURCE_1", 9999.0),
            COALESCE(f."EXT_SOURCE_2", 9999.0),
            COALESCE(f."EXT_SOURCE_3", 9999.0)
        )
    ) AS "EXT_SOURCES_RANGE",

    -- =========================================================
    -- Interactions EXT_SOURCE
    -- =========================================================
    (f."EXT_SOURCE_1" * f."EXT_SOURCE_2") AS "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_2",
    (f."EXT_SOURCE_1" * f."EXT_SOURCE_3") AS "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_3",
    (f."EXT_SOURCE_2" * f."EXT_SOURCE_3") AS "EXT_INT__EXT_SOURCE_2_x_EXT_SOURCE_3",

    POWER(f."EXT_SOURCE_1", 2) AS "EXT_POW2__EXT_SOURCE_1",
    POWER(f."EXT_SOURCE_2", 2) AS "EXT_POW2__EXT_SOURCE_2",
    POWER(f."EXT_SOURCE_3", 2) AS "EXT_POW2__EXT_SOURCE_3"

FROM features_client_test f;
"""

COUNT_ROWS_SQL = """
SELECT COUNT(*) FROM features_client_test_enriched;
"""


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def table_exists(engine, table_name: str) -> bool:
    """
    Vérifie si une table existe dans le schéma public.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    table_name : str
        Nom de la table.

    Returns
    -------
    bool
        True si la table existe, sinon False.
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


# =============================================================================
# Fonction principale de remplissage
# =============================================================================

def populate_enriched_features_table(engine) -> None:
    """
    Remplit la table `features_client_test_enriched` à partir de
    `features_client_test`.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    if not table_exists(engine, "features_client_test"):
        raise ValueError(
            "La table source 'features_client_test' n'existe pas."
        )

    if not table_exists(engine, "features_client_test_enriched"):
        raise ValueError(
            "La table cible 'features_client_test_enriched' n'existe pas. "
            "Exécute d'abord le script de création."
        )

    with engine.begin() as connection:
        connection.execute(text(TRUNCATE_ENRICHED_TABLE_SQL))
        connection.execute(text(INSERT_FEATURES_CLIENT_TEST_ENRICHED_SQL))
        row_count = connection.execute(text(COUNT_ROWS_SQL)).scalar()

    print("Table 'features_client_test_enriched' remplie avec succès.")
    print(f"Lignes insérées : {row_count}")


def main() -> None:
    """
    Point d'entrée du script.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    populate_enriched_features_table(engine)

    print("Remplissage de la table enrichie terminé.")


if __name__ == "__main__":
    main()