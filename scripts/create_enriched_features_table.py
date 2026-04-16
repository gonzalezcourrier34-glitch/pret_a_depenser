"""
Script de création de la table enrichie de features client.

Ce module crée uniquement la structure de la table
`features_client_test_enriched` à partir de la table source
`features_client_test`, sans y insérer de données.

Objectif
--------
Séparer la création de la structure de la logique d'enrichissement afin de :
- clarifier les responsabilités
- faciliter le débogage
- rendre le pipeline plus lisible

Notes
-----
- La source logique est `features_client_test`.
- La table cible est `features_client_test_enriched`.
- La table créée est vide.
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
# SQL de création de la table enrichie vide
# =============================================================================

CREATE_FEATURES_CLIENT_TEST_ENRICHED_EMPTY_SQL = """
DROP TABLE IF EXISTS features_client_test_enriched;

CREATE TABLE features_client_test_enriched AS
SELECT
    f.*,

    NULL::DOUBLE PRECISION AS "AGE_YEARS",
    NULL::DOUBLE PRECISION AS "EMPLOYED_YEARS",
    NULL::DOUBLE PRECISION AS "REGISTRATION_YEARS",
    NULL::DOUBLE PRECISION AS "ID_PUBLISH_YEARS",
    NULL::DOUBLE PRECISION AS "LAST_PHONE_CHANGE_YEARS",

    NULL::INTEGER AS "DAYS_EMPLOYED__isna",
    NULL::INTEGER AS "OWN_CAR_AGE__isna",
    NULL::INTEGER AS "EXT_SOURCE_1__isna",
    NULL::INTEGER AS "EXT_SOURCE_3__isna",
    NULL::INTEGER AS "DAYS_LAST_PHONE_CHANGE__isna",
    NULL::INTEGER AS "AMT_REQ_CREDIT_BUREAU_HOUR__isna",
    NULL::INTEGER AS "AMT_REQ_CREDIT_BUREAU_WEEK__isna",
    NULL::INTEGER AS "AMT_REQ_CREDIT_BUREAU_MON__isna",
    NULL::INTEGER AS "AMT_REQ_CREDIT_BUREAU_QRT__isna",
    NULL::INTEGER AS "AMT_REQ_CREDIT_BUREAU_YEAR__isna",

    NULL::DOUBLE PRECISION AS "CREDIT_INCOME_RATIO",
    NULL::DOUBLE PRECISION AS "ANNUITY_INCOME_RATIO",
    NULL::DOUBLE PRECISION AS "ANNUITY_CREDIT_RATIO",
    NULL::DOUBLE PRECISION AS "CREDIT_GOODS_RATIO",
    NULL::INTEGER AS "OVER_INDEBTED_40",

    NULL::DOUBLE PRECISION AS "LOG_INCOME",
    NULL::DOUBLE PRECISION AS "LOG_CREDIT",
    NULL::DOUBLE PRECISION AS "LOG_ANNUITY",
    NULL::DOUBLE PRECISION AS "LOG_GOODS",

    NULL::DOUBLE PRECISION AS "SOCIAL_DEFAULT_RATIO_30",
    NULL::DOUBLE PRECISION AS "SOCIAL_DEFAULT_RATIO_60",

    NULL::DOUBLE PRECISION AS "DOC_COUNT",
    NULL::DOUBLE PRECISION AS "CONTACT_COUNT",
    NULL::DOUBLE PRECISION AS "ADDRESS_MISMATCH_COUNT",

    NULL::DOUBLE PRECISION AS "EXT_SOURCES_MEAN",
    NULL::DOUBLE PRECISION AS "EXT_SOURCES_MIN",
    NULL::DOUBLE PRECISION AS "EXT_SOURCES_MAX",
    NULL::DOUBLE PRECISION AS "EXT_SOURCES_STD",
    NULL::DOUBLE PRECISION AS "EXT_SOURCES_RANGE",

    NULL::DOUBLE PRECISION AS "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_2",
    NULL::DOUBLE PRECISION AS "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_3",
    NULL::DOUBLE PRECISION AS "EXT_INT__EXT_SOURCE_2_x_EXT_SOURCE_3",

    NULL::DOUBLE PRECISION AS "EXT_POW2__EXT_SOURCE_1",
    NULL::DOUBLE PRECISION AS "EXT_POW2__EXT_SOURCE_2",
    NULL::DOUBLE PRECISION AS "EXT_POW2__EXT_SOURCE_3"

FROM features_client_test f
WHERE 1 = 0;
"""

CREATE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_features_client_test_enriched_sk_id_curr
ON features_client_test_enriched ("SK_ID_CURR");
"""


# =============================================================================
# Fonction principale de création
# =============================================================================

def create_enriched_features_table(engine) -> None:
    """
    Crée la structure vide de la table `features_client_test_enriched`.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_FEATURES_CLIENT_TEST_ENRICHED_EMPTY_SQL))
        connection.execute(text(CREATE_INDEX_SQL))

    print("Table 'features_client_test_enriched' créée vide avec succès.")


def main() -> None:
    """
    Point d'entrée du script.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    create_enriched_features_table(engine)

    print("Création de la table enrichie terminée.")


if __name__ == "__main__":
    main()