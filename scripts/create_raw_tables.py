"""
Script de création des tables brutes PostgreSQL.

Ce module permet de créer les tables RAW du projet à partir du schéma
des fichiers CSV sources, sans transformation métier.

Objectif
--------
Créer une couche de données brutes fidèle aux fichiers d'origine afin de :
- garantir l'intégrité des données sources
- préparer les futures tables de features
- séparer l'ingestion brute de la transformation métier

Fonctionnalités
---------------
- connexion à PostgreSQL via SQLAlchemy
- création des tables RAW
- exécution idempotente avec `CREATE TABLE IF NOT EXISTS`

Notes
-----
- Les noms de colonnes sont conservés exactement comme dans les CSV.
- Les identifiants et noms contenant des majuscules sont entourés de guillemets.
- Les tables doivent être créées avant le chargement des CSV.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

# Charge les variables définies dans le fichier .env
load_dotenv()


# =============================================================================
# Configuration de la base de données
# =============================================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL n'est pas défini dans les variables d'environnement")


# =============================================================================
# SQL de création de la table application_test
# =============================================================================

CREATE_APPLICATION_TEST_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS application_test (

    "SK_ID_CURR" BIGINT PRIMARY KEY,

    "NAME_CONTRACT_TYPE" TEXT,
    "CODE_GENDER" TEXT,
    "FLAG_OWN_CAR" TEXT,
    "FLAG_OWN_REALTY" TEXT,

    "CNT_CHILDREN" INTEGER,

    "AMT_INCOME_TOTAL" DOUBLE PRECISION,
    "AMT_CREDIT" DOUBLE PRECISION,
    "AMT_ANNUITY" DOUBLE PRECISION,
    "AMT_GOODS_PRICE" DOUBLE PRECISION,

    "NAME_TYPE_SUITE" TEXT,
    "NAME_INCOME_TYPE" TEXT,
    "NAME_EDUCATION_TYPE" TEXT,
    "NAME_FAMILY_STATUS" TEXT,
    "NAME_HOUSING_TYPE" TEXT,

    "REGION_POPULATION_RELATIVE" DOUBLE PRECISION,

    "DAYS_BIRTH" INTEGER,
    "DAYS_EMPLOYED" INTEGER,
    "DAYS_REGISTRATION" DOUBLE PRECISION,
    "DAYS_ID_PUBLISH" INTEGER,

    "OWN_CAR_AGE" DOUBLE PRECISION,

    "FLAG_MOBIL" INTEGER,
    "FLAG_EMP_PHONE" INTEGER,
    "FLAG_WORK_PHONE" INTEGER,
    "FLAG_CONT_MOBILE" INTEGER,
    "FLAG_PHONE" INTEGER,
    "FLAG_EMAIL" INTEGER,

    "OCCUPATION_TYPE" TEXT,

    "CNT_FAM_MEMBERS" DOUBLE PRECISION,

    "REGION_RATING_CLIENT" INTEGER,
    "REGION_RATING_CLIENT_W_CITY" INTEGER,

    "WEEKDAY_APPR_PROCESS_START" TEXT,
    "HOUR_APPR_PROCESS_START" INTEGER,

    "REG_REGION_NOT_LIVE_REGION" INTEGER,
    "REG_REGION_NOT_WORK_REGION" INTEGER,
    "LIVE_REGION_NOT_WORK_REGION" INTEGER,

    "REG_CITY_NOT_LIVE_CITY" INTEGER,
    "REG_CITY_NOT_WORK_CITY" INTEGER,
    "LIVE_CITY_NOT_WORK_CITY" INTEGER,

    "ORGANIZATION_TYPE" TEXT,

    "EXT_SOURCE_1" DOUBLE PRECISION,
    "EXT_SOURCE_2" DOUBLE PRECISION,
    "EXT_SOURCE_3" DOUBLE PRECISION,

    "APARTMENTS_AVG" DOUBLE PRECISION,
    "BASEMENTAREA_AVG" DOUBLE PRECISION,
    "YEARS_BEGINEXPLUATATION_AVG" DOUBLE PRECISION,
    "YEARS_BUILD_AVG" DOUBLE PRECISION,
    "COMMONAREA_AVG" DOUBLE PRECISION,
    "ELEVATORS_AVG" DOUBLE PRECISION,
    "ENTRANCES_AVG" DOUBLE PRECISION,
    "FLOORSMAX_AVG" DOUBLE PRECISION,
    "FLOORSMIN_AVG" DOUBLE PRECISION,
    "LANDAREA_AVG" DOUBLE PRECISION,
    "LIVINGAPARTMENTS_AVG" DOUBLE PRECISION,
    "LIVINGAREA_AVG" DOUBLE PRECISION,
    "NONLIVINGAPARTMENTS_AVG" DOUBLE PRECISION,
    "NONLIVINGAREA_AVG" DOUBLE PRECISION,

    "APARTMENTS_MODE" DOUBLE PRECISION,
    "BASEMENTAREA_MODE" DOUBLE PRECISION,
    "YEARS_BEGINEXPLUATATION_MODE" DOUBLE PRECISION,
    "YEARS_BUILD_MODE" DOUBLE PRECISION,
    "COMMONAREA_MODE" DOUBLE PRECISION,
    "ELEVATORS_MODE" DOUBLE PRECISION,
    "ENTRANCES_MODE" DOUBLE PRECISION,
    "FLOORSMAX_MODE" DOUBLE PRECISION,
    "FLOORSMIN_MODE" DOUBLE PRECISION,
    "LANDAREA_MODE" DOUBLE PRECISION,
    "LIVINGAPARTMENTS_MODE" DOUBLE PRECISION,
    "LIVINGAREA_MODE" DOUBLE PRECISION,
    "NONLIVINGAPARTMENTS_MODE" DOUBLE PRECISION,
    "NONLIVINGAREA_MODE" DOUBLE PRECISION,

    "APARTMENTS_MEDI" DOUBLE PRECISION,
    "BASEMENTAREA_MEDI" DOUBLE PRECISION,
    "YEARS_BEGINEXPLUATATION_MEDI" DOUBLE PRECISION,
    "YEARS_BUILD_MEDI" DOUBLE PRECISION,
    "COMMONAREA_MEDI" DOUBLE PRECISION,
    "ELEVATORS_MEDI" DOUBLE PRECISION,
    "ENTRANCES_MEDI" DOUBLE PRECISION,
    "FLOORSMAX_MEDI" DOUBLE PRECISION,
    "FLOORSMIN_MEDI" DOUBLE PRECISION,
    "LANDAREA_MEDI" DOUBLE PRECISION,
    "LIVINGAPARTMENTS_MEDI" DOUBLE PRECISION,
    "LIVINGAREA_MEDI" DOUBLE PRECISION,
    "NONLIVINGAPARTMENTS_MEDI" DOUBLE PRECISION,
    "NONLIVINGAREA_MEDI" DOUBLE PRECISION,

    "FONDKAPREMONT_MODE" TEXT,
    "HOUSETYPE_MODE" TEXT,
    "TOTALAREA_MODE" DOUBLE PRECISION,
    "WALLSMATERIAL_MODE" TEXT,
    "EMERGENCYSTATE_MODE" TEXT,

    "OBS_30_CNT_SOCIAL_CIRCLE" DOUBLE PRECISION,
    "DEF_30_CNT_SOCIAL_CIRCLE" DOUBLE PRECISION,
    "OBS_60_CNT_SOCIAL_CIRCLE" DOUBLE PRECISION,
    "DEF_60_CNT_SOCIAL_CIRCLE" DOUBLE PRECISION,

    "DAYS_LAST_PHONE_CHANGE" DOUBLE PRECISION,

    "FLAG_DOCUMENT_2" INTEGER,
    "FLAG_DOCUMENT_3" INTEGER,
    "FLAG_DOCUMENT_4" INTEGER,
    "FLAG_DOCUMENT_5" INTEGER,
    "FLAG_DOCUMENT_6" INTEGER,
    "FLAG_DOCUMENT_7" INTEGER,
    "FLAG_DOCUMENT_8" INTEGER,
    "FLAG_DOCUMENT_9" INTEGER,
    "FLAG_DOCUMENT_10" INTEGER,
    "FLAG_DOCUMENT_11" INTEGER,
    "FLAG_DOCUMENT_12" INTEGER,
    "FLAG_DOCUMENT_13" INTEGER,
    "FLAG_DOCUMENT_14" INTEGER,
    "FLAG_DOCUMENT_15" INTEGER,
    "FLAG_DOCUMENT_16" INTEGER,
    "FLAG_DOCUMENT_17" INTEGER,
    "FLAG_DOCUMENT_18" INTEGER,
    "FLAG_DOCUMENT_19" INTEGER,
    "FLAG_DOCUMENT_20" INTEGER,
    "FLAG_DOCUMENT_21" INTEGER,

    "AMT_REQ_CREDIT_BUREAU_HOUR" DOUBLE PRECISION,
    "AMT_REQ_CREDIT_BUREAU_DAY" DOUBLE PRECISION,
    "AMT_REQ_CREDIT_BUREAU_WEEK" DOUBLE PRECISION,
    "AMT_REQ_CREDIT_BUREAU_MON" DOUBLE PRECISION,
    "AMT_REQ_CREDIT_BUREAU_QRT" DOUBLE PRECISION,
    "AMT_REQ_CREDIT_BUREAU_YEAR" DOUBLE PRECISION
);
"""

CREATE_BUREAU_BALANCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bureau_balance (
    "SK_ID_BUREAU" BIGINT,

    "MONTHS_BALANCE" INTEGER,
    
    "STATUS" TEXT
);
"""

CREATE_BUREAU_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bureau (

    "SK_ID_CURR" BIGINT,
    "SK_ID_BUREAU" BIGINT,

    "CREDIT_ACTIVE" TEXT,
    "CREDIT_CURRENCY" TEXT,

    "DAYS_CREDIT" INTEGER,
    "CREDIT_DAY_OVERDUE" INTEGER,

    "DAYS_CREDIT_ENDDATE" DOUBLE PRECISION,
    "DAYS_ENDDATE_FACT" DOUBLE PRECISION,

    "AMT_CREDIT_MAX_OVERDUE" DOUBLE PRECISION,
    "CNT_CREDIT_PROLONG" INTEGER,

    "AMT_CREDIT_SUM" DOUBLE PRECISION,
    "AMT_CREDIT_SUM_DEBT" DOUBLE PRECISION,
    "AMT_CREDIT_SUM_LIMIT" DOUBLE PRECISION,
    "AMT_CREDIT_SUM_OVERDUE" DOUBLE PRECISION,

    "CREDIT_TYPE" TEXT,

    "DAYS_CREDIT_UPDATE" INTEGER,

    "AMT_ANNUITY" DOUBLE PRECISION
);
"""

CREATE_CREDIT_CARD_BALANCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS credit_card_balance (

    "SK_ID_PREV" BIGINT,
    "SK_ID_CURR" BIGINT,

    "MONTHS_BALANCE" INTEGER,

    "AMT_BALANCE" DOUBLE PRECISION,
    "AMT_CREDIT_LIMIT_ACTUAL" DOUBLE PRECISION,

    "AMT_DRAWINGS_ATM_CURRENT" DOUBLE PRECISION,
    "AMT_DRAWINGS_CURRENT" DOUBLE PRECISION,
    "AMT_DRAWINGS_OTHER_CURRENT" DOUBLE PRECISION,
    "AMT_DRAWINGS_POS_CURRENT" DOUBLE PRECISION,

    "AMT_INST_MIN_REGULARITY" DOUBLE PRECISION,
    "AMT_PAYMENT_CURRENT" DOUBLE PRECISION,
    "AMT_PAYMENT_TOTAL_CURRENT" DOUBLE PRECISION,

    "AMT_RECEIVABLE_PRINCIPAL" DOUBLE PRECISION,
    "AMT_RECIVABLE" DOUBLE PRECISION,
    "AMT_TOTAL_RECEIVABLE" DOUBLE PRECISION,

    "CNT_DRAWINGS_ATM_CURRENT" DOUBLE PRECISION,
    "CNT_DRAWINGS_CURRENT" DOUBLE PRECISION,
    "CNT_DRAWINGS_OTHER_CURRENT" DOUBLE PRECISION,
    "CNT_DRAWINGS_POS_CURRENT" DOUBLE PRECISION,

    "CNT_INSTALMENT_MATURE_CUM" DOUBLE PRECISION,

    "NAME_CONTRACT_STATUS" TEXT,

    "SK_DPD" INTEGER,
    "SK_DPD_DEF" INTEGER
);
"""

CREATE_INSTALLMENTS_PAYMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS installments_payments (

    "SK_ID_PREV" BIGINT,
    "SK_ID_CURR" BIGINT,

    "NUM_INSTALMENT_VERSION" DOUBLE PRECISION,
    "NUM_INSTALMENT_NUMBER" DOUBLE PRECISION,

    "DAYS_INSTALMENT" DOUBLE PRECISION,
    "DAYS_ENTRY_PAYMENT" DOUBLE PRECISION,

    "AMT_INSTALMENT" DOUBLE PRECISION,
    "AMT_PAYMENT" DOUBLE PRECISION
);
"""

CREATE_POS_CASH_BALANCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS "POS_CASH_balance" (

    "SK_ID_PREV" BIGINT,
    "SK_ID_CURR" BIGINT,

    "MONTHS_BALANCE" INTEGER,

    "CNT_INSTALMENT" DOUBLE PRECISION,
    "CNT_INSTALMENT_FUTURE" DOUBLE PRECISION,

    "NAME_CONTRACT_STATUS" TEXT,

    "SK_DPD" INTEGER,
    "SK_DPD_DEF" INTEGER
);
"""

CREATE_PREVIOUS_APPLICATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS previous_application (

    "SK_ID_PREV" BIGINT,
    "SK_ID_CURR" BIGINT,

    "NAME_CONTRACT_TYPE" TEXT,

    "AMT_ANNUITY" DOUBLE PRECISION,
    "AMT_APPLICATION" DOUBLE PRECISION,
    "AMT_CREDIT" DOUBLE PRECISION,
    "AMT_DOWN_PAYMENT" DOUBLE PRECISION,
    "AMT_GOODS_PRICE" DOUBLE PRECISION,

    "WEEKDAY_APPR_PROCESS_START" TEXT,
    "HOUR_APPR_PROCESS_START" INTEGER,

    "FLAG_LAST_APPL_PER_CONTRACT" TEXT,
    "NFLAG_LAST_APPL_IN_DAY" DOUBLE PRECISION,

    "RATE_DOWN_PAYMENT" DOUBLE PRECISION,
    "RATE_INTEREST_PRIMARY" DOUBLE PRECISION,
    "RATE_INTEREST_PRIVILEGED" DOUBLE PRECISION,

    "NAME_CASH_LOAN_PURPOSE" TEXT,
    "NAME_CONTRACT_STATUS" TEXT,

    "DAYS_DECISION" INTEGER,

    "NAME_PAYMENT_TYPE" TEXT,
    "CODE_REJECT_REASON" TEXT,
    "NAME_TYPE_SUITE" TEXT,
    "NAME_CLIENT_TYPE" TEXT,
    "NAME_GOODS_CATEGORY" TEXT,
    "NAME_PORTFOLIO" TEXT,
    "NAME_PRODUCT_TYPE" TEXT,
    "CHANNEL_TYPE" TEXT,

    "SELLERPLACE_AREA" INTEGER,
    "NAME_SELLER_INDUSTRY" TEXT,

    "CNT_PAYMENT" DOUBLE PRECISION,

    "NAME_YIELD_GROUP" TEXT,
    "PRODUCT_COMBINATION" TEXT,

    "DAYS_FIRST_DRAWING" DOUBLE PRECISION,
    "DAYS_FIRST_DUE" DOUBLE PRECISION,
    "DAYS_LAST_DUE_1ST_VERSION" DOUBLE PRECISION,
    "DAYS_LAST_DUE" DOUBLE PRECISION,
    "DAYS_TERMINATION" DOUBLE PRECISION,

    "NFLAG_INSURED_ON_APPROVAL" DOUBLE PRECISION
);
"""

# =============================================================================
# Fonctions de création des tables
# =============================================================================

def create_application_test_table(engine) -> None:
    """
    Crée la table brute `application_test`.

    Cette table contient les données principales des clients
    utilisées pour le scoring.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    
    with engine.begin() as connection:
        connection.execute(text(CREATE_APPLICATION_TEST_TABLE_SQL))

    print("Table 'application_test' créée ou déjà existante.")

def create_bureau_balance_table(engine) -> None:
    """
    Crée la table `bureau_balance`.

    Cette table contient l'historique mensuel des statuts
    de crédit pour chaque identifiant SK_ID_BUREAU.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    
    with engine.begin() as connection:
        connection.execute(text(CREATE_BUREAU_BALANCE_TABLE_SQL))

    print("Table 'bureau_balance' créée ou déjà existante.")

def create_bureau_table(engine) -> None:
    """
    Crée la table `bureau`.

    Cette table regroupe les informations sur les crédits
    externes des clients issues du bureau de crédit.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """

    with engine.begin() as connection:
        connection.execute(text(CREATE_BUREAU_TABLE_SQL))

    print("Table 'bureau' créée ou déjà existante.")

def create_credit_card_balance_table(engine) -> None:
    """
    Crée la table `credit_card_balance`.

    Cette table contient l'historique mensuel des cartes de crédit,
    incluant les montants, paiements et retards.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """

    with engine.begin() as connection:
        connection.execute(text(CREATE_CREDIT_CARD_BALANCE_TABLE_SQL))

    print("Table 'credit_card_balance' créée ou déjà existante.")

def create_installments_payments_table(engine) -> None:
    """
    Crée la table `installments_payments`.

    Cette table contient l'historique des paiements des échéances
    des crédits, permettant d'analyser les retards et remboursements.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """

    with engine.begin() as connection:
        connection.execute(text(CREATE_INSTALLMENTS_PAYMENTS_TABLE_SQL))

    print("Table 'installments_payments' créée ou déjà existante.")

def create_pos_cash_balance_table(engine) -> None:
    """
    Crée la table `POS_CASH_balance`.

    Cette table contient l'historique des crédits à la consommation
    (POS / cash loans) avec leur évolution mensuelle.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """

    with engine.begin() as connection:
        connection.execute(text(CREATE_POS_CASH_BALANCE_TABLE_SQL))

    print("Table 'POS_CASH_balance' créée ou déjà existante.")

def create_previous_application_table(engine) -> None:
    """
    Crée la table `previous_application`.

    Cette table contient l'historique des demandes de crédit précédentes
    des clients avec leurs caractéristiques et statuts.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """

    with engine.begin() as connection:
        connection.execute(text(CREATE_PREVIOUS_APPLICATION_TABLE_SQL))

    print("Table 'previous_application' créée ou déjà existante.")

TABLE_CREATORS = [
    create_application_test_table,
    create_bureau_balance_table,
    create_bureau_table,
    create_credit_card_balance_table,
    create_installments_payments_table,
    create_pos_cash_balance_table,
    create_previous_application_table,
]

CREATE_INDEXES_SQL = [
    'CREATE INDEX IF NOT EXISTS idx_bureau_sk_id_curr ON bureau("SK_ID_CURR");',
    'CREATE INDEX IF NOT EXISTS idx_bureau_sk_id_bureau ON bureau("SK_ID_BUREAU");',

    'CREATE INDEX IF NOT EXISTS idx_bureau_balance_sk_id_bureau ON bureau_balance("SK_ID_BUREAU");',

    'CREATE INDEX IF NOT EXISTS idx_cc_sk_id_curr ON credit_card_balance("SK_ID_CURR");',
    'CREATE INDEX IF NOT EXISTS idx_cc_sk_id_prev ON credit_card_balance("SK_ID_PREV");',

    'CREATE INDEX IF NOT EXISTS idx_inst_sk_id_curr ON installments_payments("SK_ID_CURR");',
    'CREATE INDEX IF NOT EXISTS idx_inst_sk_id_prev ON installments_payments("SK_ID_PREV");',

    'CREATE INDEX IF NOT EXISTS idx_pos_sk_id_curr ON "POS_CASH_balance"("SK_ID_CURR");',
    'CREATE INDEX IF NOT EXISTS idx_pos_sk_id_prev ON "POS_CASH_balance"("SK_ID_PREV");',

    'CREATE INDEX IF NOT EXISTS idx_prev_sk_id_curr ON previous_application("SK_ID_CURR");',
    'CREATE INDEX IF NOT EXISTS idx_prev_sk_id_prev ON previous_application("SK_ID_PREV");',
]

def create_indexes(engine) -> None:
    """
    Crée les index essentiels pour optimiser les jointures
    entre les tables RAW.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        for sql in CREATE_INDEXES_SQL:
            connection.execute(text(sql))

    print("Index créés ou déjà existants.")

def main() -> None:
    """
    Point d'entrée du script.

    Cette fonction :
    1. établit la connexion à PostgreSQL,
    2. crée les tables RAW nécessaires,
    3. affiche un message de confirmation.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    # -------------------------------------------------------------------------
    # Création des tables RAW
    # -------------------------------------------------------------------------
    for create_table in TABLE_CREATORS:
        create_table(engine)

    create_indexes(engine)

    print("Création des tables terminée.")


if __name__ == "__main__":
    main()