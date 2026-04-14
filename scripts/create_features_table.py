"""
Script de création de la table de features brute au niveau client.

Ce module construit une table intermédiaire `features_client_test_raw`
à partir des tables RAW du projet Home Credit.

Objectif
--------
Créer une table de features agrégées fidèle à la logique du notebook,
avec une ligne par client (`SK_ID_CURR`), avant les étapes finales
de nettoyage et de suppression de colonnes.

Notes
-----
- La table est reconstruite à chaque exécution.
- Cette version correspond à une couche intermédiaire.
- Le nettoyage final sera appliqué dans une seconde étape.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Configuration de la base de données
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL n'est pas défini dans les variables d'environnement")


# =============================================================================
# SQL de création de la table features_client_test_raw
# =============================================================================

CREATE_FEATURES_CLIENT_TEST_RAW_SQL = """
DROP TABLE IF EXISTS features_client_test_raw;

CREATE TABLE features_client_test_raw AS
WITH
-- ============================================================================
-- 1. bureau_balance enrichi par SK_ID_BUREAU
-- ============================================================================
bb_status AS (
    SELECT
        bb."SK_ID_BUREAU",

        MAX(
            CASE
                WHEN bb."MONTHS_BALANCE" >= -6 THEN
                    CASE bb."STATUS"
                        WHEN '1' THEN 1
                        WHEN '2' THEN 2
                        WHEN '3' THEN 3
                        WHEN '4' THEN 4
                        WHEN '5' THEN 5
                        ELSE 0
                    END
                ELSE NULL
            END
        ) AS bb__recent_max_dpd,

        AVG(
            CASE
                WHEN bb."STATUS" IN ('1', '2', '3', '4', '5') THEN 1.0
                ELSE 0.0
            END
        ) AS bb__months_late_ratio,

        SUM(
            CASE bb."STATUS"
                WHEN '1' THEN 1
                WHEN '2' THEN 2
                WHEN '3' THEN 3
                WHEN '4' THEN 4
                WHEN '5' THEN 5
                ELSE 0
            END
        ) AS bb__late_severity_sum

    FROM bureau_balance bb
    GROUP BY bb."SK_ID_BUREAU"
),

bb_num AS (
    SELECT
        bb."SK_ID_BUREAU",
        COUNT(*) AS bb__count_rows,
        AVG(bb."MONTHS_BALANCE") AS bb__MONTHS_BALANCE__mean,
        MAX(bb."MONTHS_BALANCE") AS bb__MONTHS_BALANCE__max,
        MIN(bb."MONTHS_BALANCE") AS bb__MONTHS_BALANCE__min,
        STDDEV_POP(bb."MONTHS_BALANCE") AS bb__MONTHS_BALANCE__std
    FROM bureau_balance bb
    GROUP BY bb."SK_ID_BUREAU"
),

bb_agg AS (
    SELECT
        n."SK_ID_BUREAU",
        n.bb__count_rows,
        n.bb__MONTHS_BALANCE__mean,
        n.bb__MONTHS_BALANCE__max,
        n.bb__MONTHS_BALANCE__min,
        n.bb__MONTHS_BALANCE__std,
        s.bb__recent_max_dpd,
        s.bb__months_late_ratio,
        s.bb__late_severity_sum
    FROM bb_num n
    LEFT JOIN bb_status s
        ON n."SK_ID_BUREAU" = s."SK_ID_BUREAU"
),

-- ============================================================================
-- 2. bureau enrichi puis agrégé au niveau client
-- ============================================================================
bureau_enriched AS (
    SELECT
        b."SK_ID_CURR",
        b."SK_ID_BUREAU",
        b."DAYS_CREDIT",
        b."CREDIT_DAY_OVERDUE",
        b."AMT_CREDIT_SUM",
        b."AMT_CREDIT_SUM_DEBT",
        b."AMT_CREDIT_SUM_OVERDUE",

        CASE
            WHEN b."AMT_CREDIT_SUM" IS NOT NULL THEN
                b."AMT_CREDIT_SUM_DEBT" / NULLIF(b."AMT_CREDIT_SUM", 0)
            ELSE NULL
        END AS bureau__DEBT_RATIO,

        CASE
            WHEN b."AMT_CREDIT_SUM" IS NOT NULL THEN
                b."AMT_CREDIT_SUM_OVERDUE" / NULLIF(b."AMT_CREDIT_SUM", 0)
            ELSE NULL
        END AS bureau__OVERDUE_RATIO,

        CASE
            WHEN b."CREDIT_ACTIVE" = 'Active' THEN 1
            ELSE 0
        END AS bureau__IS_ACTIVE,

        CASE
            WHEN b."CREDIT_DAY_OVERDUE" > 0 THEN 1
            ELSE 0
        END AS bureau__HAS_OVERDUE,

        -1.0 * b."DAYS_CREDIT" AS bureau__CREDIT_AGE,

        bb_agg.bb__count_rows,
        bb_agg.bb__MONTHS_BALANCE__mean,
        bb_agg.bb__MONTHS_BALANCE__max,
        bb_agg.bb__MONTHS_BALANCE__min,
        bb_agg.bb__MONTHS_BALANCE__std,
        bb_agg.bb__recent_max_dpd,
        bb_agg.bb__months_late_ratio,
        bb_agg.bb__late_severity_sum

    FROM bureau b
    LEFT JOIN bb_agg
        ON b."SK_ID_BUREAU" = bb_agg."SK_ID_BUREAU"
),

bureau_agg_curr AS (
    SELECT
        "SK_ID_CURR",
        COUNT(*) AS bureau__count_rows,
        COUNT(DISTINCT "SK_ID_BUREAU") AS bureau__nunique_SK_ID_BUREAU,

        AVG("DAYS_CREDIT") AS bureau__DAYS_CREDIT__mean,
        MAX("DAYS_CREDIT") AS bureau__DAYS_CREDIT__max,
        MIN("DAYS_CREDIT") AS bureau__DAYS_CREDIT__min,
        STDDEV_POP("DAYS_CREDIT") AS bureau__DAYS_CREDIT__std,

        AVG("AMT_CREDIT_SUM") AS bureau__AMT_CREDIT_SUM__mean,
        MAX("AMT_CREDIT_SUM") AS bureau__AMT_CREDIT_SUM__max,
        MIN("AMT_CREDIT_SUM") AS bureau__AMT_CREDIT_SUM__min,
        STDDEV_POP("AMT_CREDIT_SUM") AS bureau__AMT_CREDIT_SUM__std,

        AVG(bureau__DEBT_RATIO) AS bureau__DEBT_RATIO__mean,
        MAX(bureau__DEBT_RATIO) AS bureau__DEBT_RATIO__max,
        AVG(bureau__OVERDUE_RATIO) AS bureau__OVERDUE_RATIO__mean,
        MAX(bureau__OVERDUE_RATIO) AS bureau__OVERDUE_RATIO__max,

        AVG(bureau__IS_ACTIVE) AS bureau__IS_ACTIVE__mean,
        SUM(bureau__HAS_OVERDUE) AS bureau__HAS_OVERDUE__sum,

        AVG(bb__months_late_ratio) AS bureau__bb__months_late_ratio__mean,
        MAX(bb__recent_max_dpd) AS bureau__bb__recent_max_dpd__max,
        SUM(bb__late_severity_sum) AS bureau__bb__late_severity_sum__sum

    FROM bureau_enriched
    GROUP BY "SK_ID_CURR"
),

-- ============================================================================
-- 3. POS_CASH_balance -> SK_ID_PREV
-- ============================================================================
pos_enriched AS (
    SELECT
        p."SK_ID_PREV",
        p."SK_ID_CURR",
        p."MONTHS_BALANCE",
        p."CNT_INSTALMENT",
        p."CNT_INSTALMENT_FUTURE",
        p."SK_DPD",
        p."SK_DPD_DEF",

        p."CNT_INSTALMENT_FUTURE" / NULLIF(p."CNT_INSTALMENT", 0) AS pos__POS_REMAIN_RATIO,
        GREATEST(p."SK_DPD", 0) AS pos__POS_DPD_POS,
        CASE WHEN p."NAME_CONTRACT_STATUS" = 'Active' THEN 1 ELSE 0 END AS pos__POS_IS_ACTIVE
    FROM "POS_CASH_balance" p
),

pos_agg AS (
    SELECT
        "SK_ID_PREV",
        COUNT(*) AS pos__count_rows,
        AVG("MONTHS_BALANCE") AS pos__MONTHS_BALANCE__mean,
        MAX("MONTHS_BALANCE") AS pos__MONTHS_BALANCE__max,
        AVG(pos__POS_REMAIN_RATIO) AS pos__POS_REMAIN_RATIO__mean,
        MAX(pos__POS_DPD_POS) AS pos__POS_DPD_POS__max,
        AVG(pos__POS_IS_ACTIVE) AS pos__POS_IS_ACTIVE__mean
    FROM pos_enriched
    GROUP BY "SK_ID_PREV"
),

-- ============================================================================
-- 4. credit_card_balance -> SK_ID_PREV
-- ============================================================================
cc_enriched AS (
    SELECT
        c."SK_ID_PREV",
        c."SK_ID_CURR",
        c."MONTHS_BALANCE",
        c."AMT_BALANCE",
        c."AMT_CREDIT_LIMIT_ACTUAL",
        c."AMT_INST_MIN_REGULARITY",
        c."AMT_PAYMENT_CURRENT",
        c."AMT_PAYMENT_TOTAL_CURRENT",
        c."AMT_RECIVABLE",
        c."AMT_TOTAL_RECEIVABLE",
        c."SK_DPD",

        c."AMT_BALANCE" / NULLIF(c."AMT_CREDIT_LIMIT_ACTUAL", 0) AS cc__CC_UTILIZATION_RATIO,
        c."AMT_PAYMENT_CURRENT" / NULLIF(c."AMT_INST_MIN_REGULARITY", 0) AS cc__CC_PAYMENT_MIN_RATIO,
        c."AMT_PAYMENT_TOTAL_CURRENT" / NULLIF(c."AMT_BALANCE", 0) AS cc__CC_PAYMENT_BALANCE_RATIO,
        GREATEST(c."SK_DPD", 0) AS cc__CC_DPD_POS,
        c."AMT_RECIVABLE" / NULLIF(c."AMT_TOTAL_RECEIVABLE", 0) AS cc__CC_RECEIVABLE_RATIO
    FROM credit_card_balance c
),

cc_agg AS (
    SELECT
        "SK_ID_PREV",
        COUNT(*) AS cc__count_rows,
        AVG("MONTHS_BALANCE") AS cc__MONTHS_BALANCE__mean,
        AVG(cc__CC_UTILIZATION_RATIO) AS cc__CC_UTILIZATION_RATIO__mean,
        MAX(cc__CC_UTILIZATION_RATIO) AS cc__CC_UTILIZATION_RATIO__max,
        AVG(cc__CC_PAYMENT_MIN_RATIO) AS cc__CC_PAYMENT_MIN_RATIO__mean,
        AVG(cc__CC_PAYMENT_BALANCE_RATIO) AS cc__CC_PAYMENT_BALANCE_RATIO__mean,
        MAX(cc__CC_DPD_POS) AS cc__CC_DPD_POS__max,
        AVG(cc__CC_RECEIVABLE_RATIO) AS cc__CC_RECEIVABLE_RATIO__mean
    FROM cc_enriched
    GROUP BY "SK_ID_PREV"
),

-- ============================================================================
-- 5. installments_payments -> SK_ID_PREV
-- ============================================================================
inst_enriched AS (
    SELECT
        i."SK_ID_PREV",
        i."SK_ID_CURR",
        i."DAYS_INSTALMENT",
        i."DAYS_ENTRY_PAYMENT",
        i."AMT_INSTALMENT",
        i."AMT_PAYMENT",

        GREATEST(i."DAYS_ENTRY_PAYMENT" - i."DAYS_INSTALMENT", 0) AS inst__DPD_POS,
        CASE
            WHEN (i."DAYS_ENTRY_PAYMENT" - i."DAYS_INSTALMENT") > 30 THEN 1
            ELSE 0
        END AS inst__SEVERE_LATE_30,
        i."AMT_PAYMENT" / NULLIF(i."AMT_INSTALMENT", 0) AS inst__PAY_RATIO
    FROM installments_payments i
),

inst_agg AS (
    SELECT
        "SK_ID_PREV",
        COUNT(*) AS inst__count_rows,
        AVG(inst__DPD_POS) AS inst__DPD_POS__mean,
        MAX(inst__DPD_POS) AS inst__DPD_POS__max,
        AVG(inst__SEVERE_LATE_30) AS inst__SEVERE_LATE_30__mean,
        MAX(inst__SEVERE_LATE_30) AS inst__SEVERE_LATE_30__max,
        AVG(inst__PAY_RATIO) AS inst__PAY_RATIO__mean
    FROM inst_enriched
    GROUP BY "SK_ID_PREV"
),

-- ============================================================================
-- 6. previous_application enrichi puis agrégé au niveau client
-- ============================================================================
prev_enriched AS (
    SELECT
        p."SK_ID_PREV",
        p."SK_ID_CURR",
        p."AMT_APPLICATION",
        p."AMT_CREDIT",
        p."DAYS_DECISION",
        p."DAYS_FIRST_DUE",
        p."DAYS_LAST_DUE",
        p."CNT_PAYMENT",

        p."AMT_CREDIT" / NULLIF(p."AMT_APPLICATION", 0) AS prev__PREV_CREDIT_APPLICATION_RATIO,
        CASE WHEN p."NAME_CONTRACT_STATUS" = 'Approved' THEN 1 ELSE 0 END AS prev__PREV_IS_APPROVED,
        CASE WHEN p."NAME_CONTRACT_STATUS" = 'Refused' THEN 1 ELSE 0 END AS prev__PREV_IS_REFUSED,
        -1.0 * p."DAYS_DECISION" AS prev__PREV_DAYS_DECISION_AGE,
        p."DAYS_LAST_DUE" - p."DAYS_FIRST_DUE" AS prev__PREV_CREDIT_DURATION,

        pos_agg.pos__count_rows,
        pos_agg.pos__MONTHS_BALANCE__mean,
        pos_agg.pos__MONTHS_BALANCE__max,
        pos_agg.pos__POS_REMAIN_RATIO__mean,
        pos_agg.pos__POS_DPD_POS__max,
        pos_agg.pos__POS_IS_ACTIVE__mean,

        cc_agg.cc__count_rows,
        cc_agg.cc__MONTHS_BALANCE__mean,
        cc_agg.cc__CC_UTILIZATION_RATIO__mean,
        cc_agg.cc__CC_UTILIZATION_RATIO__max,
        cc_agg.cc__CC_PAYMENT_MIN_RATIO__mean,
        cc_agg.cc__CC_PAYMENT_BALANCE_RATIO__mean,
        cc_agg.cc__CC_DPD_POS__max,
        cc_agg.cc__CC_RECEIVABLE_RATIO__mean,

        inst_agg.inst__count_rows,
        inst_agg.inst__DPD_POS__mean,
        inst_agg.inst__DPD_POS__max,
        inst_agg.inst__SEVERE_LATE_30__mean,
        inst_agg.inst__SEVERE_LATE_30__max,
        inst_agg.inst__PAY_RATIO__mean

    FROM previous_application p
    LEFT JOIN pos_agg
        ON p."SK_ID_PREV" = pos_agg."SK_ID_PREV"
    LEFT JOIN cc_agg
        ON p."SK_ID_PREV" = cc_agg."SK_ID_PREV"
    LEFT JOIN inst_agg
        ON p."SK_ID_PREV" = inst_agg."SK_ID_PREV"
),

prev_agg_curr AS (
    SELECT
        "SK_ID_CURR",
        COUNT(*) AS prev__count_rows,
        COUNT(DISTINCT "SK_ID_PREV") AS prev__nunique_SK_ID_PREV,

        AVG("AMT_APPLICATION") AS prev__AMT_APPLICATION__mean,
        MAX("AMT_APPLICATION") AS prev__AMT_APPLICATION__max,
        AVG("AMT_CREDIT") AS prev__AMT_CREDIT__mean,
        MAX("AMT_CREDIT") AS prev__AMT_CREDIT__max,

        AVG(prev__PREV_CREDIT_APPLICATION_RATIO) AS prev__PREV_CREDIT_APPLICATION_RATIO__mean,
        AVG(prev__PREV_IS_APPROVED) AS prev__PREV_IS_APPROVED__mean,
        AVG(prev__PREV_IS_REFUSED) AS prev__PREV_IS_REFUSED__mean,
        AVG(prev__PREV_DAYS_DECISION_AGE) AS prev__PREV_DAYS_DECISION_AGE__mean,
        AVG(prev__PREV_CREDIT_DURATION) AS prev__PREV_CREDIT_DURATION__mean,
        AVG("CNT_PAYMENT") AS prev__CNT_PAYMENT__mean,

        AVG(pos__POS_REMAIN_RATIO__mean) AS prev__pos__POS_REMAIN_RATIO__mean,
        MAX(pos__POS_DPD_POS__max) AS prev__pos__POS_DPD_POS__max,
        AVG(cc__CC_UTILIZATION_RATIO__mean) AS prev__cc__CC_UTILIZATION_RATIO__mean,
        MAX(cc__CC_UTILIZATION_RATIO__max) AS prev__cc__CC_UTILIZATION_RATIO__max,
        AVG(cc__CC_PAYMENT_MIN_RATIO__mean) AS prev__cc__CC_PAYMENT_MIN_RATIO__mean,
        AVG(inst__DPD_POS__mean) AS prev__inst__DPD_POS__mean,
        MAX(inst__DPD_POS__max) AS prev__inst__DPD_POS__max,
        AVG(inst__SEVERE_LATE_30__mean) AS prev__inst__SEVERE_LATE_30__mean,
        AVG(inst__PAY_RATIO__mean) AS prev__inst__PAY_RATIO__mean

    FROM prev_enriched
    GROUP BY "SK_ID_CURR"
)

-- ============================================================================
-- 7. Jointure finale avec application_test
-- ============================================================================
SELECT
    a.*,
    b.*,
    p.*
FROM application_test a
LEFT JOIN bureau_agg_curr b
    ON a."SK_ID_CURR" = b."SK_ID_CURR"
LEFT JOIN prev_agg_curr p
    ON a."SK_ID_CURR" = p."SK_ID_CURR";
"""

CREATE_FEATURES_CLIENT_TEST_RAW_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_features_client_test_raw_sk_id_curr
ON features_client_test_raw ("SK_ID_CURR");
"""


def create_features_client_test_raw_table(engine) -> None:
    """
    Crée la table `features_client_test_raw`.

    Cette table contient une ligne par client avec l'ensemble
    des agrégations brutes issues des tables RAW, avant nettoyage final.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_FEATURES_CLIENT_TEST_RAW_SQL))
        connection.execute(text(CREATE_FEATURES_CLIENT_TEST_RAW_INDEX_SQL))

    print("Table 'features_client_test_raw' créée ou recréée avec succès.")


def main() -> None:
    """
    Point d'entrée du script.

    Cette fonction :
    1. établit la connexion à PostgreSQL,
    2. crée la table brute de features,
    3. affiche un message de confirmation.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    create_features_client_test_raw_table(engine)

    print("Création de la table 'features_client_test_raw' terminée.")


if __name__ == "__main__":
    main()