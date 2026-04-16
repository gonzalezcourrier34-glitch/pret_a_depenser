"""
Script de création des tables temporaires d'agrégation de features.

Ce module crée uniquement la structure des tables temporaires utilisées
dans le pipeline de feature engineering à partir des tables RAW.

Objectif
--------
Séparer la création des tables temporaires de leur remplissage afin de :
- clarifier les responsabilités
- faciliter le débogage
- rendre le pipeline plus lisible

Tables créées
-------------
- tmp_batch_sk_id_bureau
- tmp_batch_sk_id_prev
- tmp_batch_sk_id_curr
- bb_agg_tmp
- bureau_agg_curr_tmp
- pos_agg_tmp
- cc_agg_tmp
- inst_agg_tmp
- prev_agg_curr_tmp

Notes
-----
- Les tables sont supprimées puis recréées à chaque exécution.
- Ce script ne remplit aucune table.
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
# SQL de suppression
# =============================================================================

DROP_TEMP_TABLES_SQL = """
DROP TABLE IF EXISTS prev_agg_curr_tmp;
DROP TABLE IF EXISTS inst_agg_tmp;
DROP TABLE IF EXISTS cc_agg_tmp;
DROP TABLE IF EXISTS pos_agg_tmp;
DROP TABLE IF EXISTS bureau_agg_curr_tmp;
DROP TABLE IF EXISTS bb_agg_tmp;
DROP TABLE IF EXISTS tmp_batch_sk_id_bureau;
DROP TABLE IF EXISTS tmp_batch_sk_id_prev;
DROP TABLE IF EXISTS tmp_batch_sk_id_curr;
"""


# =============================================================================
# Création des tables batch
# =============================================================================

CREATE_TMP_BATCH_SK_ID_BUREAU_SQL = """
CREATE TABLE tmp_batch_sk_id_bureau (
    "SK_ID_BUREAU" BIGINT PRIMARY KEY
);
"""

CREATE_TMP_BATCH_SK_ID_PREV_SQL = """
CREATE TABLE tmp_batch_sk_id_prev (
    "SK_ID_PREV" BIGINT PRIMARY KEY
);
"""

CREATE_TMP_BATCH_SK_ID_CURR_SQL = """
CREATE TABLE tmp_batch_sk_id_curr (
    "SK_ID_CURR" BIGINT PRIMARY KEY
);
"""


# =============================================================================
# Création des tables temporaires vides
# =============================================================================

CREATE_BB_AGG_TMP_EMPTY_SQL = """
CREATE TABLE bb_agg_tmp (
    "SK_ID_BUREAU" BIGINT,
    bb__count_rows DOUBLE PRECISION,
    bb__MONTHS_BALANCE__mean DOUBLE PRECISION,
    bb__MONTHS_BALANCE__max DOUBLE PRECISION,
    bb__MONTHS_BALANCE__min DOUBLE PRECISION,
    bb__MONTHS_BALANCE__std DOUBLE PRECISION,
    bb__recent_max_dpd DOUBLE PRECISION,
    bb__months_late_ratio DOUBLE PRECISION,
    bb__late_severity_sum DOUBLE PRECISION
);
"""

CREATE_BUREAU_AGG_CURR_TMP_EMPTY_SQL = """
CREATE TABLE bureau_agg_curr_tmp (
    "SK_ID_CURR" BIGINT,
    bureau__count_rows DOUBLE PRECISION,
    bureau__nunique_SK_ID_BUREAU DOUBLE PRECISION,
    bureau__DAYS_CREDIT__mean DOUBLE PRECISION,
    bureau__DAYS_CREDIT__std DOUBLE PRECISION,
    bureau__CREDIT_DAY_OVERDUE__mean DOUBLE PRECISION,
    bureau__CREDIT_DAY_OVERDUE__std DOUBLE PRECISION,
    bureau__DAYS_CREDIT_ENDDATE__mean DOUBLE PRECISION,
    bureau__DAYS_CREDIT_ENDDATE__std DOUBLE PRECISION,
    bureau__DAYS_ENDDATE_FACT__mean DOUBLE PRECISION,
    bureau__DAYS_ENDDATE_FACT__std DOUBLE PRECISION,
    bureau__AMT_CREDIT_MAX_OVERDUE__mean DOUBLE PRECISION,
    bureau__AMT_CREDIT_MAX_OVERDUE__std DOUBLE PRECISION,
    bureau__CNT_CREDIT_PROLONG__mean DOUBLE PRECISION,
    bureau__CNT_CREDIT_PROLONG__std DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM__mean DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM__std DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM_DEBT__mean DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM_DEBT__std DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM_LIMIT__mean DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM_LIMIT__std DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM_OVERDUE__mean DOUBLE PRECISION,
    bureau__AMT_CREDIT_SUM_OVERDUE__std DOUBLE PRECISION,
    bureau__DAYS_CREDIT_UPDATE__mean DOUBLE PRECISION,
    bureau__DAYS_CREDIT_UPDATE__std DOUBLE PRECISION,
    bureau__AMT_ANNUITY__mean DOUBLE PRECISION,
    bureau__AMT_ANNUITY__std DOUBLE PRECISION,
    bureau__DEBT_RATIO__mean DOUBLE PRECISION,
    bureau__DEBT_RATIO__std DOUBLE PRECISION,
    bureau__OVERDUE_RATIO__mean DOUBLE PRECISION,
    bureau__OVERDUE_RATIO__std DOUBLE PRECISION,
    bureau__IS_ACTIVE__mean DOUBLE PRECISION,
    bureau__IS_ACTIVE__std DOUBLE PRECISION,
    bureau__HAS_OVERDUE__mean DOUBLE PRECISION,
    bureau__HAS_OVERDUE__std DOUBLE PRECISION,
    bureau__CREDIT_AGE__mean DOUBLE PRECISION,
    bureau__CREDIT_AGE__std DOUBLE PRECISION,
    bureau__bb__MONTHS_BALANCE__mean__mean DOUBLE PRECISION,
    bureau__bb__MONTHS_BALANCE__mean__std DOUBLE PRECISION,
    bureau__bb__MONTHS_BALANCE__std__mean DOUBLE PRECISION,
    bureau__bb__MONTHS_BALANCE__std__std DOUBLE PRECISION,
    bureau__bb__count_rows__mean DOUBLE PRECISION,
    bureau__bb__count_rows__std DOUBLE PRECISION,
    bureau__bb__recent_max_dpd__mean DOUBLE PRECISION,
    bureau__bb__recent_max_dpd__std DOUBLE PRECISION,
    bureau__bb__months_late_ratio__mean DOUBLE PRECISION,
    bureau__bb__months_late_ratio__std DOUBLE PRECISION,
    bureau__bb__late_severity_sum__mean DOUBLE PRECISION,
    bureau__bb__late_severity_sum__std DOUBLE PRECISION
);
"""

CREATE_POS_AGG_TMP_EMPTY_SQL = """
CREATE TABLE pos_agg_tmp (
    "SK_ID_PREV" BIGINT,
    pos__count_rows DOUBLE PRECISION,
    pos__MONTHS_BALANCE__mean DOUBLE PRECISION,
    pos__MONTHS_BALANCE__std DOUBLE PRECISION,
    pos__CNT_INSTALMENT__mean DOUBLE PRECISION,
    pos__CNT_INSTALMENT__std DOUBLE PRECISION,
    pos__CNT_INSTALMENT_FUTURE__mean DOUBLE PRECISION,
    pos__CNT_INSTALMENT_FUTURE__std DOUBLE PRECISION,
    pos__SK_DPD__mean DOUBLE PRECISION,
    pos__SK_DPD__max DOUBLE PRECISION,
    pos__SK_DPD__std DOUBLE PRECISION,
    pos__SK_DPD_DEF__mean DOUBLE PRECISION,
    pos__SK_DPD_DEF__max DOUBLE PRECISION,
    pos__SK_DPD_DEF__std DOUBLE PRECISION,
    pos__POS_REMAIN_RATIO__mean DOUBLE PRECISION,
    pos__POS_REMAIN_RATIO__std DOUBLE PRECISION,
    pos__POS_DPD_POS__mean DOUBLE PRECISION,
    pos__POS_DPD_POS__max DOUBLE PRECISION,
    pos__POS_DPD_POS__std DOUBLE PRECISION,
    pos__POS_IS_ACTIVE__mean DOUBLE PRECISION,
    pos__POS_IS_ACTIVE__std DOUBLE PRECISION
);
"""

CREATE_CC_AGG_TMP_EMPTY_SQL = """
CREATE TABLE cc_agg_tmp (
    "SK_ID_PREV" BIGINT,
    cc__count_rows DOUBLE PRECISION,
    cc__MONTHS_BALANCE__mean DOUBLE PRECISION,
    cc__MONTHS_BALANCE__std DOUBLE PRECISION,
    cc__AMT_BALANCE__mean DOUBLE PRECISION,
    cc__AMT_BALANCE__std DOUBLE PRECISION,
    cc__AMT_CREDIT_LIMIT_ACTUAL__mean DOUBLE PRECISION,
    cc__AMT_CREDIT_LIMIT_ACTUAL__std DOUBLE PRECISION,
    cc__AMT_DRAWINGS_ATM_CURRENT__mean DOUBLE PRECISION,
    cc__AMT_DRAWINGS_ATM_CURRENT__std DOUBLE PRECISION,
    cc__AMT_DRAWINGS_CURRENT__mean DOUBLE PRECISION,
    cc__AMT_DRAWINGS_CURRENT__std DOUBLE PRECISION,
    cc__AMT_DRAWINGS_OTHER_CURRENT__mean DOUBLE PRECISION,
    cc__AMT_DRAWINGS_OTHER_CURRENT__std DOUBLE PRECISION,
    cc__AMT_DRAWINGS_POS_CURRENT__mean DOUBLE PRECISION,
    cc__AMT_DRAWINGS_POS_CURRENT__std DOUBLE PRECISION,
    cc__AMT_INST_MIN_REGULARITY__mean DOUBLE PRECISION,
    cc__AMT_INST_MIN_REGULARITY__std DOUBLE PRECISION,
    cc__AMT_PAYMENT_CURRENT__mean DOUBLE PRECISION,
    cc__AMT_PAYMENT_CURRENT__std DOUBLE PRECISION,
    cc__AMT_PAYMENT_TOTAL_CURRENT__mean DOUBLE PRECISION,
    cc__AMT_PAYMENT_TOTAL_CURRENT__std DOUBLE PRECISION,
    cc__AMT_RECEIVABLE_PRINCIPAL__mean DOUBLE PRECISION,
    cc__AMT_RECEIVABLE_PRINCIPAL__std DOUBLE PRECISION,
    cc__AMT_RECIVABLE__mean DOUBLE PRECISION,
    cc__AMT_RECIVABLE__std DOUBLE PRECISION,
    cc__AMT_TOTAL_RECEIVABLE__mean DOUBLE PRECISION,
    cc__AMT_TOTAL_RECEIVABLE__std DOUBLE PRECISION,
    cc__CNT_DRAWINGS_ATM_CURRENT__mean DOUBLE PRECISION,
    cc__CNT_DRAWINGS_ATM_CURRENT__std DOUBLE PRECISION,
    cc__CNT_DRAWINGS_CURRENT__mean DOUBLE PRECISION,
    cc__CNT_DRAWINGS_CURRENT__std DOUBLE PRECISION,
    cc__CNT_DRAWINGS_OTHER_CURRENT__mean DOUBLE PRECISION,
    cc__CNT_DRAWINGS_OTHER_CURRENT__std DOUBLE PRECISION,
    cc__CNT_DRAWINGS_POS_CURRENT__mean DOUBLE PRECISION,
    cc__CNT_DRAWINGS_POS_CURRENT__std DOUBLE PRECISION,
    cc__CNT_INSTALMENT_MATURE_CUM__mean DOUBLE PRECISION,
    cc__CNT_INSTALMENT_MATURE_CUM__std DOUBLE PRECISION,
    cc__SK_DPD__mean DOUBLE PRECISION,
    cc__SK_DPD__max DOUBLE PRECISION,
    cc__SK_DPD__std DOUBLE PRECISION,
    cc__SK_DPD_DEF__mean DOUBLE PRECISION,
    cc__SK_DPD_DEF__max DOUBLE PRECISION,
    cc__SK_DPD_DEF__std DOUBLE PRECISION,
    cc__CC_UTILIZATION_RATIO__mean DOUBLE PRECISION,
    cc__CC_UTILIZATION_RATIO__max DOUBLE PRECISION,
    cc__CC_UTILIZATION_RATIO__std DOUBLE PRECISION,
    cc__CC_PAYMENT_MIN_RATIO__mean DOUBLE PRECISION,
    cc__CC_PAYMENT_MIN_RATIO__max DOUBLE PRECISION,
    cc__CC_PAYMENT_MIN_RATIO__std DOUBLE PRECISION,
    cc__CC_PAYMENT_BALANCE_RATIO__mean DOUBLE PRECISION,
    cc__CC_PAYMENT_BALANCE_RATIO__max DOUBLE PRECISION,
    cc__CC_PAYMENT_BALANCE_RATIO__std DOUBLE PRECISION,
    cc__CC_DPD_POS__mean DOUBLE PRECISION,
    cc__CC_DPD_POS__max DOUBLE PRECISION,
    cc__CC_DPD_POS__std DOUBLE PRECISION,
    cc__CC_RECEIVABLE_RATIO__mean DOUBLE PRECISION,
    cc__CC_RECEIVABLE_RATIO__std DOUBLE PRECISION
);
"""

CREATE_INST_AGG_TMP_EMPTY_SQL = """
CREATE TABLE inst_agg_tmp (
    "SK_ID_PREV" BIGINT,
    inst__count_rows DOUBLE PRECISION,
    inst__NUM_INSTALMENT_VERSION__mean DOUBLE PRECISION,
    inst__NUM_INSTALMENT_VERSION__std DOUBLE PRECISION,
    inst__NUM_INSTALMENT_NUMBER__mean DOUBLE PRECISION,
    inst__NUM_INSTALMENT_NUMBER__std DOUBLE PRECISION,
    inst__DAYS_INSTALMENT__mean DOUBLE PRECISION,
    inst__DAYS_INSTALMENT__std DOUBLE PRECISION,
    inst__DAYS_ENTRY_PAYMENT__mean DOUBLE PRECISION,
    inst__DAYS_ENTRY_PAYMENT__std DOUBLE PRECISION,
    inst__AMT_INSTALMENT__mean DOUBLE PRECISION,
    inst__AMT_INSTALMENT__std DOUBLE PRECISION,
    inst__AMT_PAYMENT__mean DOUBLE PRECISION,
    inst__AMT_PAYMENT__std DOUBLE PRECISION,
    inst__DPD_POS__mean DOUBLE PRECISION,
    inst__DPD_POS__max DOUBLE PRECISION,
    inst__DPD_POS__std DOUBLE PRECISION,
    inst__SEVERE_LATE_30__mean DOUBLE PRECISION,
    inst__SEVERE_LATE_30__max DOUBLE PRECISION,
    inst__SEVERE_LATE_30__std DOUBLE PRECISION,
    inst__PAY_RATIO__mean DOUBLE PRECISION,
    inst__PAY_RATIO__std DOUBLE PRECISION
);
"""

CREATE_PREV_AGG_CURR_TMP_EMPTY_SQL = """
CREATE TABLE prev_agg_curr_tmp AS
SELECT *
FROM (
    SELECT
        NULL::BIGINT AS "SK_ID_CURR",
        NULL::DOUBLE PRECISION AS prev__count_rows,
        NULL::DOUBLE PRECISION AS prev__nunique_SK_ID_PREV,
        NULL::DOUBLE PRECISION AS prev__AMT_ANNUITY__mean,
        NULL::DOUBLE PRECISION AS prev__AMT_ANNUITY__std,
        NULL::DOUBLE PRECISION AS prev__AMT_APPLICATION__mean,
        NULL::DOUBLE PRECISION AS prev__AMT_APPLICATION__std,
        NULL::DOUBLE PRECISION AS prev__AMT_CREDIT__mean,
        NULL::DOUBLE PRECISION AS prev__AMT_CREDIT__std,
        NULL::DOUBLE PRECISION AS prev__AMT_DOWN_PAYMENT__mean,
        NULL::DOUBLE PRECISION AS prev__AMT_DOWN_PAYMENT__std,
        NULL::DOUBLE PRECISION AS prev__AMT_GOODS_PRICE__mean,
        NULL::DOUBLE PRECISION AS prev__AMT_GOODS_PRICE__std,
        NULL::DOUBLE PRECISION AS prev__HOUR_APPR_PROCESS_START__mean,
        NULL::DOUBLE PRECISION AS prev__HOUR_APPR_PROCESS_START__std,
        NULL::DOUBLE PRECISION AS prev__NFLAG_LAST_APPL_IN_DAY__mean,
        NULL::DOUBLE PRECISION AS prev__NFLAG_LAST_APPL_IN_DAY__std,
        NULL::DOUBLE PRECISION AS prev__RATE_DOWN_PAYMENT__mean,
        NULL::DOUBLE PRECISION AS prev__RATE_DOWN_PAYMENT__std,
        NULL::DOUBLE PRECISION AS prev__DAYS_DECISION__mean,
        NULL::DOUBLE PRECISION AS prev__DAYS_DECISION__std,
        NULL::DOUBLE PRECISION AS prev__SELLERPLACE_AREA__mean,
        NULL::DOUBLE PRECISION AS prev__SELLERPLACE_AREA__std,
        NULL::DOUBLE PRECISION AS prev__CNT_PAYMENT__mean,
        NULL::DOUBLE PRECISION AS prev__CNT_PAYMENT__std,
        NULL::DOUBLE PRECISION AS prev__DAYS_FIRST_DRAWING__mean,
        NULL::DOUBLE PRECISION AS prev__DAYS_FIRST_DUE__mean,
        NULL::DOUBLE PRECISION AS prev__DAYS_FIRST_DUE__std,
        NULL::DOUBLE PRECISION AS prev__DAYS_LAST_DUE_1ST_VERSION__mean,
        NULL::DOUBLE PRECISION AS prev__DAYS_LAST_DUE_1ST_VERSION__std,
        NULL::DOUBLE PRECISION AS prev__DAYS_LAST_DUE__mean,
        NULL::DOUBLE PRECISION AS prev__DAYS_LAST_DUE__std,
        NULL::DOUBLE PRECISION AS prev__DAYS_TERMINATION__mean,
        NULL::DOUBLE PRECISION AS prev__DAYS_TERMINATION__std,
        NULL::DOUBLE PRECISION AS prev__NFLAG_INSURED_ON_APPROVAL__mean,
        NULL::DOUBLE PRECISION AS prev__NFLAG_INSURED_ON_APPROVAL__std,
        NULL::DOUBLE PRECISION AS prev__PREV_CREDIT_APPLICATION_RATIO__mean,
        NULL::DOUBLE PRECISION AS prev__PREV_CREDIT_APPLICATION_RATIO__std,
        NULL::DOUBLE PRECISION AS prev__PREV_IS_APPROVED__mean,
        NULL::DOUBLE PRECISION AS prev__PREV_IS_APPROVED__std,
        NULL::DOUBLE PRECISION AS prev__PREV_IS_REFUSED__mean,
        NULL::DOUBLE PRECISION AS prev__PREV_IS_REFUSED__std,
        NULL::DOUBLE PRECISION AS prev__PREV_DAYS_DECISION_AGE__mean,
        NULL::DOUBLE PRECISION AS prev__PREV_DAYS_DECISION_AGE__std,
        NULL::DOUBLE PRECISION AS prev__PREV_CREDIT_DURATION__mean,
        NULL::DOUBLE PRECISION AS prev__PREV_CREDIT_DURATION__std,
        NULL::DOUBLE PRECISION AS prev__pos__MONTHS_BALANCE__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__MONTHS_BALANCE__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__MONTHS_BALANCE__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__MONTHS_BALANCE__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT_FUTURE__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT_FUTURE__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT_FUTURE__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__CNT_INSTALMENT_FUTURE__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD__max__mean,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD__max__std,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD_DEF__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD_DEF__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD_DEF__max__mean,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD_DEF__max__std,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD_DEF__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__SK_DPD_DEF__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__POS_REMAIN_RATIO__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__POS_REMAIN_RATIO__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__POS_REMAIN_RATIO__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__POS_REMAIN_RATIO__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__POS_DPD_POS__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__POS_DPD_POS__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__POS_DPD_POS__max__mean,
        NULL::DOUBLE PRECISION AS prev__pos__POS_DPD_POS__max__std,
        NULL::DOUBLE PRECISION AS prev__pos__POS_DPD_POS__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__POS_DPD_POS__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__POS_IS_ACTIVE__mean__mean,
        NULL::DOUBLE PRECISION AS prev__pos__POS_IS_ACTIVE__mean__std,
        NULL::DOUBLE PRECISION AS prev__pos__POS_IS_ACTIVE__std__mean,
        NULL::DOUBLE PRECISION AS prev__pos__POS_IS_ACTIVE__std__std,
        NULL::DOUBLE PRECISION AS prev__pos__count_rows__mean,
        NULL::DOUBLE PRECISION AS prev__pos__count_rows__std,
        NULL::DOUBLE PRECISION AS prev__cc__MONTHS_BALANCE__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__MONTHS_BALANCE__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_BALANCE__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_BALANCE__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_CREDIT_LIMIT_ACTUAL__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_CREDIT_LIMIT_ACTUAL__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_ATM_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_ATM_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_OTHER_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_OTHER_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_POS_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_DRAWINGS_POS_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_INST_MIN_REGULARITY__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_INST_MIN_REGULARITY__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_PAYMENT_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_PAYMENT_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_PAYMENT_TOTAL_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_PAYMENT_TOTAL_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_RECEIVABLE_PRINCIPAL__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_RECEIVABLE_PRINCIPAL__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_RECIVABLE__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_RECIVABLE__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_TOTAL_RECEIVABLE__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__AMT_TOTAL_RECEIVABLE__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_ATM_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_ATM_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_OTHER_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_OTHER_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_POS_CURRENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_DRAWINGS_POS_CURRENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_INSTALMENT_MATURE_CUM__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CNT_INSTALMENT_MATURE_CUM__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__SK_DPD__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__SK_DPD__max__mean,
        NULL::DOUBLE PRECISION AS prev__cc__SK_DPD__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__SK_DPD_DEF__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__SK_DPD_DEF__max__mean,
        NULL::DOUBLE PRECISION AS prev__cc__SK_DPD_DEF__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_UTILIZATION_RATIO__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_UTILIZATION_RATIO__max__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_UTILIZATION_RATIO__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_PAYMENT_MIN_RATIO__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_PAYMENT_MIN_RATIO__max__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_PAYMENT_MIN_RATIO__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_PAYMENT_BALANCE_RATIO__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_PAYMENT_BALANCE_RATIO__max__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_PAYMENT_BALANCE_RATIO__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_DPD_POS__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_DPD_POS__max__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_DPD_POS__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_RECEIVABLE_RATIO__mean__mean,
        NULL::DOUBLE PRECISION AS prev__cc__CC_RECEIVABLE_RATIO__std__mean,
        NULL::DOUBLE PRECISION AS prev__cc__count_rows__mean,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_VERSION__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_VERSION__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_VERSION__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_VERSION__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_NUMBER__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_NUMBER__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_NUMBER__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__NUM_INSTALMENT_NUMBER__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_INSTALMENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_INSTALMENT__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_INSTALMENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_INSTALMENT__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_ENTRY_PAYMENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_ENTRY_PAYMENT__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_ENTRY_PAYMENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__DAYS_ENTRY_PAYMENT__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_INSTALMENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_INSTALMENT__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_INSTALMENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_INSTALMENT__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_PAYMENT__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_PAYMENT__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_PAYMENT__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__AMT_PAYMENT__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__DPD_POS__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__DPD_POS__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__DPD_POS__max__mean,
        NULL::DOUBLE PRECISION AS prev__inst__DPD_POS__max__std,
        NULL::DOUBLE PRECISION AS prev__inst__DPD_POS__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__DPD_POS__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__SEVERE_LATE_30__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__SEVERE_LATE_30__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__SEVERE_LATE_30__max__mean,
        NULL::DOUBLE PRECISION AS prev__inst__SEVERE_LATE_30__max__std,
        NULL::DOUBLE PRECISION AS prev__inst__SEVERE_LATE_30__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__SEVERE_LATE_30__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__PAY_RATIO__mean__mean,
        NULL::DOUBLE PRECISION AS prev__inst__PAY_RATIO__mean__std,
        NULL::DOUBLE PRECISION AS prev__inst__PAY_RATIO__std__mean,
        NULL::DOUBLE PRECISION AS prev__inst__PAY_RATIO__std__std,
        NULL::DOUBLE PRECISION AS prev__inst__count_rows__mean,
        NULL::DOUBLE PRECISION AS prev__inst__count_rows__std
) t
WHERE 1 = 0;
"""


# =============================================================================
# Index
# =============================================================================

CREATE_BB_AGG_TMP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_bb_agg_tmp_sk_id_bureau
ON bb_agg_tmp ("SK_ID_BUREAU");
"""

CREATE_BUREAU_AGG_CURR_TMP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_bureau_agg_curr_tmp_sk_id_curr
ON bureau_agg_curr_tmp ("SK_ID_CURR");
"""

CREATE_POS_AGG_TMP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_pos_agg_tmp_sk_id_prev
ON pos_agg_tmp ("SK_ID_PREV");
"""

CREATE_CC_AGG_TMP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_cc_agg_tmp_sk_id_prev
ON cc_agg_tmp ("SK_ID_PREV");
"""

CREATE_INST_AGG_TMP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_inst_agg_tmp_sk_id_prev
ON inst_agg_tmp ("SK_ID_PREV");
"""

CREATE_PREV_AGG_CURR_TMP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prev_agg_curr_tmp_sk_id_curr
ON prev_agg_curr_tmp ("SK_ID_CURR");
"""


# =============================================================================
# Fonction principale
# =============================================================================

def create_temp_feature_tables(engine) -> None:
    """
    Crée la structure des tables temporaires et les index associés.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        print("Suppression des anciennes tables temporaires...")
        connection.execute(text(DROP_TEMP_TABLES_SQL))

        print("Création des tables de batch...")
        connection.execute(text(CREATE_TMP_BATCH_SK_ID_BUREAU_SQL))
        connection.execute(text(CREATE_TMP_BATCH_SK_ID_PREV_SQL))
        connection.execute(text(CREATE_TMP_BATCH_SK_ID_CURR_SQL))

        print("Création des tables temporaires vides...")
        connection.execute(text(CREATE_BB_AGG_TMP_EMPTY_SQL))
        connection.execute(text(CREATE_BUREAU_AGG_CURR_TMP_EMPTY_SQL))
        connection.execute(text(CREATE_POS_AGG_TMP_EMPTY_SQL))
        connection.execute(text(CREATE_CC_AGG_TMP_EMPTY_SQL))
        connection.execute(text(CREATE_INST_AGG_TMP_EMPTY_SQL))
        connection.execute(text(CREATE_PREV_AGG_CURR_TMP_EMPTY_SQL))

        print("Création des index...")
        connection.execute(text(CREATE_BB_AGG_TMP_INDEX_SQL))
        connection.execute(text(CREATE_BUREAU_AGG_CURR_TMP_INDEX_SQL))
        connection.execute(text(CREATE_POS_AGG_TMP_INDEX_SQL))
        connection.execute(text(CREATE_CC_AGG_TMP_INDEX_SQL))
        connection.execute(text(CREATE_INST_AGG_TMP_INDEX_SQL))
        connection.execute(text(CREATE_PREV_AGG_CURR_TMP_INDEX_SQL))

    print("Structure des tables temporaires créée avec succès.")


def main() -> None:
    """
    Point d'entrée du script.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    create_temp_feature_tables(engine)

    print("Création des tables temporaires terminée.")


if __name__ == "__main__":
    main()