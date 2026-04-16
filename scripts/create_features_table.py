"""
Script de création de la table de features brute au niveau client.

Ce module construit la table intermédiaire `features_client_test_raw`
à partir des tables temporaires déjà agrégées du projet Home Credit.

Objectif
--------
Créer une table de features agrégées fidèle à la logique du notebook,
avec une ligne par client (`SK_ID_CURR`), avant les étapes finales
de nettoyage et de suppression de colonnes.

Tables temporaires attendues
----------------------------
- bureau_agg_curr_tmp
- prev_agg_curr_tmp

Notes
-----
- La table est reconstruite à chaque exécution.
- Les données sont insérées par lots de 500 lignes.
- Cette version suppose que les tables temporaires ont déjà été créées.
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
# Paramètres
# =============================================================================

BATCH_SIZE = 500


# =============================================================================
# SQL de création de la table vide
# =============================================================================

DROP_FEATURES_CLIENT_TEST_RAW_SQL = """
DROP TABLE IF EXISTS features_client_test_raw;
"""


CREATE_EMPTY_FEATURES_CLIENT_TEST_RAW_SQL = """
CREATE TABLE features_client_test_raw AS
SELECT
    a.*,

    -- bureau_agg_curr_tmp sans SK_ID_CURR
    b.bureau__count_rows,
    b.bureau__nunique_SK_ID_BUREAU,
    b.bureau__DAYS_CREDIT__mean,
    b.bureau__DAYS_CREDIT__std,
    b.bureau__CREDIT_DAY_OVERDUE__mean,
    b.bureau__CREDIT_DAY_OVERDUE__std,
    b.bureau__DAYS_CREDIT_ENDDATE__mean,
    b.bureau__DAYS_CREDIT_ENDDATE__std,
    b.bureau__DAYS_ENDDATE_FACT__mean,
    b.bureau__DAYS_ENDDATE_FACT__std,
    b.bureau__AMT_CREDIT_MAX_OVERDUE__mean,
    b.bureau__AMT_CREDIT_MAX_OVERDUE__std,
    b.bureau__CNT_CREDIT_PROLONG__mean,
    b.bureau__CNT_CREDIT_PROLONG__std,
    b.bureau__AMT_CREDIT_SUM__mean,
    b.bureau__AMT_CREDIT_SUM__std,
    b.bureau__AMT_CREDIT_SUM_DEBT__mean,
    b.bureau__AMT_CREDIT_SUM_DEBT__std,
    b.bureau__AMT_CREDIT_SUM_LIMIT__mean,
    b.bureau__AMT_CREDIT_SUM_LIMIT__std,
    b.bureau__AMT_CREDIT_SUM_OVERDUE__mean,
    b.bureau__AMT_CREDIT_SUM_OVERDUE__std,
    b.bureau__DAYS_CREDIT_UPDATE__mean,
    b.bureau__DAYS_CREDIT_UPDATE__std,
    b.bureau__AMT_ANNUITY__mean,
    b.bureau__AMT_ANNUITY__std,
    b.bureau__DEBT_RATIO__mean,
    b.bureau__DEBT_RATIO__std,
    b.bureau__OVERDUE_RATIO__mean,
    b.bureau__OVERDUE_RATIO__std,
    b.bureau__IS_ACTIVE__mean,
    b.bureau__IS_ACTIVE__std,
    b.bureau__HAS_OVERDUE__mean,
    b.bureau__HAS_OVERDUE__std,
    b.bureau__CREDIT_AGE__mean,
    b.bureau__CREDIT_AGE__std,
    b.bureau__bb__MONTHS_BALANCE__mean__mean,
    b.bureau__bb__MONTHS_BALANCE__mean__std,
    b.bureau__bb__MONTHS_BALANCE__std__mean,
    b.bureau__bb__MONTHS_BALANCE__std__std,
    b.bureau__bb__count_rows__mean,
    b.bureau__bb__count_rows__std,
    b.bureau__bb__recent_max_dpd__mean,
    b.bureau__bb__recent_max_dpd__std,
    b.bureau__bb__months_late_ratio__mean,
    b.bureau__bb__months_late_ratio__std,
    b.bureau__bb__late_severity_sum__mean,
    b.bureau__bb__late_severity_sum__std,

    -- prev_agg_curr_tmp sans SK_ID_CURR
    p.prev__count_rows,
    p.prev__nunique_SK_ID_PREV,
    p.prev__AMT_ANNUITY__mean,
    p.prev__AMT_ANNUITY__std,
    p.prev__AMT_APPLICATION__mean,
    p.prev__AMT_APPLICATION__std,
    p.prev__AMT_CREDIT__mean,
    p.prev__AMT_CREDIT__std,
    p.prev__AMT_DOWN_PAYMENT__mean,
    p.prev__AMT_DOWN_PAYMENT__std,
    p.prev__AMT_GOODS_PRICE__mean,
    p.prev__AMT_GOODS_PRICE__std,
    p.prev__HOUR_APPR_PROCESS_START__mean,
    p.prev__HOUR_APPR_PROCESS_START__std,
    p.prev__NFLAG_LAST_APPL_IN_DAY__mean,
    p.prev__NFLAG_LAST_APPL_IN_DAY__std,
    p.prev__RATE_DOWN_PAYMENT__mean,
    p.prev__RATE_DOWN_PAYMENT__std,
    p.prev__DAYS_DECISION__mean,
    p.prev__DAYS_DECISION__std,
    p.prev__SELLERPLACE_AREA__mean,
    p.prev__SELLERPLACE_AREA__std,
    p.prev__CNT_PAYMENT__mean,
    p.prev__CNT_PAYMENT__std,
    p.prev__DAYS_FIRST_DRAWING__mean,
    p.prev__DAYS_FIRST_DUE__mean,
    p.prev__DAYS_FIRST_DUE__std,
    p.prev__DAYS_LAST_DUE_1ST_VERSION__mean,
    p.prev__DAYS_LAST_DUE_1ST_VERSION__std,
    p.prev__DAYS_LAST_DUE__mean,
    p.prev__DAYS_LAST_DUE__std,
    p.prev__DAYS_TERMINATION__mean,
    p.prev__DAYS_TERMINATION__std,
    p.prev__NFLAG_INSURED_ON_APPROVAL__mean,
    p.prev__NFLAG_INSURED_ON_APPROVAL__std,
    p.prev__PREV_CREDIT_APPLICATION_RATIO__mean,
    p.prev__PREV_CREDIT_APPLICATION_RATIO__std,
    p.prev__PREV_IS_APPROVED__mean,
    p.prev__PREV_IS_APPROVED__std,
    p.prev__PREV_IS_REFUSED__mean,
    p.prev__PREV_IS_REFUSED__std,
    p.prev__PREV_DAYS_DECISION_AGE__mean,
    p.prev__PREV_DAYS_DECISION_AGE__std,
    p.prev__PREV_CREDIT_DURATION__mean,
    p.prev__PREV_CREDIT_DURATION__std,
    p.prev__pos__MONTHS_BALANCE__mean__mean,
    p.prev__pos__MONTHS_BALANCE__mean__std,
    p.prev__pos__MONTHS_BALANCE__std__mean,
    p.prev__pos__MONTHS_BALANCE__std__std,
    p.prev__pos__CNT_INSTALMENT__mean__mean,
    p.prev__pos__CNT_INSTALMENT__mean__std,
    p.prev__pos__CNT_INSTALMENT__std__mean,
    p.prev__pos__CNT_INSTALMENT__std__std,
    p.prev__pos__CNT_INSTALMENT_FUTURE__mean__mean,
    p.prev__pos__CNT_INSTALMENT_FUTURE__mean__std,
    p.prev__pos__CNT_INSTALMENT_FUTURE__std__mean,
    p.prev__pos__CNT_INSTALMENT_FUTURE__std__std,
    p.prev__pos__SK_DPD__mean__mean,
    p.prev__pos__SK_DPD__mean__std,
    p.prev__pos__SK_DPD__max__mean,
    p.prev__pos__SK_DPD__max__std,
    p.prev__pos__SK_DPD__std__mean,
    p.prev__pos__SK_DPD__std__std,
    p.prev__pos__SK_DPD_DEF__mean__mean,
    p.prev__pos__SK_DPD_DEF__mean__std,
    p.prev__pos__SK_DPD_DEF__max__mean,
    p.prev__pos__SK_DPD_DEF__max__std,
    p.prev__pos__SK_DPD_DEF__std__mean,
    p.prev__pos__SK_DPD_DEF__std__std,
    p.prev__pos__POS_REMAIN_RATIO__mean__mean,
    p.prev__pos__POS_REMAIN_RATIO__mean__std,
    p.prev__pos__POS_REMAIN_RATIO__std__mean,
    p.prev__pos__POS_REMAIN_RATIO__std__std,
    p.prev__pos__POS_DPD_POS__mean__mean,
    p.prev__pos__POS_DPD_POS__mean__std,
    p.prev__pos__POS_DPD_POS__max__mean,
    p.prev__pos__POS_DPD_POS__max__std,
    p.prev__pos__POS_DPD_POS__std__mean,
    p.prev__pos__POS_DPD_POS__std__std,
    p.prev__pos__POS_IS_ACTIVE__mean__mean,
    p.prev__pos__POS_IS_ACTIVE__mean__std,
    p.prev__pos__POS_IS_ACTIVE__std__mean,
    p.prev__pos__POS_IS_ACTIVE__std__std,
    p.prev__pos__count_rows__mean,
    p.prev__pos__count_rows__std,
    p.prev__cc__MONTHS_BALANCE__mean__mean,
    p.prev__cc__MONTHS_BALANCE__std__mean,
    p.prev__cc__AMT_BALANCE__mean__mean,
    p.prev__cc__AMT_BALANCE__std__mean,
    p.prev__cc__AMT_CREDIT_LIMIT_ACTUAL__mean__mean,
    p.prev__cc__AMT_CREDIT_LIMIT_ACTUAL__std__mean,
    p.prev__cc__AMT_DRAWINGS_ATM_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_ATM_CURRENT__std__mean,
    p.prev__cc__AMT_DRAWINGS_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_CURRENT__std__mean,
    p.prev__cc__AMT_DRAWINGS_OTHER_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_OTHER_CURRENT__std__mean,
    p.prev__cc__AMT_DRAWINGS_POS_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_POS_CURRENT__std__mean,
    p.prev__cc__AMT_INST_MIN_REGULARITY__mean__mean,
    p.prev__cc__AMT_INST_MIN_REGULARITY__std__mean,
    p.prev__cc__AMT_PAYMENT_CURRENT__mean__mean,
    p.prev__cc__AMT_PAYMENT_CURRENT__std__mean,
    p.prev__cc__AMT_PAYMENT_TOTAL_CURRENT__mean__mean,
    p.prev__cc__AMT_PAYMENT_TOTAL_CURRENT__std__mean,
    p.prev__cc__AMT_RECEIVABLE_PRINCIPAL__mean__mean,
    p.prev__cc__AMT_RECEIVABLE_PRINCIPAL__std__mean,
    p.prev__cc__AMT_RECIVABLE__mean__mean,
    p.prev__cc__AMT_RECIVABLE__std__mean,
    p.prev__cc__AMT_TOTAL_RECEIVABLE__mean__mean,
    p.prev__cc__AMT_TOTAL_RECEIVABLE__std__mean,
    p.prev__cc__CNT_DRAWINGS_ATM_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_ATM_CURRENT__std__mean,
    p.prev__cc__CNT_DRAWINGS_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_CURRENT__std__mean,
    p.prev__cc__CNT_DRAWINGS_OTHER_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_OTHER_CURRENT__std__mean,
    p.prev__cc__CNT_DRAWINGS_POS_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_POS_CURRENT__std__mean,
    p.prev__cc__CNT_INSTALMENT_MATURE_CUM__mean__mean,
    p.prev__cc__CNT_INSTALMENT_MATURE_CUM__std__mean,
    p.prev__cc__SK_DPD__mean__mean,
    p.prev__cc__SK_DPD__max__mean,
    p.prev__cc__SK_DPD__std__mean,
    p.prev__cc__SK_DPD_DEF__mean__mean,
    p.prev__cc__SK_DPD_DEF__max__mean,
    p.prev__cc__SK_DPD_DEF__std__mean,
    p.prev__cc__CC_UTILIZATION_RATIO__mean__mean,
    p.prev__cc__CC_UTILIZATION_RATIO__max__mean,
    p.prev__cc__CC_UTILIZATION_RATIO__std__mean,
    p.prev__cc__CC_PAYMENT_MIN_RATIO__mean__mean,
    p.prev__cc__CC_PAYMENT_MIN_RATIO__max__mean,
    p.prev__cc__CC_PAYMENT_MIN_RATIO__std__mean,
    p.prev__cc__CC_PAYMENT_BALANCE_RATIO__mean__mean,
    p.prev__cc__CC_PAYMENT_BALANCE_RATIO__max__mean,
    p.prev__cc__CC_PAYMENT_BALANCE_RATIO__std__mean,
    p.prev__cc__CC_DPD_POS__mean__mean,
    p.prev__cc__CC_DPD_POS__max__mean,
    p.prev__cc__CC_DPD_POS__std__mean,
    p.prev__cc__CC_RECEIVABLE_RATIO__mean__mean,
    p.prev__cc__CC_RECEIVABLE_RATIO__std__mean,
    p.prev__cc__count_rows__mean,
    p.prev__inst__NUM_INSTALMENT_VERSION__mean__mean,
    p.prev__inst__NUM_INSTALMENT_VERSION__mean__std,
    p.prev__inst__NUM_INSTALMENT_VERSION__std__mean,
    p.prev__inst__NUM_INSTALMENT_VERSION__std__std,
    p.prev__inst__NUM_INSTALMENT_NUMBER__mean__mean,
    p.prev__inst__NUM_INSTALMENT_NUMBER__mean__std,
    p.prev__inst__NUM_INSTALMENT_NUMBER__std__mean,
    p.prev__inst__NUM_INSTALMENT_NUMBER__std__std,
    p.prev__inst__DAYS_INSTALMENT__mean__mean,
    p.prev__inst__DAYS_INSTALMENT__mean__std,
    p.prev__inst__DAYS_INSTALMENT__std__mean,
    p.prev__inst__DAYS_INSTALMENT__std__std,
    p.prev__inst__DAYS_ENTRY_PAYMENT__mean__mean,
    p.prev__inst__DAYS_ENTRY_PAYMENT__mean__std,
    p.prev__inst__DAYS_ENTRY_PAYMENT__std__mean,
    p.prev__inst__DAYS_ENTRY_PAYMENT__std__std,
    p.prev__inst__AMT_INSTALMENT__mean__mean,
    p.prev__inst__AMT_INSTALMENT__mean__std,
    p.prev__inst__AMT_INSTALMENT__std__mean,
    p.prev__inst__AMT_INSTALMENT__std__std,
    p.prev__inst__AMT_PAYMENT__mean__mean,
    p.prev__inst__AMT_PAYMENT__mean__std,
    p.prev__inst__AMT_PAYMENT__std__mean,
    p.prev__inst__AMT_PAYMENT__std__std,
    p.prev__inst__DPD_POS__mean__mean,
    p.prev__inst__DPD_POS__mean__std,
    p.prev__inst__DPD_POS__max__mean,
    p.prev__inst__DPD_POS__max__std,
    p.prev__inst__DPD_POS__std__mean,
    p.prev__inst__DPD_POS__std__std,
    p.prev__inst__SEVERE_LATE_30__mean__mean,
    p.prev__inst__SEVERE_LATE_30__mean__std,
    p.prev__inst__SEVERE_LATE_30__max__mean,
    p.prev__inst__SEVERE_LATE_30__max__std,
    p.prev__inst__SEVERE_LATE_30__std__mean,
    p.prev__inst__SEVERE_LATE_30__std__std,
    p.prev__inst__PAY_RATIO__mean__mean,
    p.prev__inst__PAY_RATIO__mean__std,
    p.prev__inst__PAY_RATIO__std__mean,
    p.prev__inst__PAY_RATIO__std__std,
    p.prev__inst__count_rows__mean,
    p.prev__inst__count_rows__std

FROM application_test a
LEFT JOIN bureau_agg_curr_tmp b
    ON a."SK_ID_CURR" = b."SK_ID_CURR"
LEFT JOIN prev_agg_curr_tmp p
    ON a."SK_ID_CURR" = p."SK_ID_CURR"
WHERE 1 = 0;
"""


# =============================================================================
# SQL d'insertion par batch
# =============================================================================

INSERT_FEATURES_CLIENT_TEST_RAW_BATCH_SQL = """
INSERT INTO features_client_test_raw
SELECT
    a.*,

    -- bureau_agg_curr_tmp sans SK_ID_CURR
    b.bureau__count_rows,
    b.bureau__nunique_SK_ID_BUREAU,
    b.bureau__DAYS_CREDIT__mean,
    b.bureau__DAYS_CREDIT__std,
    b.bureau__CREDIT_DAY_OVERDUE__mean,
    b.bureau__CREDIT_DAY_OVERDUE__std,
    b.bureau__DAYS_CREDIT_ENDDATE__mean,
    b.bureau__DAYS_CREDIT_ENDDATE__std,
    b.bureau__DAYS_ENDDATE_FACT__mean,
    b.bureau__DAYS_ENDDATE_FACT__std,
    b.bureau__AMT_CREDIT_MAX_OVERDUE__mean,
    b.bureau__AMT_CREDIT_MAX_OVERDUE__std,
    b.bureau__CNT_CREDIT_PROLONG__mean,
    b.bureau__CNT_CREDIT_PROLONG__std,
    b.bureau__AMT_CREDIT_SUM__mean,
    b.bureau__AMT_CREDIT_SUM__std,
    b.bureau__AMT_CREDIT_SUM_DEBT__mean,
    b.bureau__AMT_CREDIT_SUM_DEBT__std,
    b.bureau__AMT_CREDIT_SUM_LIMIT__mean,
    b.bureau__AMT_CREDIT_SUM_LIMIT__std,
    b.bureau__AMT_CREDIT_SUM_OVERDUE__mean,
    b.bureau__AMT_CREDIT_SUM_OVERDUE__std,
    b.bureau__DAYS_CREDIT_UPDATE__mean,
    b.bureau__DAYS_CREDIT_UPDATE__std,
    b.bureau__AMT_ANNUITY__mean,
    b.bureau__AMT_ANNUITY__std,
    b.bureau__DEBT_RATIO__mean,
    b.bureau__DEBT_RATIO__std,
    b.bureau__OVERDUE_RATIO__mean,
    b.bureau__OVERDUE_RATIO__std,
    b.bureau__IS_ACTIVE__mean,
    b.bureau__IS_ACTIVE__std,
    b.bureau__HAS_OVERDUE__mean,
    b.bureau__HAS_OVERDUE__std,
    b.bureau__CREDIT_AGE__mean,
    b.bureau__CREDIT_AGE__std,
    b.bureau__bb__MONTHS_BALANCE__mean__mean,
    b.bureau__bb__MONTHS_BALANCE__mean__std,
    b.bureau__bb__MONTHS_BALANCE__std__mean,
    b.bureau__bb__MONTHS_BALANCE__std__std,
    b.bureau__bb__count_rows__mean,
    b.bureau__bb__count_rows__std,
    b.bureau__bb__recent_max_dpd__mean,
    b.bureau__bb__recent_max_dpd__std,
    b.bureau__bb__months_late_ratio__mean,
    b.bureau__bb__months_late_ratio__std,
    b.bureau__bb__late_severity_sum__mean,
    b.bureau__bb__late_severity_sum__std,

    -- prev_agg_curr_tmp sans SK_ID_CURR
    p.prev__count_rows,
    p.prev__nunique_SK_ID_PREV,
    p.prev__AMT_ANNUITY__mean,
    p.prev__AMT_ANNUITY__std,
    p.prev__AMT_APPLICATION__mean,
    p.prev__AMT_APPLICATION__std,
    p.prev__AMT_CREDIT__mean,
    p.prev__AMT_CREDIT__std,
    p.prev__AMT_DOWN_PAYMENT__mean,
    p.prev__AMT_DOWN_PAYMENT__std,
    p.prev__AMT_GOODS_PRICE__mean,
    p.prev__AMT_GOODS_PRICE__std,
    p.prev__HOUR_APPR_PROCESS_START__mean,
    p.prev__HOUR_APPR_PROCESS_START__std,
    p.prev__NFLAG_LAST_APPL_IN_DAY__mean,
    p.prev__NFLAG_LAST_APPL_IN_DAY__std,
    p.prev__RATE_DOWN_PAYMENT__mean,
    p.prev__RATE_DOWN_PAYMENT__std,
    p.prev__DAYS_DECISION__mean,
    p.prev__DAYS_DECISION__std,
    p.prev__SELLERPLACE_AREA__mean,
    p.prev__SELLERPLACE_AREA__std,
    p.prev__CNT_PAYMENT__mean,
    p.prev__CNT_PAYMENT__std,
    p.prev__DAYS_FIRST_DRAWING__mean,
    p.prev__DAYS_FIRST_DUE__mean,
    p.prev__DAYS_FIRST_DUE__std,
    p.prev__DAYS_LAST_DUE_1ST_VERSION__mean,
    p.prev__DAYS_LAST_DUE_1ST_VERSION__std,
    p.prev__DAYS_LAST_DUE__mean,
    p.prev__DAYS_LAST_DUE__std,
    p.prev__DAYS_TERMINATION__mean,
    p.prev__DAYS_TERMINATION__std,
    p.prev__NFLAG_INSURED_ON_APPROVAL__mean,
    p.prev__NFLAG_INSURED_ON_APPROVAL__std,
    p.prev__PREV_CREDIT_APPLICATION_RATIO__mean,
    p.prev__PREV_CREDIT_APPLICATION_RATIO__std,
    p.prev__PREV_IS_APPROVED__mean,
    p.prev__PREV_IS_APPROVED__std,
    p.prev__PREV_IS_REFUSED__mean,
    p.prev__PREV_IS_REFUSED__std,
    p.prev__PREV_DAYS_DECISION_AGE__mean,
    p.prev__PREV_DAYS_DECISION_AGE__std,
    p.prev__PREV_CREDIT_DURATION__mean,
    p.prev__PREV_CREDIT_DURATION__std,
    p.prev__pos__MONTHS_BALANCE__mean__mean,
    p.prev__pos__MONTHS_BALANCE__mean__std,
    p.prev__pos__MONTHS_BALANCE__std__mean,
    p.prev__pos__MONTHS_BALANCE__std__std,
    p.prev__pos__CNT_INSTALMENT__mean__mean,
    p.prev__pos__CNT_INSTALMENT__mean__std,
    p.prev__pos__CNT_INSTALMENT__std__mean,
    p.prev__pos__CNT_INSTALMENT__std__std,
    p.prev__pos__CNT_INSTALMENT_FUTURE__mean__mean,
    p.prev__pos__CNT_INSTALMENT_FUTURE__mean__std,
    p.prev__pos__CNT_INSTALMENT_FUTURE__std__mean,
    p.prev__pos__CNT_INSTALMENT_FUTURE__std__std,
    p.prev__pos__SK_DPD__mean__mean,
    p.prev__pos__SK_DPD__mean__std,
    p.prev__pos__SK_DPD__max__mean,
    p.prev__pos__SK_DPD__max__std,
    p.prev__pos__SK_DPD__std__mean,
    p.prev__pos__SK_DPD__std__std,
    p.prev__pos__SK_DPD_DEF__mean__mean,
    p.prev__pos__SK_DPD_DEF__mean__std,
    p.prev__pos__SK_DPD_DEF__max__mean,
    p.prev__pos__SK_DPD_DEF__max__std,
    p.prev__pos__SK_DPD_DEF__std__mean,
    p.prev__pos__SK_DPD_DEF__std__std,
    p.prev__pos__POS_REMAIN_RATIO__mean__mean,
    p.prev__pos__POS_REMAIN_RATIO__mean__std,
    p.prev__pos__POS_REMAIN_RATIO__std__mean,
    p.prev__pos__POS_REMAIN_RATIO__std__std,
    p.prev__pos__POS_DPD_POS__mean__mean,
    p.prev__pos__POS_DPD_POS__mean__std,
    p.prev__pos__POS_DPD_POS__max__mean,
    p.prev__pos__POS_DPD_POS__max__std,
    p.prev__pos__POS_DPD_POS__std__mean,
    p.prev__pos__POS_DPD_POS__std__std,
    p.prev__pos__POS_IS_ACTIVE__mean__mean,
    p.prev__pos__POS_IS_ACTIVE__mean__std,
    p.prev__pos__POS_IS_ACTIVE__std__mean,
    p.prev__pos__POS_IS_ACTIVE__std__std,
    p.prev__pos__count_rows__mean,
    p.prev__pos__count_rows__std,
    p.prev__cc__MONTHS_BALANCE__mean__mean,
    p.prev__cc__MONTHS_BALANCE__std__mean,
    p.prev__cc__AMT_BALANCE__mean__mean,
    p.prev__cc__AMT_BALANCE__std__mean,
    p.prev__cc__AMT_CREDIT_LIMIT_ACTUAL__mean__mean,
    p.prev__cc__AMT_CREDIT_LIMIT_ACTUAL__std__mean,
    p.prev__cc__AMT_DRAWINGS_ATM_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_ATM_CURRENT__std__mean,
    p.prev__cc__AMT_DRAWINGS_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_CURRENT__std__mean,
    p.prev__cc__AMT_DRAWINGS_OTHER_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_OTHER_CURRENT__std__mean,
    p.prev__cc__AMT_DRAWINGS_POS_CURRENT__mean__mean,
    p.prev__cc__AMT_DRAWINGS_POS_CURRENT__std__mean,
    p.prev__cc__AMT_INST_MIN_REGULARITY__mean__mean,
    p.prev__cc__AMT_INST_MIN_REGULARITY__std__mean,
    p.prev__cc__AMT_PAYMENT_CURRENT__mean__mean,
    p.prev__cc__AMT_PAYMENT_CURRENT__std__mean,
    p.prev__cc__AMT_PAYMENT_TOTAL_CURRENT__mean__mean,
    p.prev__cc__AMT_PAYMENT_TOTAL_CURRENT__std__mean,
    p.prev__cc__AMT_RECEIVABLE_PRINCIPAL__mean__mean,
    p.prev__cc__AMT_RECEIVABLE_PRINCIPAL__std__mean,
    p.prev__cc__AMT_RECIVABLE__mean__mean,
    p.prev__cc__AMT_RECIVABLE__std__mean,
    p.prev__cc__AMT_TOTAL_RECEIVABLE__mean__mean,
    p.prev__cc__AMT_TOTAL_RECEIVABLE__std__mean,
    p.prev__cc__CNT_DRAWINGS_ATM_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_ATM_CURRENT__std__mean,
    p.prev__cc__CNT_DRAWINGS_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_CURRENT__std__mean,
    p.prev__cc__CNT_DRAWINGS_OTHER_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_OTHER_CURRENT__std__mean,
    p.prev__cc__CNT_DRAWINGS_POS_CURRENT__mean__mean,
    p.prev__cc__CNT_DRAWINGS_POS_CURRENT__std__mean,
    p.prev__cc__CNT_INSTALMENT_MATURE_CUM__mean__mean,
    p.prev__cc__CNT_INSTALMENT_MATURE_CUM__std__mean,
    p.prev__cc__SK_DPD__mean__mean,
    p.prev__cc__SK_DPD__max__mean,
    p.prev__cc__SK_DPD__std__mean,
    p.prev__cc__SK_DPD_DEF__mean__mean,
    p.prev__cc__SK_DPD_DEF__max__mean,
    p.prev__cc__SK_DPD_DEF__std__mean,
    p.prev__cc__CC_UTILIZATION_RATIO__mean__mean,
    p.prev__cc__CC_UTILIZATION_RATIO__max__mean,
    p.prev__cc__CC_UTILIZATION_RATIO__std__mean,
    p.prev__cc__CC_PAYMENT_MIN_RATIO__mean__mean,
    p.prev__cc__CC_PAYMENT_MIN_RATIO__max__mean,
    p.prev__cc__CC_PAYMENT_MIN_RATIO__std__mean,
    p.prev__cc__CC_PAYMENT_BALANCE_RATIO__mean__mean,
    p.prev__cc__CC_PAYMENT_BALANCE_RATIO__max__mean,
    p.prev__cc__CC_PAYMENT_BALANCE_RATIO__std__mean,
    p.prev__cc__CC_DPD_POS__mean__mean,
    p.prev__cc__CC_DPD_POS__max__mean,
    p.prev__cc__CC_DPD_POS__std__mean,
    p.prev__cc__CC_RECEIVABLE_RATIO__mean__mean,
    p.prev__cc__CC_RECEIVABLE_RATIO__std__mean,
    p.prev__cc__count_rows__mean,
    p.prev__inst__NUM_INSTALMENT_VERSION__mean__mean,
    p.prev__inst__NUM_INSTALMENT_VERSION__mean__std,
    p.prev__inst__NUM_INSTALMENT_VERSION__std__mean,
    p.prev__inst__NUM_INSTALMENT_VERSION__std__std,
    p.prev__inst__NUM_INSTALMENT_NUMBER__mean__mean,
    p.prev__inst__NUM_INSTALMENT_NUMBER__mean__std,
    p.prev__inst__NUM_INSTALMENT_NUMBER__std__mean,
    p.prev__inst__NUM_INSTALMENT_NUMBER__std__std,
    p.prev__inst__DAYS_INSTALMENT__mean__mean,
    p.prev__inst__DAYS_INSTALMENT__mean__std,
    p.prev__inst__DAYS_INSTALMENT__std__mean,
    p.prev__inst__DAYS_INSTALMENT__std__std,
    p.prev__inst__DAYS_ENTRY_PAYMENT__mean__mean,
    p.prev__inst__DAYS_ENTRY_PAYMENT__mean__std,
    p.prev__inst__DAYS_ENTRY_PAYMENT__std__mean,
    p.prev__inst__DAYS_ENTRY_PAYMENT__std__std,
    p.prev__inst__AMT_INSTALMENT__mean__mean,
    p.prev__inst__AMT_INSTALMENT__mean__std,
    p.prev__inst__AMT_INSTALMENT__std__mean,
    p.prev__inst__AMT_INSTALMENT__std__std,
    p.prev__inst__AMT_PAYMENT__mean__mean,
    p.prev__inst__AMT_PAYMENT__mean__std,
    p.prev__inst__AMT_PAYMENT__std__mean,
    p.prev__inst__AMT_PAYMENT__std__std,
    p.prev__inst__DPD_POS__mean__mean,
    p.prev__inst__DPD_POS__mean__std,
    p.prev__inst__DPD_POS__max__mean,
    p.prev__inst__DPD_POS__max__std,
    p.prev__inst__DPD_POS__std__mean,
    p.prev__inst__DPD_POS__std__std,
    p.prev__inst__SEVERE_LATE_30__mean__mean,
    p.prev__inst__SEVERE_LATE_30__mean__std,
    p.prev__inst__SEVERE_LATE_30__max__mean,
    p.prev__inst__SEVERE_LATE_30__max__std,
    p.prev__inst__SEVERE_LATE_30__std__mean,
    p.prev__inst__SEVERE_LATE_30__std__std,
    p.prev__inst__PAY_RATIO__mean__mean,
    p.prev__inst__PAY_RATIO__mean__std,
    p.prev__inst__PAY_RATIO__std__mean,
    p.prev__inst__PAY_RATIO__std__std,
    p.prev__inst__count_rows__mean,
    p.prev__inst__count_rows__std

FROM (
    SELECT *
    FROM application_test
    ORDER BY "SK_ID_CURR"
    LIMIT :batch_size OFFSET :offset
) a
LEFT JOIN bureau_agg_curr_tmp b
    ON a."SK_ID_CURR" = b."SK_ID_CURR"
LEFT JOIN prev_agg_curr_tmp p
    ON a."SK_ID_CURR" = p."SK_ID_CURR";
"""


CREATE_FEATURES_CLIENT_TEST_RAW_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_features_client_test_raw_sk_id_curr
ON features_client_test_raw ("SK_ID_CURR");
"""


COUNT_APPLICATION_TEST_SQL = """
SELECT COUNT(*) AS total_rows
FROM application_test;
"""


CHECK_TEMP_TABLES_SQL = """
SELECT
    EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = 'bureau_agg_curr_tmp'
    ) AS has_bureau_tmp,
    EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = 'prev_agg_curr_tmp'
    ) AS has_prev_tmp;
"""


# =============================================================================
# Fonctions
# =============================================================================

def check_required_temp_tables(connection) -> None:
    """
    Vérifie que les tables temporaires nécessaires existent bien.
    """
    result = connection.execute(text(CHECK_TEMP_TABLES_SQL)).mappings().first()

    if not result["has_bureau_tmp"]:
        raise ValueError("La table temporaire 'bureau_agg_curr_tmp' est introuvable.")

    if not result["has_prev_tmp"]:
        raise ValueError("La table temporaire 'prev_agg_curr_tmp' est introuvable.")


def create_empty_features_client_test_raw_table(connection) -> None:
    """
    Supprime puis recrée la table cible vide avec la bonne structure.
    """
    connection.execute(text(DROP_FEATURES_CLIENT_TEST_RAW_SQL))
    connection.execute(text(CREATE_EMPTY_FEATURES_CLIENT_TEST_RAW_SQL))


def insert_features_client_test_raw_in_batches(engine, batch_size: int = 500) -> None:
    """
    Insère les lignes dans `features_client_test_raw` par paquets de `batch_size`.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    batch_size : int, default=500
        Nombre de lignes insérées à chaque batch.
    """
    with engine.begin() as connection:
        check_required_temp_tables(connection)

        print("Création de la table vide...")
        create_empty_features_client_test_raw_table(connection)

        total_rows = connection.execute(text(COUNT_APPLICATION_TEST_SQL)).scalar_one()
        print(f"Nombre total de lignes à insérer : {total_rows}")

    for offset in range(0, total_rows, batch_size):
        with engine.begin() as connection:
            connection.execute(
                text(INSERT_FEATURES_CLIENT_TEST_RAW_BATCH_SQL),
                {
                    "batch_size": batch_size,
                    "offset": offset,
                },
            )

        print(
            f"Insertion batch terminé : "
            f"{min(offset + batch_size, total_rows)}/{total_rows}"
        )

    with engine.begin() as connection:
        print("Création de l'index final...")
        connection.execute(text(CREATE_FEATURES_CLIENT_TEST_RAW_INDEX_SQL))

    print("Table 'features_client_test_raw' créée ou recréée avec succès.")


def main() -> None:
    """
    Point d'entrée du script.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    insert_features_client_test_raw_in_batches(engine, batch_size=BATCH_SIZE)

    print("Création de la table 'features_client_test_raw' terminée.")


if __name__ == "__main__":
    main()