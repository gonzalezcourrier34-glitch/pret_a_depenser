"""
Schémas Pydantic de l'application de scoring crédit.

Ce module contient :
- la liste exacte des features attendues par le modèle
- la logique de génération dynamique du schéma d'entrée modèle
- les schémas d'entrée / sortie de l'API FastAPI

Objectif
--------
Garantir une cohérence stricte entre :
- les features attendues par le pipeline ML
- les données validées par l'API
- les types Python exposés à FastAPI

Notes
-----
- Les noms de colonnes du modèle contiennent des caractères peu pratiques
  pour Python, notamment des doubles underscores.
- On génère donc un schéma dynamique avec :
  - des noms de champs Python "safe"
  - des alias correspondant aux vrais noms des features modèle
- Le payload API utilise un wrapper `PredictRequest` avec :
  - `SK_ID_CURR` optionnel
  - `features` contenant toutes les variables du modèle
"""

from __future__ import annotations

from typing import Optional, Type, cast

from pydantic import BaseModel, ConfigDict, Field, create_model


# Liste des features exactes attendues par le modèle
MODEL_FEATURES = [
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
    "bureau__DAYS_CREDIT__mean",
    "bureau__DAYS_CREDIT__std",
    "bureau__CREDIT_DAY_OVERDUE__mean",
    "bureau__CREDIT_DAY_OVERDUE__std",
    "bureau__DAYS_CREDIT_ENDDATE__mean",
    "bureau__DAYS_CREDIT_ENDDATE__std",
    "bureau__DAYS_ENDDATE_FACT__mean",
    "bureau__DAYS_ENDDATE_FACT__std",
    "bureau__AMT_CREDIT_MAX_OVERDUE__mean",
    "bureau__AMT_CREDIT_MAX_OVERDUE__std",
    "bureau__CNT_CREDIT_PROLONG__mean",
    "bureau__CNT_CREDIT_PROLONG__std",
    "bureau__AMT_CREDIT_SUM__mean",
    "bureau__AMT_CREDIT_SUM__std",
    "bureau__AMT_CREDIT_SUM_DEBT__mean",
    "bureau__AMT_CREDIT_SUM_DEBT__std",
    "bureau__AMT_CREDIT_SUM_LIMIT__mean",
    "bureau__AMT_CREDIT_SUM_LIMIT__std",
    "bureau__AMT_CREDIT_SUM_OVERDUE__mean",
    "bureau__AMT_CREDIT_SUM_OVERDUE__std",
    "bureau__DAYS_CREDIT_UPDATE__mean",
    "bureau__DAYS_CREDIT_UPDATE__std",
    "bureau__AMT_ANNUITY__mean",
    "bureau__AMT_ANNUITY__std",
    "bureau__DEBT_RATIO__mean",
    "bureau__DEBT_RATIO__std",
    "bureau__OVERDUE_RATIO__mean",
    "bureau__OVERDUE_RATIO__std",
    "bureau__IS_ACTIVE__mean",
    "bureau__IS_ACTIVE__std",
    "bureau__HAS_OVERDUE__mean",
    "bureau__HAS_OVERDUE__std",
    "bureau__CREDIT_AGE__mean",
    "bureau__CREDIT_AGE__std",
    "bureau__bb__MONTHS_BALANCE__mean__mean",
    "bureau__bb__MONTHS_BALANCE__mean__std",
    "bureau__bb__MONTHS_BALANCE__std__mean",
    "bureau__bb__MONTHS_BALANCE__std__std",
    "bureau__bb__count_rows__mean",
    "bureau__bb__count_rows__std",
    "bureau__bb__recent_max_dpd__mean",
    "bureau__bb__recent_max_dpd__std",
    "bureau__bb__months_late_ratio__mean",
    "bureau__bb__months_late_ratio__std",
    "bureau__bb__late_severity_sum__mean",
    "bureau__bb__late_severity_sum__std",
    "bureau__count_rows",
    "bureau__nunique_SK_ID_BUREAU",
    "prev__AMT_ANNUITY__mean",
    "prev__AMT_ANNUITY__std",
    "prev__AMT_APPLICATION__mean",
    "prev__AMT_APPLICATION__std",
    "prev__AMT_CREDIT__mean",
    "prev__AMT_CREDIT__std",
    "prev__AMT_DOWN_PAYMENT__mean",
    "prev__AMT_DOWN_PAYMENT__std",
    "prev__AMT_GOODS_PRICE__mean",
    "prev__AMT_GOODS_PRICE__std",
    "prev__HOUR_APPR_PROCESS_START__mean",
    "prev__HOUR_APPR_PROCESS_START__std",
    "prev__NFLAG_LAST_APPL_IN_DAY__mean",
    "prev__NFLAG_LAST_APPL_IN_DAY__std",
    "prev__RATE_DOWN_PAYMENT__mean",
    "prev__RATE_DOWN_PAYMENT__std",
    "prev__DAYS_DECISION__mean",
    "prev__DAYS_DECISION__std",
    "prev__SELLERPLACE_AREA__mean",
    "prev__SELLERPLACE_AREA__std",
    "prev__CNT_PAYMENT__mean",
    "prev__CNT_PAYMENT__std",
    "prev__DAYS_FIRST_DRAWING__mean",
    "prev__DAYS_FIRST_DUE__mean",
    "prev__DAYS_FIRST_DUE__std",
    "prev__DAYS_LAST_DUE_1ST_VERSION__mean",
    "prev__DAYS_LAST_DUE_1ST_VERSION__std",
    "prev__DAYS_LAST_DUE__mean",
    "prev__DAYS_LAST_DUE__std",
    "prev__DAYS_TERMINATION__mean",
    "prev__DAYS_TERMINATION__std",
    "prev__NFLAG_INSURED_ON_APPROVAL__mean",
    "prev__NFLAG_INSURED_ON_APPROVAL__std",
    "prev__PREV_CREDIT_APPLICATION_RATIO__mean",
    "prev__PREV_CREDIT_APPLICATION_RATIO__std",
    "prev__PREV_IS_APPROVED__mean",
    "prev__PREV_IS_APPROVED__std",
    "prev__PREV_IS_REFUSED__mean",
    "prev__PREV_IS_REFUSED__std",
    "prev__PREV_DAYS_DECISION_AGE__mean",
    "prev__PREV_DAYS_DECISION_AGE__std",
    "prev__PREV_CREDIT_DURATION__mean",
    "prev__PREV_CREDIT_DURATION__std",
    "prev__pos__MONTHS_BALANCE__mean__mean",
    "prev__pos__MONTHS_BALANCE__mean__std",
    "prev__pos__MONTHS_BALANCE__std__mean",
    "prev__pos__MONTHS_BALANCE__std__std",
    "prev__pos__CNT_INSTALMENT__mean__mean",
    "prev__pos__CNT_INSTALMENT__mean__std",
    "prev__pos__CNT_INSTALMENT__std__mean",
    "prev__pos__CNT_INSTALMENT__std__std",
    "prev__pos__CNT_INSTALMENT_FUTURE__mean__mean",
    "prev__pos__CNT_INSTALMENT_FUTURE__mean__std",
    "prev__pos__CNT_INSTALMENT_FUTURE__std__mean",
    "prev__pos__CNT_INSTALMENT_FUTURE__std__std",
    "prev__pos__SK_DPD__mean__mean",
    "prev__pos__SK_DPD__mean__std",
    "prev__pos__SK_DPD__max__mean",
    "prev__pos__SK_DPD__max__std",
    "prev__pos__SK_DPD__std__mean",
    "prev__pos__SK_DPD__std__std",
    "prev__pos__SK_DPD_DEF__mean__mean",
    "prev__pos__SK_DPD_DEF__mean__std",
    "prev__pos__SK_DPD_DEF__max__mean",
    "prev__pos__SK_DPD_DEF__max__std",
    "prev__pos__SK_DPD_DEF__std__mean",
    "prev__pos__SK_DPD_DEF__std__std",
    "prev__pos__POS_REMAIN_RATIO__mean__mean",
    "prev__pos__POS_REMAIN_RATIO__mean__std",
    "prev__pos__POS_REMAIN_RATIO__std__mean",
    "prev__pos__POS_REMAIN_RATIO__std__std",
    "prev__pos__POS_DPD_POS__mean__mean",
    "prev__pos__POS_DPD_POS__mean__std",
    "prev__pos__POS_DPD_POS__max__mean",
    "prev__pos__POS_DPD_POS__max__std",
    "prev__pos__POS_DPD_POS__std__mean",
    "prev__pos__POS_DPD_POS__std__std",
    "prev__pos__POS_IS_ACTIVE__mean__mean",
    "prev__pos__POS_IS_ACTIVE__mean__std",
    "prev__pos__POS_IS_ACTIVE__std__mean",
    "prev__pos__POS_IS_ACTIVE__std__std",
    "prev__pos__count_rows__mean",
    "prev__pos__count_rows__std",
    "prev__cc__MONTHS_BALANCE__mean__mean",
    "prev__cc__MONTHS_BALANCE__std__mean",
    "prev__cc__AMT_BALANCE__mean__mean",
    "prev__cc__AMT_BALANCE__std__mean",
    "prev__cc__AMT_CREDIT_LIMIT_ACTUAL__mean__mean",
    "prev__cc__AMT_CREDIT_LIMIT_ACTUAL__std__mean",
    "prev__cc__AMT_DRAWINGS_ATM_CURRENT__mean__mean",
    "prev__cc__AMT_DRAWINGS_ATM_CURRENT__std__mean",
    "prev__cc__AMT_DRAWINGS_CURRENT__mean__mean",
    "prev__cc__AMT_DRAWINGS_CURRENT__std__mean",
    "prev__cc__AMT_DRAWINGS_OTHER_CURRENT__mean__mean",
    "prev__cc__AMT_DRAWINGS_OTHER_CURRENT__std__mean",
    "prev__cc__AMT_DRAWINGS_POS_CURRENT__mean__mean",
    "prev__cc__AMT_DRAWINGS_POS_CURRENT__std__mean",
    "prev__cc__AMT_INST_MIN_REGULARITY__mean__mean",
    "prev__cc__AMT_INST_MIN_REGULARITY__std__mean",
    "prev__cc__AMT_PAYMENT_CURRENT__mean__mean",
    "prev__cc__AMT_PAYMENT_CURRENT__std__mean",
    "prev__cc__AMT_PAYMENT_TOTAL_CURRENT__mean__mean",
    "prev__cc__AMT_PAYMENT_TOTAL_CURRENT__std__mean",
    "prev__cc__AMT_RECEIVABLE_PRINCIPAL__mean__mean",
    "prev__cc__AMT_RECEIVABLE_PRINCIPAL__std__mean",
    "prev__cc__AMT_RECIVABLE__mean__mean",
    "prev__cc__AMT_RECIVABLE__std__mean",
    "prev__cc__AMT_TOTAL_RECEIVABLE__mean__mean",
    "prev__cc__AMT_TOTAL_RECEIVABLE__std__mean",
    "prev__cc__CNT_DRAWINGS_ATM_CURRENT__mean__mean",
    "prev__cc__CNT_DRAWINGS_ATM_CURRENT__std__mean",
    "prev__cc__CNT_DRAWINGS_CURRENT__mean__mean",
    "prev__cc__CNT_DRAWINGS_CURRENT__std__mean",
    "prev__cc__CNT_DRAWINGS_OTHER_CURRENT__mean__mean",
    "prev__cc__CNT_DRAWINGS_OTHER_CURRENT__std__mean",
    "prev__cc__CNT_DRAWINGS_POS_CURRENT__mean__mean",
    "prev__cc__CNT_DRAWINGS_POS_CURRENT__std__mean",
    "prev__cc__CNT_INSTALMENT_MATURE_CUM__mean__mean",
    "prev__cc__CNT_INSTALMENT_MATURE_CUM__std__mean",
    "prev__cc__SK_DPD__mean__mean",
    "prev__cc__SK_DPD__max__mean",
    "prev__cc__SK_DPD__std__mean",
    "prev__cc__SK_DPD_DEF__mean__mean",
    "prev__cc__SK_DPD_DEF__max__mean",
    "prev__cc__SK_DPD_DEF__std__mean",
    "prev__cc__CC_UTILIZATION_RATIO__mean__mean",
    "prev__cc__CC_UTILIZATION_RATIO__max__mean",
    "prev__cc__CC_UTILIZATION_RATIO__std__mean",
    "prev__cc__CC_PAYMENT_MIN_RATIO__mean__mean",
    "prev__cc__CC_PAYMENT_MIN_RATIO__max__mean",
    "prev__cc__CC_PAYMENT_MIN_RATIO__std__mean",
    "prev__cc__CC_PAYMENT_BALANCE_RATIO__mean__mean",
    "prev__cc__CC_PAYMENT_BALANCE_RATIO__max__mean",
    "prev__cc__CC_PAYMENT_BALANCE_RATIO__std__mean",
    "prev__cc__CC_DPD_POS__mean__mean",
    "prev__cc__CC_DPD_POS__max__mean",
    "prev__cc__CC_DPD_POS__std__mean",
    "prev__cc__CC_RECEIVABLE_RATIO__mean__mean",
    "prev__cc__CC_RECEIVABLE_RATIO__std__mean",
    "prev__cc__count_rows__mean",
    "prev__inst__NUM_INSTALMENT_VERSION__mean__mean",
    "prev__inst__NUM_INSTALMENT_VERSION__mean__std",
    "prev__inst__NUM_INSTALMENT_VERSION__std__mean",
    "prev__inst__NUM_INSTALMENT_VERSION__std__std",
    "prev__inst__NUM_INSTALMENT_NUMBER__mean__mean",
    "prev__inst__NUM_INSTALMENT_NUMBER__mean__std",
    "prev__inst__NUM_INSTALMENT_NUMBER__std__mean",
    "prev__inst__NUM_INSTALMENT_NUMBER__std__std",
    "prev__inst__DAYS_INSTALMENT__mean__mean",
    "prev__inst__DAYS_INSTALMENT__mean__std",
    "prev__inst__DAYS_INSTALMENT__std__mean",
    "prev__inst__DAYS_INSTALMENT__std__std",
    "prev__inst__DAYS_ENTRY_PAYMENT__mean__mean",
    "prev__inst__DAYS_ENTRY_PAYMENT__mean__std",
    "prev__inst__DAYS_ENTRY_PAYMENT__std__mean",
    "prev__inst__DAYS_ENTRY_PAYMENT__std__std",
    "prev__inst__AMT_INSTALMENT__mean__mean",
    "prev__inst__AMT_INSTALMENT__mean__std",
    "prev__inst__AMT_INSTALMENT__std__mean",
    "prev__inst__AMT_INSTALMENT__std__std",
    "prev__inst__AMT_PAYMENT__mean__mean",
    "prev__inst__AMT_PAYMENT__mean__std",
    "prev__inst__AMT_PAYMENT__std__mean",
    "prev__inst__AMT_PAYMENT__std__std",
    "prev__inst__DPD_POS__mean__mean",
    "prev__inst__DPD_POS__mean__std",
    "prev__inst__DPD_POS__max__mean",
    "prev__inst__DPD_POS__max__std",
    "prev__inst__DPD_POS__std__mean",
    "prev__inst__DPD_POS__std__std",
    "prev__inst__SEVERE_LATE_30__mean__mean",
    "prev__inst__SEVERE_LATE_30__mean__std",
    "prev__inst__SEVERE_LATE_30__max__mean",
    "prev__inst__SEVERE_LATE_30__max__std",
    "prev__inst__SEVERE_LATE_30__std__mean",
    "prev__inst__SEVERE_LATE_30__std__std",
    "prev__inst__PAY_RATIO__mean__mean",
    "prev__inst__PAY_RATIO__mean__std",
    "prev__inst__PAY_RATIO__std__mean",
    "prev__inst__PAY_RATIO__std__std",
    "prev__inst__count_rows__mean",
    "prev__inst__count_rows__std",
    "prev__count_rows",
    "prev__nunique_SK_ID_PREV",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "REGISTRATION_YEARS",
    "ID_PUBLISH_YEARS",
    "LAST_PHONE_CHANGE_YEARS",
    "DAYS_EMPLOYED__isna",
    "OWN_CAR_AGE__isna",
    "EXT_SOURCE_1__isna",
    "EXT_SOURCE_3__isna",
    "DAYS_LAST_PHONE_CHANGE__isna",
    "AMT_REQ_CREDIT_BUREAU_HOUR__isna",
    "AMT_REQ_CREDIT_BUREAU_WEEK__isna",
    "AMT_REQ_CREDIT_BUREAU_MON__isna",
    "AMT_REQ_CREDIT_BUREAU_QRT__isna",
    "AMT_REQ_CREDIT_BUREAU_YEAR__isna",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "CREDIT_GOODS_RATIO",
    "OVER_INDEBTED_40",
    "LOG_INCOME",
    "LOG_CREDIT",
    "LOG_ANNUITY",
    "LOG_GOODS",
    "SOCIAL_DEFAULT_RATIO_30",
    "SOCIAL_DEFAULT_RATIO_60",
    "DOC_COUNT",
    "CONTACT_COUNT",
    "ADDRESS_MISMATCH_COUNT",
    "EXT_SOURCES_MEAN",
    "EXT_SOURCES_MIN",
    "EXT_SOURCES_MAX",
    "EXT_SOURCES_STD",
    "EXT_SOURCES_RANGE",
    "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_2",
    "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_3",
    "EXT_INT__EXT_SOURCE_2_x_EXT_SOURCE_3",
    "EXT_POW2__EXT_SOURCE_1",
    "EXT_POW2__EXT_SOURCE_2",
    "EXT_POW2__EXT_SOURCE_3",
]


# Groupes de features par type
CATEGORICAL_FEATURES = {
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
}

INTEGER_FEATURES = {
    "CNT_CHILDREN",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "DAYS_EMPLOYED__isna",
    "OWN_CAR_AGE__isna",
    "EXT_SOURCE_1__isna",
    "EXT_SOURCE_3__isna",
    "DAYS_LAST_PHONE_CHANGE__isna",
    "AMT_REQ_CREDIT_BUREAU_HOUR__isna",
    "AMT_REQ_CREDIT_BUREAU_WEEK__isna",
    "AMT_REQ_CREDIT_BUREAU_MON__isna",
    "AMT_REQ_CREDIT_BUREAU_QRT__isna",
    "AMT_REQ_CREDIT_BUREAU_YEAR__isna",
    "OVER_INDEBTED_40",
    "DOC_COUNT",
    "CONTACT_COUNT",
    "ADDRESS_MISMATCH_COUNT",
}


# Fonctions utilitaires
def to_safe_field_name(feature_name: str) -> str:
    """
    Transforme un nom de feature SQL / modèle en nom de champ Python valide.

    Parameters
    ----------
    feature_name : str
        Nom d'origine de la feature.

    Returns
    -------
    str
        Nom de champ compatible avec Python.
    """
    safe_name = feature_name.lower()
    safe_name = safe_name.replace("__", "_")
    safe_name = safe_name.replace("-", "_")
    safe_name = safe_name.replace("/", "_")
    safe_name = safe_name.replace("(", "_")
    safe_name = safe_name.replace(")", "_")
    safe_name = safe_name.replace(" ", "_")

    if safe_name[0].isdigit():
        safe_name = f"f_{safe_name}"

    return safe_name


def get_field_definition(feature_name: str):
    """
    Déduit le type Pydantic à partir du nom de feature.

    Parameters
    ----------
    feature_name : str
        Nom de la feature.

    Returns
    -------
    tuple
        Tuple de la forme (type, Field(...)) pour create_model.
    """
    if feature_name in CATEGORICAL_FEATURES:
        return (Optional[str], Field(default=None, alias=feature_name))

    if feature_name in INTEGER_FEATURES:
        return (Optional[int], Field(default=None, alias=feature_name))

    return (Optional[float], Field(default=None, alias=feature_name))


# Base commune pour le schéma dynamique
class CreditModelInputBase(BaseModel):
    """
    Base commune pour le schéma des features d'entrée du modèle.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


# Génération dynamique du schéma d'entrée modèle
fields = {
    to_safe_field_name(feature_name): get_field_definition(feature_name)
    for feature_name in MODEL_FEATURES
}

CreditModelInput = cast(
    Type[BaseModel],
    create_model(
        "CreditModelInput",
        __base__=CreditModelInputBase,
        **fields,
    )
)


# Schémas API
class PredictRequest(BaseModel):
    """
    Schéma d'entrée de l'API de prédiction.

    Attributes
    ----------
    SK_ID_CURR : Optional[int]
        Identifiant client optionnel.
    features : CreditModelInput
        Ensemble des features attendues par le modèle.
    """

    SK_ID_CURR: Optional[int] = None
    features: CreditModelInput # type: ignore


class PredictResponse(BaseModel):
    """
    Schéma de réponse de l'API de prédiction.

    Attributes
    ----------
    prediction : int
        Classe prédite (0 ou 1).
    score : float
        Probabilité associée à la classe positive.
    model_version : str
        Version du modèle utilisée.
    latency_ms : float
        Temps d'inférence en millisecondes.
    """

    prediction: int
    score: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """
    Schéma de réponse pour l'endpoint de santé.

    Attributes
    ----------
    status : str
        État de santé de l'API.
    """

    status: str