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

Architecture actuelle
---------------------
- les données de prédiction sont construites exclusivement
  à partir de `application_test.csv`
- les features agrégées de type `bureau__...` et `prev__...`
  ne sont plus utilisées
- la base PostgreSQL sert uniquement au logging et au monitoring

Notes
-----
- Certains noms de colonnes contiennent des caractères peu pratiques
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


# =============================================================================
# Liste exacte des features attendues par le modèle
# =============================================================================

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


# =============================================================================
# Typage métier des features
# =============================================================================

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


# =============================================================================
# Helpers
# =============================================================================

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

    if safe_name and safe_name[0].isdigit():
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
        return Optional[str], Field(default=None, alias=feature_name)

    if feature_name in INTEGER_FEATURES:
        return Optional[int], Field(default=None, alias=feature_name)

    return Optional[float], Field(default=None, alias=feature_name)


# =============================================================================
# Schéma dynamique d'entrée modèle
# =============================================================================

class CreditModelInputBase(BaseModel):
    """
    Base commune pour le schéma des features d'entrée du modèle.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


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
    ),
)


# =============================================================================
# Schémas API
# =============================================================================

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
    features: CreditModelInput  # type: ignore


class PredictResponse(BaseModel):
    """
    Schéma de réponse de l'API de prédiction.

    Attributes
    ----------
    request_id : str
        Identifiant unique de la requête de prédiction.
    prediction : int
        Classe prédite (0 ou 1).
    score : float
        Probabilité associée à la classe positive.
    model_version : str
        Version du modèle utilisée.
    latency_ms : float
        Temps d'inférence en millisecondes.
    """

    request_id: str
    prediction: int
    score: float
    model_version: str
    latency_ms: float


class PredictBatchItemResponse(BaseModel):
    """
    Schéma de réponse pour un élément d'un batch de prédictions.

    Attributes
    ----------
    request_id : str
        Identifiant unique de la requête.
    client_id : Optional[int]
        Identifiant client éventuel.
    prediction : Optional[int]
        Classe prédite si succès.
    score : Optional[float]
        Score associé si succès.
    model_version : str
        Version du modèle utilisée.
    latency_ms : float
        Temps d'inférence en millisecondes.
    status : str
        Statut de traitement ('success' ou 'error').
    error_message : Optional[str]
        Message d'erreur éventuel.
    """

    request_id: str
    client_id: Optional[int] = None
    prediction: Optional[int] = None
    score: Optional[float] = None
    model_version: str
    latency_ms: float
    status: str
    error_message: Optional[str] = None


class PredictBatchResponse(BaseModel):
    """
    Schéma de réponse pour un batch de prédictions.

    Attributes
    ----------
    batch_size : int
        Taille totale du batch demandé.
    success_count : int
        Nombre de prédictions réussies.
    error_count : int
        Nombre de prédictions en erreur.
    model_name : str
        Nom du modèle utilisé.
    model_version : str
        Version du modèle utilisée.
    batch_latency_ms : float
        Temps total du batch en millisecondes.
    items : list[PredictBatchItemResponse]
        Détail ligne par ligne du batch.
    """

    batch_size: int
    success_count: int
    error_count: int
    model_name: str
    model_version: str
    batch_latency_ms: float
    items: list[PredictBatchItemResponse]


class HealthResponse(BaseModel):
    """
    Schéma de réponse pour l'endpoint de santé.

    Attributes
    ----------
    status : str
        État de santé de l'API.
    """

    status: str