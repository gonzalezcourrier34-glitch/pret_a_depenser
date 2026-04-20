"""
Configuration globale de l'application.

Ce module centralise les variables d'environnement utilisées
par l'API de scoring crédit et le monitoring du modèle.

Architecture actuelle
---------------------
- les données de prédiction sont construites exclusivement
  à partir de `application_test.csv`
- PostgreSQL sert uniquement au stockage des logs et du monitoring
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Helpers
# =============================================================================

def _get_bool(name: str, default: str = "False") -> bool:
    """
    Lit une variable d'environnement booléenne.

    Parameters
    ----------
    name : str
        Nom de la variable d'environnement.
    default : str
        Valeur par défaut sous forme de chaîne.

    Returns
    -------
    bool
        Valeur booléenne interprétée.
    """
    return os.getenv(name, default).strip().lower() == "true"


def _get_int(name: str, default: str) -> int:
    """
    Lit une variable d'environnement entière.

    Parameters
    ----------
    name : str
        Nom de la variable d'environnement.
    default : str
        Valeur par défaut.

    Returns
    -------
    int
        Valeur entière.
    """
    return int(os.getenv(name, default))


def _get_float(name: str, default: str) -> float:
    """
    Lit une variable d'environnement flottante.

    Parameters
    ----------
    name : str
        Nom de la variable d'environnement.
    default : str
        Valeur par défaut.

    Returns
    -------
    float
        Valeur flottante.
    """
    return float(os.getenv(name, default))


# =============================================================================
# API
# =============================================================================

API_KEY = os.getenv("API_KEY", "")
DEBUG = _get_bool("DEBUG", "False")


# =============================================================================
# Base de données
# =============================================================================
# Utilisée uniquement pour :
# - logs de prédiction
# - historique
# - monitoring
# - alertes
# - registre de modèles

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@postgres:5432/credit_api",
)


# =============================================================================
# Modèle
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "credit_scoring_model")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", "artifacts/threshold.json"))
DEBUG_MODEL = _get_bool("DEBUG_MODEL", "False")


# =============================================================================
# Données CSV
# =============================================================================
# Toutes les données utilisées pour construire les features viennent
# exclusivement du fichier application_test.csv

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

APPLICATION_TEST_CSV = Path(
    os.getenv("APPLICATION_TEST_CSV", str(DATA_DIR / "application_test.csv"))
)

SIMULATION_SOURCE_CSV = Path(
    os.getenv("SIMULATION_SOURCE_CSV", str(APPLICATION_TEST_CSV))
)


# =============================================================================
# Monitoring
# =============================================================================

REFERENCE_FEATURES_PATH = Path(
    os.getenv(
        "REFERENCE_FEATURES_PATH",
        "artifacts/monitoring/reference_features_transformed.parquet",
    )
)

CURRENT_WINDOW_DAYS = _get_int("CURRENT_WINDOW_DAYS", "7")
ANALYSIS_START = os.getenv("ANALYSIS_START")
ANALYSIS_END = os.getenv("ANALYSIS_END")

EVIDENTLY_DRIFT_SHARE = _get_float("EVIDENTLY_DRIFT_SHARE", "0.50")

EVIDENTLY_REPORT_PATH = Path(
    os.getenv(
        "EVIDENTLY_REPORT_PATH",
        "artifacts/monitoring/evidently_drift_report.html",
    )
)

ALERT_ON_RECALL_BELOW = _get_float("ALERT_ON_RECALL_BELOW", "0.60")
ALERT_ON_LATENCY_ABOVE_MS = _get_float("ALERT_ON_LATENCY_ABOVE_MS", "800")
ALERT_ON_ERROR_RATE_ABOVE = _get_float("ALERT_ON_ERROR_RATE_ABOVE", "0.05")

BUSINESS_COST_FN = _get_float("BUSINESS_COST_FN", "10")
BUSINESS_COST_FP = _get_float("BUSINESS_COST_FP", "1")


# =============================================================================
# Simulation / batch
# =============================================================================

SIMULATION_MAX_ITEMS = _get_int("SIMULATION_MAX_ITEMS", "200")
SIMULATION_DEFAULT_ITEMS = _get_int("SIMULATION_DEFAULT_ITEMS", "200")


# =============================================================================
# Validation minimale
# =============================================================================

if SIMULATION_DEFAULT_ITEMS > SIMULATION_MAX_ITEMS:
    raise ValueError(
        "SIMULATION_DEFAULT_ITEMS ne peut pas être supérieur à SIMULATION_MAX_ITEMS."
    )