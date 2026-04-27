"""
Module principal de l'application FastAPI.

Ce module constitue le point d'entrée de l'API de scoring crédit.
Il initialise l'application, configure le cycle de vie et enregistre
les routes métier.

Architecture
------------
- app.core.config : configuration
- app.core.db : connexion PostgreSQL
- app.api : routes FastAPI
- app.services : logique métier
- app.crud : persistance PostgreSQL
- app.model : modèles SQLAlchemy

Architecture actuelle
---------------------
- les données de prédiction proviennent d'un CSV local ou téléchargé
  depuis Hugging Face selon ASSETS_SOURCE
- le modèle, le modèle ONNX éventuel et le seuil sont chargés au démarrage
- PostgreSQL sert au logging, à l'historique et au monitoring
- les analyses avancées passent par la route `/analyse`

Ordre de démarrage
------------------
1. Charger les variables d'environnement
2. Vérifier / télécharger les assets nécessaires
3. Charger le modèle
4. Charger le seuil métier
5. Vérifier PostgreSQL
6. Initialiser le cache CSV
7. Démarrer l'API
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

# Important :
# Le fichier .env doit être chargé avant les imports applicatifs qui lisent
# la configuration au moment de l'import.
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from app.api.route_analyse import router as analyse_router
from app.api.route_history import router as history_router
from app.api.route_monitoring import router as monitoring_router
from app.api.route_prediction import router as predict_router
from app.core.config import DEBUG
from app.core.db import SessionLocal
from app.core.logging_config import setup_logging
from app.services.loader_services.data_loading_service import init_full_data_cache
from app.services.loader_services.huggingface_download_service import (
    ensure_assets_available,
)
from app.services.loader_services.model_loading_service import (
    get_model,
    get_threshold,
)
from app.services.logging_service import LoggingMiddleware


# =============================================================================
# Configuration du logging
# =============================================================================

setup_logging(write_file=True)
logger = logging.getLogger(__name__)


# =============================================================================
# Documentation OpenAPI
# =============================================================================

OPENAPI_TAGS = [
    {
        "name": "Predict",
        "description": "Endpoints de prédiction du risque de défaut.",
    },
    {
        "name": "History",
        "description": (
            "Consultation de l'historique des prédictions, des labels "
            "et des snapshots de features."
        ),
    },
    {
        "name": "Monitoring",
        "description": (
            "Suivi du modèle en production : synthèses, alertes, "
            "métriques de dérive, métriques d'évaluation et feature store."
        ),
    },
    {
        "name": "Analyse",
        "description": (
            "Analyses avancées de monitoring : dérive de données via Evidently "
            "et évaluation du modèle."
        ),
    },
]


# =============================================================================
# Helpers de démarrage
# =============================================================================

def _check_database_connection() -> None:
    """
    Vérifie que PostgreSQL est joignable.

    Notes
    -----
    Cette fonction ne crée pas les tables.
    Les tables doivent rester gérées par tes scripts SQL ou migrations dédiées.
    """
    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))

        logger.info(
            "PostgreSQL connection check succeeded",
            extra={
                "extra_data": {
                    "event": "database_check_ok",
                }
            },
        )

    finally:
        db.close()


def _initialize_assets() -> None:
    """
    Vérifie que les fichiers nécessaires à l'API sont disponibles.

    Selon `ASSETS_SOURCE`, cette étape peut :
    - ne rien télécharger si tout est local
    - télécharger uniquement les fichiers manquants
    - forcer le téléchargement depuis Hugging Face
    """
    logger.info(
        "Assets initialization started",
        extra={
            "extra_data": {
                "event": "assets_init_begin",
            }
        },
    )

    ensure_assets_available()

    logger.info(
        "Assets initialization completed",
        extra={
            "extra_data": {
                "event": "assets_init_success",
            }
        },
    )


def _initialize_model_and_threshold() -> None:
    """
    Charge le backend modèle configuré et le seuil métier.

    Notes
    -----
    Le backend peut être :
    - sklearn/joblib
    - ONNX
    selon la variable `MODEL_BACKEND`.
    """
    logger.info(
        "Model initialization started",
        extra={
            "extra_data": {
                "event": "model_init_begin",
            }
        },
    )

    get_model()

    logger.info(
        "Model loaded successfully",
        extra={
            "extra_data": {
                "event": "model_loaded",
            }
        },
    )

    get_threshold()

    logger.info(
        "Threshold loaded successfully",
        extra={
            "extra_data": {
                "event": "threshold_loaded",
            }
        },
    )


def _initialize_csv_cache() -> None:
    """
    Initialise le cache CSV en mémoire.

    Ce cache permet ensuite :
    - de retrouver les clients par SK_ID_CURR
    - de construire les features prêtes modèle
    - de lancer les simulations
    """
    logger.info(
        "CSV cache initialization started",
        extra={
            "extra_data": {
                "event": "csv_cache_init_begin",
            }
        },
    )

    init_full_data_cache(debug=False)

    logger.info(
        "CSV cache initialized successfully",
        extra={
            "extra_data": {
                "event": "csv_cache_init_success",
            }
        },
    )


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application FastAPI.

    Au démarrage, l'application vérifie :
    - la présence des assets nécessaires
    - le chargement du modèle
    - le chargement du seuil métier
    - la disponibilité de PostgreSQL
    - le chargement du cache CSV

    Parameters
    ----------
    app : FastAPI
        Instance FastAPI en cours d'initialisation.

    Yields
    ------
    None
        Rend la main à FastAPI une fois l'initialisation terminée.
    """
    _ = app

    logger.info(
        "Application startup initiated",
        extra={
            "extra_data": {
                "event": "app_startup_begin",
                "service": "credit_scoring_api",
            }
        },
    )

    try:
        # 1. Les assets doivent exister avant tout chargement modèle / CSV.
        _initialize_assets()

        # 2. Chargement du backend modèle et du seuil.
        _initialize_model_and_threshold()

        # 3. Vérification minimale de PostgreSQL.
        _check_database_connection()

        # 4. Chargement en mémoire du CSV et des features.
        _initialize_csv_cache()

        logger.info(
            "Application startup completed",
            extra={
                "extra_data": {
                    "event": "app_startup_complete",
                    "service": "credit_scoring_api",
                }
            },
        )

        yield

    except Exception:
        logger.exception(
            "Application startup failed",
            extra={
                "extra_data": {
                    "event": "app_startup_error",
                    "service": "credit_scoring_api",
                }
            },
        )
        raise

    finally:
        logger.info(
            "Application shutdown completed",
            extra={
                "extra_data": {
                    "event": "app_shutdown",
                    "service": "credit_scoring_api",
                }
            },
        )


# =============================================================================
# Création de l'application
# =============================================================================

app = FastAPI(
    title="API de scoring crédit",
    description=(
        "API de prédiction du risque de défaut avec journalisation "
        "PostgreSQL, endpoints de monitoring et routes d'analyse."
    ),
    version="1.0.0",
    debug=DEBUG,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)


# =============================================================================
# Middlewares
# =============================================================================

app.add_middleware(LoggingMiddleware)


# =============================================================================
# Endpoint racine
# =============================================================================

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """
    Redirige vers la documentation interactive de l'API.
    """
    return RedirectResponse(url="/docs")


# =============================================================================
# Enregistrement des routes
# =============================================================================

app.include_router(predict_router)
app.include_router(history_router)
app.include_router(monitoring_router)
app.include_router(analyse_router)