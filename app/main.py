"""
Module principal de l'application FastAPI.

Ce module constitue le point d'entrée de l'API de scoring crédit.
Il initialise l'application, configure le cycle de vie
et enregistre les routes métier.

Architecture
------------
- app.core.config : configuration
- app.core.db : connexion base de données
- app.api.routes : routes FastAPI
- app.services : logique métier
- app.crud : persistance PostgreSQL
- app.model : modèles SQLAlchemy

Architecture actuelle
---------------------
- les données de prédiction proviennent exclusivement du `.csv`
- le modèle et le seuil sont chargés au démarrage
- PostgreSQL sert uniquement au logging et au monitoring
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from app.api.route_evidently import router as evidently_router
from app.api.route_history import router as history_router
from app.api.route_monitoring import router as monitoring_router
from app.api.route_prediction import router as predict_router
from app.core.config import DEBUG
from app.core.db import SessionLocal
from app.core.logging_config import setup_logging
from app.services.logging_service import LoggingMiddleware
from app.services.data_loader_service import init_full_data_cache
from app.services.model_loading_service import get_model, get_threshold


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


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
            "Consultation de l'historique des prédictions, labels et features."
        ),
    },
    {
        "name": "Monitoring",
        "description": "Suivi du modèle, alertes, synthèses et état du monitoring.",
    },
    {
        "name": "Evidently",
        "description": "Analyses de dérive de données via Evidently.",
    },
]


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.

    Notes
    -----
    Les tables PostgreSQL sont créées via des scripts SQL dédiés.
    Aucun `create_all()` n'est exécuté ici afin d'éviter les écarts
    entre SQLAlchemy et la structure réelle de la base.

    Au démarrage, l'application vérifie :
    - l'accès au modèle de scoring
    - le chargement du seuil métier
    - la disponibilité minimale de la base PostgreSQL
    - le chargement en mémoire des données CSV métier
    """
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
        "PostgreSQL et endpoints de monitoring."
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

    Returns
    -------
    RedirectResponse
        Redirection HTTP vers `/docs`.
    """
    return RedirectResponse(url="/docs")


# =============================================================================
# Enregistrement des routes
# =============================================================================

app.include_router(predict_router)
app.include_router(history_router)
app.include_router(monitoring_router)
app.include_router(evidently_router)