"""
Module principal de l'application FastAPI.

Point d'entrée de l'API de scoring crédit.

Responsabilités
---------------
- charger la configuration
- initialiser le logging
- vérifier les dépendances critiques
- charger modèle + seuil
- charger les données en cache
- enregistrer les routes
- démarrer l'application

Architecture
------------
- core : config, db, logging
- services : logique métier
- api : routes HTTP
- crud : accès base
- model : ORM

Ordre de démarrage
------------------
1. Charger .env
2. Configurer logging
3. Vérifier assets
4. Charger modèle + seuil
5. Vérifier DB
6. Charger cache CSV
7. Lancer API
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# =============================================================================
# Chargement ENV AVANT tout
# =============================================================================

load_dotenv()

# =============================================================================
# Imports applicatifs
# =============================================================================

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from app.api.route_analyse import router as analyse_router
from app.api.route_history import router as history_router
from app.api.route_monitoring import router as monitoring_router
from app.api.route_prediction import router as predict_router

from app.core.config import DEBUG, BENCHMARK_MODE
from app.core.db import SessionLocal

from app.services.logging_service import LoggingMiddleware, setup_logging

from app.services.loader_services.data_loading_service import init_full_data_cache
from app.services.loader_services.huggingface_download_service import (
    ensure_assets_available,
)
from app.services.loader_services.model_loading_service import (
    get_model,
    get_threshold,
)

# =============================================================================
# Logging global
# =============================================================================

setup_logging(
    level=(
        logging.DEBUG
        if DEBUG
        else logging.WARNING
        if BENCHMARK_MODE
        else logging.INFO
    ),
    write_file=not BENCHMARK_MODE,
    quiet_libraries=True,
)

logger = logging.getLogger(__name__)

# =============================================================================
# OpenAPI
# =============================================================================

OPENAPI_TAGS = [
    {"name": "Predict", "description": "Endpoints de prédiction."},
    {"name": "History", "description": "Historique des prédictions."},
    {"name": "Monitoring", "description": "Monitoring du modèle."},
    {"name": "Analyse", "description": "Analyses avancées."},
]

# =============================================================================
# Helpers init
# =============================================================================

def _check_database_connection() -> None:
    """Vérifie la connexion PostgreSQL."""
    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))
        logger.info("Database OK", extra={"extra_data": {"event": "db_ok"}})
    finally:
        db.close()


def _initialize_assets() -> None:
    """Vérifie / télécharge les assets."""
    logger.info("Assets init", extra={"extra_data": {"event": "assets_start"}})
    ensure_assets_available()
    logger.info("Assets ready", extra={"extra_data": {"event": "assets_ok"}})


def _initialize_model_and_threshold() -> None:
    """Charge modèle + seuil."""
    logger.info("Model init", extra={"extra_data": {"event": "model_start"}})

    get_model()
    logger.info("Model loaded", extra={"extra_data": {"event": "model_ok"}})

    get_threshold()
    logger.info("Threshold loaded", extra={"extra_data": {"event": "threshold_ok"}})


def _initialize_csv_cache() -> None:
    """Charge les données en mémoire."""
    logger.info("Cache init", extra={"extra_data": {"event": "cache_start"}})

    init_full_data_cache(debug=False)

    logger.info("Cache ready", extra={"extra_data": {"event": "cache_ok"}})


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie FastAPI."""
    _ = app

    logger.info("Startup begin", extra={"extra_data": {"event": "startup"}})

    try:
        _initialize_assets()
        _initialize_model_and_threshold()
        _check_database_connection()
        _initialize_csv_cache()

        logger.info("Startup OK", extra={"extra_data": {"event": "startup_ok"}})

        yield

    except Exception:
        logger.exception(
            "Startup FAILED",
            extra={"extra_data": {"event": "startup_error"}},
        )
        raise

    finally:
        logger.info("Shutdown", extra={"extra_data": {"event": "shutdown"}})


# =============================================================================
# App FastAPI
# =============================================================================

app = FastAPI(
    title="API Scoring Crédit",
    description="API ML avec monitoring et logging structuré",
    version="1.0.0",
    debug=DEBUG,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)

# =============================================================================
# Middleware
# =============================================================================

if not BENCHMARK_MODE:
    app.add_middleware(LoggingMiddleware)

# =============================================================================
# Root
# =============================================================================

@app.get("/", include_in_schema=False)
def root():
    """Redirection vers docs."""
    return RedirectResponse(url="/docs")

# =============================================================================
# Routes
# =============================================================================

app.include_router(predict_router)
app.include_router(history_router)
app.include_router(monitoring_router)
app.include_router(analyse_router)