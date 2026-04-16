"""
Module principal de l'application FastAPI.

Ce module constitue le point d'entrée de l'API de scoring crédit.
Il initialise l'application, configure le cycle de vie
et enregistre les routes métier.

Architecture
------------
- app.core.config : configuration
- app.core.db : connexion base de données
- app.api : routes FastAPI
- app.services : logique métier
- app.crud : persistance PostgreSQL
- app.model : modèles SQLAlchemy
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy import text

from app.api.routes_predict import router as predict_router
from app.core.config import DEBUG
from app.core.db import SessionLocal
from app.services.model_loader import get_model, get_threshold

from fastapi.responses import RedirectResponse
# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


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
    """
    # -------------------------------------------------------------------------
    # STARTUP
    # -------------------------------------------------------------------------
    # Vérification du chargement du modèle et du seuil

    get_model()
    get_threshold()

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    finally:
        db.close()

    yield

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------
    # Aucun nettoyage spécifique nécessaire ici pour le moment.


# =============================================================================
# Création de l'application
# =============================================================================

app = FastAPI(
    title="API de scoring crédit",
    description="API de prédiction du risque de défaut avec journalisation PostgreSQL.",
    version="1.0.0",
    debug=DEBUG,
    lifespan=lifespan,
)


# =============================================================================
# Endpoint racine
# =============================================================================

@app.get("/", include_in_schema=False)
def root():
    """
    Endpoint racine de vérification.

    Returns
    -------
    dict[str, str]
        Message simple indiquant que l'API fonctionne.
    """
    return RedirectResponse(url="/docs")


# =============================================================================
# Enregistrement des routes
# =============================================================================

app.include_router(predict_router)