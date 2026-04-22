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

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from app.api.route_evidently import router as evidently_router
from app.api.route_history import router as history_router
from app.api.route_monitoring import router as monitoring_router
from app.api.route_predict import router as predict_router
from app.core.config import DEBUG
from app.core.db import SessionLocal
from app.services.data_loader_service import init_full_data_cache
from app.services.model_loader_service import get_model, get_threshold


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


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
    print("[APP] Démarrage de l'application...")

    get_model()
    print("[APP] Modèle chargé.")

    get_threshold()
    print("[APP] Seuil chargé.")

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        print("[APP] Connexion PostgreSQL OK.")
    finally:
        db.close()

    print("[APP] Chargement du cache métier CSV...")
    init_full_data_cache(debug=False)
    print("[APP] Cache CSV initialisé.")

    yield

    print("[APP] Arrêt de l'application.")


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