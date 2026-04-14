"""
Module principal de l'application FastAPI.

Ce module constitue le point d'entrée de l'API de scoring crédit.
Il initialise l'application, configure les dépendances globales
et enregistre les routes métier.

Fonctionnalités principales
---------------------------
- Chargement des variables d'environnement
- Initialisation de l'application FastAPI
- Création automatique des tables en base de données
- Enregistrement des routes (prédiction, santé, etc.)

Architecture
------------
L'application suit une architecture modulaire :

- app.config : gestion des variables d'environnement
- app.db : connexion et session base de données
- app.models : schéma SQLAlchemy
- app.schemas : validation des données (Pydantic)
- app.crud : opérations base de données
- app.services : logique métier (modèle ML)
- app.api : routes FastAPI

Notes
-----
- La création automatique des tables est adaptée pour un projet pédagogique.
- En production, il est recommandé d'utiliser Alembic pour les migrations.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.config import DEBUG
from app.db import Base, engine
from app.api.routes_predict import router as predict_router


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Lifespan (remplace on_event)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.

    Cette fonction remplace les anciens événements startup/shutdown.
    Elle permet d'initialiser les संसources avant le démarrage
    et de nettoyer proprement à l'arrêt.

    Étapes
    ------
    1. Création des tables en base de données
    2. Lancement de l'application
    3. Nettoyage éventuel à l'arrêt

    Yields
    ------
    None
        Permet à FastAPI de continuer le démarrage.
    """
    # --- STARTUP ---
    Base.metadata.create_all(bind=engine)

    yield  # ← l'application tourne ici

    # --- SHUTDOWN ---
    # (optionnel : fermer connexions, logs, etc.)


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

@app.get("/")
def root() -> dict[str, str]:
    """
    Endpoint racine de vérification.

    Returns
    -------
    dict[str, str]
        Message simple indiquant que l'API fonctionne.
    """
    return {"message": "API OK"}


# =============================================================================
# Enregistrement des routes
# =============================================================================

app.include_router(predict_router)