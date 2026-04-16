"""
Configuration globale de l'application.

Ce module centralise les variables d'environnement utilisées
par l'API de scoring crédit.

Avantages
---------
- évite les valeurs en dur dans le code
- facilite le changement d'environnement (local / docker / prod)
- améliore la lisibilité et la maintenance
"""

import os

from dotenv import load_dotenv


# Chargement des variables d'environnement
load_dotenv()


# API
# Clé secrète utilisée pour protéger les appels API.
# Le header HTTP reste "X-API-Key", mais la variable d'environnement
# doit avoir un nom simple et stable côté système.
API_KEY = os.getenv("API_KEY")

# Active ou non le mode debug FastAPI.
DEBUG = os.getenv("DEBUG", "False").lower() == "true"


# Base de données
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@postgres:5432/credit_api"
)


# Modèle
MODEL_NAME = os.getenv("MODEL_NAME", "credit_scoring_model")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "artifacts/threshold.json")


# Source des données
# Permet de choisir la source des données pour la prédiction :
# - CSV : lecture depuis fichier CSV
# - DB  : lecture depuis PostgreSQL
TYPE_ENTREE_DONNEES = os.getenv("TYPE_ENTREE_DONNEES", "DB").upper()