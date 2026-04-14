import os


# =============================================================================
# Configuration globale de l'application
# =============================================================================

"""
Centralise toutes les variables d'environnement.

Avantages :
- évite les valeurs en dur dans le code
- facilite le changement d'environnement (local / docker / prod)
- améliore la lisibilité
"""


# --- API ---
API_KEY = os.getenv("X-API-Key")

DEBUG = os.getenv("DEBUG", "False").lower() == "true"


# --- Base de données ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@postgres:5432/credit_api"
)


# --- Modèle ---
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")