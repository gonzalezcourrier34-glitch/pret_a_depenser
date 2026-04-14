import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# =============================================================================
# Configuration de la base de données
# =============================================================================

# Récupération de l'URL de connexion depuis les variables d'environnement.
# Si elle n'est pas définie, on utilise une valeur par défaut adaptée à Docker :
# - utilisateur : postgres
# - mot de passe : postgres
# - host : postgres (nom du service docker)
# - port : 5432
# - base : credit_api
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@postgres:5432/credit_api"
)


# =============================================================================
# Initialisation du moteur SQLAlchemy
# =============================================================================

# Le moteur est responsable de la connexion à la base PostgreSQL.
# echo=False : désactive les logs SQL dans la console (mettre True en debug si besoin)
engine = create_engine(DATABASE_URL, echo=False)


# =============================================================================
# Gestion des sessions de base de données
# =============================================================================

# SessionLocal est une factory de sessions.
# Chaque requête API utilisera une session indépendante.
SessionLocal = sessionmaker(
    autocommit=False,  # les transactions doivent être validées explicitement
    autoflush=False,   # évite les flush automatiques avant commit
    bind=engine
)


# =============================================================================
# Base déclarative des modèles
# =============================================================================

# Base est la classe mère de tous les modèles SQLAlchemy.
# Toutes les tables (models.py) devront hériter de cette base.
Base = declarative_base()


# =============================================================================
# Dépendance FastAPI pour accéder à la base de données
# =============================================================================

def get_db():
    """
    Fournit une session de base de données à une requête FastAPI.

    Cette fonction est utilisée comme dépendance (`Depends(get_db)`).
    Elle permet de :

    - créer une session SQLAlchemy au début de la requête
    - la rendre disponible dans les endpoints
    - garantir sa fermeture après utilisation

    Yields
    ------
    Session
        Une session SQLAlchemy active connectée à PostgreSQL.

    Notes
    -----
    - Chaque appel API obtient sa propre session (isolation des transactions).
    - La fermeture est assurée même en cas d'erreur grâce au bloc `finally`.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()