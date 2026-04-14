"""
Modèles SQLAlchemy pour la base de données.

Ce module définit les tables utilisées pour stocker les informations
liées aux prédictions du modèle de scoring crédit.

Tables
------
- PredictionLog :
    Historique des prédictions réalisées par l'API.

Fonctionnalités
---------------
- Stockage des résultats de prédiction
- Suivi des performances du modèle
- Audit des entrées et sorties

Notes
-----
- Les données d'entrée sont stockées au format JSON.
- Chaque prédiction est horodatée.
- La version du modèle est enregistrée pour le tracking.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


# =============================================================================
# Table des logs de prédiction
# =============================================================================

class PredictionLog(Base):
    """
    Représente une prédiction réalisée par le modèle.

    Cette table stocke toutes les informations nécessaires pour :
    - analyser les performances du modèle
    - tracer les décisions prises
    - détecter du drift ou des anomalies

    Attributs
    ---------
    id : int
        Identifiant unique de la prédiction.
    client_id : int | None
        Identifiant du client (optionnel).
    prediction : int
        Classe prédite (0 = non défaut, 1 = défaut).
    score : float
        Probabilité associée à la classe positive.
    model_version : str
        Version du modèle utilisé.
    input_data : dict
        Données d'entrée utilisées pour la prédiction (JSON).
    latency_ms : float
        Temps de réponse du modèle en millisecondes.
    created_at : datetime
        Date et heure de la prédiction.
    """

    __tablename__ = "prediction_logs"

    # -------------------------------------------------------------------------
    # Clé primaire
    # -------------------------------------------------------------------------
    id = Column(Integer, primary_key=True, index=True)

    # -------------------------------------------------------------------------
    # Données métier
    # -------------------------------------------------------------------------
    client_id = Column(Integer, nullable=True)
    prediction = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)

    # -------------------------------------------------------------------------
    # Métadonnées modèle
    # -------------------------------------------------------------------------
    model_version = Column(String, nullable=False)

    # -------------------------------------------------------------------------
    # Données d'entrée (JSON PostgreSQL)
    # -------------------------------------------------------------------------
    input_data = Column(JSONB, nullable=False)

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------
    latency_ms = Column(Float, nullable=False)

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------
    created_at = Column(DateTime, nullable=False)