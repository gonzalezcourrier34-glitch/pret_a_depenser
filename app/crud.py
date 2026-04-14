"""
Fonctions CRUD pour la gestion des prédictions en base de données.

Ce module centralise les opérations d'écriture liées aux logs
de prédictions du modèle de scoring crédit.

Fonctionnalités
---------------
- Enregistrement des prédictions
- Stockage des métadonnées (score, version modèle, latence, etc.)

Notes
-----
- Chaque prédiction est persistée pour permettre le monitoring,
  l'audit et l'analyse du modèle dans le temps.
"""

from sqlalchemy.orm import Session
from datetime import datetime

from app.models import PredictionLog


# =============================================================================
# Création d'un log de prédiction
# =============================================================================

def create_prediction_log(
    db: Session,
    client_id: int | None,
    prediction: int,
    score: float,
    model_version: str,
    input_data: dict,
    latency_ms: float,
) -> PredictionLog:
    """
    Enregistre une prédiction en base de données.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    client_id : int | None
        Identifiant du client (optionnel).
    prediction : int
        Classe prédite (0 ou 1).
    score : float
        Probabilité associée à la prédiction.
    model_version : str
        Version du modèle utilisé.
    input_data : dict
        Données d'entrée utilisées pour la prédiction.
    latency_ms : float
        Temps d'inférence en millisecondes.

    Returns
    -------
    PredictionLog
        Objet SQLAlchemy représentant la ligne insérée.

    Notes
    -----
    - Les données sont stockées en JSON pour faciliter l'analyse.
    - Le timestamp est généré automatiquement.
    """
    log = PredictionLog(
        client_id=client_id,
        prediction=prediction,
        score=score,
        model_version=model_version,
        input_data=input_data,
        latency_ms=latency_ms,
        created_at=datetime.utcnow(),
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return log