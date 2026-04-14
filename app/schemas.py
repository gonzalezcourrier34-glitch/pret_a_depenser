from pydantic import BaseModel
from typing import Dict, Any, Optional


# =============================================================================
# Schéma d'entrée de l'API
# =============================================================================

class PredictRequest(BaseModel):
    """
    Schéma de requête pour l'endpoint /predict.

    Attributs
    ---------
    SK_ID_CURR : int | None
        Identifiant du client (optionnel).

    features : dict
        Dictionnaire contenant les variables d'entrée du modèle.
        Exemple : {"AMT_CREDIT": 100000, "EXT_SOURCE_1": 0.5, ...}
    """

    SK_ID_CURR: Optional[int] = None
    features: Dict[str, Any]


# =============================================================================
# Schéma de réponse de l'API
# =============================================================================

class PredictResponse(BaseModel):
    """
    Schéma de réponse retourné par l'API.

    Attributs
    ---------
    prediction : int
        Classe prédite (0 ou 1).

    score : float
        Probabilité associée à la classe positive.

    model_version : str
        Version du modèle utilisé.

    latency_ms : float
        Temps d'inférence en millisecondes.
    """

    prediction: int
    score: float
    model_version: str
    latency_ms: float