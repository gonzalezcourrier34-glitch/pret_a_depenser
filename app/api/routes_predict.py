"""
Routes FastAPI liées à la prédiction et au monitoring applicatif.

Ce module expose les endpoints principaux de l'API de scoring crédit.
Il gère :

- la vérification de l'état de santé de l'application
- la sécurisation des appels via une clé API
- la réception des données de prédiction
- l'appel au modèle de machine learning
- la journalisation des résultats en base PostgreSQL

Endpoints
---------
- GET /health
    Vérifie que l'API répond correctement.
- POST /predict
    Réalise une prédiction à partir des variables d'entrée, puis
    enregistre le résultat dans la base de données.

Dépendances
-----------
- app.config : configuration globale (clé API, version modèle)
- app.db : session SQLAlchemy
- app.schemas : schémas Pydantic d'entrée et de sortie
- app.services.predictor : logique de prédiction
- app.crud : journalisation des prédictions en base

Notes
-----
- La clé API est transmise via le header HTTP `X-API-Key`.
- La latence d'inférence est mesurée en millisecondes.
- Chaque prédiction est stockée pour permettre l'auditabilité
  et le monitoring du modèle.
"""

import time

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import API_KEY, MODEL_VERSION
from app.crud import create_prediction_log
from app.db import get_db
from app.schemas import PredictRequest, PredictResponse
from app.services.predictor import make_prediction


# =============================================================================
# Initialisation du routeur FastAPI
# =============================================================================

# Le préfixe et les tags facilitent l'organisation de la documentation Swagger
# et rendent l'API plus lisible.
router = APIRouter(
    prefix="",
    tags=["Prediction"],
)


# =============================================================================
# Sécurité : vérification de la clé API
# =============================================================================

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Vérifie la validité de la clé API transmise dans les headers HTTP.

    Parameters
    ----------
    x_api_key : str
        Valeur du header `X-API-Key` envoyée par le client.

    Returns
    -------
    str
        La clé API validée.

    Raises
    ------
    HTTPException
        Levée avec un code 401 si la clé API est absente ou invalide.

    Notes
    -----
    - Si aucune clé API n'est définie côté application, la vérification
      peut être désactivée en pratique, mais ici on conserve un contrôle strict.
    - Le paramètre `alias="X-API-Key"` permet de faire correspondre
      proprement le nom du header HTTP avec le paramètre Python.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on the server.",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key.",
        )

    return x_api_key


# =============================================================================
# Endpoint de santé
# =============================================================================

@router.get("/health", summary="Vérifier l'état de santé de l'API")
def health() -> dict[str, str]:
    """
    Vérifie que l'API est accessible et opérationnelle.

    Returns
    -------
    dict[str, str]
        Un dictionnaire simple contenant le statut de l'application.

    Examples
    --------
    Réponse typique :

    ```json
    {
        "status": "ok"
    }
    ```
    """
    return {"status": "ok"}


# =============================================================================
# Endpoint principal de prédiction
# =============================================================================

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Réaliser une prédiction de risque de défaut",
)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> PredictResponse:
    """
    Réalise une prédiction à partir des données d'entrée du client.

    Cet endpoint :
    1. valide les données reçues via Pydantic,
    2. appelle le modèle de machine learning,
    3. mesure le temps d'inférence,
    4. journalise le résultat dans PostgreSQL,
    5. retourne la prédiction au client.

    Parameters
    ----------
    payload : PredictRequest
        Corps de la requête contenant l'identifiant client optionnel
        et les variables d'entrée du modèle.

    db : Session
        Session SQLAlchemy injectée par FastAPI.

    _ : str
        Clé API validée par la dépendance `verify_api_key`.
        Le nom `_` indique que cette valeur n'est pas réutilisée ensuite.

    Returns
    -------
    PredictResponse
        Objet de réponse contenant :
        - la classe prédite,
        - le score,
        - la version du modèle,
        - la latence d'inférence.

    Raises
    ------
    HTTPException
        Peut être levée indirectement en cas de clé API invalide.

    Notes
    -----
    - La latence mesurée ici concerne la phase de prédiction applicative.
    - La journalisation en base permet de conserver un historique
      utile pour le monitoring et l'audit.
    """
    # -------------------------------------------------------------------------
    # Début de la mesure de temps
    # -------------------------------------------------------------------------
    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Appel au modèle de prédiction
    # -------------------------------------------------------------------------
    prediction, score = make_prediction(payload.features)

    # -------------------------------------------------------------------------
    # Calcul de la latence en millisecondes
    # -------------------------------------------------------------------------
    latency_ms = (time.perf_counter() - start_time) * 1000

    # -------------------------------------------------------------------------
    # Journalisation du résultat en base PostgreSQL
    # -------------------------------------------------------------------------
    create_prediction_log(
        db=db,
        client_id=payload.SK_ID_CURR,
        prediction=prediction,
        score=score,
        model_version=MODEL_VERSION,
        input_data=payload.features,
        latency_ms=latency_ms,
    )

    # -------------------------------------------------------------------------
    # Construction et retour de la réponse API
    # -------------------------------------------------------------------------
    return PredictResponse(
        prediction=prediction,
        score=score,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )