"""
Gestion simple de la sécurité de l'API.

Ce module permet de protéger les routes sensibles de l'application,
en particulier l'endpoint de prédiction.

Le mécanisme retenu ici repose sur une clé API transmise dans les headers
de la requête. Cette solution est simple à mettre en place, lisible
et adaptée à un projet de déploiement de modèle en environnement contrôlé.
"""
from fastapi import Header, HTTPException, status

from app.core.config import API_KEY


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    """
    Vérifie la validité de la clé API envoyée dans le header.

    Paramètres
    ----------
    x_api_key : str | None
        Valeur transmise par le client dans le header 'X-API-Key'.

    Fonctionnement
    --------------
    - Si aucune clé n'est transmise, l'accès est refusé.
    - Si la clé transmise ne correspond pas à celle configurée
      dans les variables d'environnement, l'accès est refusé.
    - Si la clé est valide, la requête peut continuer.

    Exceptions
    ----------
    HTTPException
        Retourne une erreur 401 si la clé est absente ou invalide.
    """

    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La clé API du serveur n'est pas configurée.",
        )

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API manquante.",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide.",
        )