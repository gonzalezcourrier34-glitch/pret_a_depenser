"""
Chargement des artefacts du modèle de scoring.

Ce module a pour rôle de centraliser le chargement des éléments nécessaires
au fonctionnement de l'API :
- le pipeline de modèle entraîné
- le seuil de décision métier
- la liste des features attendues

Objectif :
Éviter de dupliquer ce code dans plusieurs fichiers et garantir
un chargement propre, lisible et réutilisable.
"""

import json
import joblib
from typing import Any

from app.config import MODEL_PATH, THRESHOLD_PATH, FEATURES_PATH


def load_model() -> Any:
    """
    Charge le pipeline de machine learning sauvegardé au format joblib.

    Retour
    ------
    Any
        Le pipeline complet entraîné, prêt à être utilisé pour la prédiction.

    Remarques
    ---------
    Dans mon projet, le modèle sauvegardé contient déjà :
    - le prétraitement
    - le modèle final

    Cela permet à l'API d'envoyer directement les données brutes
    au pipeline sans avoir à refaire manuellement les transformations.
    """
    model = joblib.load(MODEL_PATH)
    return model


def load_threshold() -> float:
    """
    Charge le seuil de décision depuis le fichier JSON.

    Retour
    ------
    float
        Le seuil de classification utilisé pour transformer
        le score en décision finale.

    Fonctionnement
    --------------
    Le fichier threshold.json contient une structure simple du type :
    {
        "threshold": 0.0545
    }
    """
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    threshold = float(data["threshold"])
    return threshold


def load_features() -> list[str]:
    """
    Charge la liste des features attendues par le modèle.

    Retour
    ------
    list[str]
        Liste ordonnée des colonnes attendues en entrée du pipeline.

    Intérêt
    -------
    Cette liste est utile pour :
    - réaligner les données reçues par l'API
    - ajouter les colonnes manquantes si besoin
    - garantir la cohérence entre l'entraînement et la prédiction
    """
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)

    return features


def load_artifacts() -> tuple[Any, float, list[str]]:
    """
    Charge l'ensemble des artefacts nécessaires à l'API.

    Retour
    ------
    tuple[Any, float, list[str]]
        Un tuple contenant :
        - le modèle
        - le seuil
        - la liste des features attendues

    Intérêt
    -------
    Cette fonction permet de charger en une seule fois tous les éléments
    utiles au démarrage de l'application.
    """
    model = load_model()
    threshold = load_threshold()
    features = load_features()

    return model, threshold, features