"""
Service de prédiction du modèle de scoring crédit.

Ce module encapsule la logique métier liée à l'inférence du modèle.
Il ne recharge pas le modèle à chaque appel : le modèle et le seuil
de décision sont récupérés depuis `app.model_loader`, qui les charge
une seule fois au démarrage de l'application.

Fonctionnalités
---------------
- prédiction unitaire à partir d'un dictionnaire de features
- prédiction batch à partir d'un DataFrame pandas
- utilisation d'un seuil métier personnalisé
- validation légère des entrées

Notes
-----
- Le score retourné correspond à la probabilité de la classe positive.
- La décision finale est calculée avec le seuil chargé depuis
  `app.model_loader`.
- Ce module est compatible avec une API FastAPI ou un usage batch.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.model_loader import get_model, get_threshold

# Fonctions utilitaires internes
def _ensure_dataframe_from_dict(features: dict[str, Any]) -> pd.DataFrame:
    """
    Convertit un dictionnaire de features en DataFrame à une seule ligne.

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire contenant les variables d'entrée du modèle.

    Returns
    -------
    pd.DataFrame
        DataFrame à une ligne, compatible avec sklearn.

    Raises
    ------
    TypeError
        Si `features` n'est pas un dictionnaire.
    ValueError
        Si le dictionnaire est vide.
    """
    if not isinstance(features, dict):
        raise TypeError("`features` doit être un dictionnaire Python.")

    if not features:
        raise ValueError("`features` est vide. Impossible de prédire sans variables d'entrée.")

    # Une entrée API unitaire devient une seule ligne de DataFrame.
    return pd.DataFrame([features])


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie qu'un objet est bien un DataFrame pandas exploitable.

    Parameters
    ----------
    df : pd.DataFrame
        Données d'entrée du modèle.

    Returns
    -------
    pd.DataFrame
        Le DataFrame validé.

    Raises
    ------
    TypeError
        Si l'objet n'est pas un DataFrame pandas.
    ValueError
        Si le DataFrame est vide.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'entrée doit être un DataFrame pandas.")

    if df.empty:
        raise ValueError("Le DataFrame fourni est vide. Impossible de lancer une prédiction.")

    return df


# Fonctions publiques de prédiction
def make_prediction(features: dict[str, Any]) -> tuple[int, float]:
    """
    Réalise une prédiction unitaire à partir d'un dictionnaire de features.

    Cette fonction est adaptée à un usage API, par exemple lorsqu'un client
    envoie un JSON contenant les variables d'entrée du modèle.

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire contenant les variables d'entrée du modèle.

    Returns
    -------
    tuple[int, float]
        Un tuple contenant :
        - prediction : classe prédite (0 ou 1)
        - score : probabilité associée à la classe positive

    Notes
    -----
    - Le modèle est récupéré depuis `app.model_loader`.
    - Le seuil de décision est récupéré depuis `app.model_loader`.
    - Le score correspond à `predict_proba(...)[0][1]`.
    """
    # -------------------------------------------------------------------------
    # Récupération du modèle et du seuil déjà chargés en mémoire
    # -------------------------------------------------------------------------
    model = get_model()
    threshold = get_threshold()

    # -------------------------------------------------------------------------
    # Transformation des données d'entrée au format attendu par sklearn
    # -------------------------------------------------------------------------
    df = _ensure_dataframe_from_dict(features)

    # -------------------------------------------------------------------------
    # Calcul de la probabilité de la classe positive
    # -------------------------------------------------------------------------
    score = float(model.predict_proba(df)[0][1])

    # -------------------------------------------------------------------------
    # Conversion du score en décision métier via le seuil sauvegardé
    # -------------------------------------------------------------------------
    prediction = int(score >= threshold)

    return prediction, score, threshold


def make_batch_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réalise des prédictions sur un lot de données.

    Cette fonction est adaptée à un usage batch, par exemple pour prédire
    sur un fichier CSV complet ou sur des données lues depuis PostgreSQL.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les variables d'entrée du modèle.

    Returns
    -------
    pd.DataFrame
        Une copie du DataFrame d'entrée enrichie avec :
        - `score` : probabilité de la classe positive
        - `prediction` : classe prédite selon le seuil métier

    Notes
    -----
    - Le DataFrame retourné est une copie du DataFrame d'entrée.
    - La prédiction finale n'utilise pas `model.predict()` mais le seuil
      métier personnalisé appliqué sur `predict_proba`.
    """
    # -------------------------------------------------------------------------
    # Récupération du modèle et du seuil déjà chargés
    # -------------------------------------------------------------------------
    model = get_model()
    threshold = get_threshold()

    # -------------------------------------------------------------------------
    # Validation légère du DataFrame d'entrée
    # -------------------------------------------------------------------------
    df = _ensure_dataframe(df)

    # On travaille sur une copie pour éviter de modifier l'objet d'origine.
    result_df = df.copy()

    # -------------------------------------------------------------------------
    # Calcul des scores pour toutes les lignes
    # -------------------------------------------------------------------------
    scores = model.predict_proba(result_df)[:, 1]

    # -------------------------------------------------------------------------
    # Application du seuil métier
    # -------------------------------------------------------------------------
    predictions = (scores >= threshold).astype(int)

    # -------------------------------------------------------------------------
    # Ajout des résultats au DataFrame
    # -------------------------------------------------------------------------
    result_df["score"] = scores
    result_df["prediction"] = predictions

    return result_df


def make_prediction_from_dataframe(df: pd.DataFrame) -> tuple[int, float]:
    """
    Réalise une prédiction unitaire à partir d'un DataFrame contenant une seule ligne.

    Cette fonction peut être utile si les données ont déjà été converties
    en DataFrame plus haut dans le pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant exactement une ligne de données.

    Returns
    -------
    tuple[int, float]
        - prediction : classe prédite (0 ou 1)
        - score : probabilité associée à la classe positive

    Raises
    ------
    ValueError
        Si le DataFrame contient zéro ligne ou plus d'une ligne.
    """
    model = get_model()
    threshold = get_threshold()

    df = _ensure_dataframe(df)

    if len(df) != 1:
        raise ValueError(
            "make_prediction_from_dataframe attend un DataFrame avec exactement une ligne."
        )

    score = float(model.predict_proba(df)[0][1])
    prediction = int(score >= threshold)

    return prediction, score

def predict_one_row(df: pd.DataFrame) -> dict[str, Any]:
    prediction, score = make_prediction_from_dataframe(df)
    threshold = get_threshold()

    return {
        "prediction": prediction,
        "score": score,
        "threshold_used": threshold,
    }