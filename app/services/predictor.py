"""
Service de prédiction du modèle de scoring crédit.

Ce module encapsule la logique métier liée à l'inférence du modèle.
Il transforme les données d'entrée en format exploitable, puis
retourne la prédiction et le score associé.

Fonctionnalités
---------------
- Chargement du modèle
- Transformation des données
- Calcul de la prédiction et du score

Notes
-----
- Le modèle est chargé une seule fois au démarrage.
- Les données sont converties en DataFrame pour compatibilité sklearn.
"""

import pandas as pd
import joblib

from app.config import MODEL_PATH


# =============================================================================
# Chargement du modèle (au niveau global)
# =============================================================================

# Chargé une seule fois à l'import du module
model = joblib.load(MODEL_PATH)


# =============================================================================
# Fonction de prédiction
# =============================================================================

def make_prediction(features: dict) -> tuple[int, float]:
    """
    Réalise une prédiction à partir des variables d'entrée.

    Parameters
    ----------
    features : dict
        Dictionnaire contenant les variables d'entrée du modèle.

    Returns
    -------
    tuple[int, float]
        - prediction : classe prédite (0 ou 1)
        - score : probabilité associée à la classe positive

    Notes
    -----
    - Les données sont converties en DataFrame (format attendu par sklearn).
    - Le score correspond à la probabilité de la classe 1.
    """
    # -------------------------------------------------------------------------
    # Transformation en DataFrame
    # -------------------------------------------------------------------------
    df = pd.DataFrame([features])

    # -------------------------------------------------------------------------
    # Prédiction
    # -------------------------------------------------------------------------
    prediction = int(model.predict(df)[0])

    # -------------------------------------------------------------------------
    # Score (probabilité)
    # -------------------------------------------------------------------------
    score = float(model.predict_proba(df)[0][1])

    return prediction, score