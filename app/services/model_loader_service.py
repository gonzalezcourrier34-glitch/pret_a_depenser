"""
Chargement centralisé du modèle et du seuil de décision.

Ce module gère :
- le chargement unique du modèle sérialisé
- le chargement unique du seuil métier
- des outils de debug optionnels
- un test rapide de fonctionnement du modèle

Notes
-----
- Le chargement repose sur un mécanisme de singleton simple.
- Le debug est activable via la variable d'environnement `DEBUG_MODEL`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import DEBUG_MODEL, MODEL_PATH, THRESHOLD_PATH


# =============================================================================
# Variables globales (singletons)
# =============================================================================

_MODEL: Any | None = None
_THRESHOLD: float | None = None


# =============================================================================
# Utils debug
# =============================================================================

def _print_separator(title: str) -> None:
    """
    Affiche un séparateur lisible en console.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def debug_model(model: Any) -> None:
    """
    Affiche des informations utiles sur le modèle chargé.
    """
    _print_separator("DEBUG MODEL")

    print(f"Type modèle          : {type(model)}")

    if hasattr(model, "n_features_in_"):
        print(f"Nb features attendues: {model.n_features_in_}")

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        print("\nFeatures attendues :")
        print(feature_names[:20])

        if len(feature_names) > 20:
            print(f"... +{len(feature_names) - 20} autres")

    try:
        size_mb = sys.getsizeof(model) / 1024**2
        print(f"Taille mémoire      : {size_mb:.2f} Mo")
    except Exception:
        pass


def debug_threshold(threshold: float) -> None:
    """
    Affiche des informations utiles sur le seuil chargé.
    """
    _print_separator("DEBUG THRESHOLD")

    print(f"Seuil utilisé : {threshold}")

    if not (0 <= threshold <= 1):
        print("WARNING : seuil hors [0,1]")


# =============================================================================
# Chargement du modèle
# =============================================================================

def load_model() -> Any:
    """
    Charge le modèle depuis le disque.

    Returns
    -------
    Any
        Modèle chargé en mémoire.

    Raises
    ------
    FileNotFoundError
        Si le fichier du modèle est introuvable.
    Exception
        Si le chargement joblib échoue.
    """
    global _MODEL

    if _MODEL is None:
        model_path = Path(MODEL_PATH)

        print(f"[MODEL] Chargement depuis : {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        try:
            _MODEL = joblib.load(model_path)
            print("[MODEL] Modèle chargé avec succès")

            if DEBUG_MODEL:
                debug_model(_MODEL)

        except Exception as exc:
            _print_separator("ERREUR CHARGEMENT MODÈLE")
            print(f"Type erreur : {type(exc).__name__}")
            print(f"Message     : {exc}")
            raise

    return _MODEL


# =============================================================================
# Chargement du seuil
# =============================================================================

def load_threshold() -> float:
    """
    Charge le seuil métier depuis le disque.

    Si le fichier est absent ou invalide, un fallback à 0.5 est utilisé.

    Returns
    -------
    float
        Seuil de décision métier.
    """
    global _THRESHOLD

    if _THRESHOLD is None:
        threshold_path = Path(THRESHOLD_PATH)

        print(f"[THRESHOLD] Chargement depuis : {threshold_path}")

        try:
            if not threshold_path.exists():
                raise FileNotFoundError(f"Seuil introuvable : {threshold_path}")

            with open(threshold_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "threshold" not in data:
                raise KeyError("La clé 'threshold' est absente du fichier JSON.")

            threshold = float(data["threshold"])

            if not 0 <= threshold <= 1:
                raise ValueError("Le seuil doit être compris entre 0 et 1.")

            _THRESHOLD = threshold
            print(f"[THRESHOLD] Chargé : {_THRESHOLD}")

        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"[THRESHOLD] Erreur ou absence de seuil ({exc}) → fallback à 0.5")
            _THRESHOLD = 0.5

        if DEBUG_MODEL:
            debug_threshold(_THRESHOLD)

    return _THRESHOLD


# =============================================================================
# Test rapide du modèle (optionnel)
# =============================================================================

def test_model_prediction() -> None:
    """
    Test rapide pour vérifier que le modèle répond.

    Notes
    -----
    Ce test reste indicatif. Pour certains pipelines sklearn,
    un DataFrame avec noms de colonnes peut être nécessaire.
    """
    model = get_model()

    _print_separator("TEST MODEL")

    if not hasattr(model, "predict_proba"):
        print("Le modèle ne possède pas de méthode `predict_proba`.")
        return

    try:
        if hasattr(model, "feature_names_in_"):
            columns = list(model.feature_names_in_)
            X = pd.DataFrame([np.zeros(len(columns))], columns=columns)

        elif hasattr(model, "n_features_in_"):
            X = np.zeros((1, model.n_features_in_))

        else:
            print("Impossible de tester automatiquement (nb features inconnu)")
            return

        proba = model.predict_proba(X)

        print("Test OK")
        print(f"Output predict_proba : {proba}")

    except Exception as exc:
        print("ERREUR TEST MODÈLE")
        print(f"{type(exc).__name__} : {exc}")


# =============================================================================
# Reset cache (utile en tests)
# =============================================================================

def reset_model_cache() -> None:
    """
    Réinitialise le cache local du modèle et du seuil.

    Utile pour les tests unitaires ou pour forcer un rechargement.
    """
    global _MODEL, _THRESHOLD
    _MODEL = None
    _THRESHOLD = None


# =============================================================================
# Accès public
# =============================================================================

def get_model() -> Any:
    """
    Retourne le modèle chargé en mémoire.
    """
    return load_model()


def get_threshold() -> float:
    """
    Retourne le seuil chargé en mémoire.
    """
    return load_threshold()