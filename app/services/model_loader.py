"""
Chargement centralisé du modèle et du seuil de décision avec debug.

Ajouts :
- affichage détaillé du modèle
- vérification des chemins
- inspection des features attendues
- debug activable via DEBUG_MODEL
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np

from app.core import config


# =============================================================================
# Configuration
# =============================================================================

DEBUG_MODEL = os.getenv("DEBUG_MODEL", "False").lower() == "true"


# =============================================================================
# Variables globales (singletons)
# =============================================================================

_MODEL = None
_THRESHOLD = None


# =============================================================================
# Utils debug
# =============================================================================

def _print_separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def debug_model(model):
    """
    Affiche des infos utiles sur le modèle.
    """
    _print_separator("DEBUG MODEL")

    print(f"Type modèle          : {type(model)}")

    # Nombre de features (si dispo)
    if hasattr(model, "n_features_in_"):
        print(f"Nb features attendues: {model.n_features_in_}")

    # Feature names (sklearn >=1.0)
    if hasattr(model, "feature_names_in_"):
        print("\nFeatures attendues :")
        print(list(model.feature_names_in_[:20]))

        if len(model.feature_names_in_) > 20:
            print(f"... +{len(model.feature_names_in_) - 20} autres")

    # Taille mémoire approximative
    try:
        import sys
        size_mb = sys.getsizeof(model) / 1024**2
        print(f"Taille mémoire      : {size_mb:.2f} Mo")
    except Exception:
        pass


def debug_threshold(threshold: float):
    _print_separator("DEBUG THRESHOLD")

    print(f"Seuil utilisé : {threshold}")

    if not (0 <= threshold <= 1):
        print("WARNING : seuil hors [0,1]")


# =============================================================================
# Chargement du modèle
# =============================================================================

def load_model():
    """
    Charge le modèle depuis le disque avec debug.
    """
    global _MODEL

    if _MODEL is None:
        model_path = Path(config.MODEL_PATH)

        print(f"[MODEL] Chargement depuis : {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        try:
            _MODEL = joblib.load(model_path)
            print("[MODEL] Modèle chargé avec succès")

            if DEBUG_MODEL:
                debug_model(_MODEL)

        except Exception as e:
            print("\n" + "=" * 80)
            print("ERREUR CHARGEMENT MODÈLE")
            print("=" * 80)
            print(f"Type erreur : {type(e).__name__}")
            print(f"Message     : {e}")
            raise

    return _MODEL


# =============================================================================
# Chargement du seuil
# =============================================================================

def load_threshold():
    """
    Charge le seuil métier avec debug.
    """
    global _THRESHOLD

    if _THRESHOLD is None:
        threshold_path = Path(config.THRESHOLD_PATH)

        print(f"[THRESHOLD] Chargement depuis : {threshold_path}")

        try:
            if not threshold_path.exists():
                raise FileNotFoundError

            with open(threshold_path, "r") as f:
                _THRESHOLD = json.load(f)["threshold"]

            print(f"[THRESHOLD] Chargé : {_THRESHOLD}")

        except Exception:
            print("[THRESHOLD] Non trouvé → fallback à 0.5")
            _THRESHOLD = 0.5

        if DEBUG_MODEL:
            debug_threshold(_THRESHOLD)

    return _THRESHOLD


# =============================================================================
# Test rapide du modèle (optionnel)
# =============================================================================

def test_model_prediction():
    """
    Test rapide pour vérifier que le modèle fonctionne.
    """
    model = get_model()

    _print_separator("TEST MODEL")

    try:
        if hasattr(model, "n_features_in_"):
            X = np.zeros((1, model.n_features_in_))
        else:
            print("Impossible de tester automatiquement (nb features inconnu)")
            return

        proba = model.predict_proba(X)

        print("Test OK")
        print(f"Output predict_proba : {proba}")

    except Exception as e:
        print("ERREUR TEST MODÈLE")
        print(f"{type(e).__name__} : {e}")


# =============================================================================
# Accès public
# =============================================================================

def get_model():
    return load_model()


def get_threshold() -> float:
    return load_threshold()