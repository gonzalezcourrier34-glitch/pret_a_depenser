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
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import DEBUG_MODEL, MODEL_PATH, THRESHOLD_PATH


logger = logging.getLogger(__name__)


# =============================================================================
# Variables globales (singletons)
# =============================================================================

_MODEL: Any | None = None
_THRESHOLD: float | None = None


# =============================================================================
# Utils debug
# =============================================================================

def _log_separator(title: str) -> None:
    """
    Écrit un séparateur lisible dans les logs de debug.
    """
    logger.debug("=" * 80)
    logger.debug("%s", title)
    logger.debug("=" * 80)


def debug_model(model: Any) -> None:
    """
    Affiche des informations utiles sur le modèle chargé.
    """
    _log_separator("DEBUG MODEL")

    logger.debug("Type modèle : %s", type(model))

    if hasattr(model, "n_features_in_"):
        logger.debug("Nb features attendues : %s", model.n_features_in_)

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        logger.debug("Features attendues (aperçu) : %s", feature_names[:20])

        if len(feature_names) > 20:
            logger.debug("... +%s autres", len(feature_names) - 20)

    try:
        size_mb = sys.getsizeof(model) / 1024**2
        logger.debug("Taille mémoire : %.2f Mo", size_mb)
    except Exception:
        logger.debug("Impossible d'estimer la taille mémoire du modèle.")


def debug_threshold(threshold: float) -> None:
    """
    Affiche des informations utiles sur le seuil chargé.
    """
    _log_separator("DEBUG THRESHOLD")

    logger.debug("Seuil utilisé : %s", threshold)

    if not (0 <= threshold <= 1):
        logger.debug("WARNING : seuil hors [0,1]")


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

        logger.info(
            "Model loading started",
            extra={
                "extra_data": {
                    "event": "model_load_start",
                    "model_path": str(model_path),
                }
            },
        )

        if not model_path.exists():
            logger.error(
                "Model file not found",
                extra={
                    "extra_data": {
                        "event": "model_load_not_found",
                        "model_path": str(model_path),
                    }
                },
            )
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        try:
            _MODEL = joblib.load(model_path)

            logger.info(
                "Model loaded successfully",
                extra={
                    "extra_data": {
                        "event": "model_load_success",
                        "model_path": str(model_path),
                        "model_type": type(_MODEL).__name__,
                        "n_features_in": getattr(_MODEL, "n_features_in_", None),
                    }
                },
            )

            if DEBUG_MODEL:
                debug_model(_MODEL)

        except Exception as exc:
            logger.exception(
                "Unexpected error while loading model",
                extra={
                    "extra_data": {
                        "event": "model_load_exception",
                        "model_path": str(model_path),
                        "error": str(exc),
                    }
                },
            )
            raise

    else:
        logger.info(
            "Model already loaded, using cache",
            extra={
                "extra_data": {
                    "event": "model_load_cache_hit",
                    "model_type": type(_MODEL).__name__ if _MODEL is not None else None,
                }
            },
        )

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

        logger.info(
            "Threshold loading started",
            extra={
                "extra_data": {
                    "event": "threshold_load_start",
                    "threshold_path": str(threshold_path),
                }
            },
        )

        try:
            if not threshold_path.exists():
                raise FileNotFoundError(f"Seuil introuvable : {threshold_path}")

            with threshold_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if "threshold" not in data:
                raise KeyError("La clé 'threshold' est absente du fichier JSON.")

            threshold = float(data["threshold"])

            if not 0 <= threshold <= 1:
                raise ValueError("Le seuil doit être compris entre 0 et 1.")

            _THRESHOLD = threshold

            logger.info(
                "Threshold loaded successfully",
                extra={
                    "extra_data": {
                        "event": "threshold_load_success",
                        "threshold_path": str(threshold_path),
                        "threshold": _THRESHOLD,
                    }
                },
            )

        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            _THRESHOLD = 0.5

            logger.warning(
                "Threshold load failed, fallback applied",
                extra={
                    "extra_data": {
                        "event": "threshold_load_fallback",
                        "threshold_path": str(threshold_path),
                        "fallback_threshold": _THRESHOLD,
                        "error": str(exc),
                    }
                },
            )

        if DEBUG_MODEL:
            debug_threshold(_THRESHOLD)

    else:
        logger.info(
            "Threshold already loaded, using cache",
            extra={
                "extra_data": {
                    "event": "threshold_load_cache_hit",
                    "threshold": _THRESHOLD,
                }
            },
        )

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

    logger.info(
        "Model prediction smoke test started",
        extra={
            "extra_data": {
                "event": "model_test_start",
                "model_type": type(model).__name__,
            }
        },
    )

    if not hasattr(model, "predict_proba"):
        logger.warning(
            "Model does not expose predict_proba",
            extra={
                "extra_data": {
                    "event": "model_test_predict_proba_missing",
                    "model_type": type(model).__name__,
                }
            },
        )
        return

    try:
        if hasattr(model, "feature_names_in_"):
            columns = list(model.feature_names_in_)
            X = pd.DataFrame([np.zeros(len(columns))], columns=columns)

        elif hasattr(model, "n_features_in_"):
            X = np.zeros((1, model.n_features_in_))

        else:
            logger.warning(
                "Model test skipped, feature count unknown",
                extra={
                    "extra_data": {
                        "event": "model_test_feature_count_unknown",
                        "model_type": type(model).__name__,
                    }
                },
            )
            return

        proba = model.predict_proba(X)

        logger.info(
            "Model prediction smoke test succeeded",
            extra={
                "extra_data": {
                    "event": "model_test_success",
                    "output_shape": list(proba.shape) if hasattr(proba, "shape") else None,
                }
            },
        )

        logger.debug("Output predict_proba : %s", proba)

    except Exception as exc:
        logger.exception(
            "Model prediction smoke test failed",
            extra={
                "extra_data": {
                    "event": "model_test_exception",
                    "error": str(exc),
                }
            },
        )


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

    logger.info(
        "Model and threshold cache reset",
        extra={
            "extra_data": {
                "event": "model_cache_reset",
            }
        },
    )


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