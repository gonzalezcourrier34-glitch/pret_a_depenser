# app/services/loader_services/model_loading_service.py
"""
Service de chargement des modèles et du seuil métier.

Ce module centralise :
- le chargement du modèle sklearn/joblib
- le chargement du modèle ONNX
- le choix du backend via MODEL_BACKEND
- le chargement du seuil métier
- les helpers de debug
- le smoke test du modèle sklearn

Backends supportés
------------------
MODEL_BACKEND=sklearn
    Charge le pipeline joblib depuis MODEL_PATH.

MODEL_BACKEND=onnx
    Charge la session ONNX Runtime depuis ONNX_MODEL_PATH.

Notes
-----
- Le cache évite de recharger le modèle à chaque prédiction.
- `load_model()` respecte MODEL_BACKEND.
- `get_sklearn_model()` et `get_onnx_session()` permettent un accès explicite.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from app.core.config import (
    DEBUG_MODEL,
    MODEL_BACKEND,
    MODEL_PATH,
    ONNX_MODEL_PATH,
    THRESHOLD_PATH,
)

logger = logging.getLogger(__name__)

_MODEL: Any | None = None
_ONNX_SESSION: ort.InferenceSession | None = None
_THRESHOLD: float | None = None


# =============================================================================
# Debug
# =============================================================================

def _log_separator(title: str) -> None:
    """
    Affiche un séparateur lisible dans les logs de debug.
    """
    logger.debug("=" * 80)
    logger.debug("%s", title)
    logger.debug("=" * 80)


def debug_model(model: Any) -> None:
    """
    Log les informations utiles sur un modèle sklearn/joblib.
    """
    _log_separator("DEBUG MODEL")
    logger.debug("Type modèle : %s", type(model))

    if hasattr(model, "n_features_in_"):
        logger.debug("Nb features attendues : %s", model.n_features_in_)

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        logger.debug("Features attendues : %s", feature_names)
        logger.debug("Features attendues aperçu : %s", feature_names[:20])

    try:
        size_mb = sys.getsizeof(model) / 1024**2
        logger.debug("Taille mémoire : %.2f Mo", size_mb)
    except Exception:
        logger.debug("Impossible d'estimer la taille mémoire du modèle.")


def debug_threshold(threshold: float) -> None:
    """
    Log le seuil métier chargé.

    Important :
    La chaîne WARNING : seuil hors [0,1] est conservée telle quelle
    pour rester compatible avec les tests unitaires.
    """
    _log_separator("DEBUG THRESHOLD")
    logger.debug("Seuil utilisé : %s", threshold)

    if not 0 <= threshold <= 1:
        logger.debug("WARNING : seuil hors [0,1]")


# =============================================================================
# Chargement sklearn / joblib
# =============================================================================

def load_sklearn_model() -> Any:
    """
    Charge le modèle sklearn/joblib depuis MODEL_PATH.

    Retourne
    --------
    Any
        Le modèle sklearn ou pipeline sklearn chargé.

    Raises
    ------
    FileNotFoundError
        Si le fichier MODEL_PATH n'existe pas.
    RuntimeError
        Si joblib échoue pendant le chargement.
    """
    global _MODEL

    if _MODEL is not None:
        logger.info("Sklearn model already loaded, using cache")
        return _MODEL

    model_path = Path(MODEL_PATH)

    logger.info(
        "Sklearn model loading started",
        extra={
            "extra_data": {
                "event": "sklearn_model_load_start",
                "model_path": str(model_path),
            }
        },
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    try:
        _MODEL = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {exc}") from exc

    logger.info(
        "Sklearn model loaded successfully",
        extra={
            "extra_data": {
                "event": "sklearn_model_load_success",
                "model_path": str(model_path),
                "model_type": type(_MODEL).__name__,
                "n_features_in": getattr(_MODEL, "n_features_in_", None),
            }
        },
    )

    if DEBUG_MODEL:
        debug_model(_MODEL)

    return _MODEL


# =============================================================================
# Chargement ONNX
# =============================================================================

def load_onnx_session() -> ort.InferenceSession:
    """
    Charge une session ONNX Runtime depuis ONNX_MODEL_PATH.

    Retourne
    --------
    ort.InferenceSession
        Session ONNX mise en cache.

    Raises
    ------
    FileNotFoundError
        Si le fichier ONNX n'existe pas.
    RuntimeError
        Si ONNX Runtime échoue pendant le chargement.
    """
    global _ONNX_SESSION

    if _ONNX_SESSION is not None:
        logger.info(
            "ONNX session already loaded, using cache",
            extra={"extra_data": {"event": "onnx_session_cache_hit"}},
        )
        return _ONNX_SESSION

    model_path = Path(ONNX_MODEL_PATH)

    logger.info(
        "ONNX session loading started",
        extra={
            "extra_data": {
                "event": "onnx_session_load_start",
                "model_path": str(model_path),
            }
        },
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle ONNX introuvable : {model_path}")

    try:
        _ONNX_SESSION = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(f"Erreur lors du chargement du modèle ONNX : {exc}") from exc

    logger.info(
        "ONNX session loaded successfully",
        extra={
            "extra_data": {
                "event": "onnx_session_load_success",
                "model_path": str(model_path),
                "inputs": [i.name for i in _ONNX_SESSION.get_inputs()],
                "outputs": [o.name for o in _ONNX_SESSION.get_outputs()],
            }
        },
    )

    return _ONNX_SESSION


# =============================================================================
# Chargement selon backend
# =============================================================================

def load_model() -> Any:
    """
    Charge le backend configuré par MODEL_BACKEND.

    MODEL_BACKEND=sklearn
        Retourne le modèle joblib.

    MODEL_BACKEND=onnx
        Retourne la session ONNX Runtime.
    """
    backend = str(MODEL_BACKEND).lower().strip()

    if backend == "sklearn":
        return load_sklearn_model()

    if backend == "onnx":
        return load_onnx_session()

    raise ValueError(
        f"MODEL_BACKEND invalide : {MODEL_BACKEND}. "
        "Valeurs attendues : 'sklearn' ou 'onnx'."
    )


# =============================================================================
# Prédiction selon backend
# =============================================================================

def predict_proba_with_backend(features_df: pd.DataFrame) -> float:
    """
    Calcule la probabilité de défaut avec le backend actif.

    Pour sklearn :
    - appelle directement predict_proba sur le pipeline joblib.

    Pour ONNX :
    - construit le dictionnaire d'inputs attendu par ONNX Runtime.
    - récupère la probabilité de la classe positive.

    Paramètres
    ----------
    features_df:
        DataFrame contenant une ligne client déjà préparée.

    Retourne
    --------
    float
        Probabilité de défaut de paiement.
    """
    if features_df.empty:
        raise ValueError("features_df est vide.")

    backend = str(MODEL_BACKEND).lower().strip()

    if backend == "sklearn":
        model = load_sklearn_model()

        if not hasattr(model, "predict_proba"):
            raise AttributeError("Le modèle sklearn ne possède pas predict_proba.")

        proba = model.predict_proba(features_df)

        if proba is None or getattr(proba, "ndim", 0) != 2 or proba.shape[1] < 2:
            raise ValueError("Format de sortie predict_proba invalide.")

        return float(proba[0, 1])

    if backend != "onnx":
        raise ValueError(
            f"MODEL_BACKEND invalide : {MODEL_BACKEND}. "
            "Valeurs attendues : 'sklearn' ou 'onnx'."
        )

    session = load_onnx_session()
    inputs = session.get_inputs()
    row = features_df.iloc[0]

    input_feed: dict[str, np.ndarray] = {}

    for input_meta in inputs:
        name = input_meta.name
        input_type = input_meta.type

        if name not in row.index:
            raise ValueError(
                f"La feature ONNX attendue `{name}` est absente du DataFrame."
            )

        value = row[name]

        if pd.isna(value):
            value = None

        if "string" in input_type:
            input_feed[name] = np.array(
                [[str(value) if value is not None else ""]],
                dtype=object,
            )

        elif "int64" in input_type:
            safe_value = 0 if value is None else int(value)
            input_feed[name] = np.array([[safe_value]], dtype=np.int64)

        else:
            safe_value = np.nan if value is None else float(value)
            input_feed[name] = np.array([[safe_value]], dtype=np.float32)

    outputs = session.run(None, input_feed)

    proba = outputs[1] if len(outputs) > 1 else outputs[0]

    if isinstance(proba, list) and proba and isinstance(proba[0], dict):
        class_one_proba = proba[0].get(1, proba[0].get("1"))

        if class_one_proba is None:
            raise ValueError("La probabilité de classe 1 est absente de la sortie ONNX.")

        return float(class_one_proba)

    proba_array = np.asarray(proba)

    if proba_array.ndim == 2 and proba_array.shape[1] >= 2:
        return float(proba_array[0, 1])

    if proba_array.ndim == 1:
        return float(proba_array[0])

    raise ValueError(f"Format de sortie ONNX inattendu : shape={proba_array.shape}")


# =============================================================================
# Seuil
# =============================================================================

def load_threshold() -> float:
    """
    Charge le seuil métier depuis THRESHOLD_PATH.

    Si le fichier est absent, invalide, ou si le seuil est hors [0,1],
    la valeur de fallback 0.5 est utilisée.
    """
    global _THRESHOLD

    if _THRESHOLD is not None:
        return _THRESHOLD

    threshold_path = Path(THRESHOLD_PATH)

    try:
        with threshold_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

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

    except Exception as exc:
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

    return _THRESHOLD


def get_threshold() -> float:
    """
    Retourne le seuil métier chargé.
    """
    return load_threshold()


# =============================================================================
# Smoke test modèle
# =============================================================================

def test_model_prediction() -> None:
    """
    Réalise un smoke test sur le modèle actif.

    Cette fonction est principalement utile pour :
    - vérifier qu'un modèle sklearn possède predict_proba
    - vérifier qu'une prédiction minimale fonctionne
    - conserver la compatibilité avec les tests unitaires existants

    Notes
    -----
    Le smoke test est pensé pour sklearn/joblib.
    Si MODEL_BACKEND vaut onnx, il est préférable de tester la session ONNX
    via `predict_proba_with_backend`.
    """
    model = get_model()

    if isinstance(model, ort.InferenceSession):
        logger.info("smoke test skipped for ONNX session")
        return

    if not hasattr(model, "predict_proba"):
        logger.warning("predict_proba unavailable on loaded model")
        return

    feature_names = getattr(model, "feature_names_in_", None)
    n_features = getattr(model, "n_features_in_", None)

    if feature_names is not None:
        columns = list(feature_names)
        sample = pd.DataFrame([[0.0] * len(columns)], columns=columns)

    elif n_features is not None:
        sample = np.zeros((1, int(n_features)))

    else:
        logger.warning("feature count unknown")
        return

    try:
        model.predict_proba(sample)
        logger.info("smoke test succeeded")
    except Exception as exc:
        logger.error("smoke test failed: %s", exc)


# =============================================================================
# Reset cache
# =============================================================================

def reset_model_cache() -> None:
    """
    Réinitialise les caches modèle, session ONNX et seuil.
    """
    global _MODEL, _ONNX_SESSION, _THRESHOLD

    _MODEL = None
    _ONNX_SESSION = None
    _THRESHOLD = None


# =============================================================================
# Accès public
# =============================================================================

def get_model() -> Any:
    """
    Retourne le backend chargé selon MODEL_BACKEND.
    """
    return load_model()


def get_sklearn_model() -> Any:
    """
    Retourne explicitement le modèle sklearn/joblib.
    """
    return load_sklearn_model()


def get_onnx_session() -> ort.InferenceSession:
    """
    Retourne explicitement la session ONNX Runtime.
    """
    return load_onnx_session()