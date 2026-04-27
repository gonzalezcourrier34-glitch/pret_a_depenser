# app/services/loader_services/model_loading_service.py

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
    logger.debug("=" * 80)
    logger.debug("%s", title)
    logger.debug("=" * 80)


def debug_model(model: Any) -> None:
    _log_separator("DEBUG MODEL")
    logger.debug("Type modèle : %s", type(model))

    if hasattr(model, "n_features_in_"):
        logger.debug("Nb features attendues : %s", model.n_features_in_)

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        logger.debug("Features attendues aperçu : %s", feature_names[:20])

    try:
        size_mb = sys.getsizeof(model) / 1024**2
        logger.debug("Taille mémoire : %.2f Mo", size_mb)
    except Exception:
        logger.debug("Impossible d'estimer la taille mémoire du modèle.")


def debug_threshold(threshold: float) -> None:
    _log_separator("DEBUG THRESHOLD")
    logger.debug("Seuil utilisé : %s", threshold)


# =============================================================================
# Chargement sklearn / joblib
# =============================================================================

def load_sklearn_model() -> Any:
    """
    Charge le modèle sklearn/joblib.
    """
    global _MODEL

    if _MODEL is not None:
        logger.info(
            "Sklearn model already loaded, using cache",
            extra={"extra_data": {"event": "sklearn_model_cache_hit"}},
        )
        return _MODEL

    model_path = Path(MODEL_PATH)

    logger.info(
        "Sklearn model loading started",
        extra={"extra_data": {"event": "sklearn_model_load_start", "model_path": str(model_path)}},
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle sklearn introuvable : {model_path}")

    _MODEL = joblib.load(model_path)

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
    Charge le modèle ONNX avec onnxruntime.
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
        extra={"extra_data": {"event": "onnx_session_load_start", "model_path": str(model_path)}},
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle ONNX introuvable : {model_path}")

    _ONNX_SESSION = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

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
    Charge le backend configuré.

    MODEL_BACKEND=sklearn -> joblib
    MODEL_BACKEND=onnx -> onnxruntime
    """
    backend = MODEL_BACKEND.lower().strip()

    if backend == "onnx":
        return load_onnx_session()

    if backend == "sklearn":
        return load_sklearn_model()

    raise ValueError(
        f"MODEL_BACKEND invalide : {MODEL_BACKEND}. Valeurs attendues : 'sklearn' ou 'onnx'."
    )


def predict_proba_with_backend(features_df: pd.DataFrame) -> float:
    """
    Calcule la probabilité de défaut avec le backend actif.

    Backend sklearn :
    - utilise directement le pipeline joblib.

    Backend ONNX :
    - alimente chaque feature attendue par ONNX séparément.
    - utile quand le modèle ONNX a été exporté avec un input par colonne.
    """
    backend = MODEL_BACKEND.lower().strip()

    if backend == "sklearn":
        model = load_sklearn_model()
        return float(model.predict_proba(features_df)[0, 1])

    if backend != "onnx":
        raise ValueError(
            f"MODEL_BACKEND invalide : {MODEL_BACKEND}. "
            "Valeurs attendues : 'sklearn' ou 'onnx'."
        )

    session = load_onnx_session()
    inputs = session.get_inputs()

    if features_df.empty:
        raise ValueError("features_df est vide.")

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

        # Tensor string
        if "string" in input_type:
            input_feed[name] = np.array(
                [[str(value) if value is not None else ""]],
                dtype=object,
            )

        # Tensor int
        elif "int64" in input_type:
            safe_value = 0 if value is None else int(value)
            input_feed[name] = np.array([[safe_value]], dtype=np.int64)

        # Tensor float/double
        else:
            safe_value = np.nan if value is None else float(value)
            input_feed[name] = np.array([[safe_value]], dtype=np.float32)

    outputs = session.run(None, input_feed)

    # Cas fréquent avec classifier ONNX :
    # outputs[0] = label
    # outputs[1] = probabilities
    proba = outputs[1] if len(outputs) > 1 else outputs[0]

    # Parfois proba est une liste de dicts si ZipMap=True
    if isinstance(proba, list) and proba and isinstance(proba[0], dict):
        return float(proba[0].get(1, proba[0].get("1")))

    proba = np.asarray(proba)

    if proba.ndim == 2 and proba.shape[1] >= 2:
        return float(proba[0, 1])

    if proba.ndim == 1:
        return float(proba[0])

    raise ValueError(f"Format de sortie ONNX inattendu : shape={proba.shape}")


# =============================================================================
# Seuil
# =============================================================================

def load_threshold() -> float:
    """
    Charge le seuil métier depuis le disque.
    """
    global _THRESHOLD

    if _THRESHOLD is not None:
        return _THRESHOLD

    threshold_path = Path(THRESHOLD_PATH)

    try:
        with threshold_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        _THRESHOLD = float(data["threshold"])

        if not 0 <= _THRESHOLD <= 1:
            raise ValueError("Le seuil doit être compris entre 0 et 1.")

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


# =============================================================================
# Reset cache
# =============================================================================

def reset_model_cache() -> None:
    """
    Réinitialise les caches modèle / ONNX / seuil.
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
    Retourne le backend chargé.
    """
    return load_model()


def get_sklearn_model() -> Any:
    """
    Retourne explicitement le modèle sklearn.
    """
    return load_sklearn_model()


def get_onnx_session() -> ort.InferenceSession:
    """
    Retourne explicitement la session ONNX.
    """
    return load_onnx_session()


def get_threshold() -> float:
    """
    Retourne le seuil métier.
    """
    return load_threshold()