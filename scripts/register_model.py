"""
Script d'enregistrement de la version de modèle déployée.

Ce script alimente la table `model_registry` avec la version réellement
servie par l'API.

Objectif
--------
Tracer proprement :
- le nom du modèle
- la version déployée
- le stage courant
- le chemin de l'artefact
- la liste des features attendues
- les métriques offline éventuelles
- le statut actif en production

Principe
--------
Ce script doit être exécuté au moment du déploiement, après que les
artefacts soient disponibles sur disque et avant ou juste après le
démarrage applicatif.

Variables d'environnement supportées
------------------------------------
- MODEL_NAME
- MODEL_VERSION
- MODEL_STAGE
- MODEL_PATH
- TRAINING_DATA_VERSION
- MODEL_RUN_ID
- MODEL_SOURCE_PATH
- MODEL_IS_ACTIVE
- MODEL_FEATURES_PATH
- MODEL_METRICS_PATH
- MODEL_HYPERPARAMETERS_PATH
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.db import SessionLocal
from app.services.monitoring_service import MonitoringService

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "credit_scoring_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
TRAINING_DATA_VERSION = os.getenv("TRAINING_DATA_VERSION")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID")
MODEL_SOURCE_PATH = os.getenv("MODEL_SOURCE_PATH", MODEL_PATH)

MODEL_IS_ACTIVE = os.getenv("MODEL_IS_ACTIVE", "true").strip().lower() in {
    "1", "true", "yes", "y"
}

MODEL_FEATURES_PATH = os.getenv(
    "MODEL_FEATURES_PATH",
    "artifacts/model_features.json",
)
MODEL_METRICS_PATH = os.getenv(
    "MODEL_METRICS_PATH",
    "artifacts/metrics.json",
)
MODEL_HYPERPARAMETERS_PATH = os.getenv(
    "MODEL_HYPERPARAMETERS_PATH",
    "artifacts/hyperparameters.json",
)


# =============================================================================
# Helpers
# =============================================================================

def _utc_now() -> datetime:
    """
    Retourne l'heure actuelle en UTC.
    """
    return datetime.now(timezone.utc)


def _read_json_file(path_str: str) -> Any | None:
    """
    Lit un fichier JSON s'il existe.

    Parameters
    ----------
    path_str : str
        Chemin du fichier JSON.

    Returns
    -------
    Any | None
        Contenu JSON ou None si absent/invalide.
    """
    path = Path(path_str)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARNING] Impossible de lire {path}: {exc}")
        return None


def _load_feature_list() -> list[str] | None:
    """
    Charge la liste des features attendues par le modèle.

    Formats supportés
    -----------------
    - JSON liste : ["f1", "f2", ...]
    - JSON dict  : {"features": ["f1", "f2", ...]}

    Returns
    -------
    list[str] | None
        Liste des features ou None.
    """
    data = _read_json_file(MODEL_FEATURES_PATH)

    if data is None:
        return None

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict) and isinstance(data.get("features"), list):
        return [str(x) for x in data["features"]]

    print(
        f"[WARNING] Format inattendu pour MODEL_FEATURES_PATH: {MODEL_FEATURES_PATH}"
    )
    return None


def _load_metrics() -> dict[str, Any] | None:
    """
    Charge les métriques offline du modèle.

    Returns
    -------
    dict[str, Any] | None
        Dictionnaire de métriques ou None.
    """
    data = _read_json_file(MODEL_METRICS_PATH)

    if data is None:
        return None

    if isinstance(data, dict):
        return data

    print(
        f"[WARNING] Format inattendu pour MODEL_METRICS_PATH: {MODEL_METRICS_PATH}"
    )
    return None


def _load_hyperparameters() -> dict[str, Any] | None:
    """
    Charge les hyperparamètres du modèle.

    Returns
    -------
    dict[str, Any] | None
        Dictionnaire d'hyperparamètres ou None.
    """
    data = _read_json_file(MODEL_HYPERPARAMETERS_PATH)

    if data is None:
        return None

    if isinstance(data, dict):
        return data

    print(
        "[WARNING] Format inattendu pour MODEL_HYPERPARAMETERS_PATH: "
        f"{MODEL_HYPERPARAMETERS_PATH}"
    )
    return None


def _validate_required_files() -> None:
    """
    Vérifie les artefacts indispensables.

    Raises
    ------
    FileNotFoundError
        Si le fichier modèle principal est absent.
    """
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Artefact modèle introuvable: {model_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Enregistre la version actuellement déployée dans model_registry.
    """
    print("=" * 80)
    print("ENREGISTREMENT DU MODÈLE DÉPLOYÉ")
    print("=" * 80)

    _validate_required_files()

    feature_list = _load_feature_list()
    metrics = _load_metrics()
    hyperparameters = _load_hyperparameters()

    print(f"MODEL_NAME              : {MODEL_NAME}")
    print(f"MODEL_VERSION           : {MODEL_VERSION}")
    print(f"MODEL_STAGE             : {MODEL_STAGE}")
    print(f"MODEL_PATH              : {MODEL_PATH}")
    print(f"MODEL_SOURCE_PATH       : {MODEL_SOURCE_PATH}")
    print(f"TRAINING_DATA_VERSION   : {TRAINING_DATA_VERSION}")
    print(f"MODEL_RUN_ID            : {MODEL_RUN_ID}")
    print(f"MODEL_IS_ACTIVE         : {MODEL_IS_ACTIVE}")
    print(f"NB FEATURES             : {len(feature_list) if feature_list else 0}")
    print(f"METRICS DISPONIBLES     : {list(metrics.keys()) if metrics else []}")

    db = SessionLocal()

    try:
        service = MonitoringService(db)

        result = service.register_model_version(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            stage=MODEL_STAGE,
            run_id=MODEL_RUN_ID,
            source_path=MODEL_SOURCE_PATH,
            training_data_version=TRAINING_DATA_VERSION,
            feature_list=feature_list,
            hyperparameters=hyperparameters,
            metrics=metrics,
            deployed_at=_utc_now(),
            is_active=MODEL_IS_ACTIVE,
        )

        db.commit()

        print("[OK] Modèle enregistré avec succès")
        print(result)

    except Exception as exc:
        db.rollback()
        print("[ERROR] Échec de l'enregistrement du modèle")
        print(f"Type    : {type(exc).__name__}")
        print(f"Message : {exc}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()