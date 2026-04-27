"""
Service de gestion des assets via Hugging Face.

Ce module permet de :
- télécharger les fichiers nécessaires (CSV, modèle, seuil)
- éviter les téléchargements inutiles grâce à un cache local
- gérer plusieurs stratégies de chargement (local / auto / huggingface)

Objectif
--------
Rendre l'application indépendante de l'environnement :
- en local → utiliser les fichiers présents
- en Docker → télécharger automatiquement si absent
- en production → pouvoir forcer la synchronisation

Stratégies disponibles
----------------------
ASSETS_SOURCE=local
    → n'utilise que les fichiers locaux, aucun téléchargement

ASSETS_SOURCE=auto (recommandé)
    → télécharge uniquement les fichiers manquants

ASSETS_SOURCE=huggingface
    → force le téléchargement (refresh complet)

Dépendances
-----------
- huggingface_hub
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.core.config import (
    ASSETS_SOURCE,
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_TOKEN,
    APPLICATION_CSV,
    MODEL_PATH,
    ONNX_MODEL_PATH,
    THRESHOLD_PATH,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Téléchargement individuel
# =============================================================================

def _download_file(filename: str, local_path: str, *, force: bool = False) -> None:
    """
    Télécharge un fichier depuis Hugging Face vers un chemin local.

    Parameters
    ----------
    filename : str
        Chemin du fichier dans le repository Hugging Face.
    local_path : str
        Chemin local où enregistrer le fichier.
    force : bool, optional
        Si True, force le téléchargement même si le fichier existe déjà.

    Notes
    -----
    - Crée automatiquement les dossiers si nécessaires
    - Utilise un cache local pour éviter les téléchargements inutiles
    """
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Skip si déjà présent sauf si force
    if destination.exists() and not force:
        logger.info(
            "Asset already exists locally, skipping download",
            extra={
                "extra_data": {
                    "event": "hf_file_skip",
                    "filename": filename,
                    "local_path": str(destination),
                }
            },
        )
        return

    logger.info(
        "Downloading file from Hugging Face",
        extra={
            "extra_data": {
                "event": "hf_file_download_start",
                "filename": filename,
            }
        },
    )

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=filename,
        token=HF_TOKEN,
    )

    destination.write_bytes(Path(downloaded_path).read_bytes())

    logger.info(
        "File downloaded successfully",
        extra={
            "extra_data": {
                "event": "hf_file_download_success",
                "filename": filename,
                "local_path": str(destination),
            }
        },
    )


# =============================================================================
# Téléchargement global
# =============================================================================

def download_required_assets_from_huggingface(*, force: bool = False) -> None:
    """
    Télécharge tous les assets nécessaires à l'application.

    Parameters
    ----------
    force : bool, optional
        Si True, force le téléchargement de tous les fichiers.
    """
    _download_file("data/application_train.csv", APPLICATION_CSV, force=force)
    _download_file("artifacts/model.joblib", MODEL_PATH, force=force)
    _download_file("artifacts/model.onnx", ONNX_MODEL_PATH, force=force)
    _download_file("artifacts/threshold.json", THRESHOLD_PATH, force=force)


# =============================================================================
# Vérification et stratégie globale
# =============================================================================

def _get_missing_assets() -> list[str]:
    """
    Vérifie quels fichiers sont absents localement.

    Returns
    -------
    list[str]
        Liste des assets manquants.
    """
    missing = []

    if not Path(APPLICATION_CSV).exists():
        missing.append("CSV")

    if not Path(MODEL_PATH).exists():
        missing.append("MODEL")

    if not Path(ONNX_MODEL_PATH).exists():
        missing.append("ONNX")

    if not Path(THRESHOLD_PATH).exists():
        missing.append("THRESHOLD")

    return missing


def ensure_assets_available() -> None:
    """
    Point d'entrée principal pour garantir la disponibilité des assets.

    Cette fonction applique la stratégie définie par `ASSETS_SOURCE`.

    Comportement
    ------------
    - local → ne fait rien
    - auto → télécharge uniquement les fichiers manquants
    - huggingface → force le téléchargement complet

    Raises
    ------
    RuntimeError
        Si des fichiers sont manquants en mode local
    """

    logger.info(
        "Checking assets availability",
        extra={
            "extra_data": {
                "event": "assets_check_start",
                "assets_source": ASSETS_SOURCE,
            }
        },
    )

    # =========================
    # MODE LOCAL
    # =========================
    if ASSETS_SOURCE == "local":
        missing = _get_missing_assets()

        if missing:
            raise RuntimeError(
                f"Missing assets in local mode: {missing}. "
                "Switch to ASSETS_SOURCE=auto or huggingface."
            )

        logger.info(
            "Using local assets only",
            extra={"extra_data": {"event": "assets_local_ok"}},
        )
        return

    # =========================
    # MODE HUGGING FACE (FORCE)
    # =========================
    if ASSETS_SOURCE == "huggingface":
        logger.info(
            "Forcing download of all assets from Hugging Face",
            extra={"extra_data": {"event": "assets_force_download"}},
        )

        download_required_assets_from_huggingface(force=True)
        return

    # =========================
    # MODE AUTO (RECOMMANDÉ)
    # =========================
    if ASSETS_SOURCE == "auto":
        missing = _get_missing_assets()

        if missing:
            logger.info(
                "Missing assets detected, downloading from Hugging Face",
                extra={
                    "extra_data": {
                        "event": "assets_missing",
                        "missing": missing,
                    }
                },
            )

            download_required_assets_from_huggingface(force=False)

        else:
            logger.info(
                "All assets already available locally",
                extra={"extra_data": {"event": "assets_already_present"}},
            )

        return

    # =========================
    # CAS INVALIDE
    # =========================
    raise ValueError(f"Invalid ASSETS_SOURCE value: {ASSETS_SOURCE}")