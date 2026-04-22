"""
Routes FastAPI dédiées aux analyses de monitoring.

Ce module expose uniquement les endpoints HTTP.
Toute la logique métier est déléguée aux services spécialisés :

- EvidentlyService
- MonitoringEvaluationService

Objectif
--------
Garder des routes FastAPI fines, lisibles et orientées HTTP :
- validation des paramètres d'entrée
- gestion des erreurs HTTP
- gestion de la transaction SQLAlchemy
- délégation de la logique métier aux services

Notes
-----
- Les analyses de dérive passent par Evidently.
- Les analyses d'évaluation monitoring reposent sur les prédictions loguées
  et les vérités terrain disponibles.
- Si `monitoring_dir` n'est pas fourni, le dossier défini dans la
  configuration applicative est utilisé.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import MONITORING_DIR
from app.core.db import get_db
from app.core.schemas import (
    EvidentlyRunResponse,
    MonitoringEvaluationRunResponse,
)
from app.core.security import verify_api_key
from app.services.analysis_services.evidently_service import EvidentlyService
from app.services.analysis_services.monitoring_evaluation_service import (
    MonitoringEvaluationService,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Initialisation du routeur
# =============================================================================

router = APIRouter(
    prefix="/analyse",
    tags=["Analyse"],
)


# =============================================================================
# Helpers
# =============================================================================

def _parse_optional_datetime(value: str | None) -> datetime | None:
    """
    Convertit une date ISO optionnelle en objet datetime.

    Parameters
    ----------
    value : str | None
        Date au format ISO-8601 ou None.

    Returns
    -------
    datetime | None
        Date convertie, ou None si aucune valeur n'est fournie.

    Raises
    ------
    HTTPException
        Si le format de date n'est pas valide.
    """
    if value is None:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Format de date invalide : {value}. "
                "Format attendu : YYYY-MM-DDTHH:MM:SS ou ISO-8601 compatible."
            ),
        ) from exc


def _resolve_monitoring_dir(monitoring_dir: str | None) -> str:
    """
    Résout le dossier de monitoring à utiliser.

    Si un chemin est fourni dans la requête, il est utilisé.
    Sinon, on utilise le chemin par défaut défini dans la configuration.

    Parameters
    ----------
    monitoring_dir : str | None
        Dossier éventuel fourni par l'appelant.

    Returns
    -------
    str
        Chemin résolu du dossier de monitoring.
    """
    if monitoring_dir is not None and str(monitoring_dir).strip():
        return str(Path(monitoring_dir))

    return str(MONITORING_DIR)


# =============================================================================
# Routes Evidently
# =============================================================================

@router.post(
    "/evidently/run",
    response_model=EvidentlyRunResponse,
    summary="Lancer une analyse Evidently",
)
def run_evidently(
    model_name: str = Query(
        ...,
        description="Nom du modèle surveillé.",
    ),
    model_version: str | None = Query(
        default=None,
        description="Version du modèle surveillé.",
    ),
    reference_kind: Literal["raw", "transformed"] = Query(
        default="transformed",
        description="Type de dataset de référence à utiliser.",
    ),
    current_kind: Literal["raw", "transformed"] = Query(
        default="transformed",
        description="Type de dataset courant à utiliser.",
    ),
    monitoring_dir: str | None = Query(
        default=None,
        description=(
            "Dossier optionnel contenant les fichiers de monitoring. "
            "Si absent, la valeur de configuration MONITORING_DIR est utilisée."
        ),
    ),
    save_html_path: str | None = Query(
        default="artifacts/evidently/report.html",
        description="Chemin optionnel de sauvegarde du rapport HTML.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> EvidentlyRunResponse:
    """
    Lance une analyse Evidently de dérive de données.

    Notes
    -----
    - `reference_kind` et `current_kind` doivent être identiques.
    - La route ne contient pas de logique métier complexe.
    - La persistance des métriques est déléguée à EvidentlyService.
    """
    resolved_monitoring_dir = _resolve_monitoring_dir(monitoring_dir)

    logger.info(
        "Evidently run requested",
        extra={
            "extra_data": {
                "event": "evidently_route_start",
                "model_name": model_name,
                "model_version": model_version,
                "reference_kind": reference_kind,
                "current_kind": current_kind,
                "monitoring_dir": resolved_monitoring_dir,
                "save_html_path": save_html_path,
            }
        },
    )

    try:
        if reference_kind != current_kind:
            logger.warning(
                "Evidently run rejected due to incompatible dataset kinds",
                extra={
                    "extra_data": {
                        "event": "evidently_route_invalid_kinds",
                        "model_name": model_name,
                        "model_version": model_version,
                        "reference_kind": reference_kind,
                        "current_kind": current_kind,
                    }
                },
            )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "reference_kind et current_kind doivent être identiques "
                    "(`raw` avec `raw` ou `transformed` avec `transformed`)."
                ),
            )

        service = EvidentlyService(db=db)

        result = service.run_and_persist_data_drift_analysis(
            model_name=model_name,
            model_version=model_version,
            reference_kind=reference_kind,
            current_kind=current_kind,
            monitoring_dir=resolved_monitoring_dir,
            save_html_path=save_html_path,
        )

        success = bool(result.get("success", False))

        if success:
            db.commit()

            logger.info(
                "Evidently run completed successfully",
                extra={
                    "extra_data": {
                        "event": "evidently_route_success",
                        "model_name": result.get("model_name", model_name),
                        "model_version": result.get("model_version", model_version),
                        "reference_kind": result.get("reference_kind", reference_kind),
                        "current_kind": result.get("current_kind", current_kind),
                        "logged_metrics": result.get("logged_metrics", 0),
                        "reference_rows": result.get("reference_rows", 0),
                        "current_rows": result.get("current_rows", 0),
                        "html_report_path": result.get("html_report_path"),
                    }
                },
            )
        else:
            db.rollback()

            logger.warning(
                "Evidently run completed without success",
                extra={
                    "extra_data": {
                        "event": "evidently_route_failed_result",
                        "model_name": result.get("model_name", model_name),
                        "model_version": result.get("model_version", model_version),
                        "reference_kind": result.get("reference_kind", reference_kind),
                        "current_kind": result.get("current_kind", current_kind),
                        "message": result.get("message"),
                        "logged_metrics": result.get("logged_metrics", 0),
                        "reference_rows": result.get("reference_rows", 0),
                        "current_rows": result.get("current_rows", 0),
                        "html_report_path": result.get("html_report_path"),
                    }
                },
            )

        return EvidentlyRunResponse(**result)

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Evidently run rejected with validation error",
            extra={
                "extra_data": {
                    "event": "evidently_route_value_error",
                    "model_name": model_name,
                    "model_version": model_version,
                    "reference_kind": reference_kind,
                    "current_kind": current_kind,
                    "monitoring_dir": resolved_monitoring_dir,
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected error during Evidently run",
            extra={
                "extra_data": {
                    "event": "evidently_route_exception",
                    "model_name": model_name,
                    "model_version": model_version,
                    "reference_kind": reference_kind,
                    "current_kind": current_kind,
                    "monitoring_dir": resolved_monitoring_dir,
                    "save_html_path": save_html_path,
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'analyse Evidently.",
        ) from exc


# =============================================================================
# Routes évaluation monitoring
# =============================================================================

@router.post(
    "/evaluation/run",
    response_model=MonitoringEvaluationRunResponse,
    summary="Lancer une évaluation monitoring",
)
def run_monitoring_evaluation(
    model_name: str = Query(
        ...,
        description="Nom du modèle surveillé.",
    ),
    model_version: str | None = Query(
        default=None,
        description="Version du modèle surveillé.",
    ),
    dataset_name: str = Query(
        default="scoring_prod",
        description="Nom logique du dataset d'évaluation.",
    ),
    window_start: str | None = Query(
        default=None,
        description="Début optionnel de la fenêtre d'évaluation au format ISO.",
    ),
    window_end: str | None = Query(
        default=None,
        description="Fin optionnelle de la fenêtre d'évaluation au format ISO.",
    ),
    beta: float = Query(
        default=2.0,
        gt=0,
        description="Paramètre beta pour la métrique F-beta.",
    ),
    cost_fn: float = Query(
        default=10.0,
        ge=0,
        description="Coût métier associé à un faux négatif.",
    ),
    cost_fp: float = Query(
        default=1.0,
        ge=0,
        description="Coût métier associé à un faux positif.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> MonitoringEvaluationRunResponse:
    """
    Lance une évaluation monitoring à partir :
    - des prédictions journalisées
    - des vérités terrain disponibles

    Notes
    -----
    Les métriques calculées sont ensuite persistées via le service métier.
    """
    parsed_window_start = _parse_optional_datetime(window_start)
    parsed_window_end = _parse_optional_datetime(window_end)

    logger.info(
        "Monitoring evaluation run requested",
        extra={
            "extra_data": {
                "event": "monitoring_evaluation_route_start",
                "model_name": model_name,
                "model_version": model_version,
                "dataset_name": dataset_name,
                "window_start": (
                    parsed_window_start.isoformat()
                    if parsed_window_start is not None
                    else None
                ),
                "window_end": (
                    parsed_window_end.isoformat()
                    if parsed_window_end is not None
                    else None
                ),
                "beta": beta,
                "cost_fn": cost_fn,
                "cost_fp": cost_fp,
            }
        },
    )

    try:
        service = MonitoringEvaluationService(db=db)

        result = service.run_and_persist_monitoring_evaluation(
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            window_start=parsed_window_start,
            window_end=parsed_window_end,
            beta=beta,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )

        success = bool(result.get("success", False))

        if success:
            db.commit()

            logger.info(
                "Monitoring evaluation run completed successfully",
                extra={
                    "extra_data": {
                        "event": "monitoring_evaluation_route_success",
                        "model_name": result.get("model_name", model_name),
                        "model_version": result.get("model_version", model_version),
                        "dataset_name": result.get("dataset_name", dataset_name),
                        "logged_metrics": result.get("logged_metrics", 0),
                        "sample_size": result.get("sample_size", 0),
                        "matched_rows": result.get("matched_rows", 0),
                        "threshold_used": result.get("threshold_used"),
                    }
                },
            )
        else:
            db.rollback()

            logger.warning(
                "Monitoring evaluation run completed without success",
                extra={
                    "extra_data": {
                        "event": "monitoring_evaluation_route_failed_result",
                        "model_name": result.get("model_name", model_name),
                        "model_version": result.get("model_version", model_version),
                        "dataset_name": result.get("dataset_name", dataset_name),
                        "message": result.get("message"),
                        "logged_metrics": result.get("logged_metrics", 0),
                        "sample_size": result.get("sample_size", 0),
                        "matched_rows": result.get("matched_rows", 0),
                    }
                },
            )

        return MonitoringEvaluationRunResponse(**result)

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Monitoring evaluation rejected with validation error",
            extra={
                "extra_data": {
                    "event": "monitoring_evaluation_route_value_error",
                    "model_name": model_name,
                    "model_version": model_version,
                    "dataset_name": dataset_name,
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected error during monitoring evaluation run",
            extra={
                "extra_data": {
                    "event": "monitoring_evaluation_route_exception",
                    "model_name": model_name,
                    "model_version": model_version,
                    "dataset_name": dataset_name,
                    "window_start": (
                        parsed_window_start.isoformat()
                        if parsed_window_start is not None
                        else None
                    ),
                    "window_end": (
                        parsed_window_end.isoformat()
                        if parsed_window_end is not None
                        else None
                    ),
                    "beta": beta,
                    "cost_fn": cost_fn,
                    "cost_fp": cost_fp,
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'évaluation monitoring.",
        ) from exc