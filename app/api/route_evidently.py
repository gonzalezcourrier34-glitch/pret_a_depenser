"""
Routes FastAPI dédiées aux analyses Evidently.

Ce module expose uniquement les endpoints HTTP.
Toute la logique métier est déléguée à EvidentlyService.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.schemas import EvidentlyRunResponse
from app.core.security import verify_api_key
from app.services.evidently_service import EvidentlyService


logger = logging.getLogger(__name__)


# =============================================================================
# Initialisation du routeur
# =============================================================================

router = APIRouter(
    prefix="/evidently",
    tags=["Evidently"],
)


# =============================================================================
# Routes
# =============================================================================

@router.post(
    "/run",
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
        description="Dossier optionnel contenant les fichiers de monitoring.",
    ),
    save_html_path: str | None = Query(
        default="artifacts/evidently/report.html",
        description="Chemin optionnel de sauvegarde du rapport HTML.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> EvidentlyRunResponse:
    """
    Lance une analyse Evidently.

    La route reste fine et délègue toute la logique
    au service Evidently.
    """
    logger.info(
        "Evidently run requested",
        extra={
            "extra_data": {
                "event": "evidently_route_start",
                "model_name": model_name,
                "model_version": model_version,
                "reference_kind": reference_kind,
                "current_kind": current_kind,
                "monitoring_dir": monitoring_dir,
                "save_html_path": save_html_path,
            }
        },
    )

    try:
        if reference_kind != current_kind:
            logger.warning(
                "Evidently run rejected بسبب incompatible dataset kinds",
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
            monitoring_dir=monitoring_dir,
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
                    "monitoring_dir": monitoring_dir,
                    "save_html_path": save_html_path,
                    "error": str(exc),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'analyse Evidently.",
        ) from exc