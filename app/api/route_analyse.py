"""
Routes FastAPI dédiées aux analyses de monitoring MLOps.

Ce module expose les endpoints permettant de lancer les analyses avancées
du système de scoring crédit :

- analyse de dérive des données avec Evidently ;
- analyse de dérive depuis les snapshots de production ;
- évaluation des performances du modèle en production.

Architecture générale
---------------------
Dans ce projet, le monitoring suit une logique MLOps simple :

1. Le modèle est utilisé en production via l'API FastAPI.
2. Chaque prédiction est journalisée dans PostgreSQL.
3. Les features utilisées en production sont stockées dans
   `feature_store_monitoring`.
4. Les analyses Evidently comparent :

   - un dataset de référence issu du training ;
   - un dataset courant issu de la production.

   Endpoints exposés
-----------------
- POST /analyse/evidently/run
    Analyse locale (debug) à partir des caches applicatifs

- POST /analyse/evidently/run-from-feature-store
    Analyse production basée sur les snapshots de features

- POST /analyse/evaluation/run
    Calcul des métriques de performance sur données réelles
    
Tables concernées
-----------------
- prediction_logs
- ground_truth_labels
- feature_store_monitoring
- drift_metrics
- evaluation_metrics

Notes
-----
- Ce module ne contient pas la logique métier lourde.
- Les calculs sont délégués aux services :
  `EvidentlyService` et `MonitoringEvaluationService`.
- Les routes sont protégées par une clé API.
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
from app.core.schemas import EvidentlyRunResponse, MonitoringEvaluationRunResponse
from app.core.security import verify_api_key
from app.services.analysis_services.evidently_service import EvidentlyService
from app.services.analysis_services.monitoring_evaluation_service import (
    MonitoringEvaluationService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyse",
    tags=["Analyse"],
)


# =============================================================================
# Helpers locaux
# =============================================================================

def _parse_optional_datetime(value: str | None) -> datetime | None:
    """
    Convertit une chaîne ISO-8601 optionnelle en objet datetime.

    Parameters
    ----------
    value : str | None
        Date reçue depuis les paramètres de requête.

    Returns
    -------
    datetime | None
        Date convertie, ou None si aucune valeur n'est fournie.

    Raises
    ------
    HTTPException
        Levée si le format de date est invalide.
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

    Parameters
    ----------
    monitoring_dir : str | None
        Dossier fourni explicitement dans la requête.

    Returns
    -------
    str
        Chemin final du dossier de monitoring.

    Notes
    -----
    Si aucun chemin n'est fourni, le dossier configuré dans
    `MONITORING_DIR` est utilisé.
    """
    if monitoring_dir is not None and str(monitoring_dir).strip():
        return str(Path(monitoring_dir))

    return str(MONITORING_DIR)


# =============================================================================
# Route Evidently depuis les caches applicatifs
# =============================================================================

@router.post(
    "/evidently/run",
    response_model=EvidentlyRunResponse,
    summary="Lancer une analyse Evidently depuis les caches applicatifs",
)
def run_evidently(
    model_name: str = Query(..., description="Nom du modèle surveillé."),
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
        description="Dossier contenant les artefacts de monitoring.",
    ),
    max_rows: int | None = Query(
        default=20000,
        ge=1,
        description="Nombre maximal de lignes analysées.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> EvidentlyRunResponse:
    """
    Lance une analyse Evidently à partir des caches applicatifs.

    Cette route sert principalement au debug et à la validation locale
    du pipeline de données.

    Principe
    --------
    Elle compare deux datasets déjà disponibles côté API :

    - un dataset de référence ;
    - un dataset courant reconstruit depuis les caches applicatifs.

    Les deux datasets peuvent être analysés sous deux formes :

    - `raw` : données brutes issues du CSV applicatif ;
    - `transformed` : features transformées réellement vues par le modèle.

    Contraintes
    -----------
    `reference_kind` et `current_kind` doivent être identiques.

    Exemple
    -------
    Comparaison correcte :

    - transformed vs transformed
    - raw vs raw

    Comparaison refusée :

    - raw vs transformed

    Notes
    -----
    Cette route n'est pas la route prioritaire pour le monitoring réel.
    Pour analyser les données réellement observées en production, utiliser :

    `/analyse/evidently/run-from-feature-store`
    """
    resolved_monitoring_dir = _resolve_monitoring_dir(monitoring_dir)

    logger.info(
        "Evidently run requested from cache",
        extra={
            "extra_data": {
                "event": "evidently_route_cache_start",
                "model_name": model_name,
                "model_version": model_version,
                "reference_kind": reference_kind,
                "current_kind": current_kind,
                "monitoring_dir": resolved_monitoring_dir,
                "max_rows": max_rows,
            }
        },
    )

    try:
        if reference_kind != current_kind:
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
            max_rows=max_rows,
        )

        if bool(result.get("success", False)):
            db.commit()
        else:
            db.rollback()

        return EvidentlyRunResponse(**result)

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()
        logger.exception(
            "Unexpected error during Evidently cache run",
            extra={
                "extra_data": {
                    "event": "evidently_route_cache_exception",
                    "model_name": model_name,
                    "model_version": model_version,
                    "error": str(exc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'analyse Evidently.",
        ) from exc


# =============================================================================
# Route Evidently depuis feature_store_monitoring
# =============================================================================

@router.post(
    "/evidently/run-from-feature-store",
    response_model=EvidentlyRunResponse,
    summary="Lancer une analyse Evidently depuis feature_store_monitoring",
)
def run_evidently_from_feature_store(
    model_name: str = Query(..., description="Nom du modèle surveillé."),
    model_version: str | None = Query(
        default=None,
        description="Version du modèle surveillé.",
    ),
    source_table: str | None = Query(
        default=None,
        description=(
            "Source à analyser dans feature_store_monitoring. "
            "Exemples : api_request, simulate_real_sample, random_simulation."
        ),
    ),
    max_rows: int = Query(
        default=1000,
        ge=1,
        le=10000,
        description="Nombre maximal de snapshots à reconstruire pour Evidently.",
    ),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> EvidentlyRunResponse:
    """
    Lance une analyse Evidently sur les données réellement observées.

    Cette route est la route principale pour le monitoring de production.

    Principe
    --------
    Evidently compare deux datasets :

    - `reference_df`
        Dataset de référence issu du training du modèle.
        Il provient des artefacts de monitoring, par exemple un fichier parquet
        comme `reference_features_raw.parquet` ou une version transformée.

    - `current_df`
        Dataset courant reconstruit depuis la table PostgreSQL
        `feature_store_monitoring`.

    Pipeline
    --------
    La table `feature_store_monitoring` stocke les features au format long :

    - request_id
    - feature_name
    - feature_value

    Le service reconstruit ensuite un DataFrame au format wide :

    - une ligne par requête ;
    - une colonne par feature.

    Ensuite :

    1. les colonnes communes sont identifiées ;
    2. les deux DataFrames sont alignés ;
    3. Evidently calcule le drift ;
    4. les métriques sont extraites ;
    5. les résultats sont enregistrés dans `drift_metrics`.

    Cas d'usage
    -----------
    Cette route doit être utilisée pour :

    - détecter une dérive après des prédictions réelles ;
    - analyser une simulation de batch ;
    - comparer le comportement actuel au dataset de référence ;
    - alimenter le dashboard Streamlit de monitoring.

    Notes
    -----
    Pour que l'analyse fonctionne, les noms de features stockées dans
    `feature_store_monitoring` doivent être compatibles avec les colonnes
    du dataset de référence.
    """
    logger.info(
        "Evidently run requested from feature store",
        extra={
            "extra_data": {
                "event": "evidently_route_feature_store_start",
                "model_name": model_name,
                "model_version": model_version,
                "source_table": source_table,
                "max_rows": max_rows,
            }
        },
    )

    try:
        service = EvidentlyService(db=db)

        result = service.run_and_persist_data_drift_from_feature_store(
            model_name=model_name,
            model_version=model_version,
            source_table=source_table,
            max_rows=max_rows,
        )

        if bool(result.get("success", False)):
            db.commit()
        else:
            db.rollback()

        logger.info(
            "Evidently run from feature store completed",
            extra={
                "extra_data": {
                    "event": "evidently_route_feature_store_completed",
                    "model_name": result.get("model_name", model_name),
                    "model_version": result.get("model_version", model_version),
                    "source_table": source_table,
                    "success": result.get("success"),
                    "logged_metrics": result.get("logged_metrics", 0),
                    "reference_rows": result.get("reference_rows", 0),
                    "current_rows": result.get("current_rows", 0),
                }
            },
        )

        return EvidentlyRunResponse(**result)

    except ValueError as exc:
        db.rollback()
        logger.warning(
            "Evidently feature store run rejected with validation error",
            extra={
                "extra_data": {
                    "event": "evidently_route_feature_store_value_error",
                    "model_name": model_name,
                    "model_version": model_version,
                    "source_table": source_table,
                    "max_rows": max_rows,
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
            "Unexpected error during Evidently run from feature store",
            extra={
                "extra_data": {
                    "event": "evidently_route_feature_store_exception",
                    "model_name": model_name,
                    "model_version": model_version,
                    "source_table": source_table,
                    "max_rows": max_rows,
                    "error": str(exc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Erreur interne lors de l'analyse Evidently depuis "
                "feature_store_monitoring."
            ),
        ) from exc


# =============================================================================
# Route évaluation monitoring
# =============================================================================

@router.post(
    "/evaluation/run",
    response_model=MonitoringEvaluationRunResponse,
    summary="Lancer une évaluation monitoring",
)
def run_monitoring_evaluation(
    model_name: str = Query(..., description="Nom du modèle évalué."),
    model_version: str | None = Query(
        default=None,
        description="Version du modèle évalué.",
    ),
    dataset_name: str = Query(
        default="scoring_prod",
        description="Nom logique du dataset d'évaluation.",
    ),
    window_start: str | None = Query(
        default=None,
        description="Début optionnel de la fenêtre d'évaluation.",
    ),
    window_end: str | None = Query(
        default=None,
        description="Fin optionnelle de la fenêtre d'évaluation.",
    ),
    beta: float = Query(
        default=2.0,
        gt=0,
        description="Coefficient beta pour le score F-beta.",
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
    Lance une évaluation des performances du modèle en production.

    Principe
    --------
    Cette route compare :

    - les prédictions stockées dans `prediction_logs` ;
    - les vérités terrain stockées dans `ground_truth_labels`.

    Elle calcule ensuite des métriques de performance utiles pour suivre
    la qualité du modèle dans le temps.

    Métriques calculées
    -------------------
    Selon les données disponibles, le service peut calculer :

    - ROC AUC ;
    - PR AUC ;
    - precision ;
    - recall ;
    - F1-score ;
    - F-beta score ;
    - matrice de confusion ;
    - coût métier.

    Coût métier
    -----------
    Le coût métier est utile pour un modèle de scoring crédit.

    Dans ce projet, un faux négatif peut être plus grave qu'un faux positif :

    - faux négatif : client risqué accepté ;
    - faux positif : client fiable refusé.

    Le coût est donc piloté par :

    - `cost_fn` ;
    - `cost_fp`.

    Persistance
    -----------
    Les résultats sont enregistrés dans la table `evaluation_metrics`.

    Notes
    -----
    Cette route nécessite que les prédictions et les vérités terrain puissent
    être reliées, généralement via `request_id` ou `client_id`.
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

        if bool(result.get("success", False)):
            db.commit()
        else:
            db.rollback()

        return MonitoringEvaluationRunResponse(**result)

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()
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
                    "error": str(exc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'évaluation monitoring.",
        ) from exc