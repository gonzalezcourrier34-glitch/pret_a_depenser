"""
Service Evidently pour l'analyse et la persistance de dérive.

Ce module encapsule :
- le chargement des datasets de référence et courants
- l'exécution d'un rapport Evidently
- l'extraction des métriques utiles
- l'enregistrement des résultats via MonitoringService

Objectif
--------
Déporter hors des routes FastAPI toute la logique Evidently
afin de garder des routes fines et lisibles.

Notes
-----
- La lecture des fichiers / caches passe par data_loader_service.
- L'écriture en base passe par MonitoringService.
- Ce service orchestre donc la chaîne complète Evidently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy.orm import Session

from app.services.data_loader_service import (
    get_features_ready_cache,
    get_input_feature_names,
    get_raw_data_cache,
    get_reference_features_raw_df,
    get_reference_features_transformed_df,
    get_transformed_feature_names,
    init_monitoring_reference_cache,
)
from app.services.monitoring_service import MonitoringService


logger = logging.getLogger(__name__)


class EvidentlyService:
    """
    Service Evidently d'analyse et de persistance de drift.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.monitoring_service = MonitoringService(db)

    # =========================================================================
    # Helpers internes
    # =========================================================================

    def _ensure_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        """
        Vérifie qu'un objet est bien un DataFrame pandas non vide.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`{name}` doit être un DataFrame pandas.")

        if df.empty:
            raise ValueError(f"`{name}` est vide.")

        return df.copy()

    def _prepare_common_columns(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Aligne les deux DataFrames sur un jeu commun de colonnes.
        """
        ref = self._ensure_dataframe(reference_df, "reference_df")
        cur = self._ensure_dataframe(current_df, "current_df")

        common_cols = sorted(set(ref.columns).intersection(set(cur.columns)))

        if feature_names is not None:
            requested = [str(col) for col in feature_names]
            common_cols = [col for col in common_cols if col in requested]

        if not common_cols:
            raise ValueError(
                "Aucune colonne commune exploitable entre reference_df et current_df."
            )

        ref = ref[common_cols].copy()
        cur = cur[common_cols].copy()

        return ref, cur, common_cols

    def _safe_as_dict(
        self,
        report_object: object,
    ) -> dict[str, Any]:
        """
        Convertit un objet Evidently en dictionnaire Python.
        """
        if report_object is None:
            return {}

        if hasattr(report_object, "as_dict"):
            try:
                result = report_object.as_dict()  # type: ignore[attr-defined]
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        if hasattr(report_object, "dict"):
            try:
                result = report_object.dict()  # type: ignore[attr-defined]
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        if hasattr(report_object, "json"):
            try:
                raw = report_object.json()  # type: ignore[attr-defined]
                if isinstance(raw, str):
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
            except Exception:
                pass

        return {}

    def _save_html_report(
        self,
        report_object: object,
        save_html_path: str | None,
    ) -> str | None:
        """
        Sauvegarde le rapport HTML si un chemin est fourni.
        """
        if not save_html_path:
            return None

        output_path = Path(save_html_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(report_object, "save_html"):
            report_object.save_html(str(output_path))  # type: ignore[attr-defined]
            return str(output_path)

        raise AttributeError(
            "Le rapport Evidently ne supporte pas `save_html` dans cette version."
        )

    def _find_dataset_drift_block(
        self,
        report_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Cherche un bloc de résumé dataset-level dans le rapport.
        """
        if not isinstance(report_dict, dict):
            return {}

        for key in ["metrics", "results", "metric_results"]:
            metrics = report_dict.get(key)

            if isinstance(metrics, list):
                for item in metrics:
                    if not isinstance(item, dict):
                        continue

                    text = json.dumps(item, ensure_ascii=False).lower()

                    if "drift" in text and (
                        "number_of_drifted_columns" in text
                        or "share_of_drifted_columns" in text
                        or "dataset_drift" in text
                    ):
                        return item

        text = json.dumps(report_dict, ensure_ascii=False).lower()
        if (
            "number_of_drifted_columns" in text
            or "share_of_drifted_columns" in text
        ):
            return report_dict

        return {}

    def _coerce_int(self, value: object, default: int = 0) -> int:
        """
        Convertit une valeur en entier de façon robuste.
        """
        try:
            if value is None or pd.isna(value):
                return default
            return int(value)
        except Exception:
            return default

    def _coerce_float(self, value: object, default: float = 0.0) -> float:
        """
        Convertit une valeur en float de façon robuste.
        """
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def _load_reference_dataframe(
        self,
        *,
        reference_kind: Literal["raw", "transformed"],
    ) -> pd.DataFrame:
        """
        Charge le DataFrame de référence via data_loader_service.
        """
        if reference_kind == "raw":
            return get_reference_features_raw_df()

        return get_reference_features_transformed_df()

    def _load_current_dataframe(
        self,
        *,
        current_kind: Literal["raw", "transformed"],
    ) -> pd.DataFrame:
        """
        Charge le DataFrame courant via data_loader_service.
        """
        if current_kind == "raw":
            raw_cache = get_raw_data_cache()

            if "application" in raw_cache:
                return raw_cache["application"].copy()

            if "application_test" in raw_cache:
                return raw_cache["application_test"].copy()

            if "app" in raw_cache:
                return raw_cache["app"].copy()

            raise ValueError(
                "Impossible de construire current_df brut : "
                "aucune source brute compatible trouvée dans RAW_DATA_CACHE."
            )

        return get_features_ready_cache().copy()

    def _load_feature_names(
        self,
        *,
        reference_kind: Literal["raw", "transformed"],
    ) -> list[str] | None:
        """
        Charge la liste des colonnes à surveiller si disponible.
        """
        if reference_kind == "raw":
            feature_names = get_input_feature_names()
        else:
            feature_names = get_transformed_feature_names()

        return feature_names if feature_names else None

    def _build_response_payload(
        self,
        *,
        success: bool,
        message: str,
        model_name: str,
        model_version: str,
        reference_kind: Literal["raw", "transformed"],
        current_kind: Literal["raw", "transformed"],
        logged_metrics: int,
        html_report_path: str | None,
        reference_rows: int,
        current_rows: int,
        analyzed_columns: list[str],
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Construit un payload de réponse homogène pour succès et échec.
        """
        return {
            "success": success,
            "message": message,
            "model_name": model_name,
            "model_version": model_version,
            "reference_kind": reference_kind,
            "current_kind": current_kind,
            "logged_metrics": logged_metrics,
            "html_report_path": html_report_path,
            "reference_rows": reference_rows,
            "current_rows": current_rows,
            "analyzed_columns": analyzed_columns,
            "report": report,
        }

    # =========================================================================
    # Calcul Evidently
    # =========================================================================

    def run_data_drift_report(
        self,
        *,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None = None,
        save_html_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Exécute un rapport de drift de données avec Evidently.
        """
        ref, cur, used_columns = self._prepare_common_columns(
            reference_df=reference_df,
            current_df=current_df,
            feature_names=feature_names,
        )

        logger.info(
            "Starting Evidently data drift report",
            extra={
                "extra_data": {
                    "event": "evidently_report_start",
                    "reference_rows": len(ref),
                    "current_rows": len(cur),
                    "analyzed_columns_count": len(used_columns),
                    "save_html_path": save_html_path,
                }
            },
        )

        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
        except Exception as exc:
            logger.exception(
                "Failed to import Evidently",
                extra={
                    "extra_data": {
                        "event": "evidently_import_exception",
                        "reference_rows": len(ref),
                        "current_rows": len(cur),
                        "analyzed_columns_count": len(used_columns),
                        "error": str(exc),
                    }
                },
            )

            return {
                "success": False,
                "message": f"Impossible d'importer Evidently : {exc}",
                "report": {},
                "html_report_path": None,
                "reference_rows": len(ref),
                "current_rows": len(cur),
                "analyzed_columns": used_columns,
            }

        try:
            report = Report([DataDriftPreset()])
            report.run(current_data=cur, reference_data=ref)

            report_dict = self._safe_as_dict(report)
            html_report_path = self._save_html_report(
                report,
                save_html_path=save_html_path,
            )

            logger.info(
                "Evidently data drift report generated successfully",
                extra={
                    "extra_data": {
                        "event": "evidently_report_success",
                        "reference_rows": len(ref),
                        "current_rows": len(cur),
                        "analyzed_columns_count": len(used_columns),
                        "html_report_path": html_report_path,
                    }
                },
            )

            return {
                "success": True,
                "message": "Rapport Evidently généré avec succès.",
                "report": report_dict,
                "html_report_path": html_report_path,
                "reference_rows": len(ref),
                "current_rows": len(cur),
                "analyzed_columns": used_columns,
            }

        except Exception as exc:
            logger.exception(
                "Unexpected error during Evidently report execution",
                extra={
                    "extra_data": {
                        "event": "evidently_report_exception",
                        "reference_rows": len(ref),
                        "current_rows": len(cur),
                        "analyzed_columns_count": len(used_columns),
                        "save_html_path": save_html_path,
                        "error": str(exc),
                    }
                },
            )

            return {
                "success": False,
                "message": f"Erreur pendant l'exécution d'Evidently : {exc}",
                "report": {},
                "html_report_path": None,
                "reference_rows": len(ref),
                "current_rows": len(cur),
                "analyzed_columns": used_columns,
            }

    def extract_dataset_drift_summary(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extrait un résumé dataset-level du drift.
        """
        block = self._find_dataset_drift_block(report)

        raw_text = json.dumps(block, ensure_ascii=False)

        number_of_drifted_columns: int | None = None
        share_of_drifted_columns: float | None = None
        drift_detected = False

        for key in [
            "number_of_drifted_columns",
            "drifted_columns",
            "n_drifted_columns",
        ]:
            if key in raw_text:
                value = block.get(key)
                if value is not None:
                    parsed_value = self._coerce_int(value, default=-1)
                    if parsed_value >= 0:
                        number_of_drifted_columns = parsed_value
                        break

        for key in [
            "share_of_drifted_columns",
            "drift_share",
            "share_drifted_columns",
        ]:
            if key in raw_text:
                value = block.get(key)
                if value is not None:
                    parsed_value = self._coerce_float(value, default=-1.0)
                    if parsed_value >= 0:
                        share_of_drifted_columns = parsed_value
                        break

        for key in ["dataset_drift", "drift_detected", "drift"]:
            value = block.get(key)
            if isinstance(value, bool):
                drift_detected = value
                break

        if number_of_drifted_columns is not None and number_of_drifted_columns > 0:
            drift_detected = True

        summary = {
            "drift_detected": drift_detected,
            "number_of_drifted_columns": number_of_drifted_columns or 0,
            "share_of_drifted_columns": share_of_drifted_columns or 0.0,
            "raw_summary": block,
        }

        logger.info(
            "Extracted dataset drift summary",
            extra={
                "extra_data": {
                    "event": "evidently_extract_dataset_summary_success",
                    "drift_detected": summary["drift_detected"],
                    "number_of_drifted_columns": summary["number_of_drifted_columns"],
                    "share_of_drifted_columns": summary["share_of_drifted_columns"],
                }
            },
        )

        return summary

    def extract_drift_metrics_from_report(
        self,
        *,
        report: dict[str, Any],
        model_name: str,
        model_version: str,
        reference_window_start: object = None,
        reference_window_end: object = None,
        current_window_start: object = None,
        current_window_end: object = None,
    ) -> list[dict[str, Any]]:
        """
        Extrait une liste normalisée de métriques de drift.
        """
        dataset_summary = self.extract_dataset_drift_summary(report)

        rows: list[dict[str, Any]] = [
            {
                "model_name": model_name,
                "model_version": model_version,
                "feature_name": "__dataset__",
                "metric_name": "share_of_drifted_columns",
                "metric_value": self._coerce_float(
                    dataset_summary.get("share_of_drifted_columns"),
                    default=0.0,
                ),
                "threshold_value": None,
                "drift_detected": bool(dataset_summary.get("drift_detected", False)),
                "details": dataset_summary,
                "reference_window_start": reference_window_start,
                "reference_window_end": reference_window_end,
                "current_window_start": current_window_start,
                "current_window_end": current_window_end,
            }
        ]

        logger.info(
            "Extracted normalized drift metrics from Evidently report",
            extra={
                "extra_data": {
                    "event": "evidently_extract_metrics_success",
                    "model_name": model_name,
                    "model_version": model_version,
                    "logged_metrics": len(rows),
                }
            },
        )

        return rows

    # =========================================================================
    # Orchestration complète
    # =========================================================================

    def run_and_persist_data_drift_analysis(
        self,
        *,
        model_name: str,
        model_version: str | None,
        reference_kind: Literal["raw", "transformed"],
        current_kind: Literal["raw", "transformed"],
        monitoring_dir: str | None = None,
        save_html_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Charge les données, exécute Evidently, puis persiste les métriques.
        """
        resolved_model_version = model_version or "unknown"

        logger.info(
            "Starting end-to-end Evidently drift analysis",
            extra={
                "extra_data": {
                    "event": "evidently_analysis_start",
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "reference_kind": reference_kind,
                    "current_kind": current_kind,
                    "monitoring_dir": monitoring_dir,
                    "save_html_path": save_html_path,
                }
            },
        )

        if monitoring_dir:
            init_monitoring_reference_cache(Path(monitoring_dir))
        else:
            init_monitoring_reference_cache()

        reference_df = self._load_reference_dataframe(
            reference_kind=reference_kind,
        )
        current_df = self._load_current_dataframe(
            current_kind=current_kind,
        )
        feature_names = self._load_feature_names(
            reference_kind=reference_kind,
        )

        logger.info(
            "Reference and current datasets loaded for Evidently analysis",
            extra={
                "extra_data": {
                    "event": "evidently_analysis_data_loaded",
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "reference_kind": reference_kind,
                    "current_kind": current_kind,
                    "reference_rows": len(reference_df),
                    "current_rows": len(current_df),
                    "feature_names_count": len(feature_names) if feature_names else 0,
                }
            },
        )

        result = self.run_data_drift_report(
            reference_df=reference_df,
            current_df=current_df,
            feature_names=feature_names,
            save_html_path=save_html_path,
        )

        if not bool(result.get("success", False)):
            logger.warning(
                "Evidently drift analysis finished without success",
                extra={
                    "extra_data": {
                        "event": "evidently_analysis_failed_result",
                        "model_name": model_name,
                        "model_version": resolved_model_version,
                        "reference_kind": reference_kind,
                        "current_kind": current_kind,
                        "message": result.get("message"),
                    }
                },
            )

            return self._build_response_payload(
                success=False,
                message=str(result.get("message", "Échec de l'analyse Evidently.")),
                model_name=model_name,
                model_version=resolved_model_version,
                reference_kind=reference_kind,
                current_kind=current_kind,
                logged_metrics=0,
                html_report_path=(
                    str(result["html_report_path"])
                    if result.get("html_report_path") is not None
                    else None
                ),
                reference_rows=self._coerce_int(result.get("reference_rows"), 0),
                current_rows=self._coerce_int(result.get("current_rows"), 0),
                analyzed_columns=[
                    str(col)
                    for col in result.get("analyzed_columns", [])
                    if isinstance(col, (str, int, float))
                ],
                report=(
                    result.get("report", {})
                    if isinstance(result.get("report"), dict)
                    else {}
                ),
            )

        report = result.get("report", {})
        if not isinstance(report, dict):
            raise ValueError("Le rapport Evidently généré n'est pas exploitable.")

        drift_rows = self.extract_drift_metrics_from_report(
            report=report,
            model_name=model_name,
            model_version=resolved_model_version,
        )

        for row in drift_rows:
            self.monitoring_service.log_drift_metric(
                model_name=str(row["model_name"]),
                model_version=str(row["model_version"]),
                feature_name=str(row["feature_name"]),
                metric_name=str(row["metric_name"]),
                metric_value=float(row["metric_value"]),
                threshold_value=(
                    float(row["threshold_value"])
                    if row.get("threshold_value") is not None
                    else None
                ),
                drift_detected=bool(row.get("drift_detected", False)),
                details=(
                    row.get("details")
                    if isinstance(row.get("details"), dict)
                    else None
                ),
                reference_window_start=row.get("reference_window_start"),
                reference_window_end=row.get("reference_window_end"),
                current_window_start=row.get("current_window_start"),
                current_window_end=row.get("current_window_end"),
            )

        logger.info(
            "Evidently drift metrics persisted successfully",
            extra={
                "extra_data": {
                    "event": "evidently_analysis_persist_success",
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "logged_metrics": len(drift_rows),
                    "html_report_path": result.get("html_report_path"),
                }
            },
        )

        return self._build_response_payload(
            success=True,
            message="Analyse Evidently exécutée et persistée avec succès.",
            model_name=model_name,
            model_version=resolved_model_version,
            reference_kind=reference_kind,
            current_kind=current_kind,
            logged_metrics=len(drift_rows),
            html_report_path=(
                str(result["html_report_path"])
                if result.get("html_report_path") is not None
                else None
            ),
            reference_rows=self._coerce_int(result.get("reference_rows"), 0),
            current_rows=self._coerce_int(result.get("current_rows"), 0),
            analyzed_columns=[
                str(col)
                for col in result.get("analyzed_columns", [])
                if isinstance(col, (str, int, float))
            ],
            report=report,
        )

    def run_and_persist_data_drift_from_dataframes(
        self,
        *,
        model_name: str,
        model_version: str | None,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None = None,
        save_html_path: str | None = None,
        reference_window_start: object = None,
        reference_window_end: object = None,
        current_window_start: object = None,
        current_window_end: object = None,
    ) -> dict[str, object]:
        """
        Exécute une analyse de drift Evidently à partir de deux DataFrames déjà
        construits, puis persiste les métriques dans la base.
        """
        resolved_model_version = model_version or "unknown"

        logger.info(
            "Starting Evidently drift analysis from provided dataframes",
            extra={
                "extra_data": {
                    "event": "evidently_analysis_from_dataframes_start",
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "reference_rows": len(reference_df) if isinstance(reference_df, pd.DataFrame) else None,
                    "current_rows": len(current_df) if isinstance(current_df, pd.DataFrame) else None,
                    "feature_names_count": len(feature_names) if feature_names else 0,
                    "save_html_path": save_html_path,
                }
            },
        )

        result = self.run_data_drift_report(
            reference_df=reference_df,
            current_df=current_df,
            feature_names=feature_names,
            save_html_path=save_html_path,
        )

        if not bool(result.get("success", False)):
            logger.warning(
                "Evidently dataframe-based analysis finished without success",
                extra={
                    "extra_data": {
                        "event": "evidently_analysis_from_dataframes_failed_result",
                        "model_name": model_name,
                        "model_version": resolved_model_version,
                        "message": result.get("message"),
                    }
                },
            )

            return {
                "success": False,
                "message": result.get("message", "Échec de l'analyse Evidently."),
                "model_name": model_name,
                "model_version": resolved_model_version,
                "logged_metrics": 0,
                "html_report_path": result.get("html_report_path"),
                "reference_rows": result.get("reference_rows", 0),
                "current_rows": result.get("current_rows", 0),
                "analyzed_columns": result.get("analyzed_columns", []),
                "report": result.get("report", {}),
            }

        report = result.get("report", {})
        if not isinstance(report, dict):
            raise ValueError("Le rapport Evidently généré n'est pas exploitable.")

        drift_rows = self.extract_drift_metrics_from_report(
            report=report,
            model_name=model_name,
            model_version=resolved_model_version,
            reference_window_start=reference_window_start,
            reference_window_end=reference_window_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
        )

        for row in drift_rows:
            self.monitoring_service.log_drift_metric(
                model_name=str(row["model_name"]),
                model_version=str(row["model_version"]),
                feature_name=str(row["feature_name"]),
                metric_name=str(row["metric_name"]),
                metric_value=float(row["metric_value"]),
                threshold_value=(
                    float(row["threshold_value"])
                    if row.get("threshold_value") is not None
                    else None
                ),
                drift_detected=bool(row.get("drift_detected", False)),
                details=(
                    row.get("details")
                    if isinstance(row.get("details"), dict)
                    else None
                ),
                reference_window_start=row.get("reference_window_start"),
                reference_window_end=row.get("reference_window_end"),
                current_window_start=row.get("current_window_start"),
                current_window_end=row.get("current_window_end"),
            )

        logger.info(
            "Evidently dataframe-based drift metrics persisted successfully",
            extra={
                "extra_data": {
                    "event": "evidently_analysis_from_dataframes_persist_success",
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "logged_metrics": len(drift_rows),
                    "html_report_path": result.get("html_report_path"),
                }
            },
        )

        return {
            "success": True,
            "message": "Analyse Evidently exécutée et persistée avec succès.",
            "model_name": model_name,
            "model_version": resolved_model_version,
            "logged_metrics": len(drift_rows),
            "html_report_path": result.get("html_report_path"),
            "reference_rows": result.get("reference_rows", 0),
            "current_rows": result.get("current_rows", 0),
            "analyzed_columns": result.get("analyzed_columns", []),
            "report": report,
        }