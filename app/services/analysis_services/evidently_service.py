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
- La lecture des fichiers / caches passe par data_loading_service.
- L'écriture en base passe par MonitoringService.
- Ce service orchestre donc la chaîne complète Evidently.
- La génération HTML locale n'est pas utilisée ici.
- Le dashboard reposera sur les résultats structurés renvoyés
  par Evidently et persistés en base.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import MONITORING_DIR
from app.services.features_builder_service import (
    build_transformed_features_from_loaded_data,
)
from app.services.loader_services.data_loading_service import (
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

    def _safe_as_dict(self, report_object: object) -> dict[str, Any]:
        """
        Convertit un objet Evidently en dictionnaire Python.

        Compatible avec plusieurs versions Evidently :
        - snapshot.json en propriété
        - snapshot.json() en méthode
        - dict()
        - as_dict()
        - model_dump()
        """
        if report_object is None:
            return {}

        def _parse_json_like(value: Any) -> dict[str, Any]:
            if isinstance(value, dict):
                return value

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return {}

                try:
                    parsed = json.loads(text)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}

            return {}

        # as_dict ou as_dict(include_render=False)
        attr = getattr(report_object, "as_dict", None)
        if attr is not None:
            try:
                result = attr(include_render=False) if callable(attr) else attr
                parsed = _parse_json_like(result)
                if parsed:
                    return parsed
            except TypeError:
                try:
                    result = attr() if callable(attr) else attr
                    parsed = _parse_json_like(result)
                    if parsed:
                        return parsed
                except Exception:
                    pass
            except Exception:
                pass

        # dict
        attr = getattr(report_object, "dict", None)
        if attr is not None:
            try:
                result = attr() if callable(attr) else attr
                parsed = _parse_json_like(result)
                if parsed:
                    return parsed
            except Exception:
                pass

        # model_dump
        attr = getattr(report_object, "model_dump", None)
        if attr is not None:
            try:
                result = attr() if callable(attr) else attr
                parsed = _parse_json_like(result)
                if parsed:
                    return parsed
            except Exception:
                pass

        # json : attention, selon Evidently ça peut être une propriété, pas une méthode
        attr = getattr(report_object, "json", None)
        if attr is not None:
            try:
                result = attr() if callable(attr) else attr
                parsed = _parse_json_like(result)
                if parsed:
                    return parsed
            except Exception:
                pass

        # _repr_json_
        attr = getattr(report_object, "_repr_json_", None)
        if attr is not None:
            try:
                result = attr() if callable(attr) else attr
                parsed = _parse_json_like(result)
                if parsed:
                    return parsed
            except Exception:
                pass

        return {}
    
    def _find_dataset_drift_block(
        self,
        report_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Recherche le vrai bloc dataset-level Evidently.

        Priorité :
        1. métrique Evidently récente DriftedColumnsCount
        2. anciens formats contenant number_of_drifted_columns / share_of_drifted_columns
        """
        if not isinstance(report_dict, dict):
            return {}

        metrics = report_dict.get("metrics", [])

        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue

                metric_name = str(metric.get("metric_name", ""))
                config = metric.get("config", {})
                value = metric.get("value", {})

                config_type = ""
                if isinstance(config, dict):
                    config_type = str(config.get("type", ""))

                if (
                    "DriftedColumnsCount" in metric_name
                    or "DriftedColumnsCount" in config_type
                ):
                    if isinstance(value, dict):
                        block = {
                            "number_of_drifted_columns": value.get("count"),
                            "share_of_drifted_columns": value.get("share"),
                            "drift_detected": self._coerce_float(value.get("count"), 0.0) > 0,
                            "raw_metric": metric,
                        }

                        logger.info(
                            "Dataset drift block found from DriftedColumnsCount",
                            extra={
                                "extra_data": {
                                    "event": "evidently_find_dataset_drift_block_from_metrics",
                                    "found": True,
                                    "number_of_drifted_columns": block["number_of_drifted_columns"],
                                    "share_of_drifted_columns": block["share_of_drifted_columns"],
                                }
                            },
                        )

                        return block

        target_keys = {
            "number_of_drifted_columns",
            "share_of_drifted_columns",
            "dataset_drift",
            "drift_detected",
            "n_drifted_columns",
            "drifted_columns",
        }

        def walk(obj: Any) -> dict[str, Any]:
            if isinstance(obj, dict):
                keys = set(obj.keys())

                # Important : ne pas accepter un simple {"type", "drift_share"}
                # car c'est souvent la config Evidently, pas le résultat.
                if keys.intersection(target_keys):
                    return obj

                for value in obj.values():
                    found = walk(value)
                    if found:
                        return found

            elif isinstance(obj, list):
                for item in obj:
                    found = walk(item)
                    if found:
                        return found

            return {}

        block = walk(report_dict)

        logger.info(
            "Dataset drift block search completed",
            extra={
                "extra_data": {
                    "event": "evidently_find_dataset_drift_block",
                    "found": bool(block),
                    "keys": list(block.keys()) if isinstance(block, dict) else [],
                }
            },
        )

        return block

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

    def _resolve_monitoring_dir(
        self,
        monitoring_dir: str | None,
    ) -> Path:
        """
        Résout le dossier de monitoring à utiliser.

        Priorité :
        1. paramètre explicite de la route
        2. variable d'environnement MONITORING_DIR
        """
        if monitoring_dir is not None and str(monitoring_dir).strip():
            return Path(str(monitoring_dir).strip())

        return Path(MONITORING_DIR)

    def _load_reference_dataframe(
        self,
        *,
        reference_kind: Literal["raw", "transformed"],
    ) -> pd.DataFrame:
        """
        Charge le DataFrame de référence via data_loading_service.
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
        Charge le DataFrame courant via les caches applicatifs.

        Notes
        -----
        - raw :
            dataset brut applicatif
        - transformed :
            dataset post-préprocessing, aligné sur les colonnes réellement
            vues par le modèle
        """
        raw_cache = get_raw_data_cache()

        if current_kind == "raw":
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

        transformed_df = build_transformed_features_from_loaded_data(
            raw_sources=raw_cache,
            client_ids=None,
            debug=False,
        )

        return self._ensure_dataframe(transformed_df, "current_df_transformed")

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

    def _limit_dataframe_rows(
        self,
        df: pd.DataFrame,
        max_rows: int | None,
        *,
        name: str,
    ) -> pd.DataFrame:
        """
        Limite le nombre de lignes d'un DataFrame si demandé.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame à limiter.
        max_rows : int | None
            Nombre maximal de lignes à conserver.
            Si None, aucune limitation n'est appliquée.
        name : str
            Nom logique du DataFrame pour les logs.

        Returns
        -------
        pd.DataFrame
            DataFrame éventuellement tronqué.
        """
        checked_df = self._ensure_dataframe(df, name)

        if max_rows is None:
            return checked_df

        if max_rows <= 0:
            raise ValueError("`max_rows` doit être strictement positif.")

        if len(checked_df) <= max_rows:
            return checked_df

        limited_df = checked_df.tail(max_rows).copy()

        logger.info(
            "Dataframe limited for Evidently analysis",
            extra={
                "extra_data": {
                    "event": "evidently_dataframe_limited",
                    "name": name,
                    "original_rows": len(checked_df),
                    "limited_rows": len(limited_df),
                    "max_rows": max_rows,
                }
            },
        )

        return limited_df

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
                "reference_rows": len(ref),
                "current_rows": len(cur),
                "analyzed_columns": used_columns,
            }

        try:
            report = Report([DataDriftPreset()])

            snapshot = report.run(
                current_data=cur,
                reference_data=ref,
            )

            # IMPORTANT :
            # Avec les versions récentes d'Evidently, le résultat exploitable
            # est souvent dans l'objet retourné par report.run(), donc `snapshot`.
            # `_safe_as_dict()` doit gérer json comme méthode OU comme propriété.
            report_dict = self._safe_as_dict(snapshot)

            if not report_dict:
                report_dict = self._safe_as_dict(report)

            logger.info(
                "Evidently objects debug",
                extra={
                    "extra_data": {
                        "event": "evidently_objects_debug",
                        "report_type": type(report).__name__,
                        "snapshot_type": type(snapshot).__name__,
                        "snapshot_is_none": snapshot is None,
                        "snapshot_has_json": hasattr(snapshot, "json"),
                        "snapshot_has_dict": hasattr(snapshot, "dict"),
                        "snapshot_has_as_dict": hasattr(snapshot, "as_dict"),
                        "snapshot_has_model_dump": hasattr(snapshot, "model_dump"),
                        "report_has_json": hasattr(report, "json"),
                        "report_has_dict": hasattr(report, "dict"),
                        "report_has_as_dict": hasattr(report, "as_dict"),
                        "report_dict_keys": (
                            list(report_dict.keys())
                            if isinstance(report_dict, dict)
                            else []
                        ),
                    }
                },
            )

            if not report_dict:
                logger.warning(
                    "Evidently report dictionary is empty after conversion attempts",
                    extra={
                        "extra_data": {
                            "event": "evidently_report_empty",
                            "reference_rows": len(ref),
                            "current_rows": len(cur),
                            "analyzed_columns_count": len(used_columns),
                        }
                    },
                )

            logger.info(
                "Evidently report top-level structure",
                extra={
                    "extra_data": {
                        "event": "evidently_report_structure_debug",
                        "top_level_keys": (
                            list(report_dict.keys())
                            if isinstance(report_dict, dict)
                            else []
                        ),
                        "preview": json.dumps(report_dict, default=str)[:3000],
                    }
                },
            )

            logger.info(
                "Evidently data drift report generated successfully",
                extra={
                    "extra_data": {
                        "event": "evidently_report_success",
                        "reference_rows": len(ref),
                        "current_rows": len(cur),
                        "analyzed_columns_count": len(used_columns),
                    }
                },
            )

            return {
                "success": True,
                "message": "Rapport Evidently généré avec succès.",
                "report": report_dict,
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
                        "error": str(exc),
                    }
                },
            )

            return {
                "success": False,
                "message": f"Erreur pendant l'exécution d'Evidently : {exc}",
                "report": {},
                "reference_rows": len(ref),
                "current_rows": len(cur),
                "analyzed_columns": used_columns,
            }

    def extract_dataset_drift_summary(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extrait un résumé global de dérive depuis le rapport Evidently.

        Compatible avec :
        - Evidently récent : {"metrics": [...], "tests": [...]}
        - anciens formats : blocs contenant number_of_drifted_columns, dataset_drift, etc.
        """
        metrics = report.get("metrics", [])

        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue

                metric_name = str(metric.get("metric_name", ""))
                config = metric.get("config", {})
                value = metric.get("value", {})

                config_type = ""
                if isinstance(config, dict):
                    config_type = str(config.get("type", ""))

                is_drifted_columns_count = (
                    "DriftedColumnsCount" in metric_name
                    or "DriftedColumnsCount" in config_type
                )

                if is_drifted_columns_count and isinstance(value, dict):
                    number_of_drifted_columns = self._coerce_int(
                        value.get("count"),
                        default=0,
                    )
                    share_of_drifted_columns = self._coerce_float(
                        value.get("share"),
                        default=0.0,
                    )

                    summary = {
                        "drift_detected": number_of_drifted_columns > 0,
                        "number_of_drifted_columns": number_of_drifted_columns,
                        "share_of_drifted_columns": share_of_drifted_columns,
                        "raw_summary": metric,
                    }

                    logger.info(
                        "Extracted dataset drift summary from Evidently metrics",
                        extra={
                            "extra_data": {
                                "event": "evidently_extract_dataset_summary_from_metrics_success",
                                "drift_detected": summary["drift_detected"],
                                "number_of_drifted_columns": summary["number_of_drifted_columns"],
                                "share_of_drifted_columns": summary["share_of_drifted_columns"],
                            }
                        },
                    )

                    return summary

        # Fallback pour les anciens formats Evidently
        block = self._find_dataset_drift_block(report)

        number_of_drifted_columns = self._coerce_int(
            block.get("number_of_drifted_columns")
            or block.get("n_drifted_columns")
            or block.get("drifted_columns"),
            default=0,
        )

        share_of_drifted_columns = self._coerce_float(
            block.get("share_of_drifted_columns")
            or block.get("drift_share"),
            default=0.0,
        )

        drift_detected_raw = (
            block.get("dataset_drift")
            if "dataset_drift" in block
            else block.get("drift_detected")
        )

        if isinstance(drift_detected_raw, bool):
            drift_detected = drift_detected_raw
        else:
            drift_detected = (
                number_of_drifted_columns > 0
                or share_of_drifted_columns > 0
            )

        summary = {
            "drift_detected": drift_detected,
            "number_of_drifted_columns": number_of_drifted_columns,
            "share_of_drifted_columns": share_of_drifted_columns,
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
                    "raw_summary_keys": list(block.keys()) if isinstance(block, dict) else [],
                }
            },
        )

        return summary
    
    def extract_feature_drift_rows_from_report(
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
        Extrait les métriques de drift par feature depuis le rapport Evidently.

        Objectif
        --------
        Alimenter le dashboard avec :
        - features les plus souvent en dérive
        - top scores de drift par feature
        - score moyen de drift dans le temps
        """
        rows: list[dict[str, Any]] = []

        metrics = report.get("metrics", [])
        if not isinstance(metrics, list):
            return rows

        for metric in metrics:
            if not isinstance(metric, dict):
                continue

            metric_name = str(metric.get("metric_name", ""))
            config = metric.get("config", {})
            value = metric.get("value", {})

            config_type = ""
            feature_name = None

            if isinstance(config, dict):
                config_type = str(config.get("type", ""))

                for key in ["column_name", "column", "feature_name"]:
                    if config.get(key) is not None:
                        feature_name = str(config.get(key))
                        break

            is_column_drift = (
                "ColumnDriftMetric" in metric_name
                or "ColumnDriftMetric" in config_type
                or "ValueDrift" in metric_name
                or "ValueDrift" in config_type
            )

            if not is_column_drift or not feature_name:
                continue

            if not isinstance(value, dict):
                continue

            drift_score = (
                value.get("drift_score")
                or value.get("score")
                or value.get("statistic")
                or value.get("p_value")
            )

            drift_detected = (
                value.get("drift_detected")
                or value.get("detected")
                or False
            )

            metric_value = self._coerce_float(drift_score, default=0.0)

            rows.append(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "feature_name": feature_name,
                    "metric_name": "feature_drift_score",
                    "metric_value": metric_value,
                    "threshold_value": None,
                    "drift_detected": bool(drift_detected),
                    "details": {
                        "metric_name": metric_name,
                        "config_type": config_type,
                        "feature_name": feature_name,
                        "raw_value": value,
                    },
                    "reference_window_start": reference_window_start,
                    "reference_window_end": reference_window_end,
                    "current_window_start": current_window_start,
                    "current_window_end": current_window_end,
                }
            )

        logger.info(
            "Extracted feature-level drift metrics from Evidently report",
            extra={
                "extra_data": {
                    "event": "evidently_extract_feature_metrics_success",
                    "model_name": model_name,
                    "model_version": model_version,
                    "feature_rows": len(rows),
                }
            },
        )

        return rows

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

        feature_rows = self.extract_feature_drift_rows_from_report(
            report=report,
            model_name=model_name,
            model_version=model_version,
            reference_window_start=reference_window_start,
            reference_window_end=reference_window_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
        )

        rows.extend(feature_rows)

        logger.info(
            "Extracted normalized drift metrics from Evidently report",
            extra={
                "extra_data": {
                    "event": "evidently_extract_metrics_success",
                    "model_name": model_name,
                    "model_version": model_version,
                    "logged_metrics": len(rows),
                    "feature_level_metrics": len(feature_rows),
                }
            },
        )

        return rows

    # =========================================================================
    # Orchestration complète
    # =========================================================================

    def run_and_persist_data_drift_from_feature_store(
        self,
        *,
        model_name: str,
        model_version: str | None,
        source_table: str | None = None,
        max_rows: int = 1000,
    ) -> dict[str, Any]:
        """
        Lance Evidently en comparant la référence modèle avec les snapshots
        stockés dans feature_store_monitoring.
        """
        resolved_model_version = model_version or "unknown"

        # IMPORTANT :
        # La référence Evidently est stockée dans MONITORING_REFERENCE_CACHE.
        # Si on lance Evidently depuis feature_store_monitoring, la route ne passe
        # pas par run_and_persist_data_drift_analysis(), donc il faut initialiser
        # explicitement ce cache ici.
        init_monitoring_reference_cache(Path(MONITORING_DIR))

        reference_df = get_reference_features_raw_df()
        feature_names = list(reference_df.columns)

        current_df = self.monitoring_service.get_feature_store_dataframe_for_drift(
            model_name=model_name,
            model_version=model_version,
            source_table=source_table,
            limit=max_rows,
        )

        common_cols_count = len(set(reference_df.columns).intersection(current_df.columns))

        logger.info(
            "Feature store dataframe reconstructed for Evidently",
            extra={
                "extra_data": {
                    "event": "evidently_feature_store_dataframe_ready",
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "source_table": source_table,
                    "reference_rows": len(reference_df),
                    "current_rows": len(current_df),
                    "reference_cols_count": len(reference_df.columns),
                    "current_cols_count": len(current_df.columns),
                    "common_cols_count": common_cols_count,
                    "reference_columns_sample": list(reference_df.columns)[:20],
                    "current_columns_sample": list(current_df.columns)[:20],
                }
            },
        )

        if current_df.empty:
            return self._build_response_payload(
                success=False,
                message="Aucune donnée trouvée dans feature_store_monitoring.",
                model_name=model_name,
                model_version=resolved_model_version,
                reference_kind="transformed",
                current_kind="transformed",
                logged_metrics=0,
                reference_rows=len(reference_df),
                current_rows=0,
                analyzed_columns=[],
                report={},
            )
        
        if common_cols_count == 0:
            return self._build_response_payload(
                success=False,
                message=(
                    "Aucune colonne commune entre la référence transformée "
                    "et les snapshots feature_store_monitoring. "
                    "Vérifie que les features loguées ont les mêmes noms que les features modèle."
                ),
                model_name=model_name,
                model_version=resolved_model_version,
                reference_kind="raw",
                current_kind="raw",
                logged_metrics=0,
                reference_rows=len(reference_df),
                current_rows=len(current_df),
                analyzed_columns=[],
                report={},
            )
        
        return self.run_and_persist_data_drift_from_dataframes(
            model_name=model_name,
            model_version=resolved_model_version,
            reference_df=reference_df,
            current_df=current_df,
            feature_names=feature_names,
            max_rows=max_rows,
        )

    def run_and_persist_data_drift_analysis(
        self,
        *,
        model_name: str,
        model_version: str | None,
        reference_kind: Literal["raw", "transformed"],
        current_kind: Literal["raw", "transformed"],
        monitoring_dir: str | None = None,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
        """
        Charge les données, exécute Evidently, puis persiste les métriques.

        Parameters
        ----------
        model_name : str
            Nom du modèle surveillé.
        model_version : str | None
            Version du modèle surveillé.
        reference_kind : Literal["raw", "transformed"]
            Nature du dataset de référence.
        current_kind : Literal["raw", "transformed"]
            Nature du dataset courant.
        monitoring_dir : str | None, default=None
            Dossier de monitoring à utiliser.
        max_rows : int | None, default=None
            Nombre maximal de lignes à analyser pour chaque dataset.
            Si None, aucune limitation n'est appliquée.
        """
        resolved_model_version = model_version or "unknown"
        resolved_monitoring_dir = self._resolve_monitoring_dir(monitoring_dir)

        logger.info(
            "Starting end-to-end Evidently drift analysis",
            extra={
                "extra_data": {
                    "event": "evidently_analysis_start",
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "reference_kind": reference_kind,
                    "current_kind": current_kind,
                    "monitoring_dir": str(resolved_monitoring_dir),
                    "max_rows": max_rows,
                }
            },
        )

        init_monitoring_reference_cache(resolved_monitoring_dir)

        reference_df = self._load_reference_dataframe(
            reference_kind=reference_kind,
        )
        current_df = self._load_current_dataframe(
            current_kind=current_kind,
        )
        feature_names = self._load_feature_names(
            reference_kind=reference_kind,
        )

        reference_df = self._limit_dataframe_rows(
            reference_df,
            max_rows,
            name="reference_df",
        )
        current_df = self._limit_dataframe_rows(
            current_df,
            max_rows,
            name="current_df",
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
                    "max_rows": max_rows,
                }
            },
        )

        result = self.run_data_drift_report(
            reference_df=reference_df,
            current_df=current_df,
            feature_names=feature_names,
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
                        "max_rows": max_rows,
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
                    "max_rows": max_rows,
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
        reference_window_start: object = None,
        reference_window_end: object = None,
        current_window_start: object = None,
        current_window_end: object = None,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
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
                    "reference_rows": (
                        len(reference_df)
                        if isinstance(reference_df, pd.DataFrame)
                        else None
                    ),
                    "current_rows": (
                        len(current_df)
                        if isinstance(current_df, pd.DataFrame)
                        else None
                    ),
                    "feature_names_count": len(feature_names) if feature_names else 0,
                    "max_rows": max_rows,
                }
            },
        )

        limited_reference_df = self._limit_dataframe_rows(
            reference_df,
            max_rows,
            name="reference_df",
        )
        limited_current_df = self._limit_dataframe_rows(
            current_df,
            max_rows,
            name="current_df",
        )

        result = self.run_data_drift_report(
            reference_df=limited_reference_df,
            current_df=limited_current_df,
            feature_names=feature_names,
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
                        "max_rows": max_rows,
                    }
                },
            )

            return {
                "success": False,
                "message": result.get("message", "Échec de l'analyse Evidently."),
                "model_name": model_name,
                "model_version": resolved_model_version,
                "logged_metrics": 0,
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
                    "max_rows": max_rows,
                }
            },
        )

        return {
            "success": True,
            "message": "Analyse Evidently exécutée et persistée avec succès.",
            "model_name": model_name,
            "model_version": resolved_model_version,
            "logged_metrics": len(drift_rows),
            "reference_rows": result.get("reference_rows", 0),
            "current_rows": result.get("current_rows", 0),
            "analyzed_columns": result.get("analyzed_columns", []),
            "report": report,
        }