"""
Job batch d'évaluation du modèle en production.

Objectif
--------
Calculer automatiquement les métriques d'évaluation à partir :
- des prédictions historisées en base
- des vérités terrain disponibles
puis enregistrer ces métriques dans la table `evaluation_metrics`.

Ce script permet d'alimenter le dashboard de monitoring MLOps
avec des indicateurs tels que :
- recall
- precision
- roc_auc
- accuracy
- fbeta
- pr_auc
- latency moyenne
- coût métier
- matrice de confusion

Principe
--------
1. récupérer le modèle actif ou le modèle demandé
2. charger les prédictions loguées dans `prediction_logs`
3. charger les vérités terrain dans `ground_truth_labels`
4. faire la jointure via `request_id` ou `client_id`
5. calculer les métriques sur la fenêtre demandée
6. insérer une ligne par métrique dans `evaluation_metrics`

Notes
-----
- le script essaie d'être tolérant aux variations de noms de colonnes
  grâce à une détection dynamique des attributs SQLAlchemy
- si ton ORM diffère légèrement, il suffira souvent d'ajuster
  la liste des noms candidats dans les helpers
- ce script est pensé pour être lancé en batch manuel ou planifié
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.model.model_SQLalchemy import (
    EvaluationMetric,
    GroundTruthLabel,
    ModelRegistry,
    PredictionLog,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATASET_NAME = "scoring_prod"
DEFAULT_THRESHOLD_PATH = Path("artifacts/threshold.json")

COST_FN = 10.0
COST_FP = 1.0
BETA = 3.0


# =============================================================================
# Structures
# =============================================================================

@dataclass
class ActiveModelInfo:
    model_name: str
    model_version: str
    threshold: float | None


# =============================================================================
# Helpers ORM / introspection
# =============================================================================

def _get_model_columns(model_cls: type[Any]) -> set[str]:
    """
    Retourne l'ensemble des attributs mappés SQLAlchemy d'un modèle.
    """
    return set(model_cls.__table__.columns.keys())


def _pick_column(model_cls: type[Any], candidates: list[str]) -> str | None:
    """
    Sélectionne le premier nom de colonne existant dans le modèle.
    """
    cols = _get_model_columns(model_cls)
    for name in candidates:
        if name in cols:
            return name
    return None


def _safe_get(obj: Any, attr: str | None) -> Any:
    """
    Lit un attribut de manière sûre.
    """
    if attr is None:
        return None
    return getattr(obj, attr, None)


def _build_metric_row(
    *,
    model_name: str,
    model_version: str,
    dataset_name: str,
    metric_name: str,
    metric_value: float,
    sample_size: int,
    threshold_used: float | None,
    window_start: datetime | None,
    window_end: datetime | None,
) -> EvaluationMetric:
    """
    Construit dynamiquement une ligne EvaluationMetric en ne passant
    que les champs réellement présents dans le modèle ORM.
    """
    cols = _get_model_columns(EvaluationMetric)

    payload: dict[str, Any] = {}

    if "model_name" in cols:
        payload["model_name"] = model_name
    if "model_version" in cols:
        payload["model_version"] = model_version
    if "dataset_name" in cols:
        payload["dataset_name"] = dataset_name
    if "metric_name" in cols:
        payload["metric_name"] = metric_name
    if "metric_value" in cols:
        payload["metric_value"] = float(metric_value)
    if "sample_size" in cols:
        payload["sample_size"] = int(sample_size)
    if "threshold_used" in cols:
        payload["threshold_used"] = threshold_used
    if "window_start" in cols:
        payload["window_start"] = window_start
    if "window_end" in cols:
        payload["window_end"] = window_end
    if "created_at" in cols:
        payload["created_at"] = datetime.now(UTC)

    return EvaluationMetric(**payload)


# =============================================================================
# Helpers métier
# =============================================================================

def _parse_dt(value: str | None) -> datetime | None:
    """
    Parse une date ISO simple.
    """
    if not value:
        return None

    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _load_threshold_from_json(path: Path) -> float | None:
    """
    Charge le seuil depuis le fichier threshold.json si disponible.
    """
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    threshold = payload.get("threshold")
    return float(threshold) if threshold is not None else None


def _get_active_model_info(
    db: Session,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
) -> ActiveModelInfo:
    """
    Récupère le modèle actif ou le modèle explicitement demandé.
    """
    model_name_col = _pick_column(ModelRegistry, ["model_name"])
    model_version_col = _pick_column(ModelRegistry, ["model_version", "version"])
    is_active_col = _pick_column(ModelRegistry, ["is_active"])
    stage_col = _pick_column(ModelRegistry, ["stage"])

    query = db.query(ModelRegistry)

    if model_name and model_name_col:
        query = query.filter(getattr(ModelRegistry, model_name_col) == model_name)

    if model_version and model_version_col:
        query = query.filter(getattr(ModelRegistry, model_version_col) == model_version)
    else:
        if is_active_col:
            query = query.filter(getattr(ModelRegistry, is_active_col).is_(True))
        elif stage_col:
            query = query.filter(getattr(ModelRegistry, stage_col) == "production")

    row = query.order_by(ModelRegistry.id.desc()).first()
    if row is None:
        raise ValueError("Aucun modèle actif ou demandé trouvé dans model_registry.")

    resolved_model_name = _safe_get(row, model_name_col)
    resolved_model_version = _safe_get(row, model_version_col)

    threshold = _load_threshold_from_json(threshold_path)

    return ActiveModelInfo(
        model_name=str(resolved_model_name),
        model_version=str(resolved_model_version),
        threshold=threshold,
    )


def _load_prediction_logs(
    db: Session,
    *,
    model_name: str,
    model_version: str,
    window_start: datetime | None,
    window_end: datetime | None,
) -> pd.DataFrame:
    """
    Charge les prédictions depuis prediction_logs.
    """
    request_id_col = _pick_column(PredictionLog, ["request_id"])
    client_id_col = _pick_column(PredictionLog, ["client_id"])
    prediction_col = _pick_column(PredictionLog, ["prediction", "prediction_value"])
    score_col = _pick_column(PredictionLog, ["score", "proba", "probability", "prediction_score"])
    threshold_col = _pick_column(PredictionLog, ["threshold_used", "threshold"])
    latency_col = _pick_column(PredictionLog, ["latency_ms"])
    model_name_col = _pick_column(PredictionLog, ["model_name"])
    model_version_col = _pick_column(PredictionLog, ["model_version"])
    created_at_col = _pick_column(PredictionLog, ["created_at", "timestamp", "requested_at"])
    status_code_col = _pick_column(PredictionLog, ["status_code"])
    has_error_col = _pick_column(PredictionLog, ["has_error"])

    query = db.query(PredictionLog)

    filters = []
    if model_name_col:
        filters.append(getattr(PredictionLog, model_name_col) == model_name)
    if model_version_col:
        filters.append(getattr(PredictionLog, model_version_col) == model_version)
    if created_at_col and window_start is not None:
        filters.append(getattr(PredictionLog, created_at_col) >= window_start)
    if created_at_col and window_end is not None:
        filters.append(getattr(PredictionLog, created_at_col) <= window_end)
    if status_code_col:
        filters.append(getattr(PredictionLog, status_code_col) == 200)
    if has_error_col:
        filters.append(getattr(PredictionLog, has_error_col).is_(False))

    if filters:
        query = query.filter(and_(*filters))

    rows = query.all()

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "request_id": _safe_get(row, request_id_col),
                "client_id": _safe_get(row, client_id_col),
                "prediction": _safe_get(row, prediction_col),
                "score": _safe_get(row, score_col),
                "threshold_used": _safe_get(row, threshold_col),
                "latency_ms": _safe_get(row, latency_col),
                "created_at": _safe_get(row, created_at_col),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "prediction" in df.columns:
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "threshold_used" in df.columns:
        df["threshold_used"] = pd.to_numeric(df["threshold_used"], errors="coerce")
    if "latency_ms" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    return df


def _load_ground_truth_labels(
    db: Session,
    *,
    window_start: datetime | None,
    window_end: datetime | None,
) -> pd.DataFrame:
    """
    Charge les vérités terrain depuis ground_truth_labels.
    """
    request_id_col = _pick_column(GroundTruthLabel, ["request_id"])
    client_id_col = _pick_column(GroundTruthLabel, ["client_id"])
    target_col = _pick_column(
        GroundTruthLabel,
        ["target", "true_label", "ground_truth", "label", "actual_target"],
    )
    created_at_col = _pick_column(GroundTruthLabel, ["created_at", "timestamp", "labeled_at"])

    if target_col is None:
        raise ValueError(
            "Impossible d'identifier la colonne de vérité terrain dans GroundTruthLabel. "
            "Colonnes candidates attendues : target / true_label / ground_truth / label / actual_target."
        )

    query = db.query(GroundTruthLabel)

    filters = []
    if created_at_col and window_start is not None:
        filters.append(getattr(GroundTruthLabel, created_at_col) >= window_start)
    if created_at_col and window_end is not None:
        filters.append(getattr(GroundTruthLabel, created_at_col) <= window_end)

    if filters:
        query = query.filter(and_(*filters))

    rows = query.all()

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "request_id": _safe_get(row, request_id_col),
                "client_id": _safe_get(row, client_id_col),
                "y_true": _safe_get(row, target_col),
                "gt_created_at": _safe_get(row, created_at_col),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    return df


def _build_evaluation_dataframe(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joint les prédictions et la vérité terrain.
    Priorité à request_id, sinon fallback sur client_id.
    """
    if predictions_df.empty:
        raise ValueError("Aucune prédiction disponible pour la fenêtre demandée.")

    if ground_truth_df.empty:
        raise ValueError("Aucune vérité terrain disponible pour la fenêtre demandée.")

    can_join_on_request_id = (
        "request_id" in predictions_df.columns
        and "request_id" in ground_truth_df.columns
        and predictions_df["request_id"].notna().any()
        and ground_truth_df["request_id"].notna().any()
    )

    can_join_on_client_id = (
        "client_id" in predictions_df.columns
        and "client_id" in ground_truth_df.columns
        and predictions_df["client_id"].notna().any()
        and ground_truth_df["client_id"].notna().any()
    )

    if can_join_on_request_id:
        df = predictions_df.merge(
            ground_truth_df,
            on="request_id",
            how="inner",
            suffixes=("", "_gt"),
        )
    elif can_join_on_client_id:
        df = predictions_df.merge(
            ground_truth_df,
            on="client_id",
            how="inner",
            suffixes=("", "_gt"),
        )
    else:
        raise ValueError(
            "Impossible de joindre prediction_logs et ground_truth_labels : "
            "ni request_id ni client_id commun exploitable."
        )

    if df.empty:
        raise ValueError("La jointure entre prédictions et vérités terrain est vide.")

    df = df.dropna(subset=["y_true"])
    if df.empty:
        raise ValueError("Aucune ligne exploitable après suppression des y_true manquants.")

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "prediction" in df.columns:
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    return df


def _compute_metrics(
    df_eval: pd.DataFrame,
    *,
    default_threshold: float | None,
    cost_fn: float,
    cost_fp: float,
    beta: float,
) -> dict[str, float]:
    """
    Calcule les métriques d'évaluation.
    """
    y_true = df_eval["y_true"].astype(int).to_numpy()

    threshold_series = df_eval.get("threshold_used")
    effective_threshold = None

    if threshold_series is not None and pd.notna(threshold_series).any():
        effective_threshold = float(pd.to_numeric(threshold_series, errors="coerce").dropna().iloc[-1])
    elif default_threshold is not None:
        effective_threshold = float(default_threshold)

    if "score" in df_eval.columns and df_eval["score"].notna().any():
        y_score = pd.to_numeric(df_eval["score"], errors="coerce").fillna(0.0).to_numpy()
        if effective_threshold is None:
            effective_threshold = 0.5
        y_pred = (y_score >= effective_threshold).astype(int)
    elif "prediction" in df_eval.columns and df_eval["prediction"].notna().any():
        y_pred = pd.to_numeric(df_eval["prediction"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_score = None
        if effective_threshold is None:
            effective_threshold = 0.5
    else:
        raise ValueError("Impossible de calculer les métriques : ni score ni prediction disponible.")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "fbeta": float(fbeta),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "cost": float(cost_fn * fn + cost_fp * fp),
        "cost_per_sample": float((cost_fn * fn + cost_fp * fp) / max(len(df_eval), 1)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }

    if y_score is not None and len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))

    if "latency_ms" in df_eval.columns and df_eval["latency_ms"].notna().any():
        metrics["latency_mean_ms"] = float(pd.to_numeric(df_eval["latency_ms"], errors="coerce").mean())

    if effective_threshold is not None:
        metrics["threshold_used"] = float(effective_threshold)

    return metrics


def _delete_existing_metrics(
    db: Session,
    *,
    model_name: str,
    model_version: str,
    dataset_name: str,
    window_start: datetime | None,
    window_end: datetime | None,
) -> int:
    """
    Supprime les métriques déjà présentes pour rendre le job idempotent.
    """
    model_name_col = _pick_column(EvaluationMetric, ["model_name"])
    model_version_col = _pick_column(EvaluationMetric, ["model_version"])
    dataset_name_col = _pick_column(EvaluationMetric, ["dataset_name"])
    window_start_col = _pick_column(EvaluationMetric, ["window_start"])
    window_end_col = _pick_column(EvaluationMetric, ["window_end"])

    query = db.query(EvaluationMetric)

    filters = []
    if model_name_col:
        filters.append(getattr(EvaluationMetric, model_name_col) == model_name)
    if model_version_col:
        filters.append(getattr(EvaluationMetric, model_version_col) == model_version)
    if dataset_name_col:
        filters.append(getattr(EvaluationMetric, dataset_name_col) == dataset_name)
    if window_start_col:
        filters.append(getattr(EvaluationMetric, window_start_col) == window_start)
    if window_end_col:
        filters.append(getattr(EvaluationMetric, window_end_col) == window_end)

    if filters:
        query = query.filter(and_(*filters))

    count = query.count()
    if count > 0:
        query.delete(synchronize_session=False)

    return count


# =============================================================================
# Job principal
# =============================================================================

def run_monitoring_evaluation(
    *,
    model_name: str | None,
    model_version: str | None,
    dataset_name: str,
    window_start: datetime | None,
    window_end: datetime | None,
    threshold_path: Path,
    replace_existing: bool,
    cost_fn: float,
    cost_fp: float,
    beta: float,
) -> None:
    """
    Exécute le job complet d'évaluation monitoring.
    """
    db = SessionLocal()

    try:
        active_model = _get_active_model_info(
            db,
            model_name=model_name,
            model_version=model_version,
            threshold_path=threshold_path,
        )

        predictions_df = _load_prediction_logs(
            db,
            model_name=active_model.model_name,
            model_version=active_model.model_version,
            window_start=window_start,
            window_end=window_end,
        )

        ground_truth_df = _load_ground_truth_labels(
            db,
            window_start=window_start,
            window_end=window_end,
        )

        df_eval = _build_evaluation_dataframe(predictions_df, ground_truth_df)

        metrics = _compute_metrics(
            df_eval,
            default_threshold=active_model.threshold,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            beta=beta,
        )

        sample_size = len(df_eval)
        threshold_used = metrics.get("threshold_used")

        if replace_existing:
            deleted = _delete_existing_metrics(
                db,
                model_name=active_model.model_name,
                model_version=active_model.model_version,
                dataset_name=dataset_name,
                window_start=window_start,
                window_end=window_end,
            )
            if deleted > 0:
                print(f"[INFO] {deleted} métrique(s) existante(s) supprimée(s) avant réinsertion.")

        rows_to_insert: list[EvaluationMetric] = []
        for metric_name, metric_value in metrics.items():
            if metric_name == "threshold_used":
                continue

            rows_to_insert.append(
                _build_metric_row(
                    model_name=active_model.model_name,
                    model_version=active_model.model_version,
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    sample_size=sample_size,
                    threshold_used=threshold_used,
                    window_start=window_start,
                    window_end=window_end,
                )
            )

        db.add_all(rows_to_insert)
        db.commit()

        print("=" * 72)
        print("Évaluation monitoring terminée avec succès")
        print(f"Modèle        : {active_model.model_name}")
        print(f"Version       : {active_model.model_version}")
        print(f"Dataset       : {dataset_name}")
        print(f"Fenêtre start : {window_start}")
        print(f"Fenêtre end   : {window_end}")
        print(f"Échantillon   : {sample_size}")
        print(f"Seuil utilisé : {threshold_used}")
        print("-" * 72)
        for k, v in metrics.items():
            print(f"{k:20s}: {v}")
        print("=" * 72)

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcule et enregistre les métriques d'évaluation monitoring."
    )

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)

    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Date ISO de début, ex: 2026-04-01T00:00:00+00:00",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default=None,
        help="Date ISO de fin, ex: 2026-04-30T23:59:59+00:00",
    )

    parser.add_argument(
        "--threshold-path",
        type=str,
        default=str(DEFAULT_THRESHOLD_PATH),
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Supprime les métriques déjà présentes sur la même fenêtre avant insertion.",
    )

    parser.add_argument("--cost-fn", type=float, default=COST_FN)
    parser.add_argument("--cost-fp", type=float, default=COST_FP)
    parser.add_argument("--beta", type=float, default=BETA)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_monitoring_evaluation(
        model_name=args.model_name,
        model_version=args.model_version,
        dataset_name=args.dataset_name,
        window_start=_parse_dt(args.window_start),
        window_end=_parse_dt(args.window_end),
        threshold_path=Path(args.threshold_path),
        replace_existing=args.replace_existing,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()