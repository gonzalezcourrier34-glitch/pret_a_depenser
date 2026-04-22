from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.evidently_service import EvidentlyService
from app.services.monitoring_service import MonitoringService

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
REFERENCE_PATH = os.getenv(
    "REFERENCE_FEATURES_PATH",
    "artifacts/reference_features.parquet",
)
MODEL_NAME = os.getenv("MODEL_NAME", "credit_scoring_model")
SAVE_HTML_PATH = os.getenv(
    "EVIDENTLY_HTML_PATH",
    "artifacts/reports/latest_drift_report.html",
)
CURRENT_WINDOW_DAYS = int(os.getenv("CURRENT_WINDOW_DAYS", "7"))

if not DATABASE_URL:
    raise ValueError("DATABASE_URL n'est pas défini.")


def load_reference_dataframe(path: str) -> pd.DataFrame:
    """
    Charge le dataset de référence utilisé pour le drift.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier de référence introuvable : {file_path}")

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(
            f"Format non supporté pour le fichier de référence : {file_path}"
        )

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Le dataset de référence est vide ou invalide.")

    return df


def _maybe_convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tente de convertir en numérique uniquement les colonnes qui semblent l'être,
    sans écraser brutalement les colonnes catégorielles.
    """
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")

            # On ne remplace que si une part non négligeable de la colonne
            # semble réellement numérique.
            if converted.notna().sum() > 0:
                out[col] = converted.where(converted.notna(), out[col])

    return out


def build_current_dataframe(
    service: MonitoringService,
    model_name: str,
    model_version: str,
    *,
    window_start: datetime,
    window_end: datetime,
) -> pd.DataFrame:
    """
    Reconstruit un dataset courant à partir du feature_store_monitoring.

    Principe
    --------
    - une ligne = un request_id
    - une colonne = une feature_name
    - la valeur = feature_value la plus récente pour cette requête/feature
    """
    feature_store = service.get_feature_store(
        limit=50000,
        model_name=model_name,
        model_version=model_version,
        window_start=window_start,
        window_end=window_end,
    )

    items = feature_store.get("items", [])
    if not items:
        return pd.DataFrame()

    raw_df = pd.DataFrame(items)

    if raw_df.empty:
        return pd.DataFrame()

    required_cols = {"request_id", "feature_name", "feature_value"}
    missing_cols = required_cols - set(raw_df.columns)
    if missing_cols:
        raise ValueError(
            f"Colonnes absentes du feature store pour construire current_df : {sorted(missing_cols)}"
        )

    pivot_df = raw_df.pivot_table(
        index="request_id",
        columns="feature_name",
        values="feature_value",
        aggfunc="last",
    ).reset_index(drop=True)

    if pivot_df.empty:
        return pd.DataFrame()

    pivot_df = _maybe_convert_numeric_columns(pivot_df)

    return pivot_df


def main() -> None:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(bind=engine)

    reference_df = load_reference_dataframe(REFERENCE_PATH)

    current_window_end = datetime.now(timezone.utc)
    current_window_start = current_window_end - timedelta(days=CURRENT_WINDOW_DAYS)

    with SessionLocal() as db:
        monitoring_service = MonitoringService(db)
        evidently_service = EvidentlyService(db)

        active_model = monitoring_service.get_active_model(model_name=MODEL_NAME)
        if active_model is None:
            print("Aucun modèle actif trouvé.")
            return

        current_df = build_current_dataframe(
            service=monitoring_service,
            model_name=active_model.model_name,
            model_version=active_model.model_version,
            window_start=current_window_start,
            window_end=current_window_end,
        )

        if current_df.empty:
            print("Aucune donnée courante disponible.")
            return

        result = evidently_service.run_and_persist_data_drift_from_dataframes(
            model_name=active_model.model_name,
            model_version=active_model.model_version,
            reference_df=reference_df,
            current_df=current_df,
            feature_names=active_model.feature_list,
            save_html_path=SAVE_HTML_PATH,
            reference_window_start=None,
            reference_window_end=None,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
        )

        if bool(result.get("success", False)):
            db.commit()
        else:
            db.rollback()

        print(result.get("message", "Analyse terminée."))
        print(result)


if __name__ == "__main__":
    main()