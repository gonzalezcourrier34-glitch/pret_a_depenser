"""
Profiling d'inférence via l'API FastAPI.

Objectif
--------
Profiler la latence réelle du endpoint /predict.

Important
---------
Ce script envoie à l'API les features d'entrée attendues par /predict.
Il ne reconstruit pas lui-même les features selon sklearn ou ONNX.

C'est l'API qui gère le backend via MODEL_BACKEND :
- sklearn
- onnx
"""

from __future__ import annotations

import cProfile
import math
import pstats
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from app.core.config import API_KEY, API_URL, MODEL_BACKEND, MONITORING_DIR
from app.services.loader_services.data_loading_service import (
    get_reference_features_raw_df,
    init_monitoring_reference_cache,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("artifacts/performance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "profiling_report.txt"


# =============================================================================
# Nettoyage JSON
# =============================================================================

def make_json_safe(value: Any) -> Any:
    """
    Convertit une valeur pandas / numpy en valeur compatible JSON.

    JSON refuse :
    - NaN
    - inf
    - -inf
    - certains types numpy
    """
    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if pd.isna(value):
        return None

    return value


def clean_features_for_json(features: dict[str, Any]) -> dict[str, Any]:
    """
    Nettoie un dictionnaire de features avant envoi à l'API.
    """
    return {
        str(key): make_json_safe(value)
        for key, value in features.items()
    }


# =============================================================================
# Chargement des features API
# =============================================================================

def load_features_for_api(row_index: int = 0) -> dict[str, Any]:
    """
    Charge les features d'entrée attendues par /predict.

    La source utilisée est la référence raw de monitoring.
    Elle doit contenir les colonnes attendues par le schéma /predict.
    """
    init_monitoring_reference_cache(Path(MONITORING_DIR))

    df = get_reference_features_raw_df()

    if df.empty:
        raise ValueError("Le DataFrame reference_features_raw est vide.")

    if row_index >= len(df):
        raise IndexError(
            f"row_index={row_index} est hors limites. "
            f"Nombre de lignes disponibles : {len(df)}"
        )

    row = df.iloc[row_index].copy()

    # Colonnes qui ne doivent pas être envoyées au modèle
    row = row.drop(labels=["TARGET", "SK_ID_CURR"], errors="ignore")

    return clean_features_for_json(row.to_dict())


# =============================================================================
# Appel API
# =============================================================================

def call_predict(row_index: int = 0) -> dict[str, Any]:
    """
    Appelle l'endpoint /predict avec les features attendues.
    """
    features = load_features_for_api(row_index=row_index)

    response = requests.post(
        f"{API_URL}/predict",
        headers={"X-API-Key": API_KEY},
        json={"features": features},
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"{response.status_code} - {response.text}"
        )

    return response.json()


# =============================================================================
# Profiling
# =============================================================================

def run_profile(n_runs: int = 100) -> None:
    features = load_features_for_api(row_index=0)

    with requests.Session() as session:
        for _ in range(n_runs):
            response = session.post(
                f"{API_URL}/predict",
                headers={"X-API-Key": API_KEY},
                json={"features": features},
                timeout=30,
            )

            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} - {response.text}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"API_URL       : {API_URL}")
    print(f"MODEL_BACKEND : {MODEL_BACKEND}")
    print(f"Profiling vers: {OUTPUT_FILE.resolve()}")

    profiler = cProfile.Profile()
    profiler.enable()

    run_profile(n_runs=100)

    profiler.disable()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats(40)

    print(f"Profiling sauvegardé : {OUTPUT_FILE.resolve()}")