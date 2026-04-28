"""
Benchmark d'inférence via l'API FastAPI.

Ce script mesure la latence du endpoint optimisé :
GET /predict/{client_id}

Avantage :
- payload très léger
- pas d'envoi de toutes les features en JSON
- utilisation du cache de features côté API
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.core.config import API_KEY, API_URL, MODEL_BACKEND


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def call_predict(
    *,
    session: requests.Session,
    client_id: int,
) -> dict[str, Any]:
    """
    Appelle l'endpoint optimisé /predict/{client_id}.
    """
    response = session.get(
        f"{API_URL}/predict/{client_id}",
        headers={"X-API-Key": API_KEY},
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} - {response.text}")

    return response.json()


def benchmark(
    *,
    client_id: int,
    n_runs: int = 300,
    warmup_runs: int = 10,
    output_file: str,
) -> pd.DataFrame:
    """
    Lance plusieurs prédictions et mesure la latence API.
    """
    rows: list[dict[str, Any]] = []

    with requests.Session() as session:
        for _ in range(warmup_runs):
            try:
                call_predict(session=session, client_id=client_id)
            except Exception:
                pass

        for i in range(n_runs):
            start = time.perf_counter()

            try:
                result = call_predict(
                    session=session,
                    client_id=client_id,
                )
                success = True
                error = None

            except Exception as exc:
                result = {}
                success = False
                error = str(exc)

            latency_ms = (time.perf_counter() - start) * 1000

            rows.append(
                {
                    "run": i + 1,
                    "backend": MODEL_BACKEND,
                    "client_id": client_id,
                    "endpoint": f"/predict/{client_id}",
                    "success": success,
                    "latency_ms": latency_ms,
                    "prediction": result.get("prediction"),
                    "probability": result.get("probability"),
                    "score": result.get("score"),
                    "model_version": result.get("model_version"),
                    "api_latency_ms": result.get("latency_ms"),
                    "error": error,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / output_file, index=False)
    return df


def print_summary(df: pd.DataFrame, output_file: str) -> None:
    """
    Affiche un résumé lisible du benchmark.
    """
    valid = df[df["success"]].copy()
    failed = df[~df["success"]].copy()

    print("\nRésultats")
    print(f"Backend : {MODEL_BACKEND}")
    print(f"Fichier : {(OUTPUT_DIR / output_file).resolve()}")
    print(f"Total runs : {len(df)}")
    print(f"Succès : {len(valid)}")
    print(f"Échecs : {len(failed)}")

    if valid.empty:
        print("\nAucune prédiction réussie.")
        if not failed.empty:
            print("\nPremière erreur :")
            print(failed.iloc[0]["error"])
        return

    print(f"\nLatence client moyenne : {valid['latency_ms'].mean():.2f} ms")
    print(f"Médiane client : {valid['latency_ms'].median():.2f} ms")
    print(f"P95 client : {valid['latency_ms'].quantile(0.95):.2f} ms")
    print(f"P99 client : {valid['latency_ms'].quantile(0.99):.2f} ms")
    print(f"Min client : {valid['latency_ms'].min():.2f} ms")
    print(f"Max client : {valid['latency_ms'].max():.2f} ms")

    if "api_latency_ms" in valid.columns and valid["api_latency_ms"].notna().any():
        print(f"\nLatence API moyenne : {valid['api_latency_ms'].mean():.2f} ms")
        print(f"P95 API : {valid['api_latency_ms'].quantile(0.95):.2f} ms")


if __name__ == "__main__":
    CLIENT_ID = 100002
    N_RUNS = 300
    WARMUP_RUNS = 10

    backend = str(MODEL_BACKEND).lower().strip()
    output_file = f"benchmark_client_id_{backend}.csv"

    print(f"API_URL : {API_URL}")
    print(f"MODEL_BACKEND : {MODEL_BACKEND}")
    print(f"API_KEY : {API_KEY[:4]}****")
    print(f"Endpoint : /predict/{CLIENT_ID}")

    df_results = benchmark(
        client_id=CLIENT_ID,
        n_runs=N_RUNS,
        warmup_runs=WARMUP_RUNS,
        output_file=output_file,
    )

    print_summary(df_results, output_file)