"""
Benchmark d'inférence via l'API FastAPI.

Ce script permet de mesurer les performances du endpoint optimisé :
GET /predict/{client_id}

Objectif pédagogique
--------------------
Simuler des appels répétés à l’API pour analyser :
- la latence réelle côté client
- la stabilité du service (erreurs / succès)
- la performance globale du modèle en conditions proches de la production

Pourquoi ce endpoint ?
---------------------
Contrairement à une API classique où l’on envoie toutes les features en JSON :
- ici on envoie uniquement un client_id
- les features sont récupérées côté serveur (base ou cache)

Avantages :
- payload très léger
- plus rapide
- plus proche d’un vrai système en production
"""

from __future__ import annotations

# Librairies standard
import time
from pathlib import Path
from typing import Any

# Librairies externes
import pandas as pd
import requests

# Configuration du projet (URL API, clé, backend modèle)
from app.core.config import API_KEY, API_URL, MODEL_BACKEND


# =============================================================================
# Configuration des chemins
# =============================================================================

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossier de sortie des benchmarks
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "performance"

# Création du dossier si nécessaire
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Fonction d'appel API
# =============================================================================

def call_predict(
    *,
    session: requests.Session,
    client_id: int,
) -> dict[str, Any]:
    """
    Appelle l'endpoint optimisé /predict/{client_id}.

    Paramètres
    ----------
    session : requests.Session
        Session HTTP réutilisée pour optimiser les appels réseau
    client_id : int
        Identifiant du client pour lequel on souhaite une prédiction

    Retour
    ------
    dict
        Réponse JSON de l'API

    Remarque
    --------
    On utilise une session persistante pour éviter de recréer
    une connexion TCP à chaque appel (gain de performance).
    """
    response = session.get(
        f"{API_URL}/predict/{client_id}",
        headers={"X-API-Key": API_KEY},
        timeout=30,
    )

    # Gestion des erreurs HTTP
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} - {response.text}")

    return response.json()


# =============================================================================
# Fonction principale de benchmark
# =============================================================================

def benchmark(
    *,
    client_id: int,
    n_runs: int = 300,
    warmup_runs: int = 10,
    output_file: str,
) -> pd.DataFrame:
    """
    Lance plusieurs appels API et mesure la latence.

    Paramètres
    ----------
    client_id : int
        Client testé
    n_runs : int
        Nombre d'appels mesurés
    warmup_runs : int
        Nombre d'appels de chauffe (non comptés)
    output_file : str
        Nom du fichier CSV de sortie

    Retour
    ------
    DataFrame
        Résultats complets du benchmark

    Logique
    -------
    1. Warmup → stabiliser l'API (cache, modèle)
    2. Boucle de test → mesurer la latence
    3. Stockage des résultats
    """

    rows: list[dict[str, Any]] = []

    # Session HTTP persistante (meilleure perf)
    with requests.Session() as session:

        # -------------------------
        # Phase de warmup
        # -------------------------
        for _ in range(warmup_runs):
            try:
                call_predict(session=session, client_id=client_id)
            except Exception:
                pass  # On ignore les erreurs de chauffe

        # -------------------------
        # Phase de benchmark
        # -------------------------
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

            # Calcul de la latence côté client (ms)
            latency_ms = (time.perf_counter() - start) * 1000

            # Stockage des résultats
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

    # Conversion en DataFrame
    df = pd.DataFrame(rows)

    # Sauvegarde en CSV
    df.to_csv(OUTPUT_DIR / output_file, index=False)

    return df


# =============================================================================
# Affichage du résumé
# =============================================================================

def print_summary(df: pd.DataFrame, output_file: str) -> None:
    """
    Affiche un résumé statistique du benchmark.

    Objectif
    --------
    Fournir des métriques lisibles pour analyse :
    - taux de succès
    - latence moyenne
    - P95 / P99 (très important en MLOps)
    """

    valid = df[df["success"]].copy()
    failed = df[~df["success"]].copy()

    print("\nRésultats")
    print(f"Backend : {MODEL_BACKEND}")
    print(f"Fichier : {(OUTPUT_DIR / output_file).resolve()}")
    print(f"Total runs : {len(df)}")
    print(f"Succès : {len(valid)}")
    print(f"Échecs : {len(failed)}")

    # Cas sans succès
    if valid.empty:
        print("\nAucune prédiction réussie.")
        if not failed.empty:
            print("\nPremière erreur :")
            print(failed.iloc[0]["error"])
        return

    # Statistiques côté client
    print(f"\nLatence client moyenne : {valid['latency_ms'].mean():.2f} ms")
    print(f"Médiane client : {valid['latency_ms'].median():.2f} ms")
    print(f"P95 client : {valid['latency_ms'].quantile(0.95):.2f} ms")
    print(f"P99 client : {valid['latency_ms'].quantile(0.99):.2f} ms")
    print(f"Min client : {valid['latency_ms'].min():.2f} ms")
    print(f"Max client : {valid['latency_ms'].max():.2f} ms")

    # Statistiques côté API (si disponibles)
    if "api_latency_ms" in valid.columns and valid["api_latency_ms"].notna().any():
        print(f"\nLatence API moyenne : {valid['api_latency_ms'].mean():.2f} ms")
        print(f"P95 API : {valid['api_latency_ms'].quantile(0.95):.2f} ms")


# =============================================================================
# Point d'entrée du script
# =============================================================================

if __name__ == "__main__":
    """
    Point d'entrée du script.

    Ce bloc permet de lancer le benchmark directement en ligne de commande :
    python benchmark.py
    """

    CLIENT_ID = 100002
    N_RUNS = 300
    WARMUP_RUNS = 10

    backend = str(MODEL_BACKEND).lower().strip()
    output_file = f"benchmark_client_id_{backend}.csv"

    # Affichage configuration
    print(f"API_URL : {API_URL}")
    print(f"MODEL_BACKEND : {MODEL_BACKEND}")
    print(f"API_KEY : {API_KEY[:4]}****")
    print(f"Endpoint : /predict/{CLIENT_ID}")

    # Lancement benchmark
    df_results = benchmark(
        client_id=CLIENT_ID,
        n_runs=N_RUNS,
        warmup_runs=WARMUP_RUNS,
        output_file=output_file,
    )

    # Affichage résumé
    print_summary(df_results, output_file)