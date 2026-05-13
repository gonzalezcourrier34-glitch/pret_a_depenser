"""
Benchmark d'inférence via l'API FastAPI.

Description
-----------
Ce script permet de mesurer les performances réelles du endpoint optimisé :

GET /predict/{client_id}

L’objectif est de simuler plusieurs appels successifs à l’API afin
d’analyser le comportement du système dans des conditions proches
de la production.

Pourquoi faire un benchmark ?
-----------------------------
En MLOps, il ne suffit pas qu’un modèle soit précis.
Il doit également être :
- rapide
- stable
- scalable
- capable de répondre sous forte charge

Ce benchmark permet donc d’évaluer :
- le temps de réponse moyen
- les pics de latence (P95 / P99)
- la stabilité globale du service
- le taux d’erreur éventuel

Pourquoi utiliser /predict/{client_id} ?
----------------------------------------
Dans cette architecture :
- le client envoie uniquement un identifiant
- les features sont récupérées automatiquement côté serveur

Cela se rapproche davantage d’un vrai système de production :
- payload HTTP plus léger
- moins de transfert réseau
- appels plus rapides
- meilleure expérience utilisateur

Concepts importants
-------------------
Warmup
~~~~~~
Les premiers appels sont souvent plus lents :
- chargement du modèle
- initialisation mémoire
- cache Python / ONNX
- ouverture des connexions

On réalise donc plusieurs appels de chauffe non comptabilisés.

Session HTTP persistante
~~~~~~~~~~~~~~~~~~~~~~~~
Le script utilise requests.Session() afin de :
- réutiliser les connexions TCP
- éviter un reconnect à chaque appel
- réduire la latence réseau

Latence client vs latence API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Deux mesures sont comparées :

1. latency_ms
   Temps total vu par le client :
   - réseau
   - sérialisation JSON
   - temps API
   - transfert HTTP

2. api_latency_ms
   Temps interne mesuré côté serveur uniquement.

Cela permet d’identifier :
- les coûts réseau
- les surcoûts applicatifs
- les vrais goulots d’étranglement

P95 / P99
~~~~~~~~~
Les percentiles élevés sont très importants en production.

Exemple :
- moyenne = système global
- P95 = cas lents fréquents
- P99 = pires cas rares

Un système peut avoir une bonne moyenne
mais de très mauvais P99.
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

# Librairies standard
import time
from pathlib import Path
from typing import Any

# Librairies externes
import pandas as pd
import requests

# Configuration du projet
from app.core.config import API_KEY, API_URL, MODEL_BACKEND


# =============================================================================
# CONFIGURATION DES DOSSIERS
# =============================================================================

"""
Organisation des fichiers de sortie.

Les résultats sont stockés dans :
artifacts/performance/

Cela permet :
- d'historiser les benchmarks
- de comparer plusieurs backends
- d’alimenter le monitoring
- de produire des graphiques
"""

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossier de sortie
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "performance"

# Création automatique du dossier
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# APPEL API
# =============================================================================

def call_predict(
    *,
    session: requests.Session,
    client_id: int,
) -> dict[str, Any]:
    """
    Appelle l’endpoint de prédiction optimisé.

    Endpoint testé
    ---------------
    GET /predict/{client_id}

    Paramètres
    ----------
    session : requests.Session
        Session HTTP persistante utilisée pour améliorer
        les performances réseau.

    client_id : int
        Identifiant du client à scorer.

    Retour
    ------
    dict[str, Any]
        Réponse JSON retournée par l’API.

    Pourquoi utiliser une Session ?
    -------------------------------
    Sans session :
    - nouvelle connexion TCP à chaque appel
    - plus de latence
    - benchmark faussé

    Avec session persistante :
    - connexion réutilisée
    - comportement plus réaliste
    - meilleures performances

    Gestion des erreurs
    -------------------
    Toute réponse HTTP >= 400 déclenche une exception
    afin d’être comptabilisée comme échec du benchmark.
    """

    response = session.get(
        f"{API_URL}/predict/{client_id}",
        headers={"X-API-Key": API_KEY},
        timeout=30,
    )

    # Détection des erreurs HTTP
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} - {response.text}")

    return response.json()


# =============================================================================
# BENCHMARK PRINCIPAL
# =============================================================================

def benchmark(
    *,
    client_id: int,
    n_runs: int = 300,
    warmup_runs: int = 10,
    output_file: str,
) -> pd.DataFrame:
    """
    Lance le benchmark principal.

    Fonctionnement
    ---------------
    Le benchmark suit plusieurs étapes :

    1. Warmup
       Stabilisation du système

    2. Appels mesurés
       Exécution répétée des prédictions

    3. Mesure des performances
       Calcul de la latence client

    4. Sauvegarde
       Export CSV des résultats

    Paramètres
    ----------
    client_id : int
        Client utilisé pour les tests.

    n_runs : int
        Nombre total d’appels mesurés.

    warmup_runs : int
        Nombre d’appels de chauffe.

    output_file : str
        Nom du CSV de sortie.

    Retour
    ------
    pd.DataFrame
        Résultats détaillés du benchmark.

    Pourquoi faire beaucoup d’appels ?
    ----------------------------------
    Un seul appel n’est pas représentatif.

    Plusieurs centaines d’appels permettent :
    - d’observer la stabilité
    - de mesurer les variations
    - d’obtenir des statistiques fiables
    """

    rows: list[dict[str, Any]] = []

    # Session HTTP persistante
    with requests.Session() as session:

        # ---------------------------------------------------------------------
        # WARMUP
        # ---------------------------------------------------------------------

        """
        Les appels de chauffe servent à stabiliser le système.

        Ils permettent notamment :
        - chargement du modèle en mémoire
        - initialisation ONNX Runtime
        - remplissage des caches
        - ouverture des connexions

        Les résultats ne sont pas comptabilisés.
        """

        for _ in range(warmup_runs):
            try:
                call_predict(session=session, client_id=client_id)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # BENCHMARK
        # ---------------------------------------------------------------------

        for i in range(n_runs):

            # Début mesure temps
            start = time.perf_counter()

            try:
                # Appel API
                result = call_predict(
                    session=session,
                    client_id=client_id,
                )

                success = True
                error = None

            except Exception as exc:

                # Gestion des erreurs benchmark
                result = {}
                success = False
                error = str(exc)

            # -----------------------------------------------------------------
            # Calcul latence client
            # -----------------------------------------------------------------

            """
            Latence mesurée côté client.

            Inclut :
            - temps réseau
            - temps API
            - sérialisation JSON
            - transfert HTTP
            """

            latency_ms = (time.perf_counter() - start) * 1000

            # -----------------------------------------------------------------
            # Stockage résultats
            # -----------------------------------------------------------------

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

                    # Latence interne API
                    "api_latency_ms": result.get("latency_ms"),

                    "error": error,
                }
            )

    # -------------------------------------------------------------------------
    # Conversion DataFrame
    # -------------------------------------------------------------------------

    df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Sauvegarde CSV
    # -------------------------------------------------------------------------

    """
    Les résultats sont exportés afin de :
    - produire des graphiques
    - comparer plusieurs backends
    - historiser les performances
    - alimenter un dashboard MLOps
    """

    df.to_csv(OUTPUT_DIR / output_file, index=False)

    return df


# =============================================================================
# RÉSUMÉ STATISTIQUE
# =============================================================================

def print_summary(df: pd.DataFrame, output_file: str) -> None:
    """
    Affiche un résumé statistique du benchmark.

    Métriques importantes
    ---------------------
    Moyenne
        Performance globale.

    Médiane
        Cas typique plus robuste que la moyenne.

    P95
        95 % des requêtes sont plus rapides que cette valeur.

    P99
        Mesure les pires cas rares.

    Pourquoi les percentiles sont importants ?
    ------------------------------------------
    En production :
    - quelques appels très lents peuvent dégrader l’expérience utilisateur
    - les moyennes seules peuvent masquer ces problèmes

    Les P95/P99 sont donc très utilisés en MLOps
    et dans les systèmes distribués.
    """

    valid = df[df["success"]].copy()
    failed = df[~df["success"]].copy()

    print("\nRésultats")
    print(f"Backend : {MODEL_BACKEND}")
    print(f"Fichier : {(OUTPUT_DIR / output_file).resolve()}")

    print(f"Total runs : {len(df)}")
    print(f"Succès : {len(valid)}")
    print(f"Échecs : {len(failed)}")

    # -------------------------------------------------------------------------
    # Cas sans succès
    # -------------------------------------------------------------------------

    if valid.empty:

        print("\nAucune prédiction réussie.")

        if not failed.empty:
            print("\nPremière erreur :")
            print(failed.iloc[0]["error"])

        return

    # -------------------------------------------------------------------------
    # Statistiques client
    # -------------------------------------------------------------------------

    print(f"\nLatence client moyenne : {valid['latency_ms'].mean():.2f} ms")
    print(f"Médiane client : {valid['latency_ms'].median():.2f} ms")
    print(f"P95 client : {valid['latency_ms'].quantile(0.95):.2f} ms")
    print(f"P99 client : {valid['latency_ms'].quantile(0.99):.2f} ms")
    print(f"Min client : {valid['latency_ms'].min():.2f} ms")
    print(f"Max client : {valid['latency_ms'].max():.2f} ms")

    # -------------------------------------------------------------------------
    # Statistiques API internes
    # -------------------------------------------------------------------------

    if (
        "api_latency_ms" in valid.columns
        and valid["api_latency_ms"].notna().any()
    ):

        print(f"\nLatence API moyenne : {valid['api_latency_ms'].mean():.2f} ms")
        print(f"P95 API : {valid['api_latency_ms'].quantile(0.95):.2f} ms")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":

    """
    Point d’entrée du script.

    Ce bloc permet d’exécuter directement le benchmark :

    python benchmark.py

    Paramètres utilisés
    -------------------
    CLIENT_ID
        Client testé.

    N_RUNS
        Nombre d’appels mesurés.

    WARMUP_RUNS
        Nombre d’appels de chauffe.
    """

    CLIENT_ID = 100002
    N_RUNS = 300
    WARMUP_RUNS = 10

    # Nom du backend
    backend = str(MODEL_BACKEND).lower().strip()

    # Nom du CSV de sortie
    output_file = f"benchmark_client_id_{backend}.csv"

    # -------------------------------------------------------------------------
    # Affichage configuration
    # -------------------------------------------------------------------------

    print(f"API_URL : {API_URL}")
    print(f"MODEL_BACKEND : {MODEL_BACKEND}")
    print(f"API_KEY : {API_KEY[:4]}****")
    print(f"Endpoint : /predict/{CLIENT_ID}")

    # -------------------------------------------------------------------------
    # Lancement benchmark
    # -------------------------------------------------------------------------

    df_results = benchmark(
        client_id=CLIENT_ID,
        n_runs=N_RUNS,
        warmup_runs=WARMUP_RUNS,
        output_file=output_file,
    )

    # -------------------------------------------------------------------------
    # Résumé final
    # -------------------------------------------------------------------------

    print_summary(df_results, output_file)