"""
Profiling d'inférence via l'API FastAPI.

Objectif
--------
Ce script permet d'analyser le temps passé dans les différentes fonctions
lors d'appels répétés au endpoint :

POST /predict

Contrairement à un simple benchmark qui mesure uniquement la latence globale,
le profiling permet d'identifier quelles parties du script consomment le plus
de temps côté client.

Important
---------
Ce script envoie à l'API les features d'entrée attendues par /predict.

Il ne reconstruit pas lui-même les features selon le backend utilisé.
C'est l'API qui choisit le moteur d'inférence via la variable MODEL_BACKEND :

- sklearn
- onnx

Le but est donc de tester le comportement réel de l'API, dans des conditions
proches d'un appel de production.
"""

from __future__ import annotations

# Outils de profiling Python
import cProfile
import pstats

# Librairies standard
import math
from pathlib import Path
from typing import Any

# Librairies externes
import numpy as np
import pandas as pd
import requests

# Configuration du projet
from app.core.config import API_KEY, API_URL, MODEL_BACKEND, MONITORING_DIR

# Services permettant de charger les features de référence
from app.services.loader_services.data_loading_service import (
    get_reference_features_raw_df,
    init_monitoring_reference_cache,
)


# =============================================================================
# Configuration des fichiers de sortie
# =============================================================================

# Dossier dans lequel le rapport de profiling sera enregistré
OUTPUT_DIR = Path("artifacts/performance")

# Création du dossier s'il n'existe pas encore
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fichier texte contenant le rapport cProfile
OUTPUT_FILE = OUTPUT_DIR / "profiling_report.txt"


# =============================================================================
# Nettoyage des valeurs pour compatibilité JSON
# =============================================================================

def make_json_safe(value: Any) -> Any:
    """
    Convertit une valeur pandas / numpy en valeur compatible JSON.

    Pourquoi cette fonction ?
    ------------------------
    Lorsqu'on récupère une ligne depuis un DataFrame pandas, certaines valeurs
    peuvent poser problème lors de l'envoi en JSON :

    - NaN
    - inf
    - -inf
    - types numpy comme np.int64 ou np.float32

    Or, une API FastAPI attend généralement un JSON propre.
    Cette fonction convertit donc les valeurs problématiques avant l'appel API.
    """

    # Valeur déjà vide
    if value is None:
        return None

    # Conversion des entiers numpy en int Python classique
    if isinstance(value, np.integer):
        return int(value)

    # Conversion des flottants numpy en float Python classique
    if isinstance(value, np.floating):
        value = float(value)

    # Gestion des float non valides pour JSON
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # Gestion générale des valeurs manquantes pandas
    if pd.isna(value):
        return None

    return value


def clean_features_for_json(features: dict[str, Any]) -> dict[str, Any]:
    """
    Nettoie un dictionnaire de features avant envoi à l'API.

    Chaque clé est convertie en chaîne de caractères.
    Chaque valeur est rendue compatible avec le format JSON.
    """

    return {
        str(key): make_json_safe(value)
        for key, value in features.items()
    }


# =============================================================================
# Chargement des features attendues par l'API
# =============================================================================

def load_features_for_api(row_index: int = 0) -> dict[str, Any]:
    """
    Charge une ligne de features utilisable par le endpoint /predict.

    Source utilisée
    ----------------
    Les données viennent de la référence brute de monitoring.

    Cette référence contient les variables d'entrée attendues par l'API.
    Elle permet donc de tester le endpoint avec un exemple réaliste.

    Paramètre
    ---------
    row_index : int
        Numéro de la ligne à utiliser dans le DataFrame de référence.

    Retour
    ------
    dict
        Dictionnaire de features nettoyé et prêt à être envoyé en JSON.
    """

    # Initialisation du cache de référence monitoring
    init_monitoring_reference_cache(Path(MONITORING_DIR))

    # Chargement du DataFrame de référence brute
    df = get_reference_features_raw_df()

    # Sécurité : on vérifie que la source n'est pas vide
    if df.empty:
        raise ValueError("Le DataFrame reference_features_raw est vide.")

    # Sécurité : on vérifie que l'index demandé existe
    if row_index >= len(df):
        raise IndexError(
            f"row_index={row_index} est hors limites. "
            f"Nombre de lignes disponibles : {len(df)}"
        )

    # Sélection d'une ligne client
    row = df.iloc[row_index].copy()

    # Ces colonnes ne doivent pas être envoyées au modèle
    # TARGET = vraie classe connue uniquement en entraînement
    # SK_ID_CURR = identifiant client, pas une feature métier
    row = row.drop(labels=["TARGET", "SK_ID_CURR"], errors="ignore")

    # Conversion en dictionnaire JSON-safe
    return clean_features_for_json(row.to_dict())


# =============================================================================
# Appel API simple
# =============================================================================

def call_predict(row_index: int = 0) -> dict[str, Any]:
    """
    Appelle le endpoint POST /predict avec les features attendues.

    Cette fonction sert à effectuer un appel unique à l'API.
    Elle est utile pour tester rapidement que l'API répond correctement.
    """

    # Chargement des features à envoyer
    features = load_features_for_api(row_index=row_index)

    # Appel du endpoint FastAPI
    response = requests.post(
        f"{API_URL}/predict",
        headers={"X-API-Key": API_KEY},
        json={"features": features},
        timeout=30,
    )

    # Gestion des erreurs HTTP
    if response.status_code >= 400:
        raise RuntimeError(
            f"{response.status_code} - {response.text}"
        )

    return response.json()


# =============================================================================
# Fonction exécutée pendant le profiling
# =============================================================================

def run_profile(n_runs: int = 100) -> None:
    """
    Lance plusieurs appels au endpoint /predict.

    Objectif
    --------
    Cette fonction est celle qui sera profilée par cProfile.

    Elle répète plusieurs appels API afin d'obtenir un rapport plus stable
    qu'avec un seul appel.

    Remarque importante
    -------------------
    Les features sont chargées une seule fois avant la boucle.
    Cela permet de profiler principalement les appels API, et non le chargement
    des données à chaque itération.
    """

    # Chargement unique des features pour ne pas fausser le profiling
    features = load_features_for_api(row_index=0)

    # Session HTTP persistante pour réutiliser la connexion
    with requests.Session() as session:

        for _ in range(n_runs):
            response = session.post(
                f"{API_URL}/predict",
                headers={"X-API-Key": API_KEY},
                json={"features": features},
                timeout=30,
            )

            # Si une requête échoue, on arrête le profiling
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} - {response.text}")


# =============================================================================
# Point d'entrée du script
# =============================================================================

if __name__ == "__main__":
    """
    Point d'entrée du script.

    Lancement typique :
    python scripts/nom_du_script.py

    Le script :
    1. affiche la configuration utilisée,
    2. active le profiler,
    3. lance 100 appels API,
    4. désactive le profiler,
    5. sauvegarde le rapport dans artifacts/performance.
    """

    print(f"API_URL       : {API_URL}")
    print(f"MODEL_BACKEND : {MODEL_BACKEND}")
    print(f"Profiling vers: {OUTPUT_FILE.resolve()}")

    # Création du profiler Python
    profiler = cProfile.Profile()

    # Début de la mesure
    profiler.enable()

    # Code réellement profilé
    run_profile(n_runs=100)

    # Fin de la mesure
    profiler.disable()

    # Sauvegarde du rapport de profiling
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f)

        # Tri par temps cumulé :
        # on voit les fonctions qui consomment le plus de temps au total
        stats.sort_stats("cumulative")

        # Affichage des 40 fonctions les plus coûteuses
        stats.print_stats(40)

    print(f"Profiling sauvegardé : {OUTPUT_FILE.resolve()}")