"""
Profiling d'inférence via l'API FastAPI.

Description
-----------
Ce script permet d'analyser précisément où le temps est consommé
lors de plusieurs appels au endpoint :

POST /predict

Contrairement au benchmark, qui mesure surtout une latence globale,
le profiling descend plus finement dans l'exécution Python.

Il permet d’identifier :
- les fonctions les plus coûteuses
- le temps passé dans les appels HTTP
- le temps passé dans requests / urllib3
- les éventuels surcoûts côté client
- les parties du script à optimiser en priorité

Différence entre benchmark et profiling
---------------------------------------
Benchmark :
    mesure combien de temps prend une requête complète.

Profiling :
    explique où ce temps est consommé.

Dans mon projet, cela permet de distinguer :
- la latence globale observée
- le coût du client Python
- le coût réseau
- le coût de sérialisation JSON
- le temps réellement passé côté API

Endpoint testé
--------------
POST /predict

Ce endpoint reçoit directement les features du client sous forme JSON.

Contrairement au endpoint optimisé GET /predict/{client_id},
ici le client envoie toutes les variables d'entrée attendues par le modèle.

Cela permet de tester un cas plus complet :
- préparation du payload JSON
- envoi des features
- traitement API
- prédiction
- réponse JSON

Rôle de cProfile
----------------
cProfile est l’outil standard de Python pour profiler un programme.

Il mesure notamment :
- le nombre d'appels par fonction
- le temps passé dans chaque fonction
- le temps cumulé incluant les sous-fonctions

Le tri par temps cumulé est particulièrement utile, car il montre
les fonctions responsables du plus gros temps total d'exécution.
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

"""
Les imports sont séparés par rôle afin de rendre le script plus lisible.

cProfile et pstats :
    outils Python utilisés pour mesurer et afficher le profiling.

math :
    utilisé pour détecter les valeurs NaN ou infinies.

Path :
    utilisé pour gérer les chemins de fichiers proprement.

Any :
    utilisé pour typer les dictionnaires de features.

numpy / pandas :
    utilisés pour manipuler les données et convertir les types.

requests :
    utilisé pour appeler l'API FastAPI.

Configuration projet :
    permet de récupérer l'URL de l'API, la clé API,
    le backend modèle et le dossier de monitoring.
"""

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
# CONFIGURATION DES FICHIERS DE SORTIE
# =============================================================================

"""
Le rapport de profiling est sauvegardé dans :

artifacts/performance/profiling_report.txt

Ce choix permet de centraliser les résultats de performance
avec les autres artefacts du projet.

Ce fichier peut ensuite être :
- consulté manuellement
- intégré dans un rapport
- comparé entre plusieurs versions
- utilisé pour justifier une optimisation
"""

# Dossier dans lequel le rapport sera enregistré
OUTPUT_DIR = Path("artifacts/performance")

# Création du dossier si nécessaire
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fichier texte final du rapport cProfile
OUTPUT_FILE = OUTPUT_DIR / "profiling_report.txt"


# =============================================================================
# NETTOYAGE DES VALEURS POUR COMPATIBILITÉ JSON
# =============================================================================

def make_json_safe(value: Any) -> Any:
    """
    Convertit une valeur pandas / numpy en valeur compatible JSON.

    Objectif
    --------
    FastAPI reçoit les données au format JSON.
    Or certaines valeurs issues de pandas ou numpy ne sont pas toujours
    directement sérialisables.

    Problèmes possibles
    -------------------
    Une ligne de DataFrame peut contenir :
    - np.int64
    - np.float32
    - NaN
    - inf
    - -inf
    - valeurs manquantes pandas

    Ces valeurs peuvent provoquer une erreur lors de l'envoi à l'API.

    Stratégie
    ---------
    - les entiers numpy sont convertis en int Python
    - les flottants numpy sont convertis en float Python
    - les NaN et inf sont remplacés par None
    - les valeurs manquantes pandas sont remplacées par None

    Retour
    ------
    Any
        Valeur nettoyée, compatible avec un payload JSON.
    """

    # Valeur déjà vide
    if value is None:
        return None

    # Conversion np.int64, np.int32, etc.
    if isinstance(value, np.integer):
        return int(value)

    # Conversion np.float32, np.float64, etc.
    if isinstance(value, np.floating):
        value = float(value)

    # Gestion des flottants invalides en JSON
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
    Nettoie un dictionnaire complet de features avant envoi à l'API.

    Objectif
    --------
    Transformer une ligne de DataFrame en payload JSON propre.

    Traitements appliqués
    ---------------------
    Pour chaque variable :
    - la clé est convertie en str
    - la valeur est convertie avec make_json_safe()

    Pourquoi convertir les clés en str ?
    ------------------------------------
    En JSON, les clés d'un objet doivent être des chaînes de caractères.
    Cette étape évite donc des problèmes de sérialisation.

    Paramètres
    ----------
    features : dict[str, Any]
        Dictionnaire brut issu d'une ligne pandas.

    Retour
    ------
    dict[str, Any]
        Dictionnaire propre et compatible JSON.
    """

    return {
        str(key): make_json_safe(value)
        for key, value in features.items()
    }


# =============================================================================
# CHARGEMENT DES FEATURES ATTENDUES PAR L'API
# =============================================================================

def load_features_for_api(row_index: int = 0) -> dict[str, Any]:
    """
    Charge une ligne de features exploitable par le endpoint POST /predict.

    Source des données
    ------------------
    Les données proviennent du fichier de référence utilisé pour le monitoring.

    Dans ce projet, ce fichier représente un jeu de features cohérent
    avec ce que le modèle attend en entrée.

    Pourquoi utiliser cette source ?
    --------------------------------
    Cela permet de tester l'API avec des données réalistes,
    sans reconstruire manuellement toutes les features.

    Étapes réalisées
    ----------------
    1. Initialisation du cache de monitoring.
    2. Chargement du DataFrame de référence.
    3. Vérification que le DataFrame n'est pas vide.
    4. Sélection d'une ligne.
    5. Suppression des colonnes non prédictives.
    6. Conversion en dictionnaire compatible JSON.

    Colonnes supprimées
    -------------------
    TARGET :
        variable cible connue uniquement pendant l'entraînement.

    SK_ID_CURR :
        identifiant client, non utilisé comme feature métier.

    Paramètres
    ----------
    row_index : int
        Index de la ligne à utiliser dans le DataFrame.

    Retour
    ------
    dict[str, Any]
        Features nettoyées et prêtes à être envoyées à l'API.
    """

    # Initialisation du cache de référence monitoring
    init_monitoring_reference_cache(Path(MONITORING_DIR))

    # Chargement du DataFrame de référence brute
    df = get_reference_features_raw_df()

    # Sécurité : DataFrame vide
    if df.empty:
        raise ValueError("Le DataFrame reference_features_raw est vide.")

    # Sécurité : index hors limites
    if row_index >= len(df):
        raise IndexError(
            f"row_index={row_index} est hors limites. "
            f"Nombre de lignes disponibles : {len(df)}"
        )

    # Sélection d'une ligne
    row = df.iloc[row_index].copy()

    # Suppression des colonnes non utilisées pour la prédiction
    row = row.drop(labels=["TARGET", "SK_ID_CURR"], errors="ignore")

    # Nettoyage JSON
    return clean_features_for_json(row.to_dict())


# =============================================================================
# APPEL API SIMPLE
# =============================================================================

def call_predict(row_index: int = 0) -> dict[str, Any]:
    """
    Effectue un appel unique au endpoint POST /predict.

    Objectif
    --------
    Cette fonction permet de tester rapidement que :
    - les features sont bien chargées
    - le payload JSON est valide
    - l'API répond correctement
    - la prédiction est bien retournée

    Elle est moins utilisée pour le profiling principal,
    car run_profile() optimise le chargement des features
    en les chargeant une seule fois.

    Paramètres
    ----------
    row_index : int
        Index de la ligne de features à envoyer.

    Retour
    ------
    dict[str, Any]
        Réponse JSON retournée par l'API.

    Erreurs
    -------
    Si l'API renvoie un code HTTP >= 400,
    une exception est levée avec le détail de l'erreur.
    """

    # Chargement des features
    features = load_features_for_api(row_index=row_index)

    # Appel POST /predict
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
# FONCTION PROFILÉE
# =============================================================================

def run_profile(n_runs: int = 100) -> None:
    """
    Lance plusieurs appels API pour produire un rapport de profiling.

    Objectif
    --------
    Cette fonction est la partie réellement analysée par cProfile.

    Elle simule plusieurs appels successifs au endpoint :

    POST /predict

    Pourquoi charger les features une seule fois ?
    ----------------------------------------------
    Le but ici est de profiler l'appel API,
    pas le chargement des données depuis le disque.

    Si les features étaient rechargées à chaque itération,
    le rapport serait pollué par :
    - la lecture de fichiers
    - la création de DataFrame
    - les conversions pandas

    En chargeant les features une seule fois,
    on concentre l'analyse sur :
    - requests
    - HTTP
    - sérialisation JSON
    - réponse API

    Pourquoi utiliser requests.Session ?
    ------------------------------------
    Une session HTTP persistante permet de réutiliser la connexion réseau.

    Cela évite :
    - d'ouvrir une nouvelle connexion à chaque requête
    - de gonfler artificiellement la latence
    - de mesurer un scénario moins réaliste

    Paramètres
    ----------
    n_runs : int
        Nombre d'appels API à exécuter pendant le profiling.

    Retour
    ------
    None
        La fonction ne retourne rien.
        Le résultat est capturé par cProfile.
    """

    # Chargement unique des features
    features = load_features_for_api(row_index=0)

    # Session persistante pour réutiliser la connexion HTTP
    with requests.Session() as session:

        for _ in range(n_runs):

            response = session.post(
                f"{API_URL}/predict",
                headers={"X-API-Key": API_KEY},
                json={"features": features},
                timeout=30,
            )

            # Arrêt immédiat si une requête échoue
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} - {response.text}")


# =============================================================================
# POINT D'ENTRÉE DU SCRIPT
# =============================================================================

if __name__ == "__main__":

    """
    Point d'entrée du script.

    Commande typique
    ----------------
    python scripts/profiling_inference.py

    Déroulé
    -------
    1. Affichage de la configuration utilisée.
    2. Création du profiler cProfile.
    3. Activation du profiler.
    4. Exécution de run_profile().
    5. Désactivation du profiler.
    6. Sauvegarde du rapport texte.

    Rapport généré
    ---------------
    Le rapport est sauvegardé dans :

    artifacts/performance/profiling_report.txt

    Lecture du rapport
    ------------------
    Les colonnes importantes sont :

    ncalls :
        nombre d'appels de la fonction.

    tottime :
        temps passé uniquement dans cette fonction.

    percall :
        temps moyen par appel.

    cumtime :
        temps cumulé incluant les sous-fonctions.

    filename:lineno(function) :
        emplacement de la fonction appelée.

    Pourquoi trier par cumulative ?
    -------------------------------
    Le temps cumulé permet d’identifier les fonctions
    qui pèsent le plus dans l'exécution globale.

    C’est souvent plus utile que le temps direct,
    car certaines fonctions délèguent beaucoup de travail
    à d'autres fonctions internes.
    """

    print(f"API_URL       : {API_URL}")
    print(f"MODEL_BACKEND : {MODEL_BACKEND}")
    print(f"Profiling vers: {OUTPUT_FILE.resolve()}")

    # Création du profiler
    profiler = cProfile.Profile()

    # Activation du profiler
    profiler.enable()

    # Exécution du code à analyser
    run_profile(n_runs=100)

    # Désactivation du profiler
    profiler.disable()

    # Sauvegarde du rapport
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        stats = pstats.Stats(profiler, stream=f)

        # Tri par temps cumulé
        stats.sort_stats("cumulative")

        # Affichage des 40 fonctions les plus coûteuses
        stats.print_stats(40)

    print(f"Profiling sauvegardé : {OUTPUT_FILE.resolve()}")