"""
Script de création des tables PostgreSQL de prédiction.

Ce module crée les tables utilisées pour journaliser les inférences
réalisées par l'API de scoring crédit.

Objectif
--------
Mettre en place une couche de persistance dédiée aux prédictions afin de :
- tracer chaque appel de l'API
- stocker les scores et classes prédites
- conserver les entrées et sorties de l'inférence
- préparer le suivi des vérités terrain
- historiser les features observées au moment de la prédiction
- accélérer les requêtes fréquentes du dashboard grâce aux index

Architecture
------------
Dans la version actuelle du projet :
- les features sont construites directement depuis les CSV
- PostgreSQL sert uniquement à stocker les logs, les vérités terrain
  et les données de monitoring
- le dashboard interroge ces tables via l'API FastAPI

Tables créées
-------------
- prediction_logs
- ground_truth_labels
- prediction_features_snapshot

Notes
-----
- Les tables sont créées avec `CREATE TABLE IF NOT EXISTS`.
- Les index sont créés avec `CREATE INDEX IF NOT EXISTS`.
- Ce script est conçu pour PostgreSQL.
- Les colonnes JSONB permettent de stocker des structures souples.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Configuration de la base de données
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL n'est pas défini dans les variables d'environnement")


# =============================================================================
# SQL de création des tables de prédiction
# =============================================================================

CREATE_PREDICTION_LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    client_id BIGINT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    score DOUBLE PRECISION NOT NULL,
    threshold_used DOUBLE PRECISION,
    latency_ms DOUBLE PRECISION,
    inference_latency_ms DOUBLE PRECISION,
    input_data JSONB NOT NULL,
    output_data JSONB,
    prediction_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status_code INTEGER,
    error_message TEXT
);
"""

CREATE_GROUND_TRUTH_LABELS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ground_truth_labels (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    client_id BIGINT,
    true_label INTEGER NOT NULL CHECK (true_label IN (0, 1)),
    label_source TEXT,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT
);
"""

CREATE_PREDICTION_FEATURES_SNAPSHOT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_features_snapshot (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    client_id BIGINT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value TEXT,
    feature_type TEXT,
    snapshot_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# =============================================================================
# SQL de création des index
# =============================================================================

CREATE_INDEXES_SQL = [
    # -------------------------------------------------------------------------
    # prediction_logs
    # -------------------------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_client_id
    ON prediction_logs(client_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version
    ON prediction_logs(model_name, model_version);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version_time
    ON prediction_logs(model_name, model_version, prediction_timestamp DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_client_time
    ON prediction_logs(client_id, prediction_timestamp DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_prediction_timestamp
    ON prediction_logs(prediction_timestamp DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_status_code
    ON prediction_logs(status_code);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_errors_recent
    ON prediction_logs(prediction_timestamp DESC)
    WHERE error_message IS NOT NULL OR status_code >= 400;
    """,

    # -------------------------------------------------------------------------
    # ground_truth_labels
    # -------------------------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_ground_truth_labels_request_id
    ON ground_truth_labels(request_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ground_truth_labels_client_id
    ON ground_truth_labels(client_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ground_truth_labels_client_observed
    ON ground_truth_labels(client_id, observed_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ground_truth_labels_observed_at
    ON ground_truth_labels(observed_at DESC);
    """,

    # -------------------------------------------------------------------------
    # prediction_features_snapshot
    # -------------------------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_features_snapshot_request_id
    ON prediction_features_snapshot(request_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_features_snapshot_client_id
    ON prediction_features_snapshot(client_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_features_snapshot_request_feature
    ON prediction_features_snapshot(request_id, feature_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_features_snapshot_model_version_time
    ON prediction_features_snapshot(model_name, model_version, snapshot_timestamp DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_features_snapshot_feature_name
    ON prediction_features_snapshot(feature_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_features_snapshot_timestamp
    ON prediction_features_snapshot(snapshot_timestamp DESC);
    """,
]


# =============================================================================
# Fonctions de création des tables
# =============================================================================

def create_prediction_logs_table(engine: Engine) -> None:
    """
    Crée la table `prediction_logs`.

    Cette table journalise chaque prédiction faite par l'API avec :
    - l'identifiant de requête
    - l'identifiant client
    - le modèle utilisé
    - la prédiction
    - le score
    - les temps de latence
    - les données d'entrée et de sortie
    - les éventuelles erreurs
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_PREDICTION_LOGS_TABLE_SQL))

    print("Table 'prediction_logs' créée ou déjà existante.")


def create_ground_truth_labels_table(engine: Engine) -> None:
    """
    Crée la table `ground_truth_labels`.

    Cette table stocke les vérités terrain observées après coup afin de
    comparer les prédictions aux résultats réels.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_GROUND_TRUTH_LABELS_TABLE_SQL))

    print("Table 'ground_truth_labels' créée ou déjà existante.")


def create_prediction_features_snapshot_table(engine: Engine) -> None:
    """
    Crée la table `prediction_features_snapshot`.

    Cette table stocke les features observées au moment de l'inférence,
    une ligne par feature, afin de faciliter :
    - l'audit d'une prédiction
    - l'analyse de dérive
    - la traçabilité fine du modèle
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_PREDICTION_FEATURES_SNAPSHOT_TABLE_SQL))

    print("Table 'prediction_features_snapshot' créée ou déjà existante.")


def create_indexes(engine: Engine) -> None:
    """
    Crée les index utiles pour accélérer les requêtes fréquentes.

    Les index ciblent principalement :
    - les recherches par request_id
    - les recherches par client_id
    - les historiques récents
    - les filtres par modèle/version
    - les snapshots de features par requête
    """
    with engine.begin() as connection:
        for sql in CREATE_INDEXES_SQL:
            connection.execute(text(sql))

    print("Index de prédiction créés ou déjà existants.")


# =============================================================================
# Orchestration
# =============================================================================

TABLE_CREATORS = [
    create_prediction_logs_table,
    create_ground_truth_labels_table,
    create_prediction_features_snapshot_table,
]


def create_prediction_tables(engine: Engine) -> None:
    """
    Crée l'ensemble des tables et index liés aux prédictions.
    """
    for create_table in TABLE_CREATORS:
        create_table(engine)

    create_indexes(engine)


def main() -> None:
    """
    Point d'entrée du script.

    Cette fonction :
    1. établit la connexion à PostgreSQL,
    2. crée les tables de prédiction,
    3. crée les index,
    4. affiche un message final.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    create_prediction_tables(engine)

    print("Création des tables de prédiction terminée.")


if __name__ == "__main__":
    main()