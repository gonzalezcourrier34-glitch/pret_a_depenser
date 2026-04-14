"""
Script de création des tables de monitoring PostgreSQL.

Ce module crée les tables de monitoring utilisées pour suivre
un modèle de machine learning en production.

Objectif
--------
Mettre en place une couche de monitoring structurée afin de :
- suivre les versions de modèles déployées
- conserver les snapshots de features
- stocker les métriques de dérive
- historiser les métriques de performance
- centraliser les alertes

Tables créées
-------------
- model_registry
- feature_store_monitoring
- drift_metrics
- evaluation_metrics
- alerts

Notes
-----
- Les tables sont créées avec `CREATE TABLE IF NOT EXISTS`.
- Ce script est conçu pour PostgreSQL.
- Les colonnes JSONB sont utilisées pour stocker des structures souples.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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
# SQL de création des tables de monitoring
# =============================================================================

CREATE_MODEL_REGISTRY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    stage TEXT NOT NULL,
    run_id TEXT,
    source_path TEXT,
    training_data_version TEXT,
    feature_list JSONB,
    hyperparameters JSONB,
    metrics JSONB,
    deployed_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, model_version)
);
"""

CREATE_FEATURE_STORE_MONITORING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feature_store_monitoring (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    client_id BIGINT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value TEXT,
    feature_type TEXT,
    source_table TEXT,
    snapshot_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_DRIFT_METRICS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    reference_window_start TIMESTAMPTZ,
    reference_window_end TIMESTAMPTZ,
    current_window_start TIMESTAMPTZ,
    current_window_end TIMESTAMPTZ,
    metric_value DOUBLE PRECISION NOT NULL,
    threshold_value DOUBLE PRECISION,
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    details JSONB,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_EVALUATION_METRICS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    roc_auc DOUBLE PRECISION,
    pr_auc DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    fbeta_score DOUBLE PRECISION,
    business_cost DOUBLE PRECISION,
    tn INTEGER,
    fp INTEGER,
    fn INTEGER,
    tp INTEGER,
    sample_size INTEGER,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_ALERTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    model_name TEXT,
    model_version TEXT,
    feature_name TEXT,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    context JSONB,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ
);
"""


# =============================================================================
# SQL de création des index
# =============================================================================

CREATE_INDEXES_SQL = [
    'CREATE INDEX IF NOT EXISTS idx_model_registry_name_version ON model_registry(model_name, model_version);',
    'CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(is_active);',

    'CREATE INDEX IF NOT EXISTS idx_feature_store_request_id ON feature_store_monitoring(request_id);',
    'CREATE INDEX IF NOT EXISTS idx_feature_store_client_id ON feature_store_monitoring(client_id);',
    'CREATE INDEX IF NOT EXISTS idx_feature_store_feature_name ON feature_store_monitoring(feature_name);',
    'CREATE INDEX IF NOT EXISTS idx_feature_store_snapshot_timestamp ON feature_store_monitoring(snapshot_timestamp);',

    'CREATE INDEX IF NOT EXISTS idx_drift_metrics_model_version ON drift_metrics(model_name, model_version);',
    'CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature_name ON drift_metrics(feature_name);',
    'CREATE INDEX IF NOT EXISTS idx_drift_metrics_detected ON drift_metrics(drift_detected);',
    'CREATE INDEX IF NOT EXISTS idx_drift_metrics_computed_at ON drift_metrics(computed_at);',

    'CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_model_version ON evaluation_metrics(model_name, model_version);',
    'CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_dataset_name ON evaluation_metrics(dataset_name);',
    'CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_computed_at ON evaluation_metrics(computed_at);',

    'CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);',
    'CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);',
    'CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);'
]


# =============================================================================
# Fonctions de création des tables
# =============================================================================

def create_model_registry_table(engine) -> None:
    """
    Crée la table `model_registry`.

    Cette table conserve l'historique des versions de modèles,
    leurs métadonnées, leur stage et leur statut de déploiement.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_MODEL_REGISTRY_TABLE_SQL))

    print("Table 'model_registry' créée ou déjà existante.")


def create_feature_store_monitoring_table(engine) -> None:
    """
    Crée la table `feature_store_monitoring`.

    Cette table stocke les valeurs de features observées au moment
    de l'inférence afin de faciliter le monitoring et l'analyse de dérive.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_FEATURE_STORE_MONITORING_TABLE_SQL))

    print("Table 'feature_store_monitoring' créée ou déjà existante.")


def create_drift_metrics_table(engine) -> None:
    """
    Crée la table `drift_metrics`.

    Cette table enregistre les métriques de dérive calculées
    par feature, par fenêtre temporelle et par version de modèle.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_DRIFT_METRICS_TABLE_SQL))

    print("Table 'drift_metrics' créée ou déjà existante.")


def create_evaluation_metrics_table(engine) -> None:
    """
    Crée la table `evaluation_metrics`.

    Cette table stocke les métriques agrégées de performance
    d'un modèle sur une période donnée.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_EVALUATION_METRICS_TABLE_SQL))

    print("Table 'evaluation_metrics' créée ou déjà existante.")


def create_alerts_table(engine) -> None:
    """
    Crée la table `alerts`.

    Cette table centralise les alertes générées par les règles
    de monitoring, de drift ou de performance.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        connection.execute(text(CREATE_ALERTS_TABLE_SQL))

    print("Table 'alerts' créée ou déjà existante.")


def create_indexes(engine) -> None:
    """
    Crée les index utiles pour accélérer les requêtes de monitoring.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    with engine.begin() as connection:
        for sql in CREATE_INDEXES_SQL:
            connection.execute(text(sql))

    print("Index créés ou déjà existants.")


# =============================================================================
# Orchestration
# =============================================================================

TABLE_CREATORS = [
    create_model_registry_table,
    create_feature_store_monitoring_table,
    create_drift_metrics_table,
    create_evaluation_metrics_table,
    create_alerts_table,
]


def main() -> None:
    """
    Point d'entrée du script.

    Cette fonction :
    1. établit la connexion à PostgreSQL,
    2. crée les tables de monitoring,
    3. crée les index,
    4. affiche un message de confirmation.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    for create_table in TABLE_CREATORS:
        create_table(engine)

    create_indexes(engine)

    print("Création des tables de monitoring terminée.")


if __name__ == "__main__":
    main()