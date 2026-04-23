"""
Script de nettoyage de la base PostgreSQL en conservant le registre des modèles.

Ce module vide les principales tables utiles au projet MLOps actuel :
- historique de prédictions
- snapshots de features
- tables de monitoring

Objectif
--------
Remettre la base dans un état propre pour rejouer des simulations,
des batchs ou des tests, tout en conservant `model_registry`
afin de ne pas perdre les versions de modèles enregistrées.

Notes
-----
- La table `model_registry` n'est jamais vidée.
- Le script utilise TRUNCATE ... RESTART IDENTITY CASCADE.
- Les tables absentes sont ignorées.
- Un lock_timeout est appliqué pour éviter un blocage infini.
- L'exécution doit être explicitement autorisée via ALLOW_DB_TRUNCATE=true.
"""

from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "credit_api")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

ALLOW_DB_TRUNCATE = os.getenv("ALLOW_DB_TRUNCATE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


# =============================================================================
# Tables à conserver / vider
# =============================================================================

TABLES_TO_KEEP: Final[list[str]] = [
    "model_registry",
]

TABLES_TO_TRUNCATE: Final[list[str]] = [
    # -------------------------------------------------------------------------
    # Historique / prédictions
    # -------------------------------------------------------------------------
    "prediction_logs",
    "ground_truth_labels",
    "prediction_features_snapshot",

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------
    "feature_store_monitoring",
    "drift_metrics",
    "evaluation_metrics",
    "alerts",

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------
    "model_registry"
]


# =============================================================================
# Helpers
# =============================================================================

def validate_configuration() -> None:
    """
    Vérifie que le script est explicitement autorisé.

    Raises
    ------
    RuntimeError
        Si ALLOW_DB_TRUNCATE n'est pas activé.
    """
    if not ALLOW_DB_TRUNCATE:
        raise RuntimeError(
            "Nettoyage refusé. Définis ALLOW_DB_TRUNCATE=true pour autoriser "
            "l'exécution de ce script."
        )


def table_exists(engine: Engine, table_name: str) -> bool:
    """
    Vérifie si une table existe dans le schéma public.

    Parameters
    ----------
    engine : Engine
        Moteur SQLAlchemy connecté à PostgreSQL.
    table_name : str
        Nom de la table à tester.

    Returns
    -------
    bool
        True si la table existe, sinon False.
    """
    sql = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = :table_name
        )
        """
    )

    with engine.connect() as connection:
        exists_flag = connection.execute(
            sql,
            {"table_name": table_name},
        ).scalar()

    return bool(exists_flag)


def truncate_table(engine: Engine, table_name: str) -> None:
    """
    Vide une table PostgreSQL avec reset d'identité.

    Parameters
    ----------
    engine : Engine
        Moteur SQLAlchemy connecté à PostgreSQL.
    table_name : str
        Nom de la table à vider.
    """
    print(f"-> Tentative de vidage : {table_name}")

    with engine.begin() as connection:
        connection.execute(text("SET lock_timeout TO '3000ms';"))
        connection.execute(
            text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;')
        )

    print(f"✅ Table vidée : {table_name}")


def print_execution_plan() -> None:
    """
    Affiche le plan d'exécution du nettoyage.
    """
    print("=" * 80)
    print("NETTOYAGE DE LA BASE POSTGRESQL")
    print("=" * 80)
    print(f"URL utilisée              : {DATABASE_URL}")
    print(f"ALLOW_DB_TRUNCATE         : {ALLOW_DB_TRUNCATE}")
    print("Tables conservées         :")
    for table_name in TABLES_TO_KEEP:
        print(f"  - {table_name}")

    print("Tables candidates au vidage :")
    for table_name in TABLES_TO_TRUNCATE:
        print(f"  - {table_name}")
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Point d'entrée principal du script de nettoyage.
    """
    validate_configuration()
    print_execution_plan()

    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    truncated_count = 0
    missing_count = 0
    error_count = 0

    print("\nDébut du nettoyage...\n")

    for table_name in TABLES_TO_TRUNCATE:
        try:
            if table_exists(engine, table_name):
                truncate_table(engine, table_name)
                truncated_count += 1
            else:
                print(f"⚪ Table absente, ignorée : {table_name}")
                missing_count += 1

        except Exception as exc:
            print(f"❌ Erreur sur {table_name} : {exc}")
            error_count += 1

    print("\n" + "=" * 80)
    print("NETTOYAGE TERMINÉ")
    print("=" * 80)
    print(f"Tables vidées    : {truncated_count}")
    print(f"Tables absentes  : {missing_count}")
    print(f"Tables en erreur : {error_count}")
    print("La table 'model_registry' a été conservée.")
    print("=" * 80)


if __name__ == "__main__":
    main()