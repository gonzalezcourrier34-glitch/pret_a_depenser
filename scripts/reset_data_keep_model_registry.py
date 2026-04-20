"""
Script de nettoyage de la base PostgreSQL en conservant le registre des modèles.

Ce module vide les principales tables du projet MLOps :
- tables RAW
- tables temporaires
- tables de features
- tables de prédictions
- tables de monitoring

Objectif
--------
Remettre la base dans un état propre pour rejouer des simulations,
des batchs ou des tests, tout en conservant `model_registry`
afin de ne pas perdre les versions de modèles enregistrées.

Notes
-----
- La table `model_registry` n'est pas vidée.
- Le script utilise TRUNCATE ... RESTART IDENTITY CASCADE.
- Les tables absentes sont ignorées.
- Un lock_timeout est appliqué pour éviter un blocage infini.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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


# =============================================================================
# Tables à vider
# =============================================================================

TABLES_TO_TRUNCATE = [
    "application_test",
    "bureau_balance",
    "bureau",
    "credit_card_balance",
    "installments_payments",
    "POS_CASH_balance",
    "previous_application",
    "bb_agg_tmp",
    "bureau_agg_curr_tmp",
    "pos_agg_tmp",
    "cc_agg_tmp",
    "inst_agg_tmp",
    "prev_agg_curr_tmp",
    "tmp_batch_sk_id_bureau",
    "tmp_batch_sk_id_prev",
    "tmp_batch_sk_id_curr",
    "features_client_test_raw",
    "features_client_test",
    "features_client_test_enriched",
    "features_client_test_model",
    "prediction_features_snapshot",
    "ground_truth_labels",
    "prediction_logs",
    "feature_store_monitoring",
    "drift_metrics",
    "evaluation_metrics",
    "alerts",
]


# =============================================================================
# Helpers
# =============================================================================

def table_exists(engine, table_name: str) -> bool:
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

    with engine.begin() as connection:
        exists_flag = connection.execute(sql, {"table_name": table_name}).scalar()

    return bool(exists_flag)


def truncate_table(engine, table_name: str) -> None:
    print(f"-> Tentative de vidage : {table_name}")

    try:
        with engine.begin() as connection:
            connection.execute(text("SET lock_timeout TO '3000ms';"))
            connection.execute(
                text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;')
            )

        print(f"✅ Table vidée : {table_name}")

    except Exception as exc:
        print(f"❌ Impossible de vider {table_name} : {exc}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Connexion à PostgreSQL...")
    print(f"URL utilisée : {DATABASE_URL}")

    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    print("\nDébut du nettoyage...")
    print("La table 'model_registry' est conservée.\n")

    for table_name in TABLES_TO_TRUNCATE:
        try:
            exists_flag = table_exists(engine, table_name)

            if exists_flag:
                truncate_table(engine, table_name)
            else:
                print(f"⚪ Table absente, ignorée : {table_name}")

        except Exception as exc:
            print(f"💥 Erreur inattendue sur {table_name} : {exc}")

    print("\nNettoyage terminé.")
    print("La table 'model_registry' a été conservée.")


if __name__ == "__main__":
    main()