"""
Script de chargement des fichiers CSV dans PostgreSQL.

Ce module permet d'importer les fichiers CSV sources dans les tables RAW
déjà créées dans PostgreSQL, en conservant strictement les noms de colonnes
d'origine.

Objectif
--------
Alimenter les tables RAW du projet avec les données des fichiers CSV sources
afin de préparer les futures étapes de feature engineering et de monitoring.

Fonctionnalités
---------------
- Chargement des variables d'environnement
- Vérification de l'existence des fichiers CSV
- Lecture des fichiers avec pandas
- Insertion des données dans PostgreSQL
- Mapping explicite entre fichiers CSV et tables cibles
- Gestion des erreurs d'import

Notes
-----
- Les tables doivent être créées au préalable.
- Les noms de colonnes du CSV doivent correspondre exactement aux colonnes SQL.
- Ce script utilise pandas + SQLAlchemy pour l'insertion.
- Pour de très gros volumes, PostgreSQL COPY sera plus rapide.
"""

from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL n'est pas défini dans les variables d'environnement")


# Mapping explicite fichier CSV -> table PostgreSQL
# Cela permet d'éviter toute ambiguïté sur les noms de tables.
CSV_TO_TABLE_MAP = {
    "application_test.csv": "application_test",
    "bureau.csv": "bureau",
    "bureau_balance.csv": "bureau_balance",
    "credit_card_balance.csv": "credit_card_balance",
    "installments_payments.csv": "installments_payments",
    "POS_CASH_balance.csv": "POS_CASH_balance",
    "previous_application.csv": "previous_application",
}


# =============================================================================
# Fonction de chargement
# =============================================================================

def load_csv_to_postgres(csv_path: Path, table_name: str, engine) -> None:
    """
    Charge un fichier CSV dans une table PostgreSQL existante.

    Parameters
    ----------
    csv_path : Path
        Chemin vers le fichier CSV.
    table_name : str
        Nom de la table PostgreSQL cible.
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.

    Raises
    ------
    Exception
        En cas d'erreur de lecture ou d'insertion.
    """
    print(f"\nChargement du fichier : {csv_path.name}")
    print(f"Table cible : {table_name}")

    try:
        # ---------------------------------------------------------------------
        # Lecture du CSV
        # ---------------------------------------------------------------------
        df = pd.read_csv(csv_path)

        print(f"{len(df)} lignes lues depuis le fichier CSV.")

        # ---------------------------------------------------------------------
        # Insertion dans PostgreSQL
        # ---------------------------------------------------------------------
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",   # important : on alimente la table existante
            index=False,
            method="multi",
            chunksize=5000,
        )

        print(f"Données insérées dans la table '{table_name}'.")

    except Exception as e:
        print(f"Erreur lors du chargement de {csv_path.name} : {e}")
        raise


# =============================================================================
# Point d'entrée
# =============================================================================

def main() -> None:
    """
    Point d'entrée du script.

    Cette fonction :
    1. se connecte à PostgreSQL,
    2. vérifie la présence des fichiers CSV,
    3. charge chaque fichier dans sa table RAW correspondante,
    4. affiche un message final de confirmation.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    for filename, table_name in CSV_TO_TABLE_MAP.items():
        csv_path = DATA_DIR / filename

        if not csv_path.exists():
            print(f"Fichier introuvable : {csv_path}")
            continue

        load_csv_to_postgres(csv_path, table_name, engine)

    print("\nChargement des tables RAW terminé.")


if __name__ == "__main__":
    main()