"""
Script de nettoyage de la table de features client.

Ce module transforme la table intermédiaire `features_client_test_raw`
en une table finale `features_client_test`, plus compacte et plus propre
pour l'inférence.

Objectif
--------
Réduire la dimension de la table de features en supprimant :
- les colonnes redondantes
- les agrégations peu utiles
- certaines colonnes techniques ou trop verbeuses

Notes
-----
- La source est la table `features_client_test_raw`.
- La table finale produite est `features_client_test`.
- Cette étape reprend l'esprit du nettoyage appliqué dans le notebook.
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
# Configuration du nettoyage
# =============================================================================

KEEP_MAX_PATTERNS = [
    "recent_max_dpd",
    "DPD_POS__max",
    "SEVERE_LATE_30__max",
    "SK_DPD__max",
    "SK_DPD_DEF__max",
    "CC_UTILIZATION_RATIO__max",
    "CC_PAYMENT_MIN_RATIO__max",
    "OVERDUE_RATIO__max",
]

DROP_EXACT_COLUMNS = [
    "EXT_INT_1_2",
    "EXT_INT_1_3",
    "EXT_INT_2_3",
]


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def get_columns(engine, table_name: str) -> list[str]:
    """
    Récupère la liste des colonnes d'une table PostgreSQL.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    table_name : str
        Nom de la table à inspecter.

    Returns
    -------
    list[str]
        Liste ordonnée des colonnes.
    """
    sql = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )

    with engine.begin() as connection:
        rows = connection.execute(sql, {"table_name": table_name}).fetchall()

    return [row[0] for row in rows]


def select_columns_to_keep(columns: list[str]) -> list[str]:
    """
    Détermine les colonnes à conserver pour la table finale.

    Parameters
    ----------
    columns : list[str]
        Liste des colonnes de `features_client_test_raw`.

    Returns
    -------
    list[str]
        Liste des colonnes à garder.
    """
    keep_cols: list[str] = []

    for col in columns:
        if col == "SK_ID_CURR":
            keep_cols.append(col)
            continue

        # Suppression MODE / MEDI
        if col.endswith("_MODE") or col.endswith("_MEDI"):
            continue

        # Suppression exacte
        if col in DROP_EXACT_COLUMNS:
            continue

        # Suppression des colonnes __sum
        if "__sum" in col:
            continue

        # Suppression de la majorité des __min
        if "__min" in col:
            continue

        # Suppression de la majorité des __max, sauf exceptions métier
        if "__max" in col and not any(pattern in col for pattern in KEEP_MAX_PATTERNS):
            continue

        keep_cols.append(col)

    # Si DOC_COUNT existe, on supprime les FLAG_DOCUMENT_*
    if "DOC_COUNT" in keep_cols:
        keep_cols = [c for c in keep_cols if not c.startswith("FLAG_DOCUMENT_")]

    return keep_cols


# =============================================================================
# Création de la table nettoyée
# =============================================================================

def create_clean_features_table(engine) -> None:
    """
    Crée la table `features_client_test` à partir de
    `features_client_test_raw` après sélection des colonnes utiles.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    """
    source_table = "features_client_test_raw"
    target_table = "features_client_test"

    columns = get_columns(engine, source_table)
    keep_cols = select_columns_to_keep(columns)

    if not keep_cols:
        raise ValueError("Aucune colonne conservée pour la table finale.")

    select_sql = ",\n    ".join(f'"{col}"' for col in keep_cols)

    create_sql = f"""
    DROP TABLE IF EXISTS {target_table};

    CREATE TABLE {target_table} AS
    SELECT
        {select_sql}
    FROM {source_table};
    """

    index_sql = f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_{target_table}_sk_id_curr
    ON {target_table} ("SK_ID_CURR");
    """

    with engine.begin() as connection:
        connection.execute(text(create_sql))
        connection.execute(text(index_sql))

    print(f"Table '{target_table}' créée avec {len(keep_cols)} colonnes conservées.")
    print(f"Colonnes supprimées : {len(columns) - len(keep_cols)}")


# =============================================================================
# Point d'entrée
# =============================================================================

def main() -> None:
    """
    Point d'entrée du script.

    Cette fonction :
    1. se connecte à PostgreSQL,
    2. lit les colonnes de `features_client_test_raw`,
    3. crée `features_client_test`,
    4. affiche un résumé.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    create_clean_features_table(engine)

    print("Nettoyage de la table de features terminé.")


if __name__ == "__main__":
    main()