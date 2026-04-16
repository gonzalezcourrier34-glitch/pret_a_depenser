"""
Script de création de la table propre de features client.

Ce module crée la structure de la table `features_client_test`
à partir de la table source `features_client_test_raw`, mais sans insérer
les données.

Objectif
--------
Séparer la création de la table cible du nettoyage métier afin de :
- clarifier les responsabilités
- faciliter les tests
- rendre le pipeline plus lisible

Notes
-----
- La source est `features_client_test_raw`.
- La cible est `features_client_test`.
- La table créée est vide.
- Les colonnes conservées sont déterminées selon les mêmes règles
  que le script de nettoyage.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Configuration de la base
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
    Récupère la liste ordonnée des colonnes d'une table PostgreSQL.

    Parameters
    ----------
    engine :
        Moteur SQLAlchemy connecté à PostgreSQL.
    table_name : str
        Nom de la table à inspecter.

    Returns
    -------
    list[str]
        Liste des colonnes.
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
        Colonnes présentes dans `features_client_test_raw`.

    Returns
    -------
    list[str]
        Colonnes à conserver.
    """
    keep_cols: list[str] = []

    for col in columns:
        if col == "SK_ID_CURR":
            keep_cols.append(col)
            continue

        if col.endswith("_MODE") or col.endswith("_MEDI"):
            continue

        if col in DROP_EXACT_COLUMNS:
            continue

        if "__sum" in col:
            continue

        if "__min" in col:
            continue

        if "__max" in col and not any(pattern in col for pattern in KEEP_MAX_PATTERNS):
            continue

        keep_cols.append(col)

    # Si un compteur de documents existe, on retire les colonnes unitaires
    if "DOC_COUNT" in keep_cols:
        keep_cols = [c for c in keep_cols if not c.startswith("FLAG_DOCUMENT_")]

    return keep_cols


# =============================================================================
# Création de la table cible vide
# =============================================================================

def create_clean_features_table(engine) -> None:
    """
    Crée la table cible vide `features_client_test`.

    La structure est dérivée de `features_client_test_raw`,
    mais seules les colonnes utiles sont conservées.

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
        raise ValueError("Aucune colonne conservée pour la table cible.")

    select_sql = ",\n    ".join(f'"{col}"' for col in keep_cols)

    create_sql = f"""
    DROP TABLE IF EXISTS {target_table};

    CREATE TABLE {target_table} AS
    SELECT
        {select_sql}
    FROM {source_table}
    WHERE 1 = 0;
    """

    index_sql = f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_{target_table}_sk_id_curr
    ON {target_table} ("SK_ID_CURR");
    """

    with engine.begin() as connection:
        connection.execute(text(create_sql))
        connection.execute(text(index_sql))

    print(f"Table '{target_table}' créée vide avec {len(keep_cols)} colonnes.")


# =============================================================================
# Point d'entrée
# =============================================================================

def main() -> None:
    """
    Point d'entrée du script.
    """
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL, echo=False)
    print("Connexion établie.")

    create_clean_features_table(engine)

    print("Création de la table cible terminée.")


if __name__ == "__main__":
    main()