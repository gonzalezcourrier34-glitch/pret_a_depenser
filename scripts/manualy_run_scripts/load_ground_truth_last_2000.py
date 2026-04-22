from __future__ import annotations

"""
Import complet des vérités terrain depuis application_train.csv
vers ground_truth_labels.

Objectif
--------
Reconstituer entièrement la table `ground_truth_labels`
à partir du fichier source `application_train.csv`
afin d'alimenter le monitoring et l'évaluation du modèle.

Principe
--------
- lecture du CSV `application_train.csv`
- vérification des colonnes attendues
- vidage complet de la table `ground_truth_labels`
- insertion en base de tous les clients du fichier

Notes
-----
- request_id = NULL car il s'agit d'un backfill historique
- label_source = "application_train_full"
- observed_at est fixé au moment de l'import
- ce script remplace totalement le contenu existant de la table
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.crud.prediction import create_ground_truth_label


CSV_PATH = PROJECT_ROOT / "data" / "application_train.csv"


def load_all_rows(path: Path) -> pd.DataFrame:
    """
    Charge toutes les lignes du CSV source.

    Parameters
    ----------
    path : Path
        Chemin du fichier CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant toutes les lignes du fichier.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si les colonnes attendues sont absentes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    required_columns = {"SK_ID_CURR", "TARGET"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Le CSV doit contenir les colonnes SK_ID_CURR et TARGET. "
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    return df.copy()


def truncate_ground_truth_table(db: Session) -> None:
    """
    Vide entièrement la table ground_truth_labels.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    """
    db.execute(text("TRUNCATE TABLE ground_truth_labels RESTART IDENTITY"))


def main() -> None:
    """
    Point d'entrée du script.

    Cette fonction :
    - charge toutes les lignes de application_train.csv
    - vide complètement ground_truth_labels
    - convertit TARGET en vérité terrain
    - insère toutes les lignes en base
    """
    df = load_all_rows(CSV_PATH)

    db: Session = SessionLocal()

    inserted = 0
    skipped = 0
    now_utc = datetime.now(timezone.utc)

    try:
        print("Chargement du CSV...")
        print(f"Lignes lues : {len(df)}")

        print("Vidage de la table ground_truth_labels...")
        truncate_ground_truth_table(db)

        print("Insertion des labels...")

        for _, row in df.iterrows():
            try:
                client_id = int(row["SK_ID_CURR"])
                true_label = int(row["TARGET"])
            except Exception:
                skipped += 1
                continue

            if true_label not in (0, 1):
                skipped += 1
                continue

            create_ground_truth_label(
                db=db,
                request_id=None,
                client_id=client_id,
                true_label=true_label,
                label_source="application_train_full",
                observed_at=now_utc,
                notes="Backfill complet depuis application_train.csv",
            )

            inserted += 1

        db.commit()

        print("=====================================")
        print("Import terminé avec succès")
        print(f"{inserted} labels insérés")
        print(f"{skipped} lignes ignorées")
        print("=====================================")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()