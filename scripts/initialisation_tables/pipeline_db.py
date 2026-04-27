from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(script_name: str) -> None:
    """
    Lance un script Python d'initialisation depuis le dossier
    scripts/initialisation_tables.
    """
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script introuvable : {script_path}")

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    """
    Lance l'initialisation complète des tables PostgreSQL.
    """
    run_script("create_prediction_tables.py")
    run_script("create_monitoring_tables.py")


if __name__ == "__main__":
    main()