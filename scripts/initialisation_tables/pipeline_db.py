"""
Pipeline d'initialisation du stockage de production.

Ce script crée uniquement les tables nécessaires
aux logs de prédiction et au monitoring.
"""

from __future__ import annotations

import subprocess


def run_script(script_name: str) -> None:
    print(f"\nExécution : {script_name}")
    subprocess.run(
        ["uv", "run", "python", f"scripts/{script_name}"],
        check=True
    )
    print(f"Terminé : {script_name}")


def main() -> None:
    run_script("create_prediction_tables.py")
    run_script("create_monitoring_tables.py")

    print("\nINITIALISATION DB TERMINÉE")


if __name__ == "__main__":
    main()