"""
Pipeline complet de préparation des données et features.

Ce script orchestre toutes les étapes :
- création des tables raw
- chargement des données
- création des features
- nettoyage
- enrichissement
- validation
- création des tables finales
"""

import subprocess


def run_script(script_name: str):
    print(f"\nExécution : {script_name}")
    result = subprocess.run(
        ["python", f"scripts/{script_name}"],
        check=True
    )
    print(f"Terminé : {script_name}")


def main():
    run_script("create_raw_tables.py")
    run_script("load_csv_to_postgres.py")
    run_script("create_features_table.py")
    run_script("clean_features_table.py")
    run_script("enrich_features_table.py")
    run_script("verify_model_features.py")
    run_script("create_model_ready_table.py")
    run_script("create_prediction_tables.py")
    run_script("create_monitoring_tables.py")

    print("\nPIPELINE COMPLET TERMINÉ")


if __name__ == "__main__":
    main()