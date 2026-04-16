"""
Page Streamlit : système et données.

Cette page regroupe :
- l'état global du système
- la disponibilité des tables
- l'exploration simple des données PostgreSQL
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st


def render_systeme_page(
    *,
    DATABASE_URL: str,
    FEATURES_TABLE: str,
    table_exists,
    get_table_columns,
    show_table_status,
    run_query,
    get_table_preview_query,
    history_limit: int,
) -> None:
    """
    Affiche la page système / données.

    Parameters
    ----------
    DATABASE_URL : str
        URL de connexion PostgreSQL.
    FEATURES_TABLE : str
        Table de features par défaut.
    table_exists :
        Fonction utilitaire de vérification d'existence de table.
    get_table_columns :
        Fonction utilitaire de lecture des colonnes d'une table.
    show_table_status :
        Fonction d'affichage de l'état d'une table.
    run_query :
        Fonction générique d'exécution SQL.
    get_table_preview_query :
        Fabrique de requêtes SQL d'aperçu.
    history_limit : int
        Nombre maximum de lignes à afficher.
    """
    tab1, tab2 = st.tabs(["💻 Système", "🗂️ Données / CRUD"])

    with tab1:
        st.markdown("### État du système")

        c1, c2 = st.columns(2)
        c1.metric("Connexion DB", "OK" if DATABASE_URL else "Non")
        c2.metric("Table features", FEATURES_TABLE)

        st.markdown("#### Tables PostgreSQL")
        left, right = st.columns(2)

        with left:
            show_table_status("prediction_logs")
            show_table_status("ground_truth_labels")
            show_table_status("prediction_features_snapshot")
            show_table_status("model_registry")

        with right:
            show_table_status("feature_store_monitoring")
            show_table_status("drift_metrics")
            show_table_status("evaluation_metrics")
            show_table_status("alerts")

        st.markdown("#### Diagnostic rapide")
        st.write("Heure de rendu :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if table_exists(FEATURES_TABLE):
            columns = get_table_columns(FEATURES_TABLE)
            st.success(f"`{FEATURES_TABLE}` disponible avec {len(columns)} colonnes.")
            st.write("Aperçu :", columns[:40])
        else:
            st.warning(f"La table `{FEATURES_TABLE}` n'existe pas.")

    with tab2:
        st.markdown("### Inspection des données")

        selected_table = st.selectbox(
            "Choisir une table à explorer",
            options=[
                "prediction_logs",
                "ground_truth_labels",
                "prediction_features_snapshot",
                "model_registry",
                "feature_store_monitoring",
                "drift_metrics",
                "evaluation_metrics",
                "alerts",
                "features_client_test",
                "features_client_test_enriched",
            ],
        )

        if not table_exists(selected_table):
            st.warning(f"La table `{selected_table}` n'existe pas.")
        else:
            columns = get_table_columns(selected_table)
            st.success(f"Table `{selected_table}` trouvée.")
            st.write(f"Nombre de colonnes : {len(columns)}")
            st.write("Colonnes :", columns)

            preview_df = run_query(
                get_table_preview_query(selected_table),
                {"limit_rows": min(history_limit, 200)},
            )
            st.dataframe(preview_df, use_container_width=True)