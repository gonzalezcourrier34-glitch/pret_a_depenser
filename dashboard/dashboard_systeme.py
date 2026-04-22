"""
Page Streamlit : système et données.

Cette page fournit une vue globale du système MLOps en production.

Objectifs
---------
- Vérifier rapidement l'état de l'API (healthcheck)
- Vérifier la présence et la validité logique de la clé API
- Suivre la disponibilité des ressources métier (logs, monitoring, etc.)
- Explorer les données exposées par l'API

Architecture
------------
⚠️ IMPORTANT :
Cette page ne lit jamais directement PostgreSQL.

Toutes les données affichées proviennent de l'API FastAPI :
- /predict/health
- /history/*
- /monitoring/*
- etc.

Le dashboard est donc totalement découplé de la base de données.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st


# =============================================================================
# Helpers UI
# =============================================================================

def _render_card(title: str, value: Any, subtitle: str = "") -> None:
    """
    Affiche une carte KPI légère.
    """
    st.markdown(
        f"""
        <div style="
            padding: 16px 18px;
            border-radius: 16px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            min-height: 115px;
        ">
            <div style="font-size: 0.9rem; color: #9CA3AF; margin-bottom: 8px;">
                {title}
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 8px;">
                {value}
            </div>
            <div style="font-size: 0.85rem; color: #D1D5DB;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_title(title: str, subtitle: str = "") -> None:
    """
    Affiche un titre de section plus visuel.
    """
    st.markdown(
        f"""
        <div style="margin-top: 8px; margin-bottom: 14px;">
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 4px;">
                {title}
            </div>
            <div style="color: #6B7280; font-size: 0.95rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_badge(label: str, color: str) -> str:
    """
    Retourne un badge HTML simple.
    """
    return (
        f"<span style='"
        f"display:inline-block;"
        f"padding:6px 12px;"
        f"border-radius:999px;"
        f"background:{color};"
        f"color:white;"
        f"font-size:0.82rem;"
        f"font-weight:600;'>"
        f"{label}</span>"
    )


def _safe_dict(value: Any) -> dict[str, Any]:
    """
    Retourne un dictionnaire vide si la valeur n'est pas un dict.
    """
    return value if isinstance(value, dict) else {}


def _safe_dataframe(value: Any) -> pd.DataFrame:
    """
    Garantit le retour d'un DataFrame.
    """
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _safe_bool(value: Any) -> bool:
    """
    Convertit une valeur en booléen de façon robuste.
    """
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "oui"}

    try:
        return bool(value)
    except Exception:
        return False


def _build_resource_summary_card(
    total_resources: int,
    available_count: int,
) -> tuple[str, str]:
    """
    Détermine le statut global du système côté ressources.
    """
    if total_resources <= 0:
        return "Aucune ressource", "#6B7280"

    if available_count == total_resources:
        return "Toutes disponibles", "#16A34A"

    if available_count == 0:
        return "Aucune disponible", "#DC2626"

    return "Disponibilité partielle", "#D97706"


def _normalize_available_tables(
    available_tables: list[str],
    preview_map: dict[str, pd.DataFrame],
) -> list[str]:
    """
    Nettoie la liste des tables disponibles.

    Si la liste fournie est vide, on se rabat sur les clés du preview_map.
    """
    if isinstance(available_tables, list) and available_tables:
        return [str(x) for x in available_tables]

    if isinstance(preview_map, dict) and preview_map:
        return [str(x) for x in preview_map.keys()]

    return []


# =============================================================================
# Page principale
# =============================================================================

def render_systeme_page(
    *,
    health_data: dict,
    tables_status_df: pd.DataFrame,
    selected_table_preview_map: dict[str, pd.DataFrame],
    available_tables: list[str],
    api_key: str | None = None,
) -> None:
    """
    Affiche la page "Système / Données" du dashboard.

    Parameters
    ----------
    health_data : dict
        Résultat de l'endpoint `/predict/health`.
    tables_status_df : pd.DataFrame
        DataFrame représentant l'état logique des ressources du système.
    selected_table_preview_map : dict[str, pd.DataFrame]
        Mapping entre une ressource et son aperçu de données.
    available_tables : list[str]
        Liste des ressources explorables côté dashboard.
    api_key : str | None, optional
        Clé API utilisée par le dashboard.
    """
    health_data = _safe_dict(health_data)
    tables_status_df = _safe_dataframe(tables_status_df)
    selected_table_preview_map = (
        selected_table_preview_map
        if isinstance(selected_table_preview_map, dict)
        else {}
    )
    available_tables = _normalize_available_tables(
        available_tables=available_tables if isinstance(available_tables, list) else [],
        preview_map=selected_table_preview_map,
    )

    st.markdown("## Système et données")
    st.caption(
        "Vue d’ensemble de l’API, des ressources exposées et des jeux de données consultables."
    )

    status_value = health_data.get("status", "unknown")
    status_upper = str(status_value).upper()

    api_key_present = bool(api_key and str(api_key).strip())
    api_key_status = "Configurée" if api_key_present else "Manquante"

    total_resources = len(tables_status_df)
    available_count = 0

    if not tables_status_df.empty and "is_available" in tables_status_df.columns:
        available_count = int(
            tables_status_df["is_available"].apply(_safe_bool).sum()
        )

    resources_label, resources_color = _build_resource_summary_card(
        total_resources=total_resources,
        available_count=available_count,
    )

    # -------------------------------------------------------------------------
    # KPI bandeau
    # -------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        _render_card(
            "Health API",
            status_upper,
            "État remonté par /predict/health",
        )
    with k2:
        _render_card(
            "Clé API",
            api_key_status,
            "Présence de la configuration côté dashboard",
        )
    with k3:
        _render_card(
            "Ressources actives",
            f"{available_count}/{total_resources}",
            "Disponibilité logique des ressources",
        )
    with k4:
        _render_card(
            "Heure de rendu",
            datetime.now().strftime("%H:%M:%S"),
            "Instant du diagnostic affiché",
        )

    tab1, tab2 = st.tabs(["💻 Système", "🗂️ Données"])

    # =========================================================================
    # ONGLET 1 - SYSTÈME
    # =========================================================================
    with tab1:
        _render_section_title(
            "État du système",
            "Diagnostic rapide de l’API et des ressources visibles par le dashboard.",
        )

        left, right = st.columns([2, 1])

        with left:
            badge_color = "#16A34A"
            badge_label = "Système opérationnel"

            if status_upper != "OK":
                badge_color = "#DC2626"
                badge_label = "Healthcheck en anomalie"
            elif total_resources > 0 and available_count < total_resources:
                badge_color = "#D97706"
                badge_label = "Système partiellement disponible"

            st.markdown(
                f"""
                <div style="
                    padding: 18px;
                    border-radius: 18px;
                    background: #F8FAFC;
                    border: 1px solid #E5E7EB;
                    margin-bottom: 12px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;">
                        Statut global
                    </div>
                    <div style="margin-bottom: 10px;">
                        {_status_badge(badge_label, badge_color)}
                    </div>
                    <div style="margin-bottom: 8px;">
                        {_status_badge(resources_label, resources_color)}
                    </div>
                    <div style="color: #4B5563; font-size: 0.92rem;">
                        Healthcheck API : {status_upper} <br>
                        Clé API dashboard : {api_key_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            if st.button("🔄 Rafraîchir et revérifier", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        _render_section_title(
            "Réponse healthcheck",
            "Contenu brut renvoyé par l’API.",
        )

        if health_data:
            st.json(health_data)
        else:
            st.info("Aucune réponse healthcheck disponible.")

        _render_section_title(
            "Ressources du système",
            "Disponibilité logique des ressources exploitées par le dashboard.",
        )

        if tables_status_df.empty:
            st.info("Aucune information de statut disponible.")
        else:
            display_df = tables_status_df.copy()

            if "is_available" in display_df.columns:
                display_df["is_available"] = display_df["is_available"].apply(
                    lambda x: "✅ Oui" if _safe_bool(x) else "❌ Non"
                )

            preferred_cols = [
                col
                for col in ["resource_name", "is_available", "row_count", "comment"]
                if col in display_df.columns
            ]

            if preferred_cols:
                st.dataframe(display_df[preferred_cols], width="stretch")
            else:
                st.dataframe(display_df, width="stretch")

            if total_resources > 0:
                if available_count == total_resources:
                    st.success(
                        f"Tous les services visibles sont disponibles ({available_count}/{total_resources})."
                    )
                else:
                    st.warning(
                        f"Système partiellement disponible : {available_count}/{total_resources} ressources actives."
                    )

    # =========================================================================
    # ONGLET 2 - DONNÉES
    # =========================================================================
    with tab2:
        _render_section_title(
            "Inspection des données",
            "Explorer les ressources exposées par l’API sans accès direct à PostgreSQL.",
        )

        if not available_tables:
            st.info("Aucune ressource explorable disponible.")
        else:
            col1, col2 = st.columns([1.2, 2])

            with col1:
                selected_table = st.selectbox(
                    "Choisir une ressource",
                    options=available_tables,
                )

            preview_df = _safe_dataframe(
                selected_table_preview_map.get(selected_table, pd.DataFrame())
            )

            with col2:
                if preview_df.empty:
                    st.warning(f"Aucune donnée disponible pour `{selected_table}`.")
                else:
                    st.success(f"Ressource `{selected_table}` chargée avec succès.")

            if preview_df.empty:
                st.info("Aucun aperçu disponible pour cette ressource.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Nombre de lignes (aperçu)", len(preview_df))
                m2.metric("Nombre de colonnes", len(preview_df.columns))
                m3.metric(
                    "Mémoire approx.",
                    f"{preview_df.memory_usage(deep=True).sum() / 1024:.1f} Ko",
                )

                with st.expander("Voir les colonnes", expanded=False):
                    st.write(list(preview_df.columns))

                with st.expander("Statistiques rapides", expanded=False):
                    numeric_df = preview_df.select_dtypes(include="number")
                    if numeric_df.empty:
                        st.info("Aucune colonne numérique disponible pour les statistiques.")
                    else:
                        try:
                            stats_df = numeric_df.describe().transpose()
                            st.dataframe(stats_df, width="stretch")
                        except Exception:
                            st.info("Impossible de calculer les statistiques sur cet aperçu.")

                st.dataframe(preview_df, width="stretch")

                csv = preview_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger l’aperçu CSV",
                    csv,
                    file_name=f"{selected_table}_preview.csv",
                    use_container_width=True,
                )