"""
Service métier lié à l'historique des prédictions.

Ce module centralise les lectures SQL utilisées par les routes
d'historique de l'API.

Objectif
--------
Séparer la logique d'accès aux données des routes FastAPI afin de :
- rendre les routes plus lisibles
- centraliser les requêtes SQL
- faciliter les tests et la maintenance

Fonctionnalités
---------------
- lecture de l'historique des prédictions
- lecture du détail d'une prédiction
- lecture des vérités terrain
- lecture du snapshot de features d'une requête
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


# =============================================================================
# Historique des prédictions
# =============================================================================

def get_prediction_history(
    db: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    client_id: int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    only_errors: bool = False,
) -> dict:
    """
    Retourne l'historique des prédictions journalisées.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    limit : int
        Nombre maximal de lignes retournées.
    offset : int
        Décalage pour la pagination.
    client_id : int | None
        Filtre optionnel sur un identifiant client.
    model_name : str | None
        Filtre optionnel sur le nom du modèle.
    model_version : str | None
        Filtre optionnel sur la version du modèle.
    only_errors : bool
        Si True, retourne uniquement les prédictions en erreur.

    Returns
    -------
    dict
        Historique des prédictions.
    """
    sql = """
    SELECT
        id,
        request_id,
        client_id,
        model_name,
        model_version,
        prediction,
        score,
        threshold_used,
        latency_ms,
        prediction_timestamp,
        status_code,
        error_message
    FROM prediction_logs
    WHERE 1 = 1
    """

    params: dict[str, object] = {
        "limit": limit,
        "offset": offset,
    }

    if client_id is not None:
        sql += " AND client_id = :client_id"
        params["client_id"] = client_id

    if model_name is not None:
        sql += " AND model_name = :model_name"
        params["model_name"] = model_name

    if model_version is not None:
        sql += " AND model_version = :model_version"
        params["model_version"] = model_version

    if only_errors:
        sql += " AND error_message IS NOT NULL"

    sql += """
    ORDER BY prediction_timestamp DESC
    LIMIT :limit OFFSET :offset
    """

    rows = db.execute(text(sql), params).mappings().all()

    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "items": [dict(row) for row in rows],
    }


def get_prediction_detail(db: Session, *, request_id: str) -> dict | None:
    """
    Retourne le détail complet d'une prédiction.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    request_id : str
        Identifiant unique de requête.

    Returns
    -------
    dict | None
        Détail de la prédiction ou None si introuvable.
    """
    sql = text(
        """
        SELECT
            id,
            request_id,
            client_id,
            model_name,
            model_version,
            prediction,
            score,
            threshold_used,
            latency_ms,
            input_data,
            output_data,
            prediction_timestamp,
            status_code,
            error_message
        FROM prediction_logs
        WHERE request_id = :request_id
        LIMIT 1
        """
    )

    row = db.execute(sql, {"request_id": request_id}).mappings().first()

    return dict(row) if row else None


# =============================================================================
# Vérités terrain
# =============================================================================

def get_ground_truth_history(
    db: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    client_id: int | None = None,
    request_id: str | None = None,
) -> dict:
    """
    Retourne l'historique des vérités terrain enregistrées.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    limit : int
        Nombre maximal de lignes retournées.
    offset : int
        Décalage pour la pagination.
    client_id : int | None
        Filtre optionnel sur l'identifiant client.
    request_id : str | None
        Filtre optionnel sur l'identifiant de requête.

    Returns
    -------
    dict
        Historique des vérités terrain.
    """
    sql = """
    SELECT
        id,
        request_id,
        client_id,
        true_label,
        label_source,
        observed_at,
        notes
    FROM ground_truth_labels
    WHERE 1 = 1
    """

    params: dict[str, object] = {
        "limit": limit,
        "offset": offset,
    }

    if client_id is not None:
        sql += " AND client_id = :client_id"
        params["client_id"] = client_id

    if request_id is not None:
        sql += " AND request_id = :request_id"
        params["request_id"] = request_id

    sql += """
    ORDER BY observed_at DESC
    LIMIT :limit OFFSET :offset
    """

    rows = db.execute(text(sql), params).mappings().all()

    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "items": [dict(row) for row in rows],
    }


# =============================================================================
# Snapshot des features
# =============================================================================

def get_prediction_features_snapshot(
    db: Session,
    *,
    request_id: str,
) -> dict | None:
    """
    Retourne le snapshot des features enregistré pour une requête.

    Parameters
    ----------
    db : Session
        Session SQLAlchemy active.
    request_id : str
        Identifiant unique de requête.

    Returns
    -------
    dict | None
        Snapshot des features ou None si introuvable.
    """
    sql = text(
        """
        SELECT
            request_id,
            client_id,
            model_name,
            model_version,
            feature_name,
            feature_value,
            feature_type,
            snapshot_timestamp
        FROM prediction_features_snapshot
        WHERE request_id = :request_id
        ORDER BY feature_name
        """
    )

    rows = db.execute(sql, {"request_id": request_id}).mappings().all()

    if not rows:
        return None

    first_row = rows[0]

    return {
        "request_id": request_id,
        "client_id": first_row["client_id"],
        "model_name": first_row["model_name"],
        "model_version": first_row["model_version"],
        "snapshot_timestamp": first_row["snapshot_timestamp"],
        "feature_count": len(rows),
        "features": [
            {
                "feature_name": row["feature_name"],
                "feature_value": row["feature_value"],
                "feature_type": row["feature_type"],
            }
            for row in rows
        ],
    }