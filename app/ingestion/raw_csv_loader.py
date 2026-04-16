"""
Service de prédiction du modèle de scoring crédit.

Ce module encapsule la logique métier liée à l'inférence du modèle.
Il ne recharge pas le modèle à chaque appel : le modèle et le seuil
de décision sont récupérés depuis `app.services.model_loader`, qui les charge
une seule fois au démarrage de l'application.

Fonctionnalités
---------------
- prédiction unitaire à partir d'un dictionnaire de features
- prédiction batch à partir d'un DataFrame pandas
- utilisation d'un seuil métier personnalisé
- validation légère des entrées
- mode debug pour faciliter le diagnostic

Notes
-----
- Le score retourné correspond à la probabilité de la classe positive.
- La décision finale est calculée avec le seuil chargé depuis
  `app.services.model_loader`.
- Ce module est compatible avec une API FastAPI ou un usage batch.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from app.services.model_loader import get_model, get_threshold


# =============================================================================
# Configuration
# =============================================================================

DEBUG_PREDICTION = os.getenv("DEBUG_PREDICTION", "False").lower() == "true"


# =============================================================================
# Fonctions utilitaires debug
# =============================================================================

def _debug_title(title: str) -> None:
    """
    Affiche un titre de section pour les logs de debug.

    Parameters
    ----------
    title : str
        Titre à afficher.
    """
    print("\n" + "=" * 80)
    print(f"[PREDICTION_SERVICE] {title}")
    print("=" * 80)


def _debug_dataframe(
    df: pd.DataFrame,
    name: str,
    *,
    preview_rows: int = 3,
    show_columns: bool = False,
) -> None:
    """
    Affiche un résumé lisible d'un DataFrame pour le debug.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à afficher.
    name : str
        Nom logique du DataFrame.
    preview_rows : int, default=3
        Nombre de lignes affichées.
    show_columns : bool, default=False
        Affiche la liste complète des colonnes si True.
    """
    print(f"\n[DEBUG] DataFrame : {name}")
    print(f"Shape               : {df.shape}")
    print(f"Nb colonnes         : {len(df.columns)}")
    print(f"Total valeurs NA    : {int(df.isna().sum().sum())}")

    if show_columns:
        print("Colonnes :")
        print(list(df.columns))

    na_cols = df.isna().sum()
    na_cols = na_cols[na_cols > 0].sort_values(ascending=False)

    if len(na_cols) > 0:
        print("Top colonnes avec NA :")
        for col, nb in na_cols.head(10).items():
            pct = (nb / len(df) * 100) if len(df) > 0 else 0
            print(f"  - {col}: {nb} ({pct:.2f} %)")

    if len(df) > 0 and preview_rows > 0:
        print("Aperçu :")
        print(df.head(preview_rows).to_string())


def _debug_model_compatibility(model: Any, df: pd.DataFrame) -> None:
    """
    Affiche quelques informations de compatibilité entre le modèle
    et le DataFrame fourni.

    Parameters
    ----------
    model : Any
        Modèle chargé.
    df : pd.DataFrame
        Données transmises au modèle.
    """
    print("\n[DEBUG] Compatibilité modèle / données")
    print(f"Type modèle         : {type(model)}")
    print(f"Nb colonnes entrée  : {len(df.columns)}")

    if hasattr(model, "n_features_in_"):
        print(f"Nb features modèle  : {model.n_features_in_}")

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        input_features = list(df.columns)

        missing = [col for col in model_features if col not in input_features]
        extra = [col for col in input_features if col not in model_features]

        print(f"Colonnes manquantes : {len(missing)}")
        print(f"Colonnes en trop    : {len(extra)}")

        if missing:
            print(f"Exemples manquantes : {missing[:10]}")
        if extra:
            print(f"Exemples en trop    : {extra[:10]}")


# =============================================================================
# Fonctions utilitaires internes
# =============================================================================

def _ensure_dataframe_from_dict(features: dict[str, Any]) -> pd.DataFrame:
    """
    Convertit un dictionnaire de features en DataFrame à une seule ligne.

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire contenant les variables d'entrée du modèle.

    Returns
    -------
    pd.DataFrame
        DataFrame à une ligne, compatible avec sklearn.

    Raises
    ------
    TypeError
        Si `features` n'est pas un dictionnaire.
    ValueError
        Si le dictionnaire est vide.
    """
    if not isinstance(features, dict):
        raise TypeError("`features` doit être un dictionnaire Python.")

    if not features:
        raise ValueError("`features` est vide. Impossible de prédire sans variables d'entrée.")

    return pd.DataFrame([features])


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie qu'un objet est bien un DataFrame pandas exploitable.

    Parameters
    ----------
    df : pd.DataFrame
        Données d'entrée du modèle.

    Returns
    -------
    pd.DataFrame
        Le DataFrame validé.

    Raises
    ------
    TypeError
        Si l'objet n'est pas un DataFrame pandas.
    ValueError
        Si le DataFrame est vide.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'entrée doit être un DataFrame pandas.")

    if df.empty:
        raise ValueError("Le DataFrame fourni est vide. Impossible de lancer une prédiction.")

    return df


def _predict_scores(model: Any, df: pd.DataFrame) -> pd.Series:
    """
    Calcule les probabilités de la classe positive pour un DataFrame.

    Parameters
    ----------
    model : Any
        Modèle de machine learning chargé.
    df : pd.DataFrame
        Données d'entrée.

    Returns
    -------
    pd.Series
        Série contenant les scores de la classe positive.

    Raises
    ------
    AttributeError
        Si le modèle ne possède pas la méthode `predict_proba`.
    RuntimeError
        Si la prédiction échoue.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Le modèle chargé ne possède pas la méthode `predict_proba`."
        )

    try:
        scores = model.predict_proba(df)[:, 1]
        return pd.Series(scores, index=df.index, name="score")

    except Exception as e:
        raise RuntimeError(
            f"Erreur lors du calcul des probabilités du modèle : {type(e).__name__} - {e}"
        ) from e


# =============================================================================
# Fonctions publiques de prédiction
# =============================================================================

def make_prediction(
    features: dict[str, Any],
    *,
    debug: bool = False,
) -> tuple[int, float, float]:
    """
    Réalise une prédiction unitaire à partir d'un dictionnaire de features.

    Cette fonction est adaptée à un usage API, par exemple lorsqu'un client
    envoie un JSON contenant les variables d'entrée du modèle.

    Parameters
    ----------
    features : dict[str, Any]
        Dictionnaire contenant les variables d'entrée du modèle.
    debug : bool, default=False
        Active l'affichage détaillé des étapes de prédiction.

    Returns
    -------
    tuple[int, float, float]
        Un tuple contenant :
        - prediction : classe prédite (0 ou 1)
        - score : probabilité associée à la classe positive
        - threshold : seuil métier utilisé

    Notes
    -----
    - Le modèle est récupéré depuis `app.services.model_loader`.
    - Le seuil de décision est récupéré depuis `app.services.model_loader`.
    - Le score correspond à `predict_proba(...)[0][1]`.
    """
    debug = debug or DEBUG_PREDICTION

    if debug:
        _debug_title("PREDICTION UNITAIRE DEPUIS DICTIONNAIRE")

    model = get_model()
    threshold = get_threshold()

    df = _ensure_dataframe_from_dict(features)

    if debug:
        _debug_dataframe(df, "input_dict_as_dataframe", preview_rows=1, show_columns=True)
        _debug_model_compatibility(model, df)
        print(f"\n[DEBUG] Threshold utilisé : {threshold}")

    score = float(_predict_scores(model, df).iloc[0])
    prediction = int(score >= threshold)

    if debug:
        print(f"[DEBUG] Score calculé      : {score}")
        print(f"[DEBUG] Prédiction finale : {prediction}")

    return prediction, score, threshold


def make_batch_prediction(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Réalise des prédictions sur un lot de données.

    Cette fonction est adaptée à un usage batch, par exemple pour prédire
    sur un fichier CSV complet ou sur des données lues depuis PostgreSQL.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les variables d'entrée du modèle.
    debug : bool, default=False
        Active l'affichage détaillé des étapes de prédiction.

    Returns
    -------
    pd.DataFrame
        Une copie du DataFrame d'entrée enrichie avec :
        - `score` : probabilité de la classe positive
        - `prediction` : classe prédite selon le seuil métier
        - `threshold_used` : seuil métier utilisé

    Notes
    -----
    - Le DataFrame retourné est une copie du DataFrame d'entrée.
    - La prédiction finale n'utilise pas `model.predict()` mais le seuil
      métier personnalisé appliqué sur `predict_proba`.
    """
    debug = debug or DEBUG_PREDICTION

    if debug:
        _debug_title("PREDICTION BATCH")

    model = get_model()
    threshold = get_threshold()

    df = _ensure_dataframe(df)
    result_df = df.copy()

    if debug:
        _debug_dataframe(result_df, "batch_input", preview_rows=3)
        _debug_model_compatibility(model, result_df)
        print(f"\n[DEBUG] Threshold utilisé : {threshold}")

    scores = _predict_scores(model, result_df)
    predictions = (scores >= threshold).astype(int)

    result_df["score"] = scores.values
    result_df["prediction"] = predictions.values
    result_df["threshold_used"] = threshold

    if debug:
        _debug_dataframe(
            result_df[["score", "prediction", "threshold_used"]],
            "batch_output_resume",
            preview_rows=5,
        )

    return result_df


def make_prediction_from_dataframe(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> tuple[int, float, float]:
    """
    Réalise une prédiction unitaire à partir d'un DataFrame contenant une seule ligne.

    Cette fonction peut être utile si les données ont déjà été converties
    en DataFrame plus haut dans le pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant exactement une ligne de données.
    debug : bool, default=False
        Active l'affichage détaillé des étapes de prédiction.

    Returns
    -------
    tuple[int, float, float]
        - prediction : classe prédite (0 ou 1)
        - score : probabilité associée à la classe positive
        - threshold : seuil métier utilisé

    Raises
    ------
    ValueError
        Si le DataFrame contient zéro ligne ou plus d'une ligne.
    """
    debug = debug or DEBUG_PREDICTION

    if debug:
        _debug_title("PREDICTION UNITAIRE DEPUIS DATAFRAME")

    model = get_model()
    threshold = get_threshold()

    df = _ensure_dataframe(df)

    if len(df) != 1:
        raise ValueError(
            "make_prediction_from_dataframe attend un DataFrame avec exactement une ligne."
        )

    if debug:
        _debug_dataframe(df, "single_row_dataframe", preview_rows=1, show_columns=True)
        _debug_model_compatibility(model, df)
        print(f"\n[DEBUG] Threshold utilisé : {threshold}")

    score = float(_predict_scores(model, df).iloc[0])
    prediction = int(score >= threshold)

    if debug:
        print(f"[DEBUG] Score calculé      : {score}")
        print(f"[DEBUG] Prédiction finale : {prediction}")

    return prediction, score, threshold


def predict_one_row(
    df: pd.DataFrame,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Retourne le résultat d'une prédiction unitaire sous forme de dictionnaire.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant exactement une ligne.
    debug : bool, default=False
        Active l'affichage détaillé des étapes de prédiction.

    Returns
    -------
    dict[str, Any]
        Dictionnaire contenant :
        - prediction
        - score
        - threshold_used
    """
    prediction, score, threshold = make_prediction_from_dataframe(df, debug=debug)

    return {
        "prediction": prediction,
        "score": score,
        "threshold_used": threshold,
    }