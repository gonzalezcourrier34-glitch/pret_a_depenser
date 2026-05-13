"""
Conversion du pipeline sklearn + XGBoost vers ONNX.

Description
-----------
Ce script permet de convertir le pipeline de machine learning
entraîné avec scikit-learn et XGBoost vers le format ONNX.

Pourquoi utiliser ONNX ?
------------------------
ONNX (Open Neural Network Exchange) est un format standard
de représentation des modèles de machine learning.

L’objectif est de rendre le modèle :
- plus portable
- plus optimisé pour l’inférence
- indépendant du framework d’entraînement
- plus facilement déployable

Dans ce projet, ONNX est utilisé pour comparer :
- le backend sklearn classique
- le backend ONNX Runtime optimisé

Avantages d’ONNX
----------------
ONNX permet notamment :
- une inférence plus rapide dans certains cas
- une meilleure portabilité
- une exécution optimisée
- une compatibilité multi-langages
- un déploiement simplifié

Particularité du projet
-----------------------
Le pipeline contient :
- du preprocessing sklearn
- un modèle XGBoost
- des variables numériques et catégorielles

Le script doit donc :
- enregistrer un convertisseur XGBoost spécifique
- déclarer correctement les types d’entrée
- exporter le pipeline complet

Résultat final
--------------
Le script génère :

artifacts/model.onnx

Ce fichier sera ensuite chargé par ONNX Runtime
dans l’API FastAPI.
"""

from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================

"""
joblib
-------
Permet de charger le pipeline sklearn sauvegardé.

XGBClassifier
-------------
Type du modèle XGBoost utilisé dans le pipeline.

convert_sklearn
---------------
Fonction principale permettant de convertir un pipeline sklearn vers ONNX.

update_registered_converter
---------------------------
Permet d’ajouter la compatibilité XGBoost dans skl2onnx.

FloatTensorType / StringTensorType
----------------------------------
Définissent les types des features attendues en entrée.

calculate_linear_classifier_output_shapes
-----------------------------------------
Fonction utilisée par skl2onnx pour calculer les shapes de sortie.

convert_xgboost
---------------
Convertisseur spécifique XGBoost → ONNX.
"""

from pathlib import Path

import joblib
from xgboost import XGBClassifier

from skl2onnx import convert_sklearn, update_registered_converter

from skl2onnx.common.data_types import (
    FloatTensorType,
    StringTensorType,
)

from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)

from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost,
)


# =============================================================================
# CHEMINS DES FICHIERS
# =============================================================================

"""
MODEL_PATH
----------
Pipeline sklearn sauvegardé au format joblib.

ONNX_PATH
---------
Chemin du futur modèle ONNX exporté.
"""

MODEL_PATH = Path("artifacts/model.joblib")
ONNX_PATH = Path("artifacts/model.onnx")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main() -> None:
    """
    Convertit le pipeline sklearn complet vers ONNX.

    Étapes principales
    ------------------
    1. Enregistrement du convertisseur XGBoost
    2. Chargement du pipeline sklearn
    3. Détection des features
    4. Déclaration des types d’entrée ONNX
    5. Conversion du pipeline
    6. Sauvegarde du fichier ONNX

    Pourquoi déclarer les types ?
    -----------------------------
    ONNX nécessite de connaître précisément :
    - le nom des features
    - leur type
    - leur shape

    Cela permet au moteur ONNX Runtime
    de préparer correctement l’inférence.

    Pourquoi gérer les variables catégorielles ?
    --------------------------------------------
    Certaines colonnes sont de type texte.

    Exemple :
    - CODE_GENDER
    - NAME_INCOME_TYPE

    Elles doivent être déclarées comme StringTensorType.

    Les autres variables numériques
    sont déclarées comme FloatTensorType.

    Résultat
    --------
    Le modèle exporté peut ensuite être chargé
    directement dans ONNX Runtime.
    """

    # -------------------------------------------------------------------------
    # ENREGISTREMENT DU CONVERTISSEUR XGBOOST
    # -------------------------------------------------------------------------

    """
    skl2onnx ne sait pas convertir automatiquement XGBoost.

    Il faut donc enregistrer manuellement
    un convertisseur compatible.

    options :
    ----------
    nocl :
        contrôle certaines optimisations.

    zipmap :
        contrôle le format de sortie des probabilités.
    """

    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={
            "nocl": [True, False],
            "zipmap": [True, False],
        },
    )

    # -------------------------------------------------------------------------
    # CHARGEMENT DU PIPELINE SKLEARN
    # -------------------------------------------------------------------------

    """
    Le pipeline contient :
    - preprocessing
    - encodage
    - scaling éventuel
    - modèle XGBoost final

    Le pipeline complet est converti en une seule fois.
    """

    model = joblib.load(MODEL_PATH)

    # Liste des features attendues par le modèle
    feature_names = list(model.feature_names_in_)

    # -------------------------------------------------------------------------
    # FEATURES CATÉGORIELLES
    # -------------------------------------------------------------------------

    """
    Ces colonnes sont de type texte.

    Elles doivent être déclarées comme StringTensorType
    dans ONNX.

    Toutes les autres colonnes seront considérées
    comme numériques.
    """

    categorical_features = {
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "ORGANIZATION_TYPE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    }

    # -------------------------------------------------------------------------
    # DÉCLARATION DES TYPES D’ENTRÉE ONNX
    # -------------------------------------------------------------------------

    """
    ONNX exige une déclaration explicite
    du schéma d’entrée.

    Chaque feature doit préciser :
    - son nom
    - son type
    - sa shape

    Shape utilisée :
    ----------------
    [None, 1]

    Cela signifie :
    - nombre de lignes variable
    - une colonne par feature
    """

    initial_types = []

    for col in feature_names:

        # Variables catégorielles → texte
        if col in categorical_features:

            initial_types.append(
                (col, StringTensorType([None, 1]))
            )

        # Variables numériques → float
        else:

            initial_types.append(
                (col, FloatTensorType([None, 1]))
            )

    # -------------------------------------------------------------------------
    # CONVERSION DU PIPELINE
    # -------------------------------------------------------------------------

    """
    Conversion complète du pipeline vers ONNX.

    target_opset :
    ---------------
    Définit les versions ONNX utilisées.

    zipmap=False :
    ----------------
    Permet d’obtenir des sorties plus simples
    et plus rapides à exploiter dans l’API.
    """

    onnx_model = convert_sklearn(
        model,

        # Définition du schéma d’entrée
        initial_types=initial_types,

        # Versions ONNX
        target_opset={
            "": 17,
            "ai.onnx.ml": 3,
        },

        # Options spécifiques au classifieur final
        options={
            id(model.steps[-1][1]): {
                "zipmap": False,
            }
        },
    )

    # -------------------------------------------------------------------------
    # CRÉATION DOSSIER DE SORTIE
    # -------------------------------------------------------------------------

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # SAUVEGARDE DU MODÈLE ONNX
    # -------------------------------------------------------------------------

    """
    Le modèle est sérialisé au format binaire ONNX.

    Ce fichier pourra ensuite être chargé directement
    dans ONNX Runtime.
    """

    with ONNX_PATH.open("wb") as f:
        f.write(onnx_model.SerializeToString())

    # -------------------------------------------------------------------------
    # AFFICHAGE FINAL
    # -------------------------------------------------------------------------

    print(f"Modèle ONNX exporté vers : {ONNX_PATH}")

    print(f"Nombre de features déclarées : {len(feature_names)}")


# =============================================================================
# POINT D’ENTRÉE
# =============================================================================

if __name__ == "__main__":

    """
    Point d’entrée du script.

    Commande typique :
    ------------------
    python convert_to_onnx.py

    Le script :
    ------------
    1. charge le pipeline sklearn
    2. prépare les types ONNX
    3. convertit le pipeline
    4. génère model.onnx
    """

    main()