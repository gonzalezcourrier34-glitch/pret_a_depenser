"""
Conversion du pipeline sklearn + XGBoost vers ONNX.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from xgboost import XGBClassifier

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost


MODEL_PATH = Path("artifacts/model.joblib")
ONNX_PATH = Path("artifacts/model.onnx")


def main() -> None:
    """
    Convertit le pipeline complet en ONNX.
    """

    # -------------------------------------------------------------------------
    # Enregistrer le convertisseur XGBoost pour skl2onnx
    # -------------------------------------------------------------------------
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
    # Charger le pipeline joblib
    # -------------------------------------------------------------------------
    model = joblib.load(MODEL_PATH)

    feature_names = list(model.feature_names_in_)

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

    initial_types = []

    for col in feature_names:
        if col in categorical_features:
            initial_types.append((col, StringTensorType([None, 1])))
        else:
            initial_types.append((col, FloatTensorType([None, 1])))

    # -------------------------------------------------------------------------
    # Conversion ONNX
    # -------------------------------------------------------------------------
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset={
            "": 17,
            "ai.onnx.ml": 3,
        },
        options={
            id(model.steps[-1][1]): {
                "zipmap": False,
            }
        },
    )

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ONNX_PATH.open("wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Modèle ONNX exporté vers : {ONNX_PATH}")
    print(f"Nombre de features déclarées : {len(feature_names)}")


if __name__ == "__main__":
    main()