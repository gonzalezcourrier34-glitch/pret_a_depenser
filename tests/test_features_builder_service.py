from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from app.services import features_builder_service as fbs


# =============================================================================
# Faux objets sklearn compatibles
# =============================================================================

class FakeTransformerPipeline(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    def get_feature_names_out(self, input_features=None):
        return np.array(["feat_a", "feat_b"], dtype=object)


class FakeTransformerNoNames(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[1.0, 0.0, 7.0], [1.0, 0.0, 7.0]])


class FakeEstimator(BaseEstimator):
    def fit(self, X, y=None):
        return self


class FakePipeline(Pipeline):
    def __init__(self):
        super().__init__(
            [
                ("prep", FakeTransformerPipeline()),
                ("model", FakeEstimator()),
            ]
        )


class FakePipelineNoNames(Pipeline):
    def __init__(self):
        super().__init__(
            [
                ("prep", FakeTransformerNoNames()),
                ("model", FakeEstimator()),
            ]
        )


class FakeShortPipeline(Pipeline):
    def __init__(self):
        super().__init__([("only_step", FakeEstimator())])


# =============================================================================
# Fixtures utilitaires
# =============================================================================

@pytest.fixture
def sample_application_df() -> pd.DataFrame:
    """
    DataFrame minimal mais riche pour tester le feature builder.
    """
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
            "CODE_GENDER": ["M", "F"],
            "FLAG_OWN_CAR": ["Y", "N"],
            "FLAG_OWN_REALTY": ["Y", "Y"],
            "CNT_CHILDREN": [1, 0],
            "AMT_INCOME_TOTAL": [100000.0, 0.0],
            "AMT_CREDIT": [50000.0, 20000.0],
            "AMT_ANNUITY": [10000.0, 4000.0],
            "AMT_GOODS_PRICE": [45000.0, 0.0],
            "NAME_TYPE_SUITE": ["Family", "Unaccompanied"],
            "NAME_INCOME_TYPE": ["Working", "Pensioner"],
            "NAME_EDUCATION_TYPE": ["Higher education", "Secondary"],
            "NAME_FAMILY_STATUS": ["Married", "Single / not married"],
            "NAME_HOUSING_TYPE": ["House / apartment", "House / apartment"],
            "REGION_POPULATION_RELATIVE": [0.018, 0.025],
            "DAYS_BIRTH": [-3650, -7300],
            "DAYS_EMPLOYED": [365243, -365],
            "DAYS_REGISTRATION": [-1000, -2000],
            "DAYS_ID_PUBLISH": [-500, -1000],
            "OWN_CAR_AGE": [5.0, np.nan],
            "FLAG_MOBIL": [1, 1],
            "FLAG_EMP_PHONE": [1, 0],
            "FLAG_WORK_PHONE": [0, 1],
            "FLAG_PHONE": [1, 0],
            "FLAG_EMAIL": [0, 1],
            "OCCUPATION_TYPE": ["Laborers", "Managers"],
            "CNT_FAM_MEMBERS": [3.0, 1.0],
            "REGION_RATING_CLIENT": [2, 1],
            "REGION_RATING_CLIENT_W_CITY": [2, 1],
            "WEEKDAY_APPR_PROCESS_START": ["MONDAY", "TUESDAY"],
            "HOUR_APPR_PROCESS_START": [10, 12],
            "REG_REGION_NOT_LIVE_REGION": [0, 1],
            "REG_REGION_NOT_WORK_REGION": [0, 0],
            "LIVE_REGION_NOT_WORK_REGION": [0, 1],
            "REG_CITY_NOT_LIVE_CITY": [1, 0],
            "REG_CITY_NOT_WORK_CITY": [0, 1],
            "LIVE_CITY_NOT_WORK_CITY": [0, 0],
            "EXT_SOURCE_1": [0.1, np.nan],
            "EXT_SOURCE_2": [0.2, 0.5],
            "EXT_SOURCE_3": [np.nan, 0.7],
            "APARTMENTS_AVG": [0.5, 0.6],
            "BASEMENTAREA_AVG": [0.1, 0.2],
            "YEARS_BEGINEXPLUATATION_AVG": [0.9, 0.95],
            "ELEVATORS_AVG": [1.0, np.nan],
            "ENTRANCES_AVG": [0.3, 0.4],
            "FLOORSMAX_AVG": [10, 15],
            "LANDAREA_AVG": [0.2, 0.25],
            "LIVINGAREA_AVG": [0.4, 0.45],
            "NONLIVINGAREA_AVG": [0.05, 0.06],
            "OBS_30_CNT_SOCIAL_CIRCLE": [2.0, 0.0],
            "DEF_30_CNT_SOCIAL_CIRCLE": [1.0, 0.0],
            "OBS_60_CNT_SOCIAL_CIRCLE": [4.0, 0.0],
            "DEF_60_CNT_SOCIAL_CIRCLE": [2.0, 0.0],
            "DAYS_LAST_PHONE_CHANGE": [0, -100],
            "AMT_REQ_CREDIT_BUREAU_HOUR": [np.nan, 1.0],
            "AMT_REQ_CREDIT_BUREAU_WEEK": [1.0, np.nan],
            "AMT_REQ_CREDIT_BUREAU_MON": [2.0, 1.0],
            "AMT_REQ_CREDIT_BUREAU_QRT": [3.0, np.nan],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [4.0, 2.0],
            "FLAG_DOCUMENT_2": [1, 0],
            "FLAG_DOCUMENT_3": [1, 1],
            "FLAG_DOCUMENT_4": [0, 0],
            "FLAG_DOCUMENT_5": [0, 1],
            "FLAG_DOCUMENT_6": [1, 0],
            "FLAG_DOCUMENT_7": [0, 0],
            "FLAG_DOCUMENT_8": [0, 0],
            "FLAG_DOCUMENT_9": [0, 0],
            "FLAG_DOCUMENT_10": [0, 0],
            "FLAG_DOCUMENT_11": [0, 0],
            "FLAG_DOCUMENT_12": [0, 0],
            "FLAG_DOCUMENT_13": [0, 0],
            "FLAG_DOCUMENT_14": [0, 0],
            "FLAG_DOCUMENT_15": [0, 0],
            "FLAG_DOCUMENT_16": [0, 0],
            "FLAG_DOCUMENT_17": [0, 0],
            "FLAG_DOCUMENT_18": [0, 0],
            "FLAG_DOCUMENT_19": [0, 0],
            "FLAG_DOCUMENT_20": [0, 0],
            "FLAG_DOCUMENT_21": [0, 0],
            "UNUSED_EXTRA_COLUMN": ["x", "y"],
        }
    )


@pytest.fixture
def patched_model_features(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    Réduit MODEL_FEATURES à un sous-ensemble contrôlé pour rendre les tests
    lisibles tout en couvrant l'alignement.
    """
    features = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "AGE_YEARS",
        "EMPLOYED_YEARS",
        "REGISTRATION_YEARS",
        "ID_PUBLISH_YEARS",
        "LAST_PHONE_CHANGE_YEARS",
        "DAYS_EMPLOYED__isna",
        "OWN_CAR_AGE__isna",
        "EXT_SOURCE_1__isna",
        "EXT_SOURCE_3__isna",
        "DAYS_LAST_PHONE_CHANGE__isna",
        "AMT_REQ_CREDIT_BUREAU_HOUR__isna",
        "AMT_REQ_CREDIT_BUREAU_WEEK__isna",
        "AMT_REQ_CREDIT_BUREAU_MON__isna",
        "AMT_REQ_CREDIT_BUREAU_QRT__isna",
        "AMT_REQ_CREDIT_BUREAU_YEAR__isna",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",
        "CREDIT_GOODS_RATIO",
        "OVER_INDEBTED_40",
        "LOG_INCOME",
        "LOG_CREDIT",
        "LOG_ANNUITY",
        "LOG_GOODS",
        "SOCIAL_DEFAULT_RATIO_30",
        "SOCIAL_DEFAULT_RATIO_60",
        "DOC_COUNT",
        "CONTACT_COUNT",
        "ADDRESS_MISMATCH_COUNT",
        "EXT_SOURCES_MEAN",
        "EXT_SOURCES_MIN",
        "EXT_SOURCES_MAX",
        "EXT_SOURCES_STD",
        "EXT_SOURCES_RANGE",
        "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_2",
        "EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_3",
        "EXT_INT__EXT_SOURCE_2_x_EXT_SOURCE_3",
        "EXT_POW2__EXT_SOURCE_1",
        "EXT_POW2__EXT_SOURCE_2",
        "EXT_POW2__EXT_SOURCE_3",
    ]
    monkeypatch.setattr(fbs, "MODEL_FEATURES", features)
    return features


# =============================================================================
# Tests helpers simples
# =============================================================================

def test_normalize_client_ids_none() -> None:
    assert fbs._normalize_client_ids(None) is None


def test_normalize_client_ids_casts_to_int() -> None:
    assert fbs._normalize_client_ids([1, "2", 3.0]) == [1, 2, 3]


def test_ensure_columns_adds_missing() -> None:
    df = pd.DataFrame({"a": [1]})
    result = fbs._ensure_columns(df, ["a", "b", "c"])

    assert list(result.columns) == ["a", "b", "c"]
    assert pd.isna(result.loc[0, "b"])
    assert pd.isna(result.loc[0, "c"])


def test_coalesce_divide_handles_zero_and_inf() -> None:
    num = pd.Series([10.0, 20.0, 30.0])
    den = pd.Series([2.0, 0.0, np.nan])

    result = fbs._coalesce_divide(num, den)

    assert result.iloc[0] == 5.0
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


# =============================================================================
# Tests extraction / résolution de source
# =============================================================================

def test_extract_application_df_from_dataframe(sample_application_df: pd.DataFrame) -> None:
    result = fbs._extract_application_df(sample_application_df)

    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_application_df)
    assert result is not sample_application_df


@pytest.mark.parametrize("key", ["application", "application_test", "app"])
def test_extract_application_df_from_dict_keys(
    sample_application_df: pd.DataFrame,
    key: str,
) -> None:
    result = fbs._extract_application_df({key: sample_application_df})
    assert result.equals(sample_application_df)


def test_extract_application_df_raises_on_missing_key() -> None:
    with pytest.raises(ValueError, match="ne contient ni 'application'"):
        fbs._extract_application_df({"wrong": pd.DataFrame()})


def test_extract_application_df_raises_on_non_dataframe() -> None:
    with pytest.raises(TypeError, match="n'est pas un DataFrame pandas"):
        fbs._extract_application_df({"application": [1, 2, 3]})


def test_extract_application_df_raises_on_invalid_type() -> None:
    with pytest.raises(TypeError, match="doit être un DataFrame ou un dictionnaire"):
        fbs._extract_application_df("not_a_dataframe")  # type: ignore[arg-type]


def test_resolve_application_source_prefers_application_df(
    sample_application_df: pd.DataFrame,
) -> None:
    other = pd.DataFrame({"x": [1]})

    result = fbs._resolve_application_source(
        raw_sources={"application": other},
        application_df=sample_application_df,
    )

    assert result.equals(sample_application_df)


def test_resolve_application_source_raises_on_bad_application_df() -> None:
    with pytest.raises(TypeError, match="`application_df` doit être un DataFrame pandas"):
        fbs._resolve_application_source(application_df="bad")  # type: ignore[arg-type]


def test_resolve_application_source_raises_when_nothing_given() -> None:
    with pytest.raises(ValueError, match="Aucune source fournie"):
        fbs._resolve_application_source()


# =============================================================================
# Tests chargement CSV
# =============================================================================

def test_load_raw_csv_sources_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "application.csv"
    pd.DataFrame({"SK_ID_CURR": [1], "AMT_CREDIT": [1000]}).to_csv(csv_path, index=False)

    sources = fbs.load_raw_csv_sources(csv_path)

    assert "application" in sources
    assert isinstance(sources["application"], pd.DataFrame)
    assert len(sources["application"]) == 1


def test_load_raw_csv_sources_missing_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
        fbs.load_raw_csv_sources(csv_path)


def test_load_raw_csv_sources_path_is_not_file(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(FileNotFoundError, match="n'est pas un fichier"):
        fbs.load_raw_csv_sources(directory)


# =============================================================================
# Tests enrichissement
# =============================================================================

def test_enrich_features_computes_expected_values(
    sample_application_df: pd.DataFrame,
) -> None:
    base = sample_application_df.copy()
    enriched = fbs._enrich_features(base)

    row0 = enriched.iloc[0]
    row1 = enriched.iloc[1]

    assert row0["AGE_YEARS"] == pytest.approx(3650 / 365.25)
    assert pd.isna(row0["EMPLOYED_YEARS"])
    assert row1["EMPLOYED_YEARS"] == pytest.approx(365 / 365.25)

    assert row0["DAYS_EMPLOYED__isna"] == 1
    assert row0["OWN_CAR_AGE__isna"] == 0
    assert row1["OWN_CAR_AGE__isna"] == 1
    assert row0["EXT_SOURCE_1__isna"] == 0
    assert row0["EXT_SOURCE_3__isna"] == 1
    assert row0["DAYS_LAST_PHONE_CHANGE__isna"] == 1

    assert row0["CREDIT_INCOME_RATIO"] == pytest.approx(0.5)
    assert row0["ANNUITY_INCOME_RATIO"] == pytest.approx(0.1)
    assert row0["ANNUITY_CREDIT_RATIO"] == pytest.approx(0.2)
    assert row0["CREDIT_GOODS_RATIO"] == pytest.approx(50000.0 / 45000.0)

    assert pd.isna(row1["CREDIT_INCOME_RATIO"])
    assert pd.isna(row1["ANNUITY_INCOME_RATIO"])
    assert pd.isna(row1["CREDIT_GOODS_RATIO"])

    assert row0["OVER_INDEBTED_40"] == 0
    assert row0["SOCIAL_DEFAULT_RATIO_30"] == pytest.approx(1.0 / 3.0)
    assert row0["SOCIAL_DEFAULT_RATIO_60"] == pytest.approx(2.0 / 5.0)

    assert row0["DOC_COUNT"] == 3
    assert row1["DOC_COUNT"] == 2
    assert row0["CONTACT_COUNT"] == 3
    assert row1["CONTACT_COUNT"] == 3
    assert row0["ADDRESS_MISMATCH_COUNT"] == 1
    assert row1["ADDRESS_MISMATCH_COUNT"] == 3

    assert row0["EXT_SOURCES_MEAN"] == pytest.approx(0.15)
    assert row0["EXT_SOURCES_MIN"] == pytest.approx(0.1)
    assert row0["EXT_SOURCES_MAX"] == pytest.approx(0.2)
    assert row0["EXT_SOURCES_RANGE"] == pytest.approx(0.1)

    assert row0["EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_2"] == pytest.approx(0.02)
    assert pd.isna(row0["EXT_INT__EXT_SOURCE_1_x_EXT_SOURCE_3"])
    assert pd.isna(row0["EXT_INT__EXT_SOURCE_2_x_EXT_SOURCE_3"])
    assert row0["EXT_POW2__EXT_SOURCE_1"] == pytest.approx(0.01)

    assert row0["LOG_INCOME"] == pytest.approx(np.log(100000.0 + 1))
    assert row1["LOG_INCOME"] == pytest.approx(np.log(1.0))


# =============================================================================
# Tests alignement final
# =============================================================================

def test_align_model_features_adds_missing_and_keeps_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fbs, "MODEL_FEATURES", ["A", "B", "C"])

    df = pd.DataFrame({"C": [3], "A": [1], "SK_ID_CURR": [42]})
    result = fbs._align_model_features(df, keep_id=True)

    assert list(result.columns) == ["SK_ID_CURR", "A", "B", "C"]
    assert pd.isna(result.loc[0, "B"])


def test_align_model_features_without_keep_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fbs, "MODEL_FEATURES", ["A", "B"])

    df = pd.DataFrame({"SK_ID_CURR": [42], "A": [1], "B": [2]})
    result = fbs._align_model_features(df, keep_id=False)

    assert list(result.columns) == ["A", "B"]


# =============================================================================
# Tests build_features_from_loaded_data
# =============================================================================

def test_build_features_from_loaded_data_from_dataframe(
    sample_application_df: pd.DataFrame,
    patched_model_features: list[str],
) -> None:
    result = fbs.build_features_from_loaded_data(
        raw_sources=sample_application_df,
        keep_id=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == patched_model_features
    assert "SK_ID_CURR" not in result.columns


def test_build_features_from_loaded_data_keep_id(
    sample_application_df: pd.DataFrame,
    patched_model_features: list[str],
) -> None:
    result = fbs.build_features_from_loaded_data(
        raw_sources={"application": sample_application_df},
        keep_id=True,
    )

    assert result.columns[0] == "SK_ID_CURR"
    assert len(result.columns) == len(patched_model_features) + 1
    assert result["SK_ID_CURR"].tolist() == [100001, 100002]


def test_build_features_from_loaded_data_filters_client_ids(
    sample_application_df: pd.DataFrame,
    patched_model_features: list[str],
) -> None:
    result = fbs.build_features_from_loaded_data(
        application_df=sample_application_df,
        client_ids=[100002],
        keep_id=True,
    )

    assert len(result) == 1
    assert result["SK_ID_CURR"].tolist() == [100002]


def test_build_features_from_loaded_data_raises_if_missing_sk_id_curr(
    patched_model_features: list[str],
) -> None:
    df = pd.DataFrame({"AMT_CREDIT": [1000.0]})

    with pytest.raises(ValueError, match="Colonne SK_ID_CURR absente"):
        fbs.build_features_from_loaded_data(
            application_df=df,
            client_ids=[1],
        )


def test_build_features_from_loaded_data_raises_if_client_not_found(
    sample_application_df: pd.DataFrame,
    patched_model_features: list[str],
) -> None:
    with pytest.raises(ValueError, match="Aucun client trouvé"):
        fbs.build_features_from_loaded_data(
            application_df=sample_application_df,
            client_ids=[999999],
        )


# =============================================================================
# Tests build_model_ready_features / build_features_for_client
# =============================================================================

def test_build_model_ready_features_uses_loader_cache(
    monkeypatch: pytest.MonkeyPatch,
    sample_application_df: pd.DataFrame,
    patched_model_features: list[str],
) -> None:
    fake_module = types.ModuleType("app.services.loader_services.data_loading_service")

    def fake_get_raw_data_cache() -> dict[str, pd.DataFrame]:
        return {"application": sample_application_df}

    fake_module.get_raw_data_cache = fake_get_raw_data_cache
    sys.modules["app.services.loader_services.data_loading_service"] = fake_module

    result = fbs.build_model_ready_features(keep_id=True)

    assert len(result) == 2
    assert result["SK_ID_CURR"].tolist() == [100001, 100002]


def test_build_features_for_client_success_without_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_model_ready_features(*, client_ids=None, debug=False, keep_id=False):
        assert client_ids == [100001]
        assert keep_id is True
        return pd.DataFrame([{"SK_ID_CURR": 100001, "feature_x": 123, "feature_y": 456}])

    monkeypatch.setattr(fbs, "build_model_ready_features", fake_build_model_ready_features)

    result = fbs.build_features_for_client(100001, keep_id=False)

    assert result == {"feature_x": 123, "feature_y": 456}


def test_build_features_for_client_success_with_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_model_ready_features(*, client_ids=None, debug=False, keep_id=False):
        return pd.DataFrame([{"SK_ID_CURR": 100001, "feature_x": 123}])

    monkeypatch.setattr(fbs, "build_model_ready_features", fake_build_model_ready_features)

    result = fbs.build_features_for_client(100001, keep_id=True)

    assert result == {"SK_ID_CURR": 100001, "feature_x": 123}


def test_build_features_for_client_raises_if_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fbs,
        "build_model_ready_features",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="Aucune feature construite"):
        fbs.build_features_for_client(100001)


def test_build_features_for_client_raises_if_multiple_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fbs,
        "build_model_ready_features",
        lambda **kwargs: pd.DataFrame(
            [{"SK_ID_CURR": 1, "x": 1}, {"SK_ID_CURR": 1, "x": 2}]
        ),
    )

    with pytest.raises(ValueError, match="a retourné 2 lignes"):
        fbs.build_features_for_client(100001)


# =============================================================================
# Tests build_transformed_features_from_loaded_data
# =============================================================================

def test_build_transformed_features_from_loaded_data_success_with_feature_names(
    monkeypatch: pytest.MonkeyPatch,
    sample_application_df: pd.DataFrame,
) -> None:
    raw_features = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        }
    )

    monkeypatch.setattr(
        fbs,
        "build_features_from_loaded_data",
        lambda **kwargs: raw_features.copy(),
    )

    fake_model_module = types.ModuleType(
        "app.services.loader_services.model_loading_service"
    )
    fake_model_module.get_model = lambda: FakePipeline()
    sys.modules["app.services.loader_services.model_loading_service"] = fake_model_module

    result = fbs.build_transformed_features_from_loaded_data(
        application_df=sample_application_df
    )

    assert list(result.columns) == ["feat_a", "feat_b"]
    assert result.shape == (2, 2)


def test_build_transformed_features_from_loaded_data_fallback_feature_names(
    monkeypatch: pytest.MonkeyPatch,
    sample_application_df: pd.DataFrame,
) -> None:
    raw_features = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        }
    )

    monkeypatch.setattr(
        fbs,
        "build_features_from_loaded_data",
        lambda **kwargs: raw_features.copy(),
    )

    fake_model_module = types.ModuleType(
        "app.services.loader_services.model_loading_service"
    )
    fake_model_module.get_model = lambda: FakePipelineNoNames()
    sys.modules["app.services.loader_services.model_loading_service"] = fake_model_module

    result = fbs.build_transformed_features_from_loaded_data(
        application_df=sample_application_df
    )

    assert list(result.columns) == ["feature_0", "feature_1", "feature_2"]
    assert result.shape == (2, 3)


def test_build_transformed_features_from_loaded_data_raises_if_model_not_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    sample_application_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        fbs,
        "build_features_from_loaded_data",
        lambda **kwargs: pd.DataFrame({"f1": [1.0]}),
    )

    fake_model_module = types.ModuleType(
        "app.services.loader_services.model_loading_service"
    )
    fake_model_module.get_model = lambda: "not_a_pipeline"
    sys.modules["app.services.loader_services.model_loading_service"] = fake_model_module

    with pytest.raises(TypeError, match="n'est pas un Pipeline sklearn"):
        fbs.build_transformed_features_from_loaded_data(
            application_df=sample_application_df
        )


def test_build_transformed_features_from_loaded_data_raises_if_pipeline_too_short(
    monkeypatch: pytest.MonkeyPatch,
    sample_application_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        fbs,
        "build_features_from_loaded_data",
        lambda **kwargs: pd.DataFrame({"f1": [1.0]}),
    )

    fake_model_module = types.ModuleType(
        "app.services.loader_services.model_loading_service"
    )
    fake_model_module.get_model = lambda: FakeShortPipeline()
    sys.modules["app.services.loader_services.model_loading_service"] = fake_model_module

    with pytest.raises(ValueError, match="ne contient pas assez d'étapes"):
        fbs.build_transformed_features_from_loaded_data(
            application_df=sample_application_df
        )