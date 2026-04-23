# tests/test_data_loading_service.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.services.loader_services import data_loading_service as dls


# =============================================================================
# Helpers / fixtures
# =============================================================================

@pytest.fixture
def sample_app_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "AMT_CREDIT": [50000.0, 20000.0],
            "EXT_SOURCE_2": [0.7, 0.2],
        }
    )


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        }
    )


@pytest.fixture(autouse=True)
def reset_all_caches():
    dls.reset_data_cache()
    dls.reset_monitoring_reference_cache()
    yield
    dls.reset_data_cache()
    dls.reset_monitoring_reference_cache()


# =============================================================================
# Helpers internes
# =============================================================================

def test_validate_dataframe_ok(sample_app_df: pd.DataFrame) -> None:
    dls._validate_dataframe(sample_app_df, "test_df")


def test_validate_dataframe_none() -> None:
    with pytest.raises(RuntimeError, match="est None"):
        dls._validate_dataframe(None, "test_df")


def test_validate_dataframe_not_dataframe() -> None:
    with pytest.raises(RuntimeError, match="n'est pas un DataFrame pandas"):
        dls._validate_dataframe("bad", "test_df")  # type: ignore[arg-type]


def test_validate_dataframe_empty() -> None:
    with pytest.raises(RuntimeError, match="est vide"):
        dls._validate_dataframe(pd.DataFrame(), "test_df")


def test_ensure_sk_id_curr_ok(sample_app_df: pd.DataFrame) -> None:
    dls._ensure_sk_id_curr(sample_app_df)


def test_ensure_sk_id_curr_missing() -> None:
    with pytest.raises(ValueError, match="SK_ID_CURR"):
        dls._ensure_sk_id_curr(pd.DataFrame({"x": [1]}))


def test_resolve_application_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dls, "APPLICATION_CSV", "data/app.csv")
    assert dls._resolve_application_csv() == Path("data/app.csv")
    assert dls._resolve_application_csv("x.csv") == Path("x.csv")


def test_resolve_monitoring_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dls, "MONITORING_DIR", "artifacts/monitoring")
    assert dls._resolve_monitoring_dir() == Path("artifacts/monitoring")
    assert dls._resolve_monitoring_dir("tmp/monitoring") == Path("tmp/monitoring")


# =============================================================================
# Chargement fichiers utilitaires
# =============================================================================

def test_load_parquet_file_success(tmp_path: Path, sample_app_df: pd.DataFrame) -> None:
    path = tmp_path / "ref.parquet"
    sample_app_df.to_parquet(path)

    result = dls._load_parquet_file(path, "reference_features_raw")

    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_app_df)


def test_load_parquet_file_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
        dls._load_parquet_file(path, "reference_features_raw")


def test_load_parquet_file_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(path)

    with pytest.raises(RuntimeError, match="est vide"):
        dls._load_parquet_file(path, "reference_features_raw")


def test_load_json_file_success(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    payload = {"a": 1, "b": ["x", "y"]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = dls._load_json_file(path, "reference_metadata")

    assert result == payload


def test_load_json_file_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
        dls._load_json_file(path, "reference_metadata")


# =============================================================================
# CSV source principal
# =============================================================================

def test_load_all_csv_success(tmp_path: Path, sample_app_df: pd.DataFrame) -> None:
    csv_path = tmp_path / "application.csv"
    sample_app_df.to_csv(csv_path, index=False)

    result = dls.load_all_csv(csv_path)

    assert list(result.keys()) == ["application"]
    assert isinstance(result["application"], pd.DataFrame)
    assert len(result["application"]) == 2


def test_load_all_csv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="introuvable"):
        dls.load_all_csv(tmp_path / "missing.csv")


def test_load_all_csv_not_a_file(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()

    with pytest.raises(FileNotFoundError, match="n'est pas un fichier CSV"):
        dls.load_all_csv(folder)


def test_load_all_csv_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["SK_ID_CURR", "AMT_CREDIT"]).to_csv(csv_path, index=False)

    with pytest.raises(RuntimeError, match="application_source"):
        dls.load_all_csv(csv_path)

# =============================================================================
# RAW_DATA_CACHE
# =============================================================================

def test_init_raw_data_cache_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_app_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(dls, "load_all_csv", lambda csv_path=None: {"application": sample_app_df})

    dls.init_raw_data_cache("dummy.csv")

    assert "application" in dls.RAW_DATA_CACHE
    assert dls.RAW_DATA_CACHE["application"].equals(sample_app_df)


def test_init_raw_data_cache_skip_when_already_initialized(
    sample_app_df: pd.DataFrame,
) -> None:
    dls.RAW_DATA_CACHE = {"application": sample_app_df.copy()}

    dls.init_raw_data_cache("dummy.csv")

    assert "application" in dls.RAW_DATA_CACHE
    assert dls.RAW_DATA_CACHE["application"].equals(sample_app_df)


def test_init_raw_data_cache_raises_if_empty_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dls, "load_all_csv", lambda csv_path=None: {})

    with pytest.raises(RuntimeError, match="Aucune source brute"):
        dls.init_raw_data_cache("dummy.csv")


def test_get_raw_data_cache_success(sample_app_df: pd.DataFrame) -> None:
    dls.RAW_DATA_CACHE = {"application": sample_app_df.copy()}

    result = dls.get_raw_data_cache()

    assert result["application"].equals(sample_app_df)


def test_get_raw_data_cache_not_initialized() -> None:
    with pytest.raises(RuntimeError, match="RAW_DATA_CACHE non initialisé"):
        dls.get_raw_data_cache()


def test_get_data_cache_alias(sample_app_df: pd.DataFrame) -> None:
    dls.RAW_DATA_CACHE = {"application": sample_app_df.copy()}

    result = dls.get_data_cache()

    assert result["application"].equals(sample_app_df)


# =============================================================================
# FEATURES_READY_CACHE
# =============================================================================

def test_init_features_ready_cache_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_app_df: pd.DataFrame,
    sample_features_df: pd.DataFrame,
) -> None:
    dls.RAW_DATA_CACHE = {"application": sample_app_df.copy()}

    monkeypatch.setattr(
        dls,
        "build_features_from_loaded_data",
        lambda raw_sources, client_ids=None, debug=False, keep_id=True: sample_features_df.copy(),
    )

    dls.init_features_ready_cache(keep_id=True, debug=False)

    assert dls.FEATURES_READY_CACHE is not None
    assert dls.FEATURES_READY_CACHE.equals(sample_features_df)


def test_init_features_ready_cache_skip_when_already_initialized(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    dls.init_features_ready_cache()

    assert dls.FEATURES_READY_CACHE.equals(sample_features_df)


def test_init_features_ready_cache_raises_if_builder_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
    sample_app_df: pd.DataFrame,
) -> None:
    dls.RAW_DATA_CACHE = {"application": sample_app_df.copy()}

    monkeypatch.setattr(
        dls,
        "build_features_from_loaded_data",
        lambda raw_sources, client_ids=None, debug=False, keep_id=True: pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="FEATURES_READY_CACHE"):
        dls.init_features_ready_cache()


def test_init_features_ready_cache_raises_if_keep_id_without_sk_id(
    monkeypatch: pytest.MonkeyPatch,
    sample_app_df: pd.DataFrame,
) -> None:
    dls.RAW_DATA_CACHE = {"application": sample_app_df.copy()}

    monkeypatch.setattr(
        dls,
        "build_features_from_loaded_data",
        lambda raw_sources, client_ids=None, debug=False, keep_id=True: pd.DataFrame({"f1": [1]}),
    )

    with pytest.raises(ValueError, match="SK_ID_CURR"):
        dls.init_features_ready_cache(keep_id=True)


def test_get_features_ready_cache_success(sample_features_df: pd.DataFrame) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    result = dls.get_features_ready_cache()

    assert result.equals(sample_features_df)


def test_get_features_ready_cache_not_initialized() -> None:
    with pytest.raises(RuntimeError, match="FEATURES_READY_CACHE non initialisé"):
        dls.get_features_ready_cache()


# =============================================================================
# Accès client
# =============================================================================

def test_get_features_for_client_from_cache_success_keep_id(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    result = dls.get_features_for_client_from_cache(100001, keep_id=True)

    assert len(result) == 1
    assert "SK_ID_CURR" in result.columns
    assert result.iloc[0]["SK_ID_CURR"] == 100001


def test_get_features_for_client_from_cache_success_without_id(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    result = dls.get_features_for_client_from_cache(100001, keep_id=False)

    assert len(result) == 1
    assert "SK_ID_CURR" not in result.columns
    assert list(result.columns) == ["f1", "f2"]


def test_get_features_for_client_from_cache_not_found(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    with pytest.raises(ValueError, match="introuvable"):
        dls.get_features_for_client_from_cache(999999)


def test_get_features_for_client_from_cache_duplicate() -> None:
    dls.FEATURES_READY_CACHE = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100001],
            "f1": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Doublon détecté"):
        dls.get_features_for_client_from_cache(100001)


def test_get_features_for_clients_from_cache_success_keep_id(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    result = dls.get_features_for_clients_from_cache([100001, 100002], keep_id=True)

    assert len(result) == 2
    assert "SK_ID_CURR" in result.columns


def test_get_features_for_clients_from_cache_success_without_id(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    result = dls.get_features_for_clients_from_cache([100001], keep_id=False)

    assert len(result) == 1
    assert "SK_ID_CURR" not in result.columns


def test_get_features_for_clients_from_cache_none_found(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    with pytest.raises(ValueError, match="Aucun client trouvé"):
        dls.get_features_for_clients_from_cache([999999])


def test_get_features_for_clients_from_cache_strict_missing_ids(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    with pytest.raises(ValueError, match="Clients absents du cache"):
        dls.get_features_for_clients_from_cache([100001, 999999], strict=True)


def test_get_features_for_clients_from_cache_non_strict_missing_ids(
    sample_features_df: pd.DataFrame,
) -> None:
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    result = dls.get_features_for_clients_from_cache([100001, 999999], strict=False)

    assert len(result) == 1
    assert result.iloc[0]["SK_ID_CURR"] == 100001


# =============================================================================
# Initialisation globale
# =============================================================================

def test_init_full_data_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"raw": False, "features": False}

    def fake_init_raw(csv_path=None):
        calls["raw"] = True

    def fake_init_features(*, keep_id=True, debug=False):
        calls["features"] = True
        assert keep_id is True
        assert debug is True

    monkeypatch.setattr(dls, "init_raw_data_cache", fake_init_raw)
    monkeypatch.setattr(dls, "init_features_ready_cache", fake_init_features)

    dls.init_full_data_cache("file.csv", debug=True)

    assert calls["raw"] is True
    assert calls["features"] is True


def test_reset_data_cache(sample_app_df: pd.DataFrame, sample_features_df: pd.DataFrame) -> None:
    dls.RAW_DATA_CACHE = {"application": sample_app_df.copy()}
    dls.FEATURES_READY_CACHE = sample_features_df.copy()

    dls.reset_data_cache()

    assert dls.RAW_DATA_CACHE == {}
    assert dls.FEATURES_READY_CACHE is None


# =============================================================================
# Monitoring reference cache
# =============================================================================

def test_init_monitoring_reference_cache_success(tmp_path: Path, sample_app_df: pd.DataFrame) -> None:
    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    raw_df = sample_app_df.copy()
    transformed_df = pd.DataFrame({"t1": [0.1, 0.2], "t2": [1.0, 2.0]})
    target_df = pd.DataFrame({"TARGET": [0, 1]})

    raw_df.to_parquet(monitoring_dir / "reference_features_raw.parquet")
    transformed_df.to_parquet(monitoring_dir / "reference_features_transformed.parquet")
    target_df.to_parquet(monitoring_dir / "reference_target.parquet")

    (monitoring_dir / "input_feature_names.json").write_text(
        json.dumps(["A", "B"]), encoding="utf-8"
    )
    (monitoring_dir / "transformed_feature_names.json").write_text(
        json.dumps(["T1", "T2"]), encoding="utf-8"
    )
    (monitoring_dir / "reference_metadata.json").write_text(
        json.dumps({"version": "v1"}), encoding="utf-8"
    )
    (monitoring_dir / "reference_stats_raw.json").write_text(
        json.dumps({"rows": 2}), encoding="utf-8"
    )
    (monitoring_dir / "reference_stats_transformed.json").write_text(
        json.dumps({"rows": 2}), encoding="utf-8"
    )

    dls.init_monitoring_reference_cache(monitoring_dir)

    cache = dls.MONITORING_REFERENCE_CACHE
    assert "reference_features_raw" in cache
    assert "reference_features_transformed" in cache
    assert "reference_target" in cache
    assert "input_feature_names" in cache
    assert "transformed_feature_names" in cache
    assert "reference_metadata" in cache
    assert "reference_stats_raw" in cache
    assert "reference_stats_transformed" in cache

    assert isinstance(cache["reference_features_raw"], pd.DataFrame)
    assert isinstance(cache["reference_features_transformed"], pd.DataFrame)
    assert cache["input_feature_names"] == ["A", "B"]


def test_init_monitoring_reference_cache_skip_when_already_initialized(
    sample_app_df: pd.DataFrame,
) -> None:
    dls.MONITORING_REFERENCE_CACHE = {"reference_features_raw": sample_app_df.copy()}

    dls.init_monitoring_reference_cache("whatever")

    assert "reference_features_raw" in dls.MONITORING_REFERENCE_CACHE


def test_init_monitoring_reference_cache_missing_dir(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing_monitoring"

    with pytest.raises(FileNotFoundError, match="Dossier monitoring introuvable"):
        dls.init_monitoring_reference_cache(missing_dir)


def test_get_monitoring_reference_cache_success(sample_app_df: pd.DataFrame) -> None:
    dls.MONITORING_REFERENCE_CACHE = {"reference_features_raw": sample_app_df.copy()}

    result = dls.get_monitoring_reference_cache()

    assert "reference_features_raw" in result


def test_get_monitoring_reference_cache_not_initialized() -> None:
    with pytest.raises(RuntimeError, match="MONITORING_REFERENCE_CACHE non initialisé"):
        dls.get_monitoring_reference_cache()


def test_get_reference_features_raw_df(sample_app_df: pd.DataFrame) -> None:
    dls.MONITORING_REFERENCE_CACHE = {"reference_features_raw": sample_app_df.copy()}

    result = dls.get_reference_features_raw_df()

    assert result.equals(sample_app_df)
    assert result is not dls.MONITORING_REFERENCE_CACHE["reference_features_raw"]


def test_get_reference_features_transformed_df() -> None:
    df = pd.DataFrame({"t1": [1, 2]})
    dls.MONITORING_REFERENCE_CACHE = {"reference_features_transformed": df.copy()}

    result = dls.get_reference_features_transformed_df()

    assert result.equals(df)
    assert result is not dls.MONITORING_REFERENCE_CACHE["reference_features_transformed"]


def test_get_reference_target_df_present() -> None:
    df = pd.DataFrame({"TARGET": [0, 1]})
    dls.MONITORING_REFERENCE_CACHE = {"reference_target": df.copy()}

    result = dls.get_reference_target_df()

    assert result is not None
    assert result.equals(df)


def test_get_reference_target_df_missing() -> None:
    dls.MONITORING_REFERENCE_CACHE = {}

    with pytest.raises(RuntimeError, match="MONITORING_REFERENCE_CACHE non initialisé"):
        dls.get_reference_target_df()


def test_get_reference_target_df_none_value(sample_app_df: pd.DataFrame) -> None:
    dls.MONITORING_REFERENCE_CACHE = {
        "reference_features_raw": sample_app_df.copy(),
        "reference_target": None,
    }

    result = dls.get_reference_target_df()

    assert result is None


def test_get_reference_target_df_not_dataframe(sample_app_df: pd.DataFrame) -> None:
    dls.MONITORING_REFERENCE_CACHE = {
        "reference_features_raw": sample_app_df.copy(),
        "reference_target": {"bad": 1},
    }

    result = dls.get_reference_target_df()

    assert result is None


def test_get_input_feature_names_list() -> None:
    dls.MONITORING_REFERENCE_CACHE = {"input_feature_names": ["A", "B", 3]}

    result = dls.get_input_feature_names()

    assert result == ["A", "B", "3"]


def test_get_input_feature_names_non_list(sample_app_df: pd.DataFrame) -> None:
    dls.MONITORING_REFERENCE_CACHE = {
        "reference_features_raw": sample_app_df.copy(),
        "input_feature_names": {"bad": 1},
    }

    result = dls.get_input_feature_names()

    assert result == []


def test_get_transformed_feature_names_list() -> None:
    dls.MONITORING_REFERENCE_CACHE = {"transformed_feature_names": ["T1", "T2", 7]}

    result = dls.get_transformed_feature_names()

    assert result == ["T1", "T2", "7"]


def test_get_transformed_feature_names_non_list(sample_app_df: pd.DataFrame) -> None:
    dls.MONITORING_REFERENCE_CACHE = {
        "reference_features_raw": sample_app_df.copy(),
        "transformed_feature_names": {"bad": 1},
    }

    result = dls.get_transformed_feature_names()

    assert result == []


def test_reset_monitoring_reference_cache(sample_app_df: pd.DataFrame) -> None:
    dls.MONITORING_REFERENCE_CACHE = {"reference_features_raw": sample_app_df.copy()}

    dls.reset_monitoring_reference_cache()

    assert dls.MONITORING_REFERENCE_CACHE == {}