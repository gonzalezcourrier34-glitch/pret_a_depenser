# tests/test_evidently_service.py

from __future__ import annotations

import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.services.analysis_services import evidently_service as es


# =============================================================================
# Faux objets
# =============================================================================

class FakeMonitoringService:
    def __init__(self, db) -> None:
        self.db = db
        self.logged_rows: list[dict[str, Any]] = []

    def log_drift_metric(self, **kwargs) -> dict[str, Any]:
        self.logged_rows.append(kwargs)
        return kwargs


class FakeReportWithAsDict:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"metrics": []}
        self.run_calls: list[dict[str, Any]] = []

    def run(self, *, current_data, reference_data) -> None:
        self.run_calls.append(
            {
                "current_rows": len(current_data),
                "reference_rows": len(reference_data),
                "current_cols": list(current_data.columns),
                "reference_cols": list(reference_data.columns),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return self.payload


class FakeReportWithDict:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"metrics": []}

    def run(self, *, current_data, reference_data) -> None:
        return None

    def dict(self) -> dict[str, Any]:
        return self.payload


class FakeReportWithJson:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"metrics": []}

    def run(self, *, current_data, reference_data) -> None:
        return None

    def json(self) -> str:
        return json.dumps(self.payload)


class FakeBrokenReport:
    def run(self, *, current_data, reference_data) -> None:
        raise RuntimeError("report crashed")


class FakeDataDriftPreset:
    pass


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fake_db() -> object:
    return object()


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch, fake_db: object) -> es.EvidentlyService:
    monkeypatch.setattr(es, "MonitoringService", FakeMonitoringService)
    return es.EvidentlyService(fake_db)


@pytest.fixture
def reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [10.0, 20.0, 30.0],
            "C": ["x", "y", "z"],
        }
    )


@pytest.fixture
def current_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.5, 2.5],
            "B": [11.0, 21.0],
            "C": ["x", "y"],
            "D": [999, 888],
        }
    )


# =============================================================================
# Helpers internes
# =============================================================================

def test_ensure_dataframe_success(service: es.EvidentlyService, reference_df: pd.DataFrame) -> None:
    result = service._ensure_dataframe(reference_df, "reference_df")

    assert isinstance(result, pd.DataFrame)
    assert result.equals(reference_df)
    assert result is not reference_df


def test_ensure_dataframe_raises_type_error(service: es.EvidentlyService) -> None:
    with pytest.raises(TypeError, match="doit être un DataFrame pandas"):
        service._ensure_dataframe({"A": [1]}, "reference_df")  # type: ignore[arg-type]


def test_ensure_dataframe_raises_value_error(service: es.EvidentlyService) -> None:
    with pytest.raises(ValueError, match="est vide"):
        service._ensure_dataframe(pd.DataFrame(), "reference_df")


def test_prepare_common_columns_without_feature_names(
    service: es.EvidentlyService,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    ref, cur, cols = service._prepare_common_columns(
        reference_df=reference_df,
        current_df=current_df,
        feature_names=None,
    )

    assert cols == ["A", "B", "C"]
    assert list(ref.columns) == ["A", "B", "C"]
    assert list(cur.columns) == ["A", "B", "C"]


def test_prepare_common_columns_with_feature_names(
    service: es.EvidentlyService,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    ref, cur, cols = service._prepare_common_columns(
        reference_df=reference_df,
        current_df=current_df,
        feature_names=["B", "A", "Z"],
    )

    assert cols == ["A", "B"]
    assert list(ref.columns) == ["A", "B"]
    assert list(cur.columns) == ["A", "B"]


def test_prepare_common_columns_raises_if_no_common_columns(service: es.EvidentlyService) -> None:
    ref = pd.DataFrame({"A": [1]})
    cur = pd.DataFrame({"B": [2]})

    with pytest.raises(ValueError, match="Aucune colonne commune exploitable"):
        service._prepare_common_columns(reference_df=ref, current_df=cur)


def test_safe_as_dict_from_as_dict(service: es.EvidentlyService) -> None:
    payload = {"hello": "world"}
    obj = FakeReportWithAsDict(payload)

    result = service._safe_as_dict(obj)

    assert result == payload


def test_safe_as_dict_from_dict_method(service: es.EvidentlyService) -> None:
    payload = {"x": 1}
    obj = FakeReportWithDict(payload)

    result = service._safe_as_dict(obj)

    assert result == payload


def test_safe_as_dict_from_json_method(service: es.EvidentlyService) -> None:
    payload = {"x": 2}
    obj = FakeReportWithJson(payload)

    result = service._safe_as_dict(obj)

    assert result == payload


def test_safe_as_dict_returns_empty_dict_on_failure(service: es.EvidentlyService) -> None:
    class BrokenObject:
        def as_dict(self):
            raise RuntimeError("boom")

        def dict(self):
            raise RuntimeError("boom")

        def json(self):
            raise RuntimeError("boom")

    result = service._safe_as_dict(BrokenObject())

    assert result == {}


def test_find_dataset_drift_block_from_metrics_list(service: es.EvidentlyService) -> None:
    report = {
        "metrics": [
            {"name": "something_else"},
            {
                "dataset_drift": True,
                "number_of_drifted_columns": 3,
                "share_of_drifted_columns": 0.5,
            },
        ]
    }

    result = service._find_dataset_drift_block(report)

    assert result["dataset_drift"] is True
    assert result["number_of_drifted_columns"] == 3


def test_find_dataset_drift_block_from_root_dict(service: es.EvidentlyService) -> None:
    report = {
        "number_of_drifted_columns": 2,
        "share_of_drifted_columns": 0.25,
    }

    result = service._find_dataset_drift_block(report)

    assert result == report


def test_find_dataset_drift_block_returns_empty_dict(service: es.EvidentlyService) -> None:
    result = service._find_dataset_drift_block({"metrics": [{"name": "nope"}]})

    assert result == {}


def test_coerce_int(service: es.EvidentlyService) -> None:
    assert service._coerce_int(5) == 5
    assert service._coerce_int("7") == 7
    assert service._coerce_int(None, default=9) == 9
    assert service._coerce_int("bad", default=3) == 3


def test_coerce_float(service: es.EvidentlyService) -> None:
    assert service._coerce_float(1.5) == 1.5
    assert service._coerce_float("2.75") == 2.75
    assert service._coerce_float(None, default=9.0) == 9.0
    assert service._coerce_float("bad", default=3.5) == 3.5


def test_resolve_monitoring_dir(service: es.EvidentlyService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(es, "MONITORING_DIR", Path("artifacts/monitoring"))

    assert service._resolve_monitoring_dir("custom/monitoring") == Path("custom/monitoring")
    assert service._resolve_monitoring_dir(None) == Path("artifacts/monitoring")


def test_load_reference_dataframe_raw(service: es.EvidentlyService, monkeypatch: pytest.MonkeyPatch, reference_df: pd.DataFrame) -> None:
    monkeypatch.setattr(es, "get_reference_features_raw_df", lambda: reference_df.copy())

    result = service._load_reference_dataframe(reference_kind="raw")

    assert result.equals(reference_df)


def test_load_reference_dataframe_transformed(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "get_reference_features_transformed_df", lambda: reference_df.copy())

    result = service._load_reference_dataframe(reference_kind="transformed")

    assert result.equals(reference_df)


def test_load_current_dataframe_raw_application(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "get_raw_data_cache", lambda: {"application": current_df.copy()})

    result = service._load_current_dataframe(current_kind="raw")

    assert result.equals(current_df)


def test_load_current_dataframe_raw_application_test(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "get_raw_data_cache", lambda: {"application_test": current_df.copy()})

    result = service._load_current_dataframe(current_kind="raw")

    assert result.equals(current_df)


def test_load_current_dataframe_raw_app_key(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "get_raw_data_cache", lambda: {"app": current_df.copy()})

    result = service._load_current_dataframe(current_kind="raw")

    assert result.equals(current_df)


def test_load_current_dataframe_raw_raises_if_missing(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(es, "get_raw_data_cache", lambda: {"wrong": pd.DataFrame({"A": [1]})})

    with pytest.raises(ValueError, match="aucune source brute compatible"):
        service._load_current_dataframe(current_kind="raw")


def test_load_current_dataframe_transformed(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "get_raw_data_cache", lambda: {"application": reference_df.copy()})
    monkeypatch.setattr(
        es,
        "build_transformed_features_from_loaded_data",
        lambda raw_sources, client_ids=None, debug=False: pd.DataFrame({"T1": [1.0, 2.0]}),
    )

    result = service._load_current_dataframe(current_kind="transformed")

    assert list(result.columns) == ["T1"]


def test_load_feature_names(service: es.EvidentlyService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(es, "get_input_feature_names", lambda: ["A", "B"])
    monkeypatch.setattr(es, "get_transformed_feature_names", lambda: ["T1", "T2"])

    assert service._load_feature_names(reference_kind="raw") == ["A", "B"]
    assert service._load_feature_names(reference_kind="transformed") == ["T1", "T2"]


def test_load_feature_names_returns_none_if_empty(service: es.EvidentlyService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(es, "get_input_feature_names", lambda: [])

    assert service._load_feature_names(reference_kind="raw") is None


def test_limit_dataframe_rows_without_limit(service: es.EvidentlyService, reference_df: pd.DataFrame) -> None:
    result = service._limit_dataframe_rows(reference_df, None, name="reference_df")

    assert result.equals(reference_df)


def test_limit_dataframe_rows_with_limit(service: es.EvidentlyService) -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})

    result = service._limit_dataframe_rows(df, 2, name="reference_df")

    assert len(result) == 2
    assert result["A"].tolist() == [4, 5]


def test_limit_dataframe_rows_raises_on_non_positive_limit(service: es.EvidentlyService, reference_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        service._limit_dataframe_rows(reference_df, 0, name="reference_df")


def test_build_response_payload(service: es.EvidentlyService) -> None:
    result = service._build_response_payload(
        success=True,
        message="ok",
        model_name="xgb",
        model_version="v1",
        reference_kind="raw",
        current_kind="raw",
        logged_metrics=1,
        reference_rows=10,
        current_rows=8,
        analyzed_columns=["A", "B"],
        report={"hello": "world"},
    )

    assert result["success"] is True
    assert result["message"] == "ok"
    assert result["model_name"] == "xgb"
    assert result["report"] == {"hello": "world"}


# =============================================================================
# Calcul Evidently
# =============================================================================

def test_run_data_drift_report_import_failure(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "evidently":
            raise ImportError("evidently missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    result = service.run_data_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        feature_names=["A", "B"],
    )

    assert result["success"] is False
    assert "Impossible d'importer Evidently" in result["message"]
    assert result["report"] == {}


def test_run_data_drift_report_success(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    payload = {
        "metrics": [
            {
                "dataset_drift": True,
                "number_of_drifted_columns": 2,
                "share_of_drifted_columns": 0.5,
            }
        ]
    }

    fake_report_instance = FakeReportWithAsDict(payload)

    evidently_module = types.ModuleType("evidently")
    evidently_module.Report = lambda presets: fake_report_instance

    evidently_presets_module = types.ModuleType("evidently.presets")
    evidently_presets_module.DataDriftPreset = FakeDataDriftPreset

    monkeypatch.setitem(sys.modules, "evidently", evidently_module)
    monkeypatch.setitem(sys.modules, "evidently.presets", evidently_presets_module)

    result = service.run_data_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        feature_names=["A", "B"],
    )

    assert result["success"] is True
    assert result["message"] == "Rapport Evidently généré avec succès."
    assert result["reference_rows"] == 3
    assert result["current_rows"] == 2
    assert result["analyzed_columns"] == ["A", "B"]
    assert result["report"] == payload
    assert fake_report_instance.run_calls[0]["reference_cols"] == ["A", "B"]


def test_run_data_drift_report_runtime_failure(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    evidently_module = types.ModuleType("evidently")
    evidently_module.Report = lambda presets: FakeBrokenReport()

    evidently_presets_module = types.ModuleType("evidently.presets")
    evidently_presets_module.DataDriftPreset = FakeDataDriftPreset

    monkeypatch.setitem(sys.modules, "evidently", evidently_module)
    monkeypatch.setitem(sys.modules, "evidently.presets", evidently_presets_module)

    result = service.run_data_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        feature_names=["A", "B"],
    )

    assert result["success"] is False
    assert "Erreur pendant l'exécution d'Evidently" in result["message"]
    assert result["report"] == {}


def test_extract_dataset_drift_summary(service: es.EvidentlyService) -> None:
    report = {
        "metrics": [
            {
                "dataset_drift": True,
                "number_of_drifted_columns": 4,
                "share_of_drifted_columns": 0.8,
            }
        ]
    }

    result = service.extract_dataset_drift_summary(report)

    assert result["drift_detected"] is True
    assert result["number_of_drifted_columns"] == 4
    assert result["share_of_drifted_columns"] == 0.8


def test_extract_dataset_drift_summary_defaults(service: es.EvidentlyService) -> None:
    result = service.extract_dataset_drift_summary({"metrics": [{"name": "nothing"}]})

    assert result["drift_detected"] is False
    assert result["number_of_drifted_columns"] == 0
    assert result["share_of_drifted_columns"] == 0.0


def test_extract_drift_metrics_from_report(service: es.EvidentlyService) -> None:
    report = {
        "metrics": [
            {
                "dataset_drift": True,
                "number_of_drifted_columns": 3,
                "share_of_drifted_columns": 0.6,
            }
        ]
    }

    rows = service.extract_drift_metrics_from_report(
        report=report,
        model_name="xgb",
        model_version="v1",
        reference_window_start="ref_start",
        reference_window_end="ref_end",
        current_window_start="cur_start",
        current_window_end="cur_end",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["feature_name"] == "__dataset__"
    assert row["metric_name"] == "share_of_drifted_columns"
    assert row["metric_value"] == 0.6
    assert row["drift_detected"] is True
    assert row["reference_window_start"] == "ref_start"
    assert row["current_window_end"] == "cur_end"


# =============================================================================
# Orchestration complète
# =============================================================================

def test_run_and_persist_data_drift_analysis_success(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "init_monitoring_reference_cache", lambda path=None: None)
    monkeypatch.setattr(service, "_resolve_monitoring_dir", lambda monitoring_dir: Path("monitoring"))
    monkeypatch.setattr(service, "_load_reference_dataframe", lambda reference_kind: reference_df.copy())
    monkeypatch.setattr(service, "_load_current_dataframe", lambda current_kind: current_df.copy())
    monkeypatch.setattr(service, "_load_feature_names", lambda reference_kind: ["A", "B", "C"])
    monkeypatch.setattr(service, "_limit_dataframe_rows", lambda df, max_rows, name: df.copy())
    monkeypatch.setattr(
        service,
        "run_data_drift_report",
        lambda **kwargs: {
            "success": True,
            "message": "Rapport Evidently généré avec succès.",
            "report": {
                "metrics": [
                    {
                        "dataset_drift": True,
                        "number_of_drifted_columns": 2,
                        "share_of_drifted_columns": 0.5,
                    }
                ]
            },
            "reference_rows": 3,
            "current_rows": 2,
            "analyzed_columns": ["A", "B"],
        },
    )

    result = service.run_and_persist_data_drift_analysis(
        model_name="xgb",
        model_version="v1",
        reference_kind="raw",
        current_kind="raw",
        monitoring_dir=None,
        max_rows=100,
    )

    assert result["success"] is True
    assert result["model_name"] == "xgb"
    assert result["model_version"] == "v1"
    assert result["logged_metrics"] == 1
    assert result["reference_rows"] == 3
    assert result["current_rows"] == 2
    assert result["analyzed_columns"] == ["A", "B"]

    logged = service.monitoring_service.logged_rows
    assert len(logged) == 1
    assert logged[0]["model_name"] == "xgb"
    assert logged[0]["model_version"] == "v1"
    assert logged[0]["feature_name"] == "__dataset__"
    assert logged[0]["metric_name"] == "share_of_drifted_columns"
    assert logged[0]["metric_value"] == 0.5


def test_run_and_persist_data_drift_analysis_failed_result(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "init_monitoring_reference_cache", lambda path=None: None)
    monkeypatch.setattr(service, "_resolve_monitoring_dir", lambda monitoring_dir: Path("monitoring"))
    monkeypatch.setattr(service, "_load_reference_dataframe", lambda reference_kind: reference_df.copy())
    monkeypatch.setattr(service, "_load_current_dataframe", lambda current_kind: current_df.copy())
    monkeypatch.setattr(service, "_load_feature_names", lambda reference_kind: ["A", "B"])
    monkeypatch.setattr(service, "_limit_dataframe_rows", lambda df, max_rows, name: df.copy())
    monkeypatch.setattr(
        service,
        "run_data_drift_report",
        lambda **kwargs: {
            "success": False,
            "message": "échec",
            "report": {},
            "reference_rows": 3,
            "current_rows": 2,
            "analyzed_columns": ["A"],
        },
    )

    result = service.run_and_persist_data_drift_analysis(
        model_name="xgb",
        model_version=None,
        reference_kind="raw",
        current_kind="raw",
    )

    assert result["success"] is False
    assert result["model_version"] == "unknown"
    assert result["logged_metrics"] == 0
    assert service.monitoring_service.logged_rows == []


def test_run_and_persist_data_drift_analysis_invalid_report(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(es, "init_monitoring_reference_cache", lambda path=None: None)
    monkeypatch.setattr(service, "_resolve_monitoring_dir", lambda monitoring_dir: Path("monitoring"))
    monkeypatch.setattr(service, "_load_reference_dataframe", lambda reference_kind: reference_df.copy())
    monkeypatch.setattr(service, "_load_current_dataframe", lambda current_kind: current_df.copy())
    monkeypatch.setattr(service, "_load_feature_names", lambda reference_kind: ["A", "B"])
    monkeypatch.setattr(service, "_limit_dataframe_rows", lambda df, max_rows, name: df.copy())
    monkeypatch.setattr(
        service,
        "run_data_drift_report",
        lambda **kwargs: {
            "success": True,
            "message": "ok",
            "report": "not_a_dict",
            "reference_rows": 3,
            "current_rows": 2,
            "analyzed_columns": ["A"],
        },
    )

    with pytest.raises(ValueError, match="rapport Evidently généré n'est pas exploitable"):
        service.run_and_persist_data_drift_analysis(
            model_name="xgb",
            model_version="v1",
            reference_kind="raw",
            current_kind="raw",
        )


def test_run_and_persist_data_drift_from_dataframes_success(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(service, "_limit_dataframe_rows", lambda df, max_rows, name: df.copy())
    monkeypatch.setattr(
        service,
        "run_data_drift_report",
        lambda **kwargs: {
            "success": True,
            "message": "ok",
            "report": {
                "metrics": [
                    {
                        "dataset_drift": True,
                        "number_of_drifted_columns": 1,
                        "share_of_drifted_columns": 0.33,
                    }
                ]
            },
            "reference_rows": 3,
            "current_rows": 2,
            "analyzed_columns": ["A", "B"],
        },
    )

    result = service.run_and_persist_data_drift_from_dataframes(
        model_name="xgb",
        model_version=None,
        reference_df=reference_df,
        current_df=current_df,
        feature_names=["A", "B"],
        reference_window_start="ref_start",
        reference_window_end="ref_end",
        current_window_start="cur_start",
        current_window_end="cur_end",
        max_rows=50,
    )

    assert result["success"] is True
    assert result["model_version"] == "unknown"
    assert result["logged_metrics"] == 1

    logged = service.monitoring_service.logged_rows
    assert len(logged) == 1
    assert logged[0]["reference_window_start"] == "ref_start"
    assert logged[0]["current_window_end"] == "cur_end"


def test_run_and_persist_data_drift_from_dataframes_failed_result(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(service, "_limit_dataframe_rows", lambda df, max_rows, name: df.copy())
    monkeypatch.setattr(
        service,
        "run_data_drift_report",
        lambda **kwargs: {
            "success": False,
            "message": "report failed",
            "report": {},
            "reference_rows": 3,
            "current_rows": 2,
            "analyzed_columns": ["A"],
        },
    )

    result = service.run_and_persist_data_drift_from_dataframes(
        model_name="xgb",
        model_version="v1",
        reference_df=reference_df,
        current_df=current_df,
    )

    assert result["success"] is False
    assert result["message"] == "report failed"
    assert result["logged_metrics"] == 0
    assert service.monitoring_service.logged_rows == []


def test_run_and_persist_data_drift_from_dataframes_invalid_report(
    service: es.EvidentlyService,
    monkeypatch: pytest.MonkeyPatch,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(service, "_limit_dataframe_rows", lambda df, max_rows, name: df.copy())
    monkeypatch.setattr(
        service,
        "run_data_drift_report",
        lambda **kwargs: {
            "success": True,
            "message": "ok",
            "report": "bad",
            "reference_rows": 3,
            "current_rows": 2,
            "analyzed_columns": ["A"],
        },
    )

    with pytest.raises(ValueError, match="rapport Evidently généré n'est pas exploitable"):
        service.run_and_persist_data_drift_from_dataframes(
            model_name="xgb",
            model_version="v1",
            reference_df=reference_df,
            current_df=current_df,
        )