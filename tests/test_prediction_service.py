# tests/test_prediction_service.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any

import pandas as pd
import pytest

from app.services import prediction_service as ps


# =============================================================================
# Faux objets
# =============================================================================

class DummyModel:
    def __init__(self, proba_matrix):
        self.proba_matrix = proba_matrix

    def predict_proba(self, df: pd.DataFrame):
        return self.proba_matrix


class DummyModelNoProba:
    pass


class FakePredictionLoggingService:
    def __init__(self, db=None):
        self.db = db
        self.logged_success = []
        self.logged_errors = []

    def log_full_prediction_event(self, **kwargs):
        self.logged_success.append(kwargs)

    def log_prediction_error(self, **kwargs):
        self.logged_errors.append(kwargs)


@dataclass
class FakeGroundTruth:
    id: int
    request_id: str | None
    client_id: int | None
    true_label: int
    label_source: str | None
    observed_at: datetime
    notes: str | None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fake_db():
    return object()


@pytest.fixture
def sample_features() -> dict[str, Any]:
    return {
        "AMT_CREDIT": 50000.0,
        "AMT_INCOME_TOTAL": 100000.0,
        "EXT_SOURCE_2": 0.7,
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"AMT_CREDIT": 50000.0, "AMT_INCOME_TOTAL": 100000.0},
            {"AMT_CREDIT": 20000.0, "AMT_INCOME_TOTAL": 50000.0},
        ]
    )


def build_perf_counter(step: float = 0.1):
    """
    Retourne une fonction perf_counter robuste pour les tests.
    """
    values = count(start=1.0, step=step)
    return lambda: next(values)


# =============================================================================
# Validation helpers
# =============================================================================

def test_ensure_dataframe_from_dict_success(sample_features: dict[str, Any]) -> None:
    df = ps._ensure_dataframe_from_dict(sample_features)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_ensure_dataframe_from_dict_raises_type_error() -> None:
    with pytest.raises(TypeError, match="doit être un dictionnaire"):
        ps._ensure_dataframe_from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_ensure_dataframe_from_dict_raises_value_error() -> None:
    with pytest.raises(ValueError, match="est vide"):
        ps._ensure_dataframe_from_dict({})


def test_ensure_dataframe_success(sample_df: pd.DataFrame) -> None:
    result = ps._ensure_dataframe(sample_df)
    assert result.equals(sample_df)


def test_ensure_dataframe_raises_type_error() -> None:
    with pytest.raises(TypeError, match="doit être un DataFrame pandas"):
        ps._ensure_dataframe({"a": 1})  # type: ignore[arg-type]


def test_ensure_dataframe_raises_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="est vide"):
        ps._ensure_dataframe(pd.DataFrame())


def test_ensure_batch_size_valid() -> None:
    ps._ensure_batch_size(1)
    ps._ensure_batch_size(ps.MAX_BATCH_SIZE)


def test_ensure_batch_size_raises_empty() -> None:
    with pytest.raises(ValueError, match="batch est vide"):
        ps._ensure_batch_size(0)


def test_ensure_batch_size_raises_too_large() -> None:
    with pytest.raises(ValueError, match="limite autorisée"):
        ps._ensure_batch_size(ps.MAX_BATCH_SIZE + 1)


def test_ensure_non_string_sequence_valid() -> None:
    ps._ensure_non_string_sequence([1, 2, 3], "values")


def test_ensure_non_string_sequence_raises_string() -> None:
    with pytest.raises(TypeError, match="non une chaîne"):
        ps._ensure_non_string_sequence("abc", "values")  # type: ignore[arg-type]


def test_ensure_non_string_sequence_raises_not_sequence() -> None:
    with pytest.raises(TypeError, match="doit être une séquence Python"):
        ps._ensure_non_string_sequence(123, "values")  # type: ignore[arg-type]


def test_ensure_non_string_sequence_raises_empty() -> None:
    with pytest.raises(ValueError, match="au moins un élément"):
        ps._ensure_non_string_sequence([], "values")


def test_clean_feature_dict_replaces_nan_with_none() -> None:
    result = ps._clean_feature_dict({"a": 1.0, "b": float("nan"), "c": None})
    assert result == {"a": 1.0, "b": None, "c": None}


def test_validate_ground_truth_inputs_ok() -> None:
    ps._validate_ground_truth_inputs(
        request_id="req_1",
        client_id=None,
        true_label=1,
        observed_at=datetime.utcnow(),
    )


def test_validate_ground_truth_inputs_ok_with_observed_at_none() -> None:
    ps._validate_ground_truth_inputs(
        request_id="req_1",
        client_id=None,
        true_label=1,
        observed_at=None,
    )


def test_validate_ground_truth_inputs_requires_identifier() -> None:
    with pytest.raises(ValueError, match="Au moins un identifiant doit être fourni"):
        ps._validate_ground_truth_inputs(
            request_id=None,
            client_id=None,
            true_label=1,
            observed_at=datetime.utcnow(),
        )


def test_validate_ground_truth_inputs_true_label_invalid() -> None:
    with pytest.raises(ValueError, match="doit valoir 0 ou 1"):
        ps._validate_ground_truth_inputs(
            request_id="req_1",
            client_id=None,
            true_label=2,
            observed_at=datetime.utcnow(),
        )


def test_validate_ground_truth_inputs_observed_at_invalid() -> None:
    with pytest.raises(ValueError, match="doit être un datetime valide"):
        ps._validate_ground_truth_inputs(
            request_id="req_1",
            client_id=None,
            true_label=1,
            observed_at="bad",  # type: ignore[arg-type]
        )


# =============================================================================
# Helpers techniques
# =============================================================================

def test_safe_request_id_returns_string() -> None:
    result = ps._safe_request_id()
    assert isinstance(result, str)
    assert result


def test_safe_request_id_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps.uuid, "uuid4", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    result = ps._safe_request_id()
    assert isinstance(result, str)
    assert result.startswith("fallback-")


# =============================================================================
# Helpers modèle
# =============================================================================

def test_predict_scores_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(ps, "get_model", lambda: DummyModel([[0.2, 0.8], [0.7, 0.3]]))

    scores = ps._predict_scores(sample_df)

    assert list(scores) == [0.8, 0.3]


def test_predict_scores_raises_if_no_predict_proba(
    monkeypatch: pytest.MonkeyPatch,
    sample_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(ps, "get_model", lambda: DummyModelNoProba())

    with pytest.raises(AttributeError, match="ne possède pas de méthode `predict_proba`"):
        ps._predict_scores(sample_df)


def test_predict_scores_raises_if_invalid_predict_proba_output(
    monkeypatch: pytest.MonkeyPatch,
    sample_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(ps, "get_model", lambda: DummyModel([[0.8], [0.3]]))

    with pytest.raises(ValueError, match="ne contient pas deux colonnes"):
        ps._predict_scores(sample_df)


def test_predict_raw_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_features: dict[str, Any],
) -> None:
    monkeypatch.setattr(ps, "get_threshold", lambda: 0.5)
    monkeypatch.setattr(ps, "_predict_scores", lambda df: pd.Series([0.9], index=df.index))

    prediction, score, threshold = ps._predict_raw(sample_features)

    assert prediction == 1
    assert score == 0.9
    assert threshold == 0.5


# =============================================================================
# Cache / extraction helpers
# =============================================================================

def test_extract_application_df_from_dataframe(
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ps, "get_data_cache", lambda: sample_df)

    result = ps._extract_application_df()

    assert result.equals(sample_df)


@pytest.mark.parametrize("key", ["application", "application_test", "app"])
def test_extract_application_df_from_dict(
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    monkeypatch.setattr(ps, "get_data_cache", lambda: {key: sample_df})

    result = ps._extract_application_df()

    assert result.equals(sample_df)


def test_extract_application_df_raises_if_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "get_data_cache", lambda: {"wrong": pd.DataFrame()})

    with pytest.raises(ValueError, match="ne contient ni 'application'"):
        ps._extract_application_df()


def test_extract_application_df_raises_if_invalid_cache_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "get_data_cache", lambda: 123)

    with pytest.raises(TypeError, match="doit être un DataFrame ou un dictionnaire"):
        ps._extract_application_df()


def test_extract_existing_client_ids_success() -> None:
    df = pd.DataFrame({"SK_ID_CURR": [1, 2, 2, None, 3]})

    result = ps.extract_existing_client_ids(df)

    assert result == [1, 2, 3]


def test_extract_existing_client_ids_missing_column() -> None:
    with pytest.raises(ValueError, match="colonne `SK_ID_CURR` est absente"):
        ps.extract_existing_client_ids(pd.DataFrame({"x": [1]}))


def test_extract_existing_client_ids_no_usable_ids() -> None:
    df = pd.DataFrame({"SK_ID_CURR": [None, None]})

    with pytest.raises(ValueError, match="Aucun identifiant client exploitable"):
        ps.extract_existing_client_ids(df)


def test_get_random_existing_client_ids_success() -> None:
    df = pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4, 5]})

    result = ps.get_random_existing_client_ids(df, limit=3, random_seed=42)

    assert len(result) == 3
    assert len(set(result)) == 3


def test_get_random_existing_client_ids_not_enough_clients() -> None:
    df = pd.DataFrame({"SK_ID_CURR": [1, 2]})

    with pytest.raises(ValueError, match="Seulement 2 clients disponibles"):
        ps.get_random_existing_client_ids(df, limit=3)


def test_load_client_source_dataframe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"SK_ID_CURR": [1, 2], "x": [10, 20]})
    monkeypatch.setattr(ps, "_extract_application_df", lambda: df)

    result = ps._load_client_source_dataframe()

    assert list(result.columns) == ["SK_ID_CURR"]
    assert len(result) == 2


def test_load_client_source_dataframe_missing_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "_extract_application_df", lambda: pd.DataFrame({"x": [1]}))

    with pytest.raises(ValueError, match="colonne `SK_ID_CURR` est absente"):
        ps._load_client_source_dataframe()


def test_generate_random_value_from_series_numeric_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps.random, "randint", lambda a, b: 7)
    series = pd.Series([1, 5, 9])

    result = ps._generate_random_value_from_series(series)

    assert result == 7


def test_generate_random_value_from_series_numeric_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps.random, "uniform", lambda a, b: 2.5)
    series = pd.Series([1.1, 2.2, 3.3])

    result = ps._generate_random_value_from_series(series)

    assert result == 2.5


def test_generate_random_value_from_series_constant_numeric() -> None:
    series = pd.Series([5, 5, 5])

    result = ps._generate_random_value_from_series(series)

    assert result == 5


def test_generate_random_value_from_series_categorical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps.random, "choice", lambda values: values[0])
    series = pd.Series(["A", "B", "C"])

    result = ps._generate_random_value_from_series(series)

    assert result == "A"


def test_generate_random_value_from_series_empty() -> None:
    series = pd.Series([None, None])

    result = ps._generate_random_value_from_series(series)

    assert result is None


def test_build_random_feature_rows_from_application(monkeypatch: pytest.MonkeyPatch) -> None:
    app_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "AMT_CREDIT": [1000, 2000],
            "NAME_CONTRACT_TYPE": ["Cash", "Revolving"],
        }
    )

    monkeypatch.setattr(ps, "_extract_application_df", lambda: app_df)
    monkeypatch.setattr(
        ps,
        "_generate_random_value_from_series",
        lambda s: "X" if s.dtype == "object" else 123,
    )

    rows = ps._build_random_feature_rows_from_application(limit=2)

    assert len(rows) == 2
    assert rows[0]["SK_ID_CURR"] is None
    assert rows[0]["AMT_CREDIT"] == 123
    assert rows[0]["NAME_CONTRACT_TYPE"] == "X"


def test_sanitize_feature_row() -> None:
    client_id, features = ps._sanitize_feature_row(
        {
            "SK_ID_CURR": "123",
            "A": 1,
            "B": float("nan"),
        }
    )

    assert client_id == 123
    assert features == {"A": 1, "B": None}


def test_get_single_row_feature_dict_success() -> None:
    df = pd.DataFrame([{"A": 1, "B": float("nan")}])

    result = ps._get_single_row_feature_dict(df, context="Test")

    assert result == {"A": 1, "B": None}


def test_get_single_row_feature_dict_raises_if_multiple_rows() -> None:
    df = pd.DataFrame([{"A": 1}, {"A": 2}])

    with pytest.raises(ValueError, match="doit contenir exactement une ligne"):
        ps._get_single_row_feature_dict(df, context="Test")


def test_get_features_for_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ps,
        "get_features_for_client_from_cache",
        lambda client_id, keep_id=False: pd.DataFrame([{"A": 1, "B": float("nan")}]),
    )

    result = ps._get_features_for_client(100001)

    assert result == {"A": 1, "B": None}


# =============================================================================
# Logging helpers
# =============================================================================

def test_log_success_prediction() -> None:
    logger = FakePredictionLoggingService()
    ps._log_success_prediction(
        logger,
        request_id="req_1",
        client_id=1,
        features={"A": 1},
        prediction=1,
        score=0.8,
        threshold_used=0.5,
        latency_ms=12.3,
        source_table="api_request",
    )

    assert len(logger.logged_success) == 1
    assert logger.logged_success[0]["request_id"] == "req_1"
    assert logger.logged_success[0]["prediction"] == 1


def test_log_error_prediction() -> None:
    logger = FakePredictionLoggingService()
    ps._log_error_prediction(
        logger,
        request_id="req_2",
        client_id=2,
        input_data={"A": 1},
        error_message="boom",
        latency_ms=5.0,
    )

    assert len(logger.logged_errors) == 1
    assert logger.logged_errors[0]["error_message"] == "boom"


def test_build_success_item() -> None:
    result = ps._build_success_item(
        request_id="req",
        client_id=1,
        prediction=1,
        score=0.7,
        threshold_used=0.5,
        latency_ms=1.2,
    )
    assert result["status"] == "success"
    assert result["prediction"] == 1


def test_build_error_item() -> None:
    result = ps._build_error_item(
        request_id="req",
        client_id=1,
        error_message="boom",
        latency_ms=1.2,
    )
    assert result["status"] == "error"
    assert result["prediction"] is None


# =============================================================================
# Single prediction public API
# =============================================================================

def test_make_prediction_without_db(
    monkeypatch: pytest.MonkeyPatch,
    sample_features: dict[str, Any],
) -> None:
    monkeypatch.setattr(ps, "_predict_raw", lambda features: (1, 0.88, 0.5))

    result = ps.make_prediction(sample_features, db=None)

    assert result == (1, 0.88, 0.5)


def test_make_prediction_with_db_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_features: dict[str, Any],
    fake_db,
) -> None:
    fake_logger = FakePredictionLoggingService()

    monkeypatch.setattr(ps, "PredictionLoggingService", lambda db: fake_logger)
    monkeypatch.setattr(ps, "_predict_raw", lambda features: (1, 0.91, 0.5))
    monkeypatch.setattr(ps, "_safe_request_id", lambda: "req_uuid")
    monkeypatch.setattr(ps.time, "perf_counter", build_perf_counter())

    result = ps.make_prediction(sample_features, client_id=100001, db=fake_db)

    assert result["request_id"] == "req_uuid"
    assert result["client_id"] == 100001
    assert result["prediction"] == 1
    assert result["status"] == "success"
    assert len(fake_logger.logged_success) == 1


def test_make_prediction_with_db_error_logs_and_raises(
    monkeypatch: pytest.MonkeyPatch,
    sample_features: dict[str, Any],
    fake_db,
) -> None:
    fake_logger = FakePredictionLoggingService()

    def raise_predict(features):
        raise RuntimeError("predict failed")

    monkeypatch.setattr(ps, "PredictionLoggingService", lambda db: fake_logger)
    monkeypatch.setattr(ps, "_predict_raw", raise_predict)
    monkeypatch.setattr(ps, "_safe_request_id", lambda: "req_uuid")
    monkeypatch.setattr(ps.time, "perf_counter", build_perf_counter())

    with pytest.raises(RuntimeError, match="predict failed"):
        ps.make_prediction(sample_features, client_id=123, db=fake_db)

    assert len(fake_logger.logged_errors) == 1
    assert fake_logger.logged_errors[0]["request_id"] == "req_uuid"


def test_make_prediction_from_client_id(monkeypatch: pytest.MonkeyPatch, fake_db) -> None:
    monkeypatch.setattr(ps, "_get_features_for_client", lambda client_id: {"A": 1})
    monkeypatch.setattr(
        ps,
        "make_prediction",
        lambda features, client_id=None, db=None, source_table="x": {
            "prediction": 1,
            "client_id": client_id,
        },
    )

    result = ps.make_prediction_from_client_id(10, db=fake_db)

    assert result["client_id"] == 10


def test_make_prediction_from_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "_predict_raw", lambda features: (0, 0.2, 0.5))

    result = ps.make_prediction_from_dataframe(pd.DataFrame([{"A": 1}]))

    assert result == (0, 0.2, 0.5)


def test_make_prediction_from_dataframe_raises_if_not_one_row() -> None:
    with pytest.raises(ValueError, match="exactement une ligne"):
        ps.make_prediction_from_dataframe(pd.DataFrame([{"A": 1}, {"A": 2}]))


def test_predict_one_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "make_prediction_from_dataframe", lambda df: (1, 0.8, 0.5))

    result = ps.predict_one_row(pd.DataFrame([{"A": 1}]))

    assert result == {
        "prediction": 1,
        "score": 0.8,
        "threshold_used": 0.5,
    }


# =============================================================================
# Batch public API
# =============================================================================

def test_make_batch_prediction(monkeypatch: pytest.MonkeyPatch, sample_df: pd.DataFrame) -> None:
    monkeypatch.setattr(ps, "_predict_raw", lambda features: (1, 0.75, 0.5))

    result = ps.make_batch_prediction(sample_df)

    assert len(result) == 2
    assert "score" in result.columns
    assert "prediction" in result.columns
    assert "threshold_used" in result.columns


def test_run_batch_prediction_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_db,
) -> None:
    fake_logger = FakePredictionLoggingService()

    def fake_predict_raw(features):
        if features.get("A") == "bad":
            raise RuntimeError("bad row")
        return (1, 0.9, 0.5)

    monkeypatch.setattr(ps, "PredictionLoggingService", lambda db: fake_logger)
    monkeypatch.setattr(ps, "_predict_raw", fake_predict_raw)
    monkeypatch.setattr(ps, "_safe_request_id", lambda: "req_id")
    monkeypatch.setattr(ps.time, "perf_counter", build_perf_counter())

    payloads = [
        {"client_id": 1, "features": {"A": 1}},
        {"client_id": 2, "features": {"A": "bad"}},
    ]

    result = ps.run_batch_prediction(payloads, db=fake_db, source_table="api_batch")

    assert result["batch_size"] == 2
    assert result["success_count"] == 1
    assert result["error_count"] == 1
    assert result["items"][0]["status"] == "success"
    assert result["items"][1]["status"] == "error"
    assert len(fake_logger.logged_success) == 1
    assert len(fake_logger.logged_errors) == 1


def test_run_batch_prediction_raises_if_features_not_dict(
    monkeypatch: pytest.MonkeyPatch,
    fake_db,
) -> None:
    fake_logger = FakePredictionLoggingService()

    monkeypatch.setattr(ps, "PredictionLoggingService", lambda db: fake_logger)
    monkeypatch.setattr(ps, "_safe_request_id", lambda: "req_id")
    monkeypatch.setattr(ps.time, "perf_counter", build_perf_counter())

    result = ps.run_batch_prediction(
        [{"client_id": 1, "features": "not_a_dict"}],
        db=fake_db,
        source_table="api_batch",
    )

    assert result["success_count"] == 0
    assert result["error_count"] == 1
    assert result["items"][0]["status"] == "error"


def test_predict_batch_from_client_ids_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_db,
) -> None:
    fake_logger = FakePredictionLoggingService()

    def fake_get_features(client_id):
        if client_id == 2:
            raise RuntimeError("client missing")
        return {"A": client_id}

    monkeypatch.setattr(ps, "_get_features_for_client", fake_get_features)
    monkeypatch.setattr(ps, "_predict_raw", lambda features: (1, 0.8, 0.5))
    monkeypatch.setattr(ps, "PredictionLoggingService", lambda db: fake_logger)
    monkeypatch.setattr(ps, "_safe_request_id", lambda: "req_id")
    monkeypatch.setattr(ps.time, "perf_counter", build_perf_counter())

    result = ps.predict_batch_from_client_ids([1, 2], db=fake_db, source_table="cache")

    assert result["batch_size"] == 2
    assert result["success_count"] == 1
    assert result["error_count"] == 1
    assert result["items"][0]["status"] == "success"
    assert result["items"][1]["status"] == "error"


# =============================================================================
# Simulations
# =============================================================================

def test_run_real_client_simulation(monkeypatch: pytest.MonkeyPatch, fake_db) -> None:
    monkeypatch.setattr(
        ps,
        "_load_client_source_dataframe",
        lambda: pd.DataFrame({"SK_ID_CURR": [1, 2, 3]}),
    )
    monkeypatch.setattr(ps, "get_random_existing_client_ids", lambda **kwargs: [1, 2])
    monkeypatch.setattr(
        ps,
        "predict_batch_from_client_ids",
        lambda client_ids, db, source_table: {
            "batch_size": 2,
            "success_count": 2,
            "error_count": 0,
            "items": [],
        },
    )

    result = ps.run_real_client_simulation(
        limit=2,
        random_seed=42,
        db=fake_db,
        source_table="real_clients",
    )

    assert result["selected_client_ids"] == [1, 2]
    assert result["success_count"] == 2


def test_run_real_client_simulation_raises_if_source_empty(
    monkeypatch: pytest.MonkeyPatch,
    fake_db,
) -> None:
    monkeypatch.setattr(
        ps,
        "_load_client_source_dataframe",
        lambda: pd.DataFrame({"SK_ID_CURR": []}),
    )

    with pytest.raises(ValueError, match="Aucune source client disponible"):
        ps.run_real_client_simulation(
            limit=1,
            random_seed=42,
            db=fake_db,
            source_table="real_clients",
        )


def test_run_random_feature_simulation_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_db,
) -> None:
    fake_logger = FakePredictionLoggingService()

    generated_rows = [
        {"SK_ID_CURR": None, "A": 1},
        {"SK_ID_CURR": None, "A": "bad"},
    ]

    def fake_predict_raw(features):
        if features["A"] == "bad":
            raise RuntimeError("bad generated row")
        return (1, 0.77, 0.5)

    monkeypatch.setattr(ps, "_build_random_feature_rows_from_application", lambda limit: generated_rows)
    monkeypatch.setattr(
        ps,
        "build_features_from_loaded_data",
        lambda application_df, client_ids=None, debug=False, keep_id=True: pd.DataFrame(generated_rows),
    )
    monkeypatch.setattr(ps, "_predict_raw", fake_predict_raw)
    monkeypatch.setattr(ps, "PredictionLoggingService", lambda db: fake_logger)
    monkeypatch.setattr(ps, "_safe_request_id", lambda: "req_id")
    monkeypatch.setattr(ps.time, "perf_counter", build_perf_counter())

    result = ps.run_random_feature_simulation(
        limit=2,
        db=fake_db,
        source_table="random_generated",
    )

    assert result["batch_size"] == 2
    assert result["success_count"] == 1
    assert result["error_count"] == 1
    assert result["items"][0]["status"] == "success"
    assert result["items"][1]["status"] == "error"


def test_run_random_feature_simulation_raises_if_no_generated_rows(
    monkeypatch: pytest.MonkeyPatch,
    fake_db,
) -> None:
    monkeypatch.setattr(ps, "_build_random_feature_rows_from_application", lambda limit: [])

    with pytest.raises(ValueError, match="Impossible de générer des données artificielles"):
        ps.run_random_feature_simulation(
            limit=1,
            db=fake_db,
            source_table="random_generated",
        )


# =============================================================================
# Ground truth
# =============================================================================

def test_create_ground_truth_label(monkeypatch: pytest.MonkeyPatch, fake_db) -> None:
    observed_at = datetime.utcnow()

    monkeypatch.setattr(
        ps.prediction_crud,
        "create_ground_truth_label",
        lambda db, request_id, client_id, true_label, label_source, observed_at, notes: FakeGroundTruth(
            id=1,
            request_id=request_id,
            client_id=client_id,
            true_label=true_label,
            label_source=label_source,
            observed_at=observed_at,
            notes=notes,
        ),
    )

    result = ps.create_ground_truth_label(
        db=fake_db,
        request_id="req_1",
        client_id=100001,
        true_label=1,
        label_source="manual",
        observed_at=observed_at,
        notes="ok",
    )

    assert result["id"] == 1
    assert result["request_id"] == "req_1"
    assert result["client_id"] == 100001
    assert result["true_label"] == 1
    assert result["message"] == "Ground truth enregistré avec succès."


def test_create_ground_truth_label_with_observed_at_none(
    monkeypatch: pytest.MonkeyPatch,
    fake_db,
) -> None:
    monkeypatch.setattr(
        ps.prediction_crud,
        "create_ground_truth_label",
        lambda db, request_id, client_id, true_label, label_source, observed_at, notes: FakeGroundTruth(
            id=2,
            request_id=request_id,
            client_id=client_id,
            true_label=true_label,
            label_source=label_source,
            observed_at=observed_at,
            notes=notes,
        ),
    )

    result = ps.create_ground_truth_label(
        db=fake_db,
        request_id="req_2",
        client_id=None,
        true_label=0,
        label_source="manual",
        observed_at=None,
        notes=None,
    )

    assert result["id"] == 2
    assert result["request_id"] == "req_2"
    assert result["true_label"] == 0
    assert isinstance(result["observed_at"], datetime)


# =============================================================================
# Summary helper
# =============================================================================

def test_summarize_batch_results() -> None:
    results = [
        {"status": "success", "x": 1},
        {"status": "error", "x": 2},
        {"status": "success", "x": 3},
    ]

    result = ps.summarize_batch_results(results)

    assert result["batch_size"] == 3
    assert result["success_count"] == 2
    assert result["error_count"] == 1
    assert len(result["items"]) == 3