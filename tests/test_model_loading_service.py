# tests/test_model_loading_service.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.loader_services import model_loading_service as mls


# =============================================================================
# Faux modèles
# =============================================================================

class DummyModel:
    def __init__(self, n_features_in_=3, feature_names_in_=None, proba=None):
        self.n_features_in_ = n_features_in_
        if feature_names_in_ is not None:
            self.feature_names_in_ = np.array(feature_names_in_, dtype=object)
        self._proba = proba if proba is not None else np.array([[0.2, 0.8]])

    def predict_proba(self, X):
        return self._proba


class DummyModelNoPredictProba:
    def __init__(self):
        self.n_features_in_ = 2


class DummyModelUnknownFeatures:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


class DummyBrokenModel:
    def __init__(self, n_features_in_=2):
        self.n_features_in_ = n_features_in_

    def predict_proba(self, X):
        raise RuntimeError("predict failed")


# =============================================================================
# Fixture reset cache
# =============================================================================

@pytest.fixture(autouse=True)
def reset_cache(monkeypatch: pytest.MonkeyPatch):
    """
    Force le backend sklearn pour ces tests.

    Le service peut charger ONNX ou joblib selon MODEL_BACKEND.
    Ici, on teste volontairement le chemin joblib/sklearn.
    """
    mls.reset_model_cache()
    monkeypatch.setattr(mls, "MODEL_BACKEND", "sklearn")
    yield
    mls.reset_model_cache()


# =============================================================================
# Helpers debug
# =============================================================================

def test_log_separator(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        mls._log_separator("TEST TITLE")

    assert "TEST TITLE" in caplog.text


def test_debug_model_logs_basic_info(caplog: pytest.LogCaptureFixture) -> None:
    model = DummyModel(n_features_in_=4, feature_names_in_=["a", "b", "c", "d"])

    with caplog.at_level("DEBUG"):
        mls.debug_model(model)

    assert "DEBUG MODEL" in caplog.text
    assert "Nb features attendues" in caplog.text
    assert "Features attendues" in caplog.text


def test_debug_threshold_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        mls.debug_threshold(0.7)

    assert "DEBUG THRESHOLD" in caplog.text
    assert "Seuil utilisé" in caplog.text


def test_debug_threshold_logs_warning_for_out_of_range(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("DEBUG"):
        mls.debug_threshold(1.5)

    assert "WARNING : seuil hors [0,1]" in caplog.text


# =============================================================================
# load_model / get_model en backend sklearn
# =============================================================================

def test_load_model_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_text("dummy", encoding="utf-8")

    model = DummyModel()

    monkeypatch.setattr(mls, "MODEL_PATH", model_path)
    monkeypatch.setattr(mls.joblib, "load", lambda path: model)

    result = mls.load_model()

    assert result is model
    assert mls._MODEL is model


def test_load_model_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel()
    mls._MODEL = model

    called = {"count": 0}

    def fake_load(path):
        called["count"] += 1
        return DummyModel()

    monkeypatch.setattr(mls.joblib, "load", fake_load)

    result = mls.load_model()

    assert result is model
    assert called["count"] == 0


def test_load_model_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.joblib"

    monkeypatch.setattr(mls, "MODEL_BACKEND", "sklearn")
    monkeypatch.setattr(mls, "MODEL_PATH", missing_path)

    with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
        mls.load_model()


def test_load_model_joblib_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(mls, "MODEL_BACKEND", "sklearn")
    monkeypatch.setattr(mls, "MODEL_PATH", model_path)

    def fake_load(path):
        raise RuntimeError("joblib crashed")

    monkeypatch.setattr(mls.joblib, "load", fake_load)

    with pytest.raises(RuntimeError, match="joblib crashed"):
        mls.load_model()


def test_get_model_calls_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel()

    monkeypatch.setattr(mls, "load_model", lambda: model)

    result = mls.get_model()

    assert result is model


def test_load_model_invalid_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mls, "MODEL_BACKEND", "bad_backend")

    with pytest.raises(ValueError, match="MODEL_BACKEND invalide"):
        mls.load_model()


# =============================================================================
# load_threshold / get_threshold
# =============================================================================

def test_load_threshold_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text(json.dumps({"threshold": 0.42}), encoding="utf-8")

    monkeypatch.setattr(mls, "THRESHOLD_PATH", threshold_path)

    result = mls.load_threshold()

    assert result == 0.42
    assert mls._THRESHOLD == 0.42


def test_load_threshold_uses_cache() -> None:
    mls._THRESHOLD = 0.77

    result = mls.load_threshold()

    assert result == 0.77


def test_load_threshold_fallback_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    threshold_path = tmp_path / "missing.json"

    monkeypatch.setattr(mls, "THRESHOLD_PATH", threshold_path)

    result = mls.load_threshold()

    assert result == 0.5
    assert mls._THRESHOLD == 0.5


def test_load_threshold_fallback_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text(json.dumps({"wrong": 0.42}), encoding="utf-8")

    monkeypatch.setattr(mls, "THRESHOLD_PATH", threshold_path)

    result = mls.load_threshold()

    assert result == 0.5


def test_load_threshold_fallback_when_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text(json.dumps({"threshold": 1.8}), encoding="utf-8")

    monkeypatch.setattr(mls, "THRESHOLD_PATH", threshold_path)

    result = mls.load_threshold()

    assert result == 0.5


def test_load_threshold_fallback_when_json_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text("{bad json", encoding="utf-8")

    monkeypatch.setattr(mls, "THRESHOLD_PATH", threshold_path)

    result = mls.load_threshold()

    assert result == 0.5


def test_get_threshold_calls_load_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mls, "load_threshold", lambda: 0.33)

    result = mls.get_threshold()

    assert result == 0.33


# =============================================================================
# test_model_prediction
# =============================================================================

def test_model_prediction_logs_warning_if_no_predict_proba(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(mls, "get_model", lambda: DummyModelNoPredictProba())

    with caplog.at_level("WARNING"):
        mls.test_model_prediction()

    assert "predict_proba" in caplog.text


def test_model_prediction_with_feature_names(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = DummyModel(
        n_features_in_=3,
        feature_names_in_=["f1", "f2", "f3"],
        proba=np.array([[0.3, 0.7]]),
    )

    monkeypatch.setattr(mls, "get_model", lambda: model)

    with caplog.at_level("INFO"):
        mls.test_model_prediction()

    assert "smoke test succeeded" in caplog.text


def test_model_prediction_with_n_features_only(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = DummyModel(
        n_features_in_=2,
        feature_names_in_=None,
        proba=np.array([[0.4, 0.6]]),
    )

    if hasattr(model, "feature_names_in_"):
        delattr(model, "feature_names_in_")

    monkeypatch.setattr(mls, "get_model", lambda: model)

    with caplog.at_level("INFO"):
        mls.test_model_prediction()

    assert "smoke test succeeded" in caplog.text


def test_model_prediction_skipped_if_feature_count_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(mls, "get_model", lambda: DummyModelUnknownFeatures())

    with caplog.at_level("WARNING"):
        mls.test_model_prediction()

    assert "feature count unknown" in caplog.text


def test_model_prediction_exception_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(mls, "get_model", lambda: DummyBrokenModel())

    with caplog.at_level("ERROR"):
        mls.test_model_prediction()

    assert "smoke test failed" in caplog.text


# =============================================================================
# reset cache
# =============================================================================

def test_reset_model_cache() -> None:
    mls._MODEL = DummyModel()
    mls._THRESHOLD = 0.8

    mls.reset_model_cache()

    assert mls._MODEL is None
    assert mls._THRESHOLD is None