# tests/test_monitoring_evaluation_service.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pytest

from app.services.analysis_services import monitoring_evaluation_service as mes


# =============================================================================
# Faux outils ORM
# =============================================================================

class FakeColumnCollection:
    def __init__(self, keys_list: list[str]) -> None:
        self._keys = keys_list

    def keys(self):
        return list(self._keys)


class FakeTable:
    def __init__(self, keys_list: list[str]) -> None:
        self.columns = FakeColumnCollection(keys_list)


class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __ge__(self, other):
        return ("ge", self.name, other)

    def __le__(self, other):
        return ("le", self.name, other)

    def is_(self, other):
        return ("is", self.name, other)

    def desc(self):
        return ("desc", self.name)


class FakeAndExpression:
    def __init__(self, conditions):
        self.conditions = list(conditions)


def fake_and_(*conditions):
    return FakeAndExpression(conditions)


# =============================================================================
# Faux modèles
# =============================================================================

class FakeModelRegistry:
    __table__ = FakeTable(
        [
            "id",
            "model_name",
            "model_version",
            "is_active",
            "stage",
        ]
    )
    id = FakeColumn("id")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    is_active = FakeColumn("is_active")
    stage = FakeColumn("stage")


class FakePredictionLog:
    __table__ = FakeTable(
        [
            "request_id",
            "client_id",
            "prediction",
            "score",
            "threshold_used",
            "latency_ms",
            "prediction_timestamp",
            "model_name",
            "model_version",
            "status_code",
        ]
    )
    request_id = FakeColumn("request_id")
    client_id = FakeColumn("client_id")
    prediction = FakeColumn("prediction")
    score = FakeColumn("score")
    threshold_used = FakeColumn("threshold_used")
    latency_ms = FakeColumn("latency_ms")
    prediction_timestamp = FakeColumn("prediction_timestamp")
    model_name = FakeColumn("model_name")
    model_version = FakeColumn("model_version")
    status_code = FakeColumn("status_code")
    id = FakeColumn("id")


class FakeGroundTruthLabel:
    __table__ = FakeTable(
        [
            "request_id",
            "client_id",
            "true_label",
            "created_at",
        ]
    )
    request_id = FakeColumn("request_id")
    client_id = FakeColumn("client_id")
    true_label = FakeColumn("true_label")
    created_at = FakeColumn("created_at")


# =============================================================================
# Faux objets de ligne
# =============================================================================

@dataclass
class FakeRegistryRow:
    id: int = 1
    model_name: str = "xgb"
    model_version: str = "v1"
    is_active: bool = True
    stage: str = "production"


@dataclass
class FakePredictionRow:
    request_id: str | None = None
    client_id: int | None = None
    prediction: int | None = None
    score: float | None = None
    threshold_used: float | None = None
    latency_ms: float | None = None
    prediction_timestamp: datetime | None = None
    model_name: str | None = None
    model_version: str | None = None
    status_code: int | None = None


@dataclass
class FakeGroundTruthRow:
    request_id: str | None = None
    client_id: int | None = None
    true_label: int | None = None
    created_at: datetime | None = None


# =============================================================================
# Faux Query / Session
# =============================================================================

class FakeQuery:
    def __init__(self, *, rows=None, first_result=None):
        self.rows = rows or []
        self.first_result = first_result
        self.filters = []
        self.order_args = []

    def filter(self, *conditions):
        for condition in conditions:
            if isinstance(condition, FakeAndExpression):
                self.filters.extend(condition.conditions)
            else:
                self.filters.append(condition)
        return self

    def order_by(self, *args):
        self.order_args.extend(args)
        return self

    def first(self):
        if self.first_result is not None:
            return self.first_result
        return self.rows[0] if self.rows else None

    def all(self):
        return list(self.rows)


class FakeSession:
    def __init__(self) -> None:
        self.query_map = {}

    def set_query(self, model, query: FakeQuery) -> None:
        self.query_map[model] = query

    def query(self, model):
        return self.query_map[model]


# =============================================================================
# Faux MonitoringService
# =============================================================================

class FakeMonitoringService:
    def __init__(self, db) -> None:
        self.db = db
        self.logged_evaluations = []

    def log_evaluation_metrics(self, **kwargs):
        self.logged_evaluations.append(kwargs)
        return kwargs


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mes, "ModelRegistry", FakeModelRegistry)
    monkeypatch.setattr(mes, "PredictionLog", FakePredictionLog)
    monkeypatch.setattr(mes, "GroundTruthLabel", FakeGroundTruthLabel)
    monkeypatch.setattr(mes, "MonitoringService", FakeMonitoringService)
    monkeypatch.setattr(mes, "and_", fake_and_)


@pytest.fixture
def db() -> FakeSession:
    return FakeSession()


@pytest.fixture
def service(db: FakeSession) -> mes.MonitoringEvaluationService:
    return mes.MonitoringEvaluationService(db)


@pytest.fixture
def prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "request_id": ["r1", "r2", "r3"],
            "client_id": [1, 2, 3],
            "prediction": [0, 1, 1],
            "score": [0.2, 0.8, 0.9],
            "threshold_used": [0.5, 0.5, 0.5],
            "latency_ms": [100.0, 120.0, 140.0],
            "created_at": [
                datetime(2026, 4, 20),
                datetime(2026, 4, 21),
                datetime(2026, 4, 22),
            ],
        }
    )


@pytest.fixture
def ground_truth_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "request_id": ["r1", "r2", "r3"],
            "client_id": [1, 2, 3],
            "y_true": [0, 1, 0],
            "gt_created_at": [
                datetime(2026, 4, 20),
                datetime(2026, 4, 21),
                datetime(2026, 4, 22),
            ],
        }
    )


# =============================================================================
# Helpers internes
# =============================================================================

def test_get_model_columns(service: mes.MonitoringEvaluationService) -> None:
    result = service._get_model_columns(FakePredictionLog)
    assert "request_id" in result
    assert "score" in result


def test_pick_column_found(service: mes.MonitoringEvaluationService) -> None:
    assert service._pick_column(FakePredictionLog, ["foo", "score", "bar"]) == "score"


def test_pick_column_not_found(service: mes.MonitoringEvaluationService) -> None:
    assert service._pick_column(FakePredictionLog, ["foo", "bar"]) is None


def test_safe_get(service: mes.MonitoringEvaluationService) -> None:
    row = FakePredictionRow(score=0.9)
    assert service._safe_get(row, "score") == 0.9
    assert service._safe_get(row, None) is None


def test_coerce_int(service: mes.MonitoringEvaluationService) -> None:
    assert service._coerce_int(5) == 5
    assert service._coerce_int("6") == 6
    assert service._coerce_int(None, default=9) == 9
    assert service._coerce_int("bad", default=7) == 7


def test_coerce_float(service: mes.MonitoringEvaluationService) -> None:
    assert service._coerce_float(1.5) == 1.5
    assert service._coerce_float("2.75") == 2.75
    assert service._coerce_float(None, default=9.5) == 9.5
    assert service._coerce_float("bad", default=7.5) == 7.5


def test_ensure_dataframe_success(service: mes.MonitoringEvaluationService, prediction_df: pd.DataFrame) -> None:
    result = service._ensure_dataframe(prediction_df, "prediction_df")
    assert isinstance(result, pd.DataFrame)
    assert result.equals(prediction_df)
    assert result is not prediction_df


def test_ensure_dataframe_type_error(service: mes.MonitoringEvaluationService) -> None:
    with pytest.raises(TypeError, match="doit être un DataFrame pandas"):
        service._ensure_dataframe({"a": 1}, "prediction_df")  # type: ignore[arg-type]


def test_ensure_dataframe_value_error(service: mes.MonitoringEvaluationService) -> None:
    with pytest.raises(ValueError, match="est vide"):
        service._ensure_dataframe(pd.DataFrame(), "prediction_df")


def test_build_response_payload(service: mes.MonitoringEvaluationService) -> None:
    result = service._build_response_payload(
        success=True,
        message="ok",
        model_name="xgb",
        model_version="v1",
        dataset_name="prod",
        logged_metrics=1,
        sample_size=10,
        matched_rows=10,
        threshold_used=0.5,
        window_start=None,
        window_end=None,
        metrics={"precision": 0.8},
    )

    assert result["success"] is True
    assert result["model_name"] == "xgb"
    assert result["metrics"]["precision"] == 0.8


# =============================================================================
# Résolution identité modèle
# =============================================================================

def test_resolve_model_identity_with_explicit_version(
    service: mes.MonitoringEvaluationService,
    db: FakeSession,
) -> None:
    row = FakeRegistryRow(model_name="xgb", model_version="v2")
    query = FakeQuery(first_result=row)
    db.set_query(FakeModelRegistry, query)

    result = service._resolve_model_identity(model_name="xgb", model_version="v2")

    assert result == ("xgb", "v2")
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v2") in query.filters


def test_resolve_model_identity_with_active_model(
    service: mes.MonitoringEvaluationService,
    db: FakeSession,
) -> None:
    row = FakeRegistryRow(model_name="xgb", model_version="v1", is_active=True)
    query = FakeQuery(first_result=row)
    db.set_query(FakeModelRegistry, query)

    result = service._resolve_model_identity(model_name="xgb", model_version=None)

    assert result == ("xgb", "v1")
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("is", "is_active", True) in query.filters


def test_resolve_model_identity_returns_explicit_when_not_found(
    service: mes.MonitoringEvaluationService,
    db: FakeSession,
) -> None:
    query = FakeQuery(first_result=None)
    db.set_query(FakeModelRegistry, query)

    result = service._resolve_model_identity(model_name="xgb", model_version="v9")

    assert result == ("xgb", "v9")


def test_resolve_model_identity_raises_when_no_active_model(
    service: mes.MonitoringEvaluationService,
    db: FakeSession,
) -> None:
    query = FakeQuery(first_result=None)
    db.set_query(FakeModelRegistry, query)

    with pytest.raises(ValueError, match="Aucun modèle actif trouvé"):
        service._resolve_model_identity(model_name="xgb", model_version=None)


# =============================================================================
# Chargement des données
# =============================================================================

def test_load_prediction_logs(
    service: mes.MonitoringEvaluationService,
    db: FakeSession,
) -> None:
    rows = [
        FakePredictionRow(
            request_id="r1",
            client_id=1,
            prediction=1,
            score=0.8,
            threshold_used=0.5,
            latency_ms=120.0,
            prediction_timestamp=datetime(2026, 4, 20),
            model_name="xgb",
            model_version="v1",
            status_code=200,
        ),
        FakePredictionRow(
            request_id="r2",
            client_id=2,
            prediction=0,
            score=0.2,
            threshold_used=0.5,
            latency_ms=95.0,
            prediction_timestamp=datetime(2026, 4, 21),
            model_name="xgb",
            model_version="v1",
            status_code=200,
        ),
    ]
    query = FakeQuery(rows=rows)
    db.set_query(FakePredictionLog, query)

    result = service._load_prediction_logs(
        model_name="xgb",
        model_version="v1",
    )

    assert len(result) == 2
    assert list(result.columns) == [
        "request_id",
        "client_id",
        "prediction",
        "score",
        "threshold_used",
        "latency_ms",
        "created_at",
    ]
    assert ("eq", "model_name", "xgb") in query.filters
    assert ("eq", "model_version", "v1") in query.filters
    assert ("eq", "status_code", 200) in query.filters


def test_load_ground_truth_labels(
    service: mes.MonitoringEvaluationService,
    db: FakeSession,
) -> None:
    rows = [
        FakeGroundTruthRow(
            request_id="r1",
            client_id=1,
            true_label=1,
            created_at=datetime(2026, 4, 20),
        ),
        FakeGroundTruthRow(
            request_id="r2",
            client_id=2,
            true_label=0,
            created_at=datetime(2026, 4, 21),
        ),
    ]
    query = FakeQuery(rows=rows)
    db.set_query(FakeGroundTruthLabel, query)

    result = service._load_ground_truth_labels()

    assert len(result) == 2
    assert list(result.columns) == ["request_id", "client_id", "y_true", "gt_created_at"]


def test_load_ground_truth_labels_raises_without_target_column(
    service: mes.MonitoringEvaluationService,
    monkeypatch: pytest.MonkeyPatch,
    db: FakeSession,
) -> None:
    def fake_pick_column(model_cls, candidates):
        if model_cls is FakeGroundTruthLabel and candidates == ["target", "true_label", "ground_truth", "label", "actual_target"]:
            return None
        return service._get_model_columns(model_cls) and next((c for c in candidates if c in service._get_model_columns(model_cls)), None)

    monkeypatch.setattr(service, "_pick_column", fake_pick_column)
    db.set_query(FakeGroundTruthLabel, FakeQuery(rows=[]))

    with pytest.raises(ValueError, match="Impossible d'identifier la colonne de vérité terrain"):
        service._load_ground_truth_labels()


# =============================================================================
# Jointure évaluation
# =============================================================================

def test_build_evaluation_dataframe_join_on_request_id(
    service: mes.MonitoringEvaluationService,
    prediction_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> None:
    result = service._build_evaluation_dataframe(
        predictions_df=prediction_df,
        ground_truth_df=ground_truth_df,
    )

    assert len(result) == 3
    assert "y_true" in result.columns


def test_build_evaluation_dataframe_join_on_client_id(
    service: mes.MonitoringEvaluationService,
) -> None:
    pred = pd.DataFrame(
        {
            "client_id": [1, 2],
            "prediction": [0, 1],
            "score": [0.2, 0.8],
        }
    )
    gt = pd.DataFrame(
        {
            "client_id": [1, 2],
            "y_true": [0, 1],
        }
    )

    result = service._build_evaluation_dataframe(
        predictions_df=pred,
        ground_truth_df=gt,
    )

    assert len(result) == 2
    assert "y_true" in result.columns


def test_build_evaluation_dataframe_raises_without_join_key(
    service: mes.MonitoringEvaluationService,
) -> None:
    pred = pd.DataFrame({"prediction": [0, 1]})
    gt = pd.DataFrame({"y_true": [0, 1]})

    with pytest.raises(ValueError, match="Impossible de joindre"):
        service._build_evaluation_dataframe(
            predictions_df=pred,
            ground_truth_df=gt,
        )


def test_build_evaluation_dataframe_raises_if_join_empty(
    service: mes.MonitoringEvaluationService,
) -> None:
    pred = pd.DataFrame({"request_id": ["r1"], "prediction": [0]})
    gt = pd.DataFrame({"request_id": ["r2"], "y_true": [1]})

    with pytest.raises(ValueError, match="jointure entre les prédictions et les vérités terrain est vide"):
        service._build_evaluation_dataframe(
            predictions_df=pred,
            ground_truth_df=gt,
        )


def test_build_evaluation_dataframe_raises_if_all_y_true_missing(
    service: mes.MonitoringEvaluationService,
) -> None:
    pred = pd.DataFrame({"request_id": ["r1"], "prediction": [0]})
    gt = pd.DataFrame({"request_id": ["r1"], "y_true": [None]})

    with pytest.raises(ValueError, match="Aucune ligne exploitable après suppression des y_true manquants"):
        service._build_evaluation_dataframe(
            predictions_df=pred,
            ground_truth_df=gt,
        )


# =============================================================================
# Calcul des métriques
# =============================================================================

def test_compute_evaluation_metrics_with_scores(
    service: mes.MonitoringEvaluationService,
    prediction_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> None:
    eval_df = service._build_evaluation_dataframe(
        predictions_df=prediction_df,
        ground_truth_df=ground_truth_df,
    )

    result = service.compute_evaluation_metrics(
        evaluation_df=eval_df,
        beta=2.0,
        cost_fn=10.0,
        cost_fp=1.0,
    )

    assert result["sample_size"] == 3
    assert result["matched_rows"] == 3
    assert result["threshold_used"] == 0.5
    assert "precision" in result
    assert "recall" in result
    assert "accuracy" in result
    assert "f1" in result
    assert "fbeta" in result
    assert "business_cost" in result
    assert "roc_auc" in result
    assert "pr_auc" in result
    assert "latency_mean_ms" in result


def test_compute_evaluation_metrics_with_prediction_only(
    service: mes.MonitoringEvaluationService,
) -> None:
    eval_df = pd.DataFrame(
        {
            "y_true": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
        }
    )

    result = service.compute_evaluation_metrics(
        evaluation_df=eval_df,
        beta=2.0,
        cost_fn=10.0,
        cost_fp=1.0,
    )

    assert result["sample_size"] == 4
    assert result["threshold_used"] is None
    assert "roc_auc" not in result
    assert "pr_auc" not in result


def test_compute_evaluation_metrics_raises_without_score_or_prediction(
    service: mes.MonitoringEvaluationService,
) -> None:
    eval_df = pd.DataFrame({"y_true": [0, 1]})

    with pytest.raises(ValueError, match="ni `score` ni `prediction` disponible"):
        service.compute_evaluation_metrics(evaluation_df=eval_df)


# =============================================================================
# Orchestration complète
# =============================================================================

def test_run_and_persist_monitoring_evaluation_success(
    service: mes.MonitoringEvaluationService,
    monkeypatch: pytest.MonkeyPatch,
    prediction_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        service,
        "_resolve_model_identity",
        lambda model_name, model_version: ("xgb", "v1"),
    )
    monkeypatch.setattr(
        service,
        "_load_prediction_logs",
        lambda **kwargs: prediction_df.copy(),
    )
    monkeypatch.setattr(
        service,
        "_load_ground_truth_labels",
        lambda **kwargs: ground_truth_df.copy(),
    )

    result = service.run_and_persist_monitoring_evaluation(
        model_name="xgb",
        model_version=None,
        dataset_name="scoring_prod",
        beta=2.0,
        cost_fn=10.0,
        cost_fp=1.0,
    )

    assert result["success"] is True
    assert result["model_name"] == "xgb"
    assert result["model_version"] == "v1"
    assert result["dataset_name"] == "scoring_prod"
    assert result["logged_metrics"] == 1
    assert result["sample_size"] == 3
    assert result["matched_rows"] == 3
    assert isinstance(service.monitoring_service.logged_evaluations, list)
    assert len(service.monitoring_service.logged_evaluations) == 1

    logged = service.monitoring_service.logged_evaluations[0]
    assert logged["model_name"] == "xgb"
    assert logged["model_version"] == "v1"
    assert logged["dataset_name"] == "scoring_prod"


def test_run_and_persist_monitoring_evaluation_failure(
    service: mes.MonitoringEvaluationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service,
        "_resolve_model_identity",
        lambda model_name, model_version: ("xgb", "v1"),
    )

    def broken_load_predictions(**kwargs):
        raise RuntimeError("db read failed")

    monkeypatch.setattr(service, "_load_prediction_logs", broken_load_predictions)

    result = service.run_and_persist_monitoring_evaluation(
        model_name="xgb",
        model_version=None,
    )

    assert result["success"] is False
    assert "Erreur pendant l'évaluation monitoring" in result["message"]
    assert result["logged_metrics"] == 0
    assert result["sample_size"] == 0
    assert result["matched_rows"] == 0
    assert result["metrics"] == {}


def test_run_and_persist_monitoring_evaluation_from_dataframes_success(
    service: mes.MonitoringEvaluationService,
    prediction_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> None:
    result = service.run_and_persist_monitoring_evaluation_from_dataframes(
        model_name="xgb",
        model_version=None,
        prediction_df=prediction_df,
        ground_truth_df=ground_truth_df,
        dataset_name="batch_eval",
        beta=2.0,
        cost_fn=10.0,
        cost_fp=1.0,
    )

    assert result["success"] is True
    assert result["model_name"] == "xgb"
    assert result["model_version"] == "unknown"
    assert result["dataset_name"] == "batch_eval"
    assert result["logged_metrics"] == 1
    assert len(service.monitoring_service.logged_evaluations) == 1


def test_run_and_persist_monitoring_evaluation_from_dataframes_failure(
    service: mes.MonitoringEvaluationService,
) -> None:
    pred = pd.DataFrame({"prediction": [0, 1]})
    gt = pd.DataFrame({"y_true": [0, 1]})

    result = service.run_and_persist_monitoring_evaluation_from_dataframes(
        model_name="xgb",
        model_version="v1",
        prediction_df=pred,
        ground_truth_df=gt,
    )

    assert result["success"] is False
    assert "Erreur pendant l'évaluation monitoring" in result["message"]
    assert result["logged_metrics"] == 0
    assert result["metrics"] == {}