from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# =============================================================================
# 1. Ajout du projet au PYTHONPATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 2. Variables d'environnement SAFE pour les tests
# =============================================================================

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch: pytest.MonkeyPatch):
    """
    Définit un environnement stable pour les tests.
    Évite de dépendre de ton .env réel.
    """
    monkeypatch.setenv("API_KEY", "test_key")
    monkeypatch.setenv("DEBUG", "True")

    monkeypatch.setenv(
        "DATABASE_URL",
        "sqlite:///./test.db",  # jamais postgres en test
    )

    monkeypatch.setenv("MODEL_NAME", "test_model")
    monkeypatch.setenv("MODEL_VERSION", "v_test")

    monkeypatch.setenv("DATA_DIR", "tests/data")
    monkeypatch.setenv("SOURCE_CSV", "test.csv")

    monkeypatch.setenv("SIMULATION_MAX_ITEMS", "100")
    monkeypatch.setenv("SIMULATION_DEFAULT_ITEMS", "50")

    monkeypatch.setenv("BUSINESS_COST_FN", "10")
    monkeypatch.setenv("BUSINESS_COST_FP", "1")

    yield


# =============================================================================
# 3. Création automatique d’un CSV minimal pour les tests
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def create_test_data():
    """
    Crée un CSV minimal pour éviter les erreurs de load.
    """
    data_dir = PROJECT_ROOT / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "test.csv"

    if not csv_path.exists():
        csv_path.write_text(
            "SK_ID_CURR,AMT_CREDIT\n100001,50000\n",
            encoding="utf-8",
        )

    yield


# =============================================================================
# 4. Reset éventuel des caches globaux (si utilisés)
# =============================================================================

@pytest.fixture(autouse=True)
def reset_caches():
    """
    Reset des caches globaux si ton projet en utilise.
    Évite les effets de bord entre tests.
    """
    try:
        from app.services.loader_services import data_loading_service

        if hasattr(data_loading_service, "RAW_DATA_CACHE"):
            data_loading_service.RAW_DATA_CACHE.clear()

        if hasattr(data_loading_service, "FEATURES_READY_CACHE"):
            data_loading_service.FEATURES_READY_CACHE = None

    except Exception:
        # Si module pas encore chargé, on ignore
        pass

    yield