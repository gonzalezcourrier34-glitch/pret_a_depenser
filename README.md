create_raw_tables.py
    ↓
load_csv_to_postgres.py
    ↓
create_features_table.py
    ↓
clean_features_table.py
    ↓
enrich_features_table.py
    ↓
verify_model_features.py
    ↓
create_model_ready_table.py
    ↓
create_prediction_tables.py
    ↓
create_monitoring_tables.py

## lancelment serveur mlflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///./mlflow_backend/mlflow.db --artifacts-destination ./mlflow_artifacts