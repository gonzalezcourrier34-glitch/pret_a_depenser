# Historique des versions

## v1.3.0 - Monitoring et optimisation
- Ajout du monitoring des prédictions en production simulée.
- Ajout de l’analyse de Data Drift avec Evidently.
- Ajout des métriques de performance modèle : ROC AUC, recall, precision, F1, F-beta.
- Ajout du suivi de latence API : moyenne, P95, P99.
- Ajout des alertes de monitoring.
- Optimisation de l’inférence avec backend ONNX.
- Ajout des rapports de performance et de dérive.

## v1.2.0 - Traçabilité et persistance
- Ajout de la base PostgreSQL.
- Persistance des prédictions dans `prediction_logs`.
- Persistance des features dans `prediction_features_snapshot`.
- Ajout du suivi des modèles dans `model_registry`.
- Ajout du stockage des métriques dans `drift_metrics` et `evaluation_metrics`.
- Ajout des logs applicatifs structurés.

## v1.1.0 - API de prédiction
- Création de l’API FastAPI.
- Ajout du endpoint de prédiction unitaire.
- Ajout du endpoint de prédiction batch.
- Ajout du endpoint de simulation sur clients réels.
- Ajout de la sécurisation par clé API.
- Ajout des schémas Pydantic de validation.

## v1.0.0 - Modèle initial
- Préparation du dataset Home Credit.
- Nettoyage des données.
- Feature engineering.
- Entraînement des premiers modèles.
- Suivi des expériences avec MLflow.
- Sélection du modèle de scoring crédit.
- Sauvegarde du modèle final et du seuil de décision.