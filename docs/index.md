# Credit Scoring API – Système MLOps complet

API de scoring crédit déployée en production avec une architecture MLOps complète, intégrant prédiction, monitoring, traçabilité et analyse de dérive.

---

## Objectif

Ce projet vise à reproduire un système industriel de scoring crédit :

- prédire le risque client en temps réel
- suivre les performances du modèle
- détecter les dérives de données
- garantir la traçabilité des décisions

---

## Fonctionnalités principales

### Prédiction
- API REST avec FastAPI
- prédiction unitaire et batch
- seuil configurable
- latence optimisée

### Monitoring MLOps
- suivi des performances (precision, recall, ROC AUC)
- suivi de la latence (P95, P99)
- détection de drift global et par feature
- génération d’alertes

### Traçabilité
- stockage des prédictions en base PostgreSQL
- historique consultable
- audit des décisions modèle

### Analyse avancée
- rapports de dérive avec Evidently
- comparaison données référence vs production
- analyse fine par variable

### Dashboard
- visualisation des métriques
- exploration des prédictions
- suivi des alertes

---

## Architecture du système

Le projet suit une architecture MLOps complète :

Client → API FastAPI → Modèle ML
↓
PostgreSQL
↓
Monitoring & Drift (Evidently)
↓
Dashboard Streamlit


---

## Stack technique

- FastAPI → API REST
- PostgreSQL → stockage
- SQLAlchemy → ORM
- Evidently → monitoring & drift
- MLflow → gestion du modèle
- Streamlit → dashboard
- Docker → containerisation

---

## Sécurité & bonnes pratiques

- authentification via API Key
- validation stricte des entrées (Pydantic)
- logs structurés JSON
- séparation claire des couches (API / services / CRUD)

---

## Organisation du projet

```text
app/
├── api/                # routes FastAPI
├── services/           # logique métier
├── crud/               # accès base de données
├── models/             # modèles SQLAlchemy
├── schemas/            # validation Pydantic
├── core/               # config & sécurité

artifacts/
├── model/              # modèle ML
├── monitoring/         # données de drift

dashboard/
├── Streamlit UI

docs/
├── documentation MkDocs

---

## Monitoring en production

Le système surveille en continu :

- dérive des données
- performances du modèle
- latence des prédictions
- taux d’erreur

Des alertes sont générées automatiquement en cas d’anomalie.

---

## Tests & CI/CD

- tests unitaires avec pytest
- pipeline CI GitHub Actions
- build Docker automatisé

---

## Évolutions possibles

- déploiement cloud (AWS / GCP)
- feature store réel
- retraining automatique
- alerting avancé (Slack / email)

---

## Auteur

Projet réalisé dans le cadre d’un parcours MLOps / IA Engineering.