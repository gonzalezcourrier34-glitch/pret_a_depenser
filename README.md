<h1 align="center">Système de scoring crédit avec API, monitoring et pipeline MLOps</h1>

<p align="center">
  <strong>Déploiement d’un modèle de machine learning en production avec traçabilité, monitoring, optimisation et dashboard</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-API-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/PostgreSQL-Database-blue" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b" alt="Streamlit">
  <img src="https://img.shields.io/badge/SQLAlchemy-ORM-red" alt="SQLAlchemy">
  <img src="https://img.shields.io/badge/MLflow-Tracking-lightgrey" alt="MLflow">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED" alt="Docker">
  <img src="https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF" alt="CI/CD">
  <img src="https://img.shields.io/badge/Monitoring-Production-orange" alt="Monitoring">
  <img src="https://img.shields.io/badge/MLOps-Credit_Scoring-purple" alt="MLOps">
</p>

<hr>

<h2>Sommaire</h2>

<ul>
  <li><a href="#section-1">1. Quick Start</a></li>
  <li><a href="#section-2">2. Présentation du projet</a></li>
  <li><a href="#section-3">3. Objectifs du projet</a></li>
  <li><a href="#section-4">4. Architecture du système</a></li>
  <li><a href="#section-5">5. Structure du projet</a></li>
  <li><a href="#section-6">6. Pipeline de préparation des données</a></li>
  <li><a href="#section-7">7. Scripts de création des tables</a></li>
  <li><a href="#section-8">8. Scripts d’exécution et scripts manuels</a></li>
  <li><a href="#section-9">9. Base de données et persistance</a></li>
  <li><a href="#section-10">10. API de prédiction</a></li>
  <li><a href="#section-11">11. Monitoring MLOps</a></li>
  <li><a href="#section-12">12. Dashboard Streamlit</a></li>
  <li><a href="#section-13">13. Optimisation post-déploiement</a></li>
  <li><a href="#section-14">14. Tests et qualité du code</a></li>
  <li><a href="#section-15">15. CI/CD et déploiement</a></li>
  <li><a href="#section-16">16. Configuration</a></li>
  <li><a href="#section-17">17. Lancer le projet</a></li>
  <li><a href="#section-18">18. Limites du prototype</a></li>
  <li><a href="#section-19">19. Pistes d’amélioration</a></li>
  <li><a href="#section-20">20. Auteur</a></li>
  <li><a href="#section-21">21. Licence</a></li>
</ul>

<hr>

<h2 id="section-1">1. Quick Start</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Lancer rapidement le projet dans un environnement local reproductible, initialiser PostgreSQL, enregistrer le modèle actif, démarrer l’API FastAPI puis visualiser les prédictions et le monitoring dans le dashboard Streamlit.
</div>

<pre><code>git clone repo_url
cd pret_a_depenser
cp .env.example .env
docker compose up --build</code></pre>

<p><strong>Ordre recommandé :</strong></p>

<ol>
  <li>créer les tables PostgreSQL nécessaires</li>
  <li>vérifier que le modèle et les fichiers CSV sont disponibles</li>
  <li>enregistrer le modèle actif dans le registre</li>
  <li>démarrer l’API FastAPI</li>
  <li>démarrer le dashboard Streamlit</li>
  <li>simuler des prédictions pour alimenter le monitoring</li>
  <li>associer ou simuler les vérités terrain</li>
  <li>consulter les métriques de latence, drift, performance et alertes</li>
</ol>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Important</strong><br><br>
  Le monitoring devient exploitable uniquement lorsque des prédictions sont journalisées en base. Les simulations de prédictions servent donc à générer de la donnée de production simulée.
</div>

<hr>

<h2 id="section-2">2. Présentation du projet</h2>

<p>
Ce projet met en place un système complet de <strong>scoring crédit en production</strong>. Le modèle de machine learning est exposé via une API FastAPI, les prédictions sont historisées dans PostgreSQL, et un dashboard Streamlit permet de suivre les métriques métier, techniques et MLOps.
</p>

<p>
L’objectif n’est pas seulement de prédire un risque de défaut, mais de construire une chaîne complète de mise en production :
</p>

<ul>
  <li>chargement des données client depuis CSV</li>
  <li>construction des features attendues par le modèle</li>
  <li>alignement strict des colonnes avec le modèle entraîné</li>
  <li>exposition du modèle via une API FastAPI</li>
  <li>sécurisation simple par clé API</li>
  <li>journalisation des prédictions</li>
  <li>stockage des snapshots de features</li>
  <li>suivi du modèle actif dans un registre</li>
  <li>monitoring de la performance, de la dérive et de la latence</li>
  <li>dashboard Streamlit pour piloter et démontrer le système</li>
  <li>optimisation post-déploiement du temps de réponse</li>
  <li>pipeline CI/CD pour automatiser les contrôles</li>
</ul>

<hr>

<h2 id="section-3">3. Objectifs du projet</h2>

<h3 style="color: #48C9B0;">Contexte</h3>

<p>
Dans un contexte de scoring crédit, un modèle peut perdre en fiabilité après son déploiement. Les profils clients peuvent évoluer, les distributions des variables peuvent dériver, et les performances observées en production peuvent s’éloigner des performances mesurées en phase d’entraînement.
</p>

<h3 style="color: #48C9B0;">Problématique</h3>

<p>
Le problème consiste donc à construire un système capable de :
</p>

<ul>
  <li>servir une prédiction fiable</li>
  <li>tracer les appels API</li>
  <li>conserver les scores et les classes prédites</li>
  <li>mesurer la latence API et le temps d’inférence</li>
  <li>suivre les versions de modèle</li>
  <li>analyser les dérives de données</li>
  <li>rapprocher les prédictions des vérités terrain</li>
  <li>générer des alertes de monitoring</li>
  <li>documenter les optimisations réalisées après déploiement</li>
</ul>

<h3 style="color: #48C9B0;">Objectif final</h3>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Construire un prototype MLOps complet, capable de déployer un modèle de scoring crédit, de tracer ses prédictions, de surveiller son comportement en production simulée et de démontrer une amélioration mesurable du temps de réponse.
</div>

<hr>

<h2 id="section-4">4. Architecture du système</h2>

<pre><code>CSV Home Credit
      │
      ▼
Service de chargement des données
      │
      ├── lecture du CSV source
      ├── cache des données brutes
      ├── construction des features
      └── cache des features prêtes
      │
      ▼
Modèle entraîné joblib
      │
      ▼
API FastAPI
      │
      ├── /predict
      ├── /predict/batch
      ├── /predict/simulate/real-sample
      ├── /history
      └── /monitoring
      │
      ▼
PostgreSQL
      │
      ├── prediction_logs
      ├── prediction_features_snapshot
      ├── ground_truth_labels
      ├── model_registry
      ├── drift_metrics
      ├── evaluation_metrics
      ├── feature_store_monitoring
      └── alerts
      │
      ▼
Dashboard Streamlit
      │
      ├── prédictions
      ├── historique
      ├── monitoring
      ├── drift
      ├── alertes
      └── registre modèle</code></pre>

<hr>

<h2 id="section-5">5. Structure du projet</h2>

<pre><code>pret_a_depenser/
│
├── app/
│   ├── api/
│   │   ├── route_prediction.py
│   │   ├── route_history.py
│   │   ├── route_monitoring.py
│   │   └── route_analyse.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── db.py
│   │   ├── logging_config.py
│   │   ├── model_features.py
│   │   ├── schemas.py
│   │   └── security.py
│   │
│   ├── crud/
│   │   ├── monitoring.py
│   │   └── prediction.py
│   │
│   ├── model/
│   │   └── model_SQLalchemy.py
│   │
│   ├── services/
│   │   ├── analysis_services/
│   │   │   ├── evidently_service.py
│   │   │   └── monitoring_evaluation_service.py
│   │   │
│   │   ├── loader_services/
│   │   │   ├── data_loading_service.py
│   │   │   └── model_loading_service.py
│   │   │
│   │   ├── features_builder_service.py
│   │   ├── history_service.py
│   │   ├── logging_service.py
│   │   ├── monitoring_service.py
│   │   ├── prediction_logging_service.py
│   │   └── prediction_service.py
│   │
│   └── main.py
│
├── dashboard/
│   ├── dashboard.py
│   ├── dashboard_predictions.py
│   ├── dashboard_monitoring.py
│   ├── dashboard_systeme.py
│   └── dashboard_request.py
│
├── scripts/
│   ├── create_tables/
│   │   ├── create_prediction_tables.py
│   │   ├── create_monitoring_tables.py
│   │   └── create_raw_tables.py
│   │
│   ├── manualy_run_scripts/
│   │   ├── run_previeu_ground_truth.py
│   │   ├── run_monitoring_analysis.py
│   │   └── run_evidently_analysis.py
│   │
│   ├── register_model.py
│   ├── benchmark_inference.py
│   └── profile_inference.py
│
├── artifacts/
│   ├── model.joblib
│   ├── threshold.json
│   ├── benchmark_baseline.csv
│   ├── benchmark_optimized.csv
│   └── profiling_report.txt
│
├── data/
│   ├── application_test.csv
│   └── autres fichiers Home Credit
│
├── docs/
│   └── rapport_optimisation_inference.md
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md</code></pre>

<hr>

<h2 id="section-6">6. Pipeline de préparation des données</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Transformer les données Home Credit en features exploitables par le modèle, tout en garantissant que les colonnes envoyées au modèle correspondent exactement aux features attendues.
</div>

<p>
Dans la version actuelle du projet, l’API fonctionne principalement à partir du CSV source et d’un service de chargement en mémoire. Les données sont chargées une seule fois, les features sont construites puis conservées en cache afin d’éviter de recalculer toute la préparation à chaque prédiction.
</p>

<ol>
  <li>lecture du fichier défini par <code>APPLICATION_CSV</code></li>
  <li>mise en cache des données brutes</li>
  <li>construction des features via le service dédié</li>
  <li>alignement avec <code>MODEL_FEATURES</code></li>
  <li>mise en cache des features prêtes</li>
  <li>réutilisation des features pour les prédictions unitaires et batch</li>
</ol>

<h3 style="color: #48C9B0;">Exemples de features créées</h3>

<ul>
  <li><code>AGE_YEARS</code></li>
  <li><code>EMPLOYED_YEARS</code></li>
  <li><code>CREDIT_INCOME_RATIO</code></li>
  <li><code>ANNUITY_INCOME_RATIO</code></li>
  <li><code>ANNUITY_CREDIT_RATIO</code></li>
  <li><code>OVER_INDEBTED_40</code></li>
  <li><code>LOG_INCOME</code></li>
  <li><code>LOG_CREDIT</code></li>
  <li><code>DOC_COUNT</code></li>
  <li><code>EXT_SOURCES_MEAN</code></li>
  <li><code>EXT_SOURCES_MIN</code></li>
  <li><code>EXT_SOURCES_MAX</code></li>
</ul>

<hr>

<h2 id="section-7">7. Scripts de création des tables</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Créer les tables PostgreSQL nécessaires à la journalisation des prédictions, au stockage des vérités terrain, au monitoring et au registre des modèles.
</div>

<h3 style="color: #48C9B0;">Créer les tables de prédiction</h3>

<pre><code>uv run python -m scripts.create_tables.create_prediction_tables</code></pre>

<p>Tables concernées :</p>

<ul>
  <li><code>prediction_logs</code></li>
  <li><code>prediction_features_snapshot</code></li>
  <li><code>ground_truth_labels</code></li>
</ul>

<h3 style="color: #48C9B0;">Créer les tables de monitoring</h3>

<pre><code>uv run python -m scripts.create_tables.create_monitoring_tables</code></pre>

<p>Tables concernées :</p>

<ul>
  <li><code>model_registry</code></li>
  <li><code>feature_store_monitoring</code></li>
  <li><code>drift_metrics</code></li>
  <li><code>evaluation_metrics</code></li>
  <li><code>alerts</code></li>
</ul>

<h3 style="color: #48C9B0;">Créer les tables RAW</h3>

<pre><code>uv run python -m scripts.create_tables.create_raw_tables</code></pre>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Note</strong><br><br>
  Les tables RAW existent pour une architecture plus complète basée sur PostgreSQL. Dans la version priorisée actuellement, le projet fonctionne principalement à partir du CSV et du cache applicatif.
</div>

<hr>

<h2 id="section-8">8. Scripts d’exécution et scripts manuels</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Utiliser des scripts ponctuels pour enregistrer un modèle, associer les prédictions aux vérités terrain, déclencher une analyse de monitoring ou lancer une analyse Evidently.
</div>

<h3 style="color: #48C9B0;">Enregistrer le modèle actif</h3>

<pre><code>docker exec -e PYTHONPATH=/app -e MODEL_VERSION=v1 -it fastapi_credit_api uv run python -m scripts.register_model</code></pre>

<p>
Ce script ajoute le modèle courant dans <code>model_registry</code> et permet à l’API ou au dashboard d’identifier le modèle actif.
</p>

<h3 style="color: #48C9B0;">Associer les prédictions à une vérité terrain simulée</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_previeu_ground_truth</code></pre>

<p>
Ce script rapproche les prédictions existantes des labels disponibles ou simulés. Il permet ensuite de calculer des métriques de performance sur des données de production simulée.
</p>

<h3 style="color: #48C9B0;">Lancer une analyse de monitoring</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_monitoring_analysis</code></pre>

<p>
Ce script calcule des métriques de performance, crée des alertes ou alimente les tables de monitoring.
</p>

<h3 style="color: #48C9B0;">Lancer une analyse Evidently</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_evidently_analysis</code></pre>

<p>
Ce script produit ou simule une analyse de dérive des données avec Evidently.
</p>

<hr>

<h2 id="section-9">9. Base de données et persistance</h2>

<p>
PostgreSQL est utilisé comme base centrale du projet.
</p>

<p>Elle stocke :</p>

<ul>
  <li>les logs de prédiction</li>
  <li>les snapshots de features</li>
  <li>les vérités terrain</li>
  <li>les métriques de dérive</li>
  <li>les métriques de performance</li>
  <li>les alertes</li>
  <li>le registre des modèles</li>
</ul>

<p>
Cette séparation permet de distinguer les données opérationnelles, les données de monitoring et les métadonnées du modèle.
</p>

<hr>

<h2 id="section-10">10. API de prédiction</h2>

<p>
L’API FastAPI expose le modèle de scoring crédit.
</p>

<h3 style="color: #48C9B0;">Endpoints principaux</h3>

<ul>
  <li><code>GET /health</code> : état général de l’API</li>
  <li><code>GET /predict/health</code> : état du service de prédiction</li>
  <li><code>POST /predict</code> : prédiction unitaire</li>
  <li><code>POST /predict/batch</code> : prédiction batch</li>
  <li><code>POST /predict/simulate/real-sample</code> : simulation de prédictions sur clients réels</li>
  <li><code>GET /history/predictions</code> : historique des prédictions</li>
  <li><code>GET /monitoring/summary</code> : synthèse du monitoring</li>
  <li><code>GET /monitoring/models</code> : registre des modèles</li>
  <li><code>GET /monitoring/alerts</code> : alertes de monitoring</li>
</ul>

<h3 style="color: #48C9B0;">Traçabilité</h3>

<p>
Chaque prédiction peut être journalisée avec :
</p>

<ul>
  <li>le score prédit</li>
  <li>la classe prédite</li>
  <li>le seuil utilisé</li>
  <li>la latence API</li>
  <li>le temps d’inférence pur</li>
  <li>les informations modèle</li>
  <li>les éventuelles erreurs</li>
</ul>

<hr>

<h2 id="section-11">11. Monitoring MLOps</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Suivre le comportement du modèle en production simulée : performance, dérive, latence, temps d’inférence, erreurs et alertes.
</div>

<h3 style="color: #48C9B0;">Métriques suivies</h3>

<ul>
  <li>nombre total de prédictions</li>
  <li>distribution des scores prédits</li>
  <li>latence moyenne API</li>
  <li>latence p95</li>
  <li>latence p99</li>
  <li>temps d’inférence moyen</li>
  <li>taux d’erreur API</li>
  <li>drift par feature</li>
  <li>ROC AUC</li>
  <li>PR AUC</li>
  <li>précision</li>
  <li>rappel</li>
  <li>F1-score</li>
  <li>F-beta score</li>
  <li>coût métier</li>
</ul>

<h3 style="color: #48C9B0;">Différence entre latence et inférence</h3>

<p>
La <strong>latence API</strong> mesure le temps total perçu par l’utilisateur.
</p>

<pre><code>requête reçue
+ préparation des données
+ prédiction du modèle
+ journalisation
+ réponse HTTP</code></pre>

<p>
Le <strong>temps d’inférence</strong> mesure uniquement le temps passé dans le modèle.
</p>

<pre><code>model.predict_proba(features)</code></pre>

<p>
Les deux métriques sont utiles : la latence mesure l’expérience réelle utilisateur, tandis que le temps d’inférence permet d’identifier si le modèle lui-même est lent.
</p>

<hr>

<h2 id="section-12">12. Dashboard Streamlit</h2>

<p>
Le dashboard Streamlit permet de visualiser et piloter le système.
</p>

<h3 style="color: #48C9B0;">Pages principales</h3>

<ul>
  <li>vue système</li>
  <li>prédiction unitaire</li>
  <li>simulation batch</li>
  <li>historique des prédictions</li>
  <li>détail d’une prédiction</li>
  <li>monitoring global</li>
  <li>métriques de dérive</li>
  <li>métriques de performance</li>
  <li>alertes</li>
  <li>registre des modèles</li>
</ul>

<h3 style="color: #48C9B0;">Indicateurs affichés</h3>

<ul>
  <li>total des prédictions</li>
  <li>latence moyenne</li>
  <li>latence p95</li>
  <li>latence p99</li>
  <li>temps moyen d’inférence</li>
  <li>taux d’erreur</li>
  <li>distribution des scores</li>
  <li>répartition des classes prédites</li>
  <li>métriques de performance</li>
  <li>alertes ouvertes</li>
</ul>

<hr>

<h2 id="section-13">13. Optimisation post-déploiement</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Analyser les performances réelles ou simulées du modèle après déploiement, identifier les goulots d’étranglement et tester des optimisations pour améliorer le temps de réponse.
</div>

<h3 style="color: #48C9B0;">Métriques utilisées</h3>

<ul>
  <li>latence moyenne API</li>
  <li>latence p95</li>
  <li>latence p99</li>
  <li>temps d’inférence pur</li>
  <li>temps de préparation des features</li>
  <li>taux d’erreur</li>
  <li>utilisation CPU / mémoire si disponible</li>
</ul>

<h3 style="color: #48C9B0;">Profiling</h3>

<pre><code>uv run python -m cProfile -o artifacts/profile_inference.prof scripts/profile_inference.py</code></pre>

<pre><code>uv run python scripts/profile_inference.py</code></pre>

<h3 style="color: #48C9B0;">Benchmark baseline</h3>

<pre><code>uv run python scripts/benchmark_inference.py --mode baseline</code></pre>

<h3 style="color: #48C9B0;">Benchmark optimisé</h3>

<pre><code>uv run python scripts/benchmark_inference.py --mode optimized</code></pre>

<h3 style="color: #48C9B0;">Optimisations testées</h3>

<ul>
  <li>chargement du modèle une seule fois au démarrage de l’API</li>
  <li>cache des données et des features client en mémoire</li>
  <li>alignement strict des colonnes avant prédiction</li>
  <li>réduction des conversions pandas inutiles</li>
  <li>conversion des données numériques en <code>float32</code></li>
  <li>journalisation du temps d’inférence pur</li>
  <li>séparation entre latence API et inférence modèle</li>
</ul>

<h3 style="color: #48C9B0;">Rapport d’optimisation</h3>

<pre><code>docs/rapport_optimisation_inference.md</code></pre>

<hr>

<h2 id="section-14">14. Tests et qualité du code</h2>

<p>
Le projet intègre ou prévoit des tests automatisés sur :
</p>

<ul>
  <li>les endpoints FastAPI</li>
  <li>le chargement du modèle</li>
  <li>l’alignement des features</li>
  <li>les services de prédiction</li>
  <li>les routes de monitoring</li>
  <li>les appels du dashboard</li>
  <li>les scripts de création de tables</li>
</ul>

<h3 style="color: #48C9B0;">Lancer les tests</h3>

<pre><code>uv run pytest</code></pre>

<h3 style="color: #48C9B0;">Linting avec Ruff</h3>

<pre><code>uv run ruff check .</code></pre>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Attention</strong><br><br>
  Si la commande Ruff échoue avec <code>Failed to spawn: ruff</code>, cela signifie que Ruff n’est pas installé dans l’environnement. Il faut l’ajouter dans les dépendances du projet.
</div>

<hr>

<h2 id="section-15">15. CI/CD et déploiement</h2>

<p>
Le projet peut être contrôlé par GitHub Actions afin d’automatiser :
</p>

<ul>
  <li>l’installation de l’environnement</li>
  <li>les tests unitaires</li>
  <li>le linting</li>
  <li>la construction Docker</li>
  <li>le déploiement de la version optimisée</li>
</ul>

<h3 style="color: #48C9B0;">Pipeline attendu</h3>

<pre><code>push GitHub
      │
      ▼
GitHub Actions
      │
      ├── uv sync
      ├── ruff check
      ├── pytest
      ├── build Docker
      └── deploy</code></pre>

<hr>

<h2 id="section-16">16. Configuration</h2>

<h3 style="color: #48C9B0;">Prérequis</h3>

<ul>
  <li>Python 3.11</li>
  <li>uv</li>
  <li>Docker</li>
  <li>Docker Compose</li>
  <li>PostgreSQL</li>
  <li>un modèle sérialisé au format <code>joblib</code></li>
</ul>

<h3 style="color: #48C9B0;">Variables d’environnement</h3>

<pre><code>API_KEY=votre_token_api
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/credit_api

MODEL_PATH=artifacts/model.joblib
MODEL_NAME=credit_scoring_model
MODEL_VERSION=v1
MODEL_THRESHOLD=0.5

APPLICATION_CSV=data/application_test.csv

DEBUG=True</code></pre>

<hr>

<h2 id="section-17">17. Lancer le projet</h2>

<h3 style="color: #48C9B0;">1. Installer les dépendances</h3>

<pre><code>uv sync</code></pre>

<h3 style="color: #48C9B0;">2. Démarrer les services Docker</h3>

<pre><code>docker compose up --build</code></pre>

<h3 style="color: #48C9B0;">3. Créer les tables de prédiction</h3>

<pre><code>uv run python -m scripts.create_tables.create_prediction_tables</code></pre>

<h3 style="color: #48C9B0;">4. Créer les tables de monitoring</h3>

<pre><code>uv run python -m scripts.create_tables.create_monitoring_tables</code></pre>

<h3 style="color: #48C9B0;">5. Enregistrer le modèle actif</h3>

<pre><code>docker exec -e PYTHONPATH=/app -e MODEL_VERSION=v1 -it fastapi_credit_api uv run python -m scripts.register_model</code></pre>

<h3 style="color: #48C9B0;">6. Lancer l’API en local</h3>

<pre><code>uv run uvicorn app.main:app --reload</code></pre>

<h3 style="color: #48C9B0;">7. Lancer le dashboard</h3>

<pre><code>uv run streamlit run dashboard/dashboard.py</code></pre>

<h3 style="color: #48C9B0;">8. Simuler des prédictions</h3>

<pre><code>POST /predict/simulate/real-sample?limit=200</code></pre>

<h3 style="color: #48C9B0;">9. Générer ou associer les vérités terrain</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_previeu_ground_truth</code></pre>

<h3 style="color: #48C9B0;">10. Lancer le monitoring</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_monitoring_analysis</code></pre>

<h3 style="color: #48C9B0;">11. Lancer les benchmarks d’optimisation</h3>

<pre><code>uv run python scripts/benchmark_inference.py --mode baseline
uv run python scripts/benchmark_inference.py --mode optimized</code></pre>

<hr>

<h2 id="section-18">18. Limites du prototype</h2>

<ul>
  <li>le monitoring dépend du volume de prédictions journalisées</li>
  <li>les métriques de performance nécessitent une vérité terrain fiable</li>
  <li>les données de production sont simulées dans le cadre du projet</li>
  <li>le monitoring CPU/GPU reste basique si aucun outil système dédié n’est branché</li>
  <li>les optimisations testées restent adaptées à un prototype local</li>
  <li>le drift dépend fortement du choix de la fenêtre de référence</li>
  <li>le dashboard dépend de la disponibilité de l’API FastAPI</li>
</ul>

<hr>

<h2 id="section-19">19. Pistes d’amélioration</h2>

<ul>
  <li>ajouter un vrai ordonnanceur de jobs de monitoring</li>
  <li>automatiser le calcul périodique du drift</li>
  <li>ajouter Prometheus et Grafana pour les métriques système</li>
  <li>brancher un monitoring CPU / mémoire plus complet</li>
  <li>tester ONNX Runtime pour optimiser l’inférence</li>
  <li>ajouter un système de rollback modèle</li>
  <li>améliorer la gestion des seuils de décision</li>
  <li>déployer sur une infrastructure cloud</li>
  <li>ajouter des tests de non-régression sur les prédictions</li>
  <li>mettre en place des alertes automatiques</li>
</ul>

<hr>

<h2 id="section-20">20. Auteur</h2>

<p>
Projet réalisé par <strong>Stéphane GONZALEZ</strong> dans le cadre d’un apprentissage autour du déploiement de modèles de machine learning, du monitoring MLOps, de l’optimisation post-déploiement et des architectures applicatives de scoring crédit.
</p>

<hr>

<h2 id="section-21">21. Licence</h2>

<p>Usage éducatif.</p>