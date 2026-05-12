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
  <img src="https://img.shields.io/badge/ONNX_Runtime-Inference-005CED" alt="ONNX Runtime">
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
  <li><a href="#section-18">18. Livrables de la mission</a></li>
  <li><a href="#section-19">19. Limites du prototype</a></li>
  <li><a href="#section-20">20. Pistes d’amélioration</a></li>
  <li><a href="#section-21">21. Auteur</a></li>
  <li><a href="#section-22">22. Licence</a></li>
</ul>

<hr>

<h2 id="section-1">1. Quick Start</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Lancer rapidement le projet dans un environnement local reproductible, initialiser PostgreSQL, enregistrer le modèle actif, démarrer l’API FastAPI puis visualiser les prédictions, la latence, le temps d’inférence et le monitoring dans le dashboard Streamlit.
</div>

<pre><code>git clone repo_url
cd pret_a_depenser
cp .env.example .env
docker compose up --build</code></pre>

<p><strong>Ordre recommandé :</strong></p>

<ol>
  <li>créer les tables PostgreSQL nécessaires</li>
  <li>vérifier que les fichiers CSV sont disponibles</li>
  <li>vérifier que le modèle <code>joblib</code> ou <code>ONNX</code> est disponible</li>
  <li>configurer le backend avec <code>MODEL_BACKEND</code></li>
  <li>enregistrer le modèle actif dans le registre</li>
  <li>démarrer l’API FastAPI</li>
  <li>démarrer le dashboard Streamlit</li>
  <li>simuler des prédictions pour alimenter le monitoring</li>
  <li>associer ou simuler les vérités terrain</li>
  <li>consulter les métriques de latence, temps d’inférence, drift, performance et alertes</li>
</ol>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Important</strong><br><br>
  Le monitoring devient exploitable uniquement lorsque des prédictions sont journalisées en base. Les simulations de prédictions servent donc à générer de la donnée de production simulée.
</div>

<h3 style="color: #48C9B0;">URLs utiles</h3>

<ul>
  <li><code>http://127.0.0.1:8000/docs</code> : documentation Swagger de l’API</li>
  <li><code>http://127.0.0.1:8501</code> : dashboard Streamlit</li>
  <li><code>http://127.0.0.1:8000/predict/health</code> : healthcheck prédiction</li>
</ul>

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
  <li>support de deux backends d’inférence : <code>sklearn/joblib</code> et <code>ONNX Runtime</code></li>
  <li>exposition du modèle via une API FastAPI</li>
  <li>sécurisation simple par clé API</li>
  <li>journalisation des prédictions</li>
  <li>stockage des snapshots de features</li>
  <li>suivi du modèle actif dans un registre</li>
  <li>monitoring de la performance, de la dérive, de la latence API et du temps d’inférence pur</li>
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
  <li>mesurer la latence API globale</li>
  <li>mesurer le temps d’inférence pur du modèle avec <code>inference_latency_ms</code></li>
  <li>suivre les versions de modèle</li>
  <li>basculer entre un backend <code>sklearn/joblib</code> et un backend <code>ONNX Runtime</code></li>
  <li>analyser les dérives de données</li>
  <li>rapprocher les prédictions des vérités terrain</li>
  <li>générer des alertes de monitoring</li>
  <li>documenter les optimisations réalisées après déploiement</li>
</ul>

<h3 style="color: #48C9B0;">Objectif final</h3>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Construire un prototype MLOps complet, capable de déployer un modèle de scoring crédit, de tracer ses prédictions, de surveiller son comportement en production simulée, de comparer plusieurs backends d’inférence et de démontrer une amélioration mesurable du temps de réponse.
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
Service de chargement du modèle
      │
      ├── backend sklearn / joblib
      ├── backend ONNX Runtime
      ├── cache du modèle sklearn
      ├── cache de la session ONNX
      └── chargement du seuil métier
      │
      ▼
Service de prédiction
      │
      ├── nettoyage des features
      ├── appel predict_proba_with_backend
      ├── calcul du score
      ├── application du seuil
      ├── mesure inference_latency_ms
      └── retour prediction / score / seuil / temps d’inférence
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
      ├── latence API
      ├── temps d’inférence
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
│   │   │   ├── huggingface_download_service.py
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
│   ├── dashboard_onglet/
│   │   ├── dashboard_monitoring.py
│   │   ├── dashboard_monitoring_performance.py
│   │   ├── dashboard_monitoring_drift.py
│   │   ├── dashboard_predictions.py
│   │   └── dashboard_systeme.py
│   ├── dashboard_config.py
│   ├── dashboard_main.py
│   └── dashboard_request.py
│
├── scripts/
│   ├── initialisation_tables/
│   ├── create_monitoring_tables.py
│   ├── create_prediction_tables.py
│   ├── pipeline_db.py
│   └── reset_data.py
│
├── manualy_run_scripts/
│   ├── load_ground_truth_last_2000.py
│   ├── run_evidently_monitoring.py
│   ├── run_ground_truth_association.py
│   ├── run_monitoring_evaluation.py
│   └── run_preview_ground_truth.py
│
├── benchmark_inference.py
├── convert_model_to_onnx.py
├── profile_inference.py
├── register_model.py
│
├── artifacts/
│   ├── monitoring/
│   ├── performance/
│   │   ├── benchmark_baseline.csv
│   │   ├── benchmark_optimized.csv
│   │   └── profiling_report.txt
│   ├── model.joblib
│   ├── model.onnx
│   └── threshold.json
│
├── data/
│   ├── application_test.csv
│   └── autres fichiers Home Credit...
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
  <li>nettoyage des valeurs manquantes avant prédiction</li>
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

<pre><code>uv run python -m scripts.initialisation_tables.create_prediction_tables</code></pre>

<p>Tables concernées :</p>

<ul>
  <li><code>prediction_logs</code></li>
  <li><code>prediction_features_snapshot</code></li>
  <li><code>ground_truth_labels</code></li>
</ul>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Note</strong><br><br>
  La table <code>prediction_logs</code> contient deux métriques distinctes :
  <ul>
    <li><code>latency_ms</code> : latence globale API</li>
    <li><code>inference_latency_ms</code> : temps d’inférence pur du modèle</li>
  </ul>
</div>

<h3 style="color: #48C9B0;">Créer les tables de monitoring</h3>

<pre><code>uv run python -m scripts.initialisation_tables.create_monitoring_tables</code></pre>

<p>Tables concernées :</p>

<ul>
  <li><code>model_registry</code></li>
  <li><code>feature_store_monitoring</code></li>
  <li><code>drift_metrics</code></li>
  <li><code>evaluation_metrics</code></li>
  <li><code>alerts</code></li>
</ul>

<h3 style="color: #48C9B0;">Initialiser la base complète</h3>

<pre><code>uv run python -m scripts.initialisation_tables.pipeline_db</code></pre>

<p>
Ce script permet d’enchaîner la création complète de la base (tables + initialisation éventuelle).
</p>

<h3 style="color: #48C9B0;">Réinitialiser les données</h3>

<pre><code>uv run python -m scripts.initialisation_tables.reset_data</code></pre>

<p>
Permet de nettoyer ou réinitialiser les données de test pour repartir sur un environnement propre.
</p>

<hr>

<h2 id="section-8">8. Scripts d’exécution et scripts manuels</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Utiliser des scripts ponctuels pour enregistrer un modèle, associer les prédictions aux vérités terrain, déclencher une analyse de monitoring ou lancer une analyse Evidently.
</div>

<h3 style="color: #48C9B0;">Enregistrer le modèle actif</h3>

<pre><code>uv run python -m scripts.register_model</code></pre>

<p>
Ajoute le modèle courant dans <code>model_registry</code> et le rend actif pour l’API.
</p>

<h3 style="color: #48C9B0;">Associer les prédictions aux vérités terrain</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_preview_ground_truth</code></pre>

<p>
Associe les prédictions existantes aux labels disponibles ou simulés afin de permettre le calcul des métriques de performance.
</p>

<h3 style="color: #48C9B0;">Charger des vérités terrain récentes</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.load_ground_truth_last_2000</code></pre>

<p>
Injecte un jeu de vérité terrain simulé pour alimenter les métriques de monitoring.
</p>

<h3 style="color: #48C9B0;">Association avancée des vérités terrain</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_ground_truth_assiociation</code></pre>

<p>
Script avancé pour reconstruire les correspondances entre prédictions et labels.
</p>

<h3 style="color: #48C9B0;">Lancer une évaluation monitoring</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_monitoring_evaluation</code></pre>

<p>
Calcule les métriques de performance (precision, recall, coût métier, etc.).
</p>

<h3 style="color: #48C9B0;">Lancer une analyse de dérive (Evidently)</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_evidently_monitoring</code></pre>

<p>
Génère une analyse de dérive des données (feature drift) et alimente la table <code>drift_metrics</code>.
</p>

<hr>

<h2 id="section-9">9. Base de données et persistance</h2>

<p>
PostgreSQL est utilisé comme base centrale du projet.
</p>

<p>Elle stocke :</p>

<ul>
  <li>les logs de prédiction</li>
  <li>la latence API globale</li>
  <li>le temps d’inférence pur</li>
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
L’API FastAPI expose le modèle de scoring crédit. Le backend utilisé pour l’inférence est sélectionné par configuration.
</p>

<h3 style="color: #48C9B0;">Backends supportés</h3>

<ul>
  <li><code>MODEL_BACKEND=sklearn</code> : chargement du pipeline via <code>joblib</code></li>
  <li><code>MODEL_BACKEND=onnx</code> : chargement du modèle via <code>ONNX Runtime</code></li>
</ul>

<p>
Le service de prédiction utilise la fonction <code>predict_proba_with_backend</code>, ce qui permet de garder la même API applicative tout en changeant le moteur d’inférence.
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
  <li>la latence API globale avec <code>latency_ms</code></li>
  <li>le temps d’inférence pur avec <code>inference_latency_ms</code></li>
  <li>les informations modèle</li>
  <li>le backend d’inférence configuré</li>
  <li>les éventuelles erreurs</li>
</ul>

<h3 style="color: #48C9B0;">Exemple de réponse</h3>

<pre><code>{
  "request_id": "uuid",
  "client_id": 100001,
  "prediction": 1,
  "score": 0.82,
  "threshold_used": 0.5,
  "model_name": "credit_scoring_model",
  "model_version": "v1",
  "latency_ms": 12.3,
  "inference_latency_ms": 3.1,
  "status": "success"
}</code></pre>

<hr>

<h2 id="section-11">11. Monitoring MLOps</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Suivre en continu le comportement du modèle en production simulée afin de détecter toute dégradation :
  performance métier, dérive des données, latence, erreurs et anomalies système.
</div>

<h3 style="color: #48C9B0;">Métriques suivies</h3>

<p>
Le système de monitoring combine des métriques <strong>techniques</strong>, <strong>statistiques</strong> et <strong>métier</strong> afin d’obtenir une vision complète du modèle en production.
</p>

<ul>
  <li><strong>Volume</strong> : nombre total de prédictions</li>
  <li><strong>Distribution</strong> : scores prédits, classes, dérive des probabilités</li>
  <li><strong>Performance API</strong> : latence moyenne, p95, p99</li>
  <li><strong>Performance modèle</strong> : temps d’inférence moyen et distribution</li>
  <li><strong>Fiabilité</strong> : taux d’erreur API</li>
  <li><strong>Drift</strong> : dérive des features (feature drift)</li>
  <li><strong>Performance ML</strong> :
    <ul>
      <li>ROC AUC</li>
      <li>PR AUC</li>
      <li>précision</li>
      <li>rappel</li>
      <li>F1-score / F-beta</li>
    </ul>
  </li>
  <li><strong>Coût métier</strong> : pondération FN / FP</li>
</ul>

<h3 style="color: #48C9B0;">Différence entre latence et inférence</h3>

<p>
La distinction entre latence API et temps d’inférence est essentielle pour identifier les goulots d’étranglement.
</p>

<p><strong>Latence API (expérience utilisateur)</strong></p>

<pre><code>requête reçue
+ validation
+ préparation des données
+ prédiction du modèle
+ journalisation
+ réponse HTTP</code></pre>

<p><strong>Temps d’inférence (performance modèle)</strong></p>

<pre><code>predict_proba_with_backend(features)</code></pre>

<p>
👉 Une latence élevée avec une inférence faible indique un problème applicatif  
👉 Une inférence élevée indique un problème lié au modèle ou au backend
</p>

<hr>

<h2 id="section-12">12. Dashboard Streamlit</h2>

<p>
Le dashboard Streamlit agit comme un <strong>poste de pilotage du modèle en production</strong>.
Il permet de visualiser les métriques, analyser les anomalies et déclencher des actions de monitoring.
</p>

<h3 style="color: #48C9B0;">Pages principales</h3>

<ul>
  <li>vue système (état global API + modèle)</li>
  <li>prédiction unitaire</li>
  <li>simulation batch</li>
  <li>historique des prédictions</li>
  <li>détail d’une prédiction (traçabilité complète)</li>
  <li>monitoring global</li>
  <li>analyse de dérive (drift)</li>
  <li>performance du modèle</li>
  <li>gestion des alertes</li>
  <li>registre des modèles</li>
</ul>

<h3 style="color: #48C9B0;">Indicateurs affichés</h3>

<ul>
  <li>KPI globaux (volume, erreurs, drift détecté)</li>
  <li>latence API (moyenne, p95, p99)</li>
  <li>temps d’inférence</li>
  <li>distribution des scores</li>
  <li>répartition des classes</li>
  <li>évolution temporelle des métriques</li>
  <li>métriques de performance ML</li>
  <li>alertes ouvertes et historiques</li>
</ul>

<p>
👉 Le dashboard permet de détecter rapidement :
<ul>
  <li>une dérive de données</li>
  <li>une dégradation de performance</li>
  <li>un problème de latence</li>
</ul>
</p>

<hr>

<h2 id="section-13">13. Optimisation post-déploiement</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Identifier les goulots d’étranglement après déploiement et améliorer les performances réelles du système (latence et inférence).
</div>

<h3 style="color: #48C9B0;">Métriques utilisées</h3>

<ul>
  <li>latence moyenne, p95, p99</li>
  <li>temps d’inférence pur</li>
  <li>temps de préparation des features</li>
  <li>taux d’erreur</li>
  <li>utilisation CPU / mémoire (si instrumenté)</li>
</ul>

<h3 style="color: #48C9B0;">Profiling</h3>

<pre><code>uv run python -m cProfile -o artifacts/profile_inference.prof scripts/profile_inference.py</code></pre>

<pre><code>uv run python scripts/profile_inference.py</code></pre>

<p>
Permet d’identifier précisément les fonctions les plus coûteuses.
</p>

<h3 style="color: #48C9B0;">Benchmark</h3>

<p><strong>Baseline</strong></p>

<pre><code>uv run python scripts/benchmark_inference.py --mode baseline</code></pre>

<p><strong>Version optimisée</strong></p>

<pre><code>uv run python scripts/benchmark_inference.py --mode optimized</code></pre>

<p>
👉 Comparaison directe des performances avant / après optimisation.
</p>

<h3 style="color: #48C9B0;">Optimisations mises en place</h3>

<ul>
  <li>chargement unique du modèle au démarrage</li>
  <li>mise en cache du modèle et du seuil</li>
  <li>cache des features en mémoire</li>
  <li>réduction des transformations pandas</li>
  <li>conversion en <code>float32</code></li>
  <li>séparation latence API / inférence</li>
  <li>support ONNX Runtime pour accélération CPU</li>
</ul>

<h3 style="color: #48C9B0;">Rapport d’analyse</h3>

<pre><code>docs/rapport_optimisation_inference.md</code></pre>

<hr>

<h2 id="section-14">14. Tests et qualité du code</h2>

<p>
Le projet inclut une stratégie de tests visant à garantir la robustesse du système en production.
</p>

<h3 style="color: #48C9B0;">Couverture des tests</h3>

<ul>
  <li>endpoints FastAPI</li>
  <li>chargement et cache du modèle</li>
  <li>choix dynamique du backend (sklearn / ONNX)</li>
  <li>chargement du seuil métier</li>
  <li>alignement des features</li>
  <li>services de prédiction</li>
  <li>mesure du <code>inference_latency_ms</code></li>
  <li>routes de monitoring</li>
  <li>scripts d’initialisation</li>
</ul>

<h3 style="color: #48C9B0;">Exécution des tests</h3>

<pre><code>uv run pytest</code></pre>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Note</strong><br><br>
  Certains tests forcent <code>MODEL_BACKEND=sklearn</code> afin de garantir la stabilité des tests unitaires.
</div>

<h3 style="color: #48C9B0;">Contrat des fonctions de prédiction</h3>

<pre><code>prediction, score, threshold_used, inference_latency_ms = _predict_raw(features)</code></pre>

<pre><code>scores, inference_latency_ms = _predict_scores(df)</code></pre>

<p>
Le temps d’inférence est désormais une métrique de premier niveau dans le système.
</p>

<p>
Cela permet de mesurer précisément la performance du modèle indépendamment du reste de la chaîne applicative.
</p>

<hr>

<h2 id="section-15">15. CI/CD et déploiement</h2>

<p>
Le projet est conçu pour s’intégrer dans un pipeline CI/CD automatisé via GitHub Actions afin de garantir la qualité et la reproductibilité des déploiements.
</p>

<h3 style="color: #48C9B0;">Objectifs du pipeline</h3>

<ul>
  <li>assurer la reproductibilité de l’environnement</li>
  <li>valider le code via des tests automatisés</li>
  <li>construire une image Docker versionnée</li>
  <li>préparer un déploiement continu</li>
</ul>

<h3 style="color: #48C9B0;">Pipeline CI/CD</h3>

<pre><code>push GitHub
      │
      ▼
GitHub Actions
      │
      ├── installation environnement (uv sync)
      ├── linting (optionnel mais recommandé)
      ├── tests unitaires (pytest)
      ├── build image Docker
      ├── push image (registry)
      └── déploiement (optionnel)</code></pre>

<p>
👉 Ce pipeline permet de sécuriser chaque modification avant mise en production.
</p>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Amélioration possible</strong><br><br>
  Ajouter une étape de linting avec <code>ruff</code> ou <code>flake8</code> pour garantir la qualité du code.
</div>

<hr>

<h2 id="section-16">16. Configuration</h2>

<h3 style="color: #48C9B0;">Prérequis</h3>

<ul>
  <li>Python 3.11</li>
  <li>uv (gestionnaire d’environnement rapide)</li>
  <li>Docker & Docker Compose</li>
  <li>PostgreSQL</li>
  <li>modèle sérialisé (<code>joblib</code> ou <code>ONNX</code>)</li>
</ul>

<h3 style="color: #48C9B0;">Variables d’environnement principales</h3>

<p>
La configuration du système repose sur un fichier <code>.env</code> permettant de piloter dynamiquement le comportement de l’API et du monitoring.
</p>

<pre><code># API
API_KEY=xxxxxxxxxxxxxx
DEBUG=True
DEBUG_MODEL=False

# Base de données
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/credit_api

# Source des données
DATA_DIR=data
SOURCE_CSV=application_train.csv

# Modèle
MODEL_NAME=credit_scoring_model
MODEL_VERSION=v1
MODEL_BACKEND=onnx
MODEL_PATH=artifacts/model.joblib
ONNX_MODEL_PATH=artifacts/model.onnx
THRESHOLD_PATH=artifacts/threshold.json

# Dashboard
API_URL=http://127.0.0.1:8000
DEFAULT_LIMIT=200

# Monitoring
REFERENCE_FEATURES_PATH=artifacts/monitoring/reference_features_transformed.parquet
MONITORING_DIR=artifacts/monitoring
CURRENT_WINDOW_DAYS=7

# Seuils de drift et alertes
EVIDENTLY_DRIFT_SHARE=0.50
ALERT_ON_RECALL_BELOW=0.60
ALERT_ON_LATENCY_ABOVE_MS=800
ALERT_ON_ERROR_RATE_ABOVE=0.05

# Coût métier
BUSINESS_COST_FN=10
BUSINESS_COST_FP=1

# Simulation
SIMULATION_MAX_ITEMS=200
SIMULATION_DEFAULT_ITEMS=200

# Evidently
EVIDENTLY_REPORT_PATH=artifacts/monitoring/evidently_drift_report.html
DEFAULT_EVIDENTLY_TIMEOUT=300

# Artefacts (local / Hugging Face)
ASSETS_SOURCE=auto
HF_REPO_ID=SteGONZALEZ/pret-a-depenser-artifacts
HF_REPO_TYPE=dataset
HF_TOKEN=hf_xxxxx...</code></pre>

<h3 style="color: #48C9B0;">Choix du backend modèle</h3>

<p>
Le système supporte deux moteurs d’inférence interchangeables :
</p>

<ul>
  <li><strong>sklearn/joblib</strong> : simple et fidèle au modèle d’entraînement</li>
  <li><strong>ONNX Runtime</strong> : optimisé pour l’inférence rapide</li>
</ul>

<p><strong>Configuration sklearn</strong></p>

<pre><code>MODEL_BACKEND=sklearn
MODEL_PATH=artifacts/model.joblib</code></pre>

<p><strong>Configuration ONNX</strong></p>

<pre><code>MODEL_BACKEND=onnx
ONNX_MODEL_PATH=artifacts/model.onnx</code></pre>

<p>
👉 Le changement de backend ne nécessite aucune modification du code applicatif.
</p>

<h3 style="color: #48C9B0;">Gestion des artefacts</h3>

<p>
Les modèles et artefacts peuvent être chargés dynamiquement :
</p>

<ul>
  <li><code>ASSETS_SOURCE=local</code> : utilisation des fichiers locaux</li>
  <li><code>ASSETS_SOURCE=huggingface</code> : téléchargement distant</li>
  <li><code>ASSETS_SOURCE=auto</code> : fallback intelligent</li>
</ul>

<p>
👉 Permet de simuler une architecture proche d’un environnement cloud réel.
</p>

<hr>

<h2 id="section-17">17. Lancer le projet</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Démarrer rapidement l’ensemble du système : base de données, API, modèle et dashboard, puis générer des données pour activer le monitoring.
</div>

<h3 style="color: #48C9B0;">1. Installer les dépendances</h3>

<pre><code>uv sync</code></pre>

<h3 style="color: #48C9B0;">2. Démarrer les services Docker</h3>

<pre><code>docker compose up --build</code></pre>

<h3 style="color: #48C9B0;">3. Initialiser la base de données</h3>

<pre><code>uv run python -m scripts.initialisation_tables.pipeline_db</code></pre>

<p>
👉 Crée automatiquement les tables nécessaires (prédiction + monitoring).
</p>

<h3 style="color: #48C9B0;">4. Enregistrer le modèle actif</h3>

<pre><code>uv run python -m scripts.register_model</code></pre>

<h3 style="color: #48C9B0;">5. Lancer l’API</h3>

<pre><code>uv run uvicorn app.main:app --reload</code></pre>

<h3 style="color: #48C9B0;">6. Lancer le dashboard</h3>

<pre><code>uv run streamlit run dashboard/dashboard_main.py</code></pre>

<h3 style="color: #48C9B0;">7. Générer des prédictions (simulation)</h3>

<p>
Depuis Swagger ou via requête HTTP :
</p>

<pre><code>POST /predict/simulate/real-sample?limit=200</code></pre>


<h3 style="color: #48C9B0;">8. Générer les vérités terrain</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_preview_ground_truth</code></pre>

<h3 style="color: #48C9B0;">9. Calculer les métriques de monitoring</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_monitoring_evaluation</code></pre>

<h3 style="color: #48C9B0;">10. Lancer l’analyse de dérive (drift)</h3>

<pre><code>uv run python -m scripts.manualy_run_scripts.run_evidently_monitoring</code></pre>

<h3 style="color: #48C9B0;">11. Benchmark des performances</h3>

<pre><code>uv run python scripts/benchmark_inference.py --mode baseline
uv run python scripts/benchmark_inference.py --mode optimized</code></pre>

<hr>

<h2 id="section-18">18. Livrables de la mission</h2>

<table>
  <thead>
    <tr>
      <th>Livrable demandé</th>
      <th>Emplacement dans le dépôt</th>
      <th>Statut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Historique des versions</td>
      <td><code>model_registry</code>, <code>MLflow</code>, <code>scripts/register_model.py</code></td>
      <td>Présent</td>
    </tr>
    <tr>
      <td>Scripts API</td>
      <td><code>app/api/</code>, <code>app/services/</code>, <code>app/main.py</code></td>
      <td>Présent</td>
    </tr>
    <tr>
      <td>Dockerfile</td>
      <td><code>Dockerfile</code>, <code>docker-compose.yml</code></td>
      <td>Présent</td>
    </tr>
    <tr>
      <td>Scripts de tests automatisés</td>
      <td><code>tests/</code></td>
      <td>Présent</td>
    </tr>
    <tr>
      <td>Pipeline CI/CD YAML</td>
      <td><code>.github/workflows/</code></td>
      <td>Présent</td>
    </tr>
    <tr>
      <td>Analyse du Data Drift au format notebook</td>
      <td><code>notebooks/</code> ou <code>docs/</code></td>
      <td>Présent</td>
    </tr>
    <tr>
      <td>Screenshots stockage données production</td>
      <td><code>docs/images/</code> ou <code>screenshots/</code></td>
      <td>Présent</td>
    </tr>
  </tbody>
</table>

<h2 id="section-19">19. Limites du prototype</h2>

<ul>
  <li>le monitoring dépend du volume de prédictions disponibles</li>
  <li>les métriques de performance nécessitent une vérité terrain fiable</li>
  <li>les données de production sont simulées</li>
  <li>le monitoring système (CPU / RAM) reste limité sans outil externe</li>
  <li>les optimisations sont testées dans un environnement local</li>
  <li>le drift dépend du choix de la fenêtre de référence</li>
  <li>le dashboard dépend de la disponibilité de l’API</li>
  <li>la compatibilité ONNX dépend du modèle exporté</li>
</ul>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; color: #000000; padding: 14px 18px; margin: 18px 0;">
  <strong>Lecture critique</strong><br><br>
  Ces limites reflètent un environnement de prototype contrôlé. Elles seraient levées dans une architecture production (cloud, monitoring système, orchestration).
</div>

<hr>

<h2 id="section-20">20. Pistes d’amélioration</h2>

<ul>
  <li>ajouter un orchestrateur (Airflow / Prefect) pour automatiser le monitoring</li>
  <li>planifier le calcul du drift (batch journalier)</li>
  <li>intégrer Prometheus + Grafana pour le monitoring système</li>
  <li>monitorer CPU / mémoire / throughput</li>
  <li>comparer formellement sklearn vs ONNX</li>
  <li>tester la quantification ONNX</li>
  <li>mettre en place un rollback automatique de modèle</li>
  <li>améliorer la gestion dynamique des seuils</li>
  <li>déployer sur une infrastructure cloud (AWS / GCP)</li>
  <li>ajouter des tests de non-régression sur les prédictions</li>
  <li>implémenter un système d’alertes automatisées</li>
</ul>

<hr>

<h2 id="section-21">21. Auteur</h2>

<p>
Projet réalisé par <strong>Stéphane GONZALEZ</strong> dans le cadre d’un apprentissage avancé du déploiement de modèles de machine learning, du monitoring MLOps et de l’optimisation post-déploiement.
</p>

<hr>

<h2 id="section-22">22. Licence</h2>

<p>Projet à usage éducatif.</p>