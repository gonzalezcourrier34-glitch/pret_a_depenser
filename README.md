<h1 align="center">Système de scoring crédit avec API, monitoring et pipeline MLOps</h1>

<p align="center">
  <strong>Déploiement d’un modèle de machine learning en production avec traçabilité, monitoring et dashboard</strong>
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
  <li><a href="#section-5">5. Pipeline de préparation des données</a></li>
  <li><a href="#section-6">6. Base de données et persistance</a></li>
  <li><a href="#section-7">7. API de prédiction</a></li>
  <li><a href="#section-8">8. Monitoring MLOps</a></li>
  <li><a href="#section-9">9. Dashboard</a></li>
  <li><a href="#section-10">10. Tables de prédiction</a></li>
  <li><a href="#section-11">11. Tables de monitoring</a></li>
  <li><a href="#section-12">12. Structure du projet</a></li>
  <li><a href="#section-13">13. Tests</a></li>
  <li><a href="#section-14">14. Infrastructure</a></li>
  <li><a href="#section-15">15. Limites du prototype</a></li>
  <li><a href="#section-16">16. Pistes d’amélioration</a></li>
  <li><a href="#section-17">17. Configuration</a></li>
  <li><a href="#section-18">18. Lancer le projet</a></li>
  <li><a href="#section-19">19. Auteur</a></li>
  <li><a href="#section-20">20. Licence</a></li>
</ul>

<hr>

<h2 id="section-1">1. Quick Start</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Lancer rapidement le projet dans un environnement local reproductible, créer les tables PostgreSQL, préparer les features, démarrer l’API puis visualiser les résultats dans le dashboard.
</div>

<pre><code>git clone repo_url
cd project
cp .env.example .env
docker compose up --build</code></pre>

<p><strong>Puis :</strong></p>

<ol>
  <li>initialiser la base PostgreSQL</li>
  <li>exécuter le pipeline de préparation des données</li>
  <li>démarrer l’API FastAPI</li>
  <li>ouvrir le dashboard Streamlit</li>
  <li>faire une prédiction ou consulter le monitoring</li>
</ol>

<div style="border-left: 5px solid #F5B041; background: #FFF8E8; padding: 14px 18px; margin: 18px 0;">
  <strong>Important</strong><br><br>
  Le système de monitoring n’est réellement utile que si les prédictions sont journalisées en base et si les tables de suivi sont créées au préalable.
</div>

<hr>

<h2 id="section-2">2. Présentation du projet</h2>

<p>
Dans ce projet, j’ai conçu un système complet de <strong>scoring crédit en production</strong>, capable de servir un modèle de machine learning via une API, de stocker les prédictions dans PostgreSQL et de suivre le comportement du modèle à travers un dispositif de monitoring MLOps.
</p>

<p>
L’objectif est de ne pas se limiter à la modélisation, mais d’implémenter une chaîne plus réaliste de mise en production, avec :
</p>

<ul>
  <li>une API de prédiction sécurisée</li>
  <li>une base PostgreSQL pour stocker les données techniques et métier</li>
  <li>un pipeline de préparation de features aligné sur le modèle</li>
  <li>un historique des prédictions</li>
  <li>un système de monitoring dédié aux dérives, performances et alertes</li>
  <li>un dashboard de consultation et de pilotage</li>
</ul>

<p>
Ce projet m’a permis de travailler sur une logique MLOps complète, allant de la structuration des données jusqu’au suivi d’un modèle en environnement applicatif. Les tables de prédiction et de monitoring sont créées par des scripts dédiés, avec index SQL pour faciliter les lectures côté dashboard. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
</p>

<hr>

<h2 id="section-3">3. Objectifs du projet</h2>

<h3 style="color: #48C9B0;">Contexte</h3>

<p>
Dans le cadre de ce projet de scoring crédit, l’enjeu n’est pas seulement de produire une probabilité de défaut, mais de concevoir un système capable de <strong>servir, tracer et surveiller</strong> le modèle dans le temps.
</p>

<p>
Un modèle peut perdre en qualité une fois déployé si les données changent, si les comportements clients évoluent ou si les distributions observées en production s’éloignent de celles du jeu d’entraînement.
</p>

<h3 style="color: #48C9B0;">Problématique</h3>

<p>
La difficulté consiste donc à construire une architecture capable de :
</p>

<ul>
  <li>réaliser des prédictions de manière fiable</li>
  <li>conserver les entrées et sorties utiles</li>
  <li>suivre les versions de modèle</li>
  <li>mesurer les dérives de données</li>
  <li>suivre les métriques de performance</li>
  <li>générer des alertes exploitables</li>
</ul>

<h3 style="color: #48C9B0;">Objectif du projet</h3>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Avec ce projet, j’ai cherché à mettre en place un système de scoring crédit déployable, structuré et traçable, capable de journaliser les prédictions, de stocker les features observées en production et de suivre les signaux de dérive et de performance dans un cadre MLOps.
</div>

<ul>
  <li>servir un modèle via FastAPI</li>
  <li>persister les données de prédiction dans PostgreSQL</li>
  <li>structurer un pipeline de préparation des features</li>
  <li>créer une couche de monitoring dédiée</li>
  <li>préparer un dashboard d’analyse</li>
  <li>séparer proprement les tables de prédiction et les tables de monitoring</li>
</ul>

<hr>

<h2 id="section-4">4. Architecture du système</h2>

<p>
L’architecture repose sur plusieurs briques distinctes, chacune ayant un rôle clair dans la chaîne de mise en production.
</p>

<pre><code>CSV / Tables RAW
      │
      ▼
Pipeline de préparation des données
      │
      ├── création des tables brutes
      ├── chargement des CSV
      ├── agrégations temporaires
      ├── création des features
      ├── nettoyage
      ├── enrichissement
      ├── validation du schéma modèle
      └── création de la table finale
      │
      ▼
Modèle entraîné (joblib)
      │
      ▼
API FastAPI
      │
      ├── prédiction
      ├── log des requêtes
      ├── snapshot des features
      └── accès aux métadonnées modèle
      │
      ▼
PostgreSQL
      │
      ├── tables de prédiction
      ├── tables de monitoring
      └── tables techniques intermédiaires
      │
      ▼
Dashboard Streamlit</code></pre>

<p>
Le pipeline global est orchestré par un script dédié qui enchaîne les étapes de création des features, nettoyage, enrichissement, validation et création des tables de prédiction et de monitoring. :contentReference[oaicite:2]{index=2}
</p>

<hr>

<h2 id="section-5">5. Pipeline de préparation des données</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Transformer les données brutes Home Credit en une table de features compatible avec le modèle entraîné, tout en conservant une logique claire, reproductible et découpée en étapes.
</div>

<p>
Le pipeline de préparation repose sur plusieurs scripts spécialisés. L’idée est de séparer la donnée brute, les agrégations intermédiaires, les enrichissements métier et la validation finale.
</p>

<pre><code>1. Création des tables RAW PostgreSQL
2. Chargement des CSV dans la base
3. Création des tables temporaires d’agrégation
4. Création de la table de features client
5. Nettoyage de la table de features
6. Enrichissement avec ratios, logs, flags et interactions
7. Vérification des features attendues par le modèle
8. Création de la table finale prête pour la prédiction
9. Création des tables de prédiction
10. Création des tables de monitoring</code></pre>

<p>
La structure brute de chargement des CSV vers PostgreSQL est explicitement définie dans un mapping fichier → table, avec insertion par chunks. :contentReference[oaicite:3]{index=3}
</p>

<p>
Les tables temporaires d’agrégation servent de couche intermédiaire entre les tables RAW et la table finale de features client, notamment pour les historiques bureau, POS, cartes de crédit, paiements et previous applications. :contentReference[oaicite:4]{index=4}
</p>

<h3 style="color: #48C9B0;">Enrichissement métier</h3>

<p>
Une étape spécifique ajoute des variables dérivées utiles au modèle : conversions temporelles, flags de valeurs manquantes, ratios financiers, logs, compteurs documentaires et interactions entre scores externes. Cette logique est matérialisée dans la table <code>features_client_test_enriched</code>. :contentReference[oaicite:5]{index=5}
</p>

<ul>
  <li><code>AGE_YEARS</code></li>
  <li><code>EMPLOYED_YEARS</code></li>
  <li><code>CREDIT_INCOME_RATIO</code></li>
  <li><code>ANNUITY_INCOME_RATIO</code></li>
  <li><code>OVER_INDEBTED_40</code></li>
  <li><code>LOG_INCOME</code></li>
  <li><code>DOC_COUNT</code></li>
  <li><code>EXT_SOURCES_MEAN</code></li>
</ul>

<hr>

<h2 id="section-6">6. Base de données et persistance</h2>

<p>
J’ai utilisé <strong>PostgreSQL</strong> comme base centrale du projet afin de disposer d’une persistance robuste pour :
</p>

<ul>
  <li>les données sources</li>
  <li>les tables intermédiaires</li>
  <li>les features finales</li>
  <li>les logs de prédiction</li>
  <li>les métriques de monitoring</li>
  <li>les alertes</li>
</ul>

<p>
Les tables RAW sont créées pour refléter fidèlement les schémas des fichiers CSV sources, avec conservation stricte des noms de colonnes d’origine. :contentReference[oaicite:6]{index=6}
</p>

<p>
Cette séparation entre données brutes, features, prédictions et monitoring permet d’éviter le mélange des responsabilités et rend le système plus maintenable.
</p>

<hr>

<h2 id="section-7">7. API de prédiction</h2>

<p>
L’API a été développée avec <strong>FastAPI</strong> afin d’exposer le modèle sous forme de service HTTP.
</p>

<p>
Elle a pour rôle de :
</p>

<ul>
  <li>recevoir les données d’entrée</li>
  <li>vérifier la sécurité d’accès</li>
  <li>appeler le modèle</li>
  <li>retourner un score et une classe prédite</li>
  <li>journaliser les informations utiles en base</li>
</ul>

<h3 style="color: #48C9B0;">Endpoints attendus</h3>

<ul>
  <li><code>/health</code> pour vérifier l’état du service</li>
  <li><code>/predict</code> pour réaliser une prédiction</li>
</ul>

<p>
Le projet est pensé pour tracer chaque appel de l’API, stocker les scores, conserver les données d’entrée et préparer le suivi des vérités terrain. C’est précisément le rôle des tables de prédiction créées dans le script dédié. :contentReference[oaicite:7]{index=7}
</p>

<hr>

<h2 id="section-8">8. Monitoring MLOps</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Suivre le comportement du modèle en production, détecter les dérives de données, mesurer les performances observées et centraliser les alertes dans une couche de monitoring dédiée.
</div>

<p>
Le monitoring repose sur plusieurs tables spécialisées :
</p>

<ul>
  <li><code>model_registry</code></li>
  <li><code>feature_store_monitoring</code></li>
  <li><code>drift_metrics</code></li>
  <li><code>evaluation_metrics</code></li>
  <li><code>alerts</code></li>
</ul>

<p>
Le script de création de ces tables met en place une couche de monitoring structurée, avec index SQL pour accélérer les requêtes de lecture. :contentReference[oaicite:8]{index=8}
</p>

<h3 style="color: #48C9B0;">Axes suivis</h3>

<ul>
  <li><strong>versioning modèle</strong> : suivi des versions déployées et du modèle actif</li>
  <li><strong>snapshot de features</strong> : valeurs observées en production</li>
  <li><strong>dérive</strong> : métriques calculées par variable et par fenêtre temporelle</li>
  <li><strong>performance</strong> : ROC AUC, PR AUC, précision, rappel, F1, F-beta, coût métier</li>
  <li><strong>alertes</strong> : événements anormaux centralisés dans une table dédiée</li>
</ul>

<hr>

<h2 id="section-9">9. Dashboard</h2>

<p>
Le projet prévoit un dashboard <strong>Streamlit</strong> permettant de piloter les usages principaux du système.
</p>

<p>
Il doit permettre de :
</p>

<ul>
  <li>faire une prédiction sur un client ou un JSON</li>
  <li>consulter l’historique des prédictions</li>
  <li>afficher les snapshots de features</li>
  <li>visualiser les métriques de drift</li>
  <li>consulter les métriques d’évaluation</li>
  <li>lire les alertes ouvertes ou résolues</li>
</ul>

<p>
L’idée est de proposer une vue exploitable à la fois pour l’usage démonstratif, le pilotage MLOps et la soutenance.
</p>

<hr>

<h2 id="section-10">10. Tables de prédiction</h2>

<p>
Les tables de prédiction ont été conçues pour tracer les appels d’inférence et préparer le suivi post-déploiement. Elles sont créées par un script dédié. :contentReference[oaicite:9]{index=9}
</p>

<h3 style="color: #48C9B0;"><code>prediction_logs</code></h3>

<p>
Cette table journalise chaque appel de l’API avec :
</p>

<ul>
  <li>un identifiant de requête</li>
  <li>un client éventuel</li>
  <li>le nom et la version du modèle</li>
  <li>la classe prédite</li>
  <li>le score</li>
  <li>le seuil utilisé</li>
  <li>la latence</li>
  <li>les données d’entrée et de sortie</li>
  <li>le code de statut et les erreurs éventuelles</li>
</ul>

<h3 style="color: #48C9B0;"><code>ground_truth_labels</code></h3>

<p>
Cette table stocke les vérités terrain observées a posteriori, afin de comparer les prédictions aux résultats réels.
</p>

<h3 style="color: #48C9B0;"><code>prediction_features_snapshot</code></h3>

<p>
Cette table conserve les features observées au moment de l’inférence, une ligne par feature, pour faciliter les analyses futures de dérive ou d’audit.
</p>

<hr>

<h2 id="section-11">11. Tables de monitoring</h2>

<p>
Les tables de monitoring sont séparées des tables de prédiction afin de distinguer clairement :
</p>

<ul>
  <li>les événements opérationnels liés à l’inférence</li>
  <li>les indicateurs de suivi et de gouvernance du modèle</li>
</ul>

<h3 style="color: #48C9B0;"><code>model_registry</code></h3>

<p>
Conserve l’historique des modèles, leurs métadonnées, leur stage, leur chemin source, leur date de déploiement et leur statut actif. :contentReference[oaicite:10]{index=10}
</p>

<h3 style="color: #48C9B0;"><code>feature_store_monitoring</code></h3>

<p>
Stocke les snapshots de features observées en production pour faciliter le suivi statistique.
</p>

<h3 style="color: #48C9B0;"><code>drift_metrics</code></h3>

<p>
Enregistre les métriques de dérive calculées par feature, par fenêtre de référence et fenêtre courante.
</p>

<h3 style="color: #48C9B0;"><code>evaluation_metrics</code></h3>

<p>
Stocke les métriques de performance agrégées sur une période donnée.
</p>

<h3 style="color: #48C9B0;"><code>alerts</code></h3>

<p>
Centralise les alertes générées par les règles de drift, de performance ou de monitoring.
</p>

<hr>

<h2 id="section-12">12. Structure du projet</h2>

<pre><code>project/
│
├── app/
│   ├── main.py
│   ├── core/
│   ├── routes/
│   ├── services/
│   ├── model/
│   └── dashboard/
│
├── scripts/
│   ├── create_raw_tables.py
│   ├── load_csv_to_postgres.py
│   ├── create_temp_feature_tables.py
│   ├── create_prediction_tables.py
│   ├── create_monitoring_tables.py
│   ├── enrich_features_table.py
│   └── verify_model_features.py
│
├── artifacts/
├── data/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md</code></pre>

<p>
Le pipeline de création des tables et de préparation des données est porté par plusieurs scripts spécialisés, ce qui rend l’ensemble plus lisible et plus maintenable. 
</p>

<hr>

<h2 id="section-13">13. Tests</h2>

<p>
Le projet a vocation à intégrer des tests automatisés pour fiabiliser :
</p>

<ul>
  <li>les endpoints FastAPI</li>
  <li>les services de préparation des données</li>
  <li>la cohérence des features avec le modèle</li>
  <li>les accès à la base PostgreSQL</li>
  <li>les requêtes utilisées par le dashboard</li>
</ul>

<p>
Une étape importante du projet consiste à vérifier l’alignement exact entre la table SQL de features et la liste attendue par le modèle. Le script de vérification compare les colonnes disponibles à la liste de features attendues et signale les manquants et les extras. :contentReference[oaicite:12]{index=12}
</p>

<hr>

<h2 id="section-14">14. Infrastructure</h2>

<h3 style="color: #48C9B0;">PostgreSQL</h3>

<p>
PostgreSQL constitue la base centrale de persistance du projet.
</p>

<h3 style="color: #48C9B0;">FastAPI</h3>

<p>
FastAPI expose le modèle via une API HTTP sécurisée et documentable.
</p>

<h3 style="color: #48C9B0;">Streamlit</h3>

<p>
Streamlit sert de couche d’interface pour consulter les prédictions et les indicateurs de monitoring.
</p>

<h3 style="color: #48C9B0;">Docker</h3>

<p>
L’application peut être conteneurisée pour obtenir un environnement reproductible et plus proche d’un déploiement réel.
</p>

<h3 style="color: #48C9B0;">MLflow</h3>

<p>
MLflow peut être utilisé pour la partie suivi expérimental et historique de modélisation, dans une logique cohérente avec l’approche MLOps du projet.
</p>

<hr>

<h2 id="section-15">15. Limites du prototype</h2>

<ul>
  <li>le monitoring dépend de la qualité et du volume des données réellement journalisées</li>
  <li>certaines métriques ne prennent pleinement sens qu’une fois la vérité terrain disponible</li>
  <li>la détection de drift nécessite une bonne définition de la fenêtre de référence</li>
  <li>le dashboard dépend fortement de la qualité des requêtes SQL et de la disponibilité des tables</li>
  <li>la logique d’alerte peut encore être enrichie</li>
</ul>

<hr>

<h2 id="section-16">16. Pistes d’amélioration</h2>

<ul>
  <li>ajout d’un monitoring automatisé planifié</li>
  <li>calcul régulier de métriques de drift en batch</li>
  <li>alimentation automatique des vérités terrain</li>
  <li>gestion plus fine des modèles actifs et des versions</li>
  <li>déploiement cloud</li>
  <li>pipeline CI/CD complet</li>
  <li>tests plus complets sur l’API et le dashboard</li>
  <li>alertes temps réel</li>
</ul>

<hr>

<h2 id="section-17">17. Configuration</h2>

<h3 style="color: #48C9B0;">Prérequis</h3>

<ul>
  <li>Python 3.11</li>
  <li>PostgreSQL</li>
  <li>Docker et Docker Compose si conteneurisation</li>
  <li>un modèle sérialisé au format joblib</li>
</ul>

<h3 style="color: #48C9B0;">Variables d’environnement</h3>

<p>
Créer un fichier <code>.env</code> avec les variables nécessaires au projet :
</p>

<pre><code>API_KEY=votre_token_api
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/credit_api
MODEL_PATH=artifacts/model.joblib
MODEL_NAME=credit_scoring_model
MODEL_VERSION=v1</code></pre>

<p>
Les scripts de création des tables, de chargement des CSV et d’enrichissement dépendent tous de la variable <code>DATABASE_URL</code>. 
</p>

<hr>

<h2 id="section-18">18. Lancer le projet</h2>

<h3 style="color: #48C9B0;">1. Créer les tables RAW</h3>

<pre><code>uv run python scripts/create_raw_tables.py</code></pre>

<h3 style="color: #48C9B0;">2. Charger les CSV dans PostgreSQL</h3>

<pre><code>uv run python scripts/load_csv_to_postgres.py</code></pre>

<h3 style="color: #48C9B0;">3. Exécuter le pipeline de préparation</h3>

<pre><code>uv run python scripts/pipeline_all.py</code></pre>

<p>
Selon ton arborescence réelle, adapte le nom du script d’orchestration si nécessaire. Le pipeline observé dans le projet enchaîne notamment la création des features, l’enrichissement, la vérification du schéma et la création des tables de prédiction et de monitoring. :contentReference[oaicite:14]{index=14}
</p>

<h3 style="color: #48C9B0;">4. Lancer l’API</h3>

<pre><code>uv run uvicorn app.main:app --reload</code></pre>

<h3 style="color: #48C9B0;">5. Lancer le dashboard</h3>

<pre><code>uv run streamlit run dashboard/dashboard.py</code></pre>

<h3 style="color: #48C9B0;">6. Vérifier les services</h3>

<ul>
  <li><strong>API FastAPI :</strong> <code>http://localhost:8000</code></li>
  <li><strong>Swagger :</strong> <code>http://localhost:8000/docs</code></li>
  <li><strong>Dashboard Streamlit :</strong> <code>http://localhost:8501</code></li>
</ul>

<hr>

<h2>Résumé rapide</h2>

<pre><code>git clone repo_url
cd project
cp .env.example .env
uv sync
uv run python scripts/create_raw_tables.py
uv run python scripts/load_csv_to_postgres.py
uv run python scripts/pipeline_all.py
uv run uvicorn app.main:app --reload</code></pre>

<p>Puis :</p>

<ol>
  <li>ouvrir la documentation API</li>
  <li>tester l’endpoint <code>/predict</code></li>
  <li>ouvrir le dashboard</li>
  <li>consulter les tables de prédiction et de monitoring</li>
</ol>

<hr>

<h2 id="section-19">19. Auteur</h2>

<p>
Projet réalisé par <strong>Stéphane GONZALEZ</strong> dans le cadre d’un apprentissage autour du déploiement de modèles de machine learning, du monitoring MLOps et des architectures applicatives de scoring crédit.
</p>

<hr>

<h2 id="section-20">20. Licence</h2>

<p>Usage éducatif.</p>