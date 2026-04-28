<h2 style="color: #48C9B0;">Analyse des décisions : sécurisation et monitoring du modèle</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Je cherche à expliquer les choix d’architecture et de conception que j’ai mis en place pour sécuriser le modèle de scoring crédit, assurer la traçabilité des prédictions et surveiller son comportement en production simulée.
</div>

---

<h3 style="color: #48C9B0;">1. Sécurisation du système</h3>

<p>
J’ai structuré mon application en plusieurs couches bien séparées afin de limiter les risques et clarifier les responsabilités :
</p>

<ul>
  <li>une API FastAPI qui sert de point d’entrée unique</li>
  <li>des services métier pour la logique (prédiction, monitoring, dérive)</li>
  <li>une base PostgreSQL pour la persistance</li>
  <li>un dashboard Streamlit pour la visualisation</li>
</ul>

<p>
Le dashboard ne communique jamais directement avec la base de données. Toutes les interactions passent par l’API, ce qui me permet de centraliser les contrôles d’accès et de réduire les risques d’exposition.
</p>

<h4 style="color: #48C9B0;">Authentification API</h4>

<ul>
  <li>j’utilise une clé API transmise dans l’en-tête <code>X-API-Key</code></li>
  <li>je protège les endpoints sensibles comme la prédiction et le monitoring</li>
</ul>

<p>
Ce mécanisme reste simple mais il est suffisant dans le cadre d’un prototype sécurisé.
</p>

<h4 style="color: #48C9B0;">Gestion du modèle actif</h4>

<ul>
  <li>j’utilise une table <code>model_registry</code></li>
  <li>je stocke le nom, la version et le statut du modèle</li>
</ul>

<p>
Cela me permet de savoir précisément quel modèle est en production et d’éviter toute ambiguïté. Cette approche facilite aussi les évolutions futures comme un rollback ou la gestion de plusieurs modèles.
</p>

<h4 style="color: #48C9B0;">Backend d’inférence configurable</h4>

<ul>
  <li><code>sklearn / joblib</code> pour rester fidèle à l’entraînement</li>
  <li><code>ONNX Runtime</code> pour améliorer les performances</li>
</ul>

<p>
Le choix du backend est piloté par configuration via un fichier <code>.env</code>. Je peux donc changer le moteur d’inférence sans modifier le code.
</p>

<h4 style="color: #48C9B0;">Seuil métier externalisé</h4>

<ul>
  <li>le seuil de décision est stocké dans <code>threshold.json</code></li>
</ul>

<p>
Cela me permet d’ajuster les décisions métier sans avoir à réentraîner le modèle.
</p>

---

<h3 style="color: #48C9B0;">2. Traçabilité des prédictions</h3>

<p>
Chaque prédiction est enregistrée dans la table <code>prediction_logs</code>.
</p>

<p>
Pour chaque requête, je stocke notamment :
</p>

<ul>
  <li>un identifiant unique <code>request_id</code></li>
  <li>l’identifiant client</li>
  <li>le score et la décision finale</li>
  <li>le seuil utilisé</li>
  <li>le modèle et sa version</li>
  <li>la latence totale et le temps d’inférence</li>
  <li>le statut et les éventuelles erreurs</li>
</ul>

<p>
Le <code>request_id</code> me permet de suivre une prédiction de bout en bout dans le système.
</p>

<h4 style="color: #48C9B0;">Snapshot des features</h4>

<p>
Je stocke également les variables utilisées lors de la prédiction dans une table dédiée.
</p>

<p>
Cela me permet de reconstruire exactement le contexte d’une décision, ce qui est essentiel pour l’audit et le debug.
</p>

---

<h3 style="color: #48C9B0;">3. Monitoring technique</h3>

<p>
Je mesure deux métriques principales :
</p>

<ul>
  <li>la latence globale de l’API (<code>latency_ms</code>)</li>
  <li>le temps d’inférence du modèle (<code>inference_latency_ms</code>)</li>
</ul>

<h4 style="color: #48C9B0;">Latence API</h4>

<pre><code>requête reçue
+ validation
+ préparation des données
+ prédiction
+ logging
+ réponse HTTP</code></pre>

<h4 style="color: #48C9B0;">Temps d’inférence</h4>

<pre><code>predict_proba_with_backend(features)</code></pre>

<p>
Cette séparation me permet d’identifier l’origine d’un problème de performance :
</p>

<ul>
  <li>si la latence est élevée mais l’inférence faible, le problème vient de l’API</li>
  <li>si l’inférence est élevée, le problème vient du modèle ou du backend</li>
</ul>

---

<h3 style="color: #48C9B0;">4. Monitoring de dérive</h3>

<p>
J’utilise Evidently pour détecter les dérives entre les données de référence et les données en production.
</p>

<p>
Les résultats sont stockés dans la table <code>drift_metrics</code> avec :
</p>

<ul>
  <li>le nom de la feature</li>
  <li>la métrique calculée</li>
  <li>la valeur mesurée</li>
  <li>le seuil de référence</li>
  <li>un indicateur de dérive</li>
</ul>

<p>
Je distingue deux niveaux d’analyse :
</p>

<ul>
  <li>global (ensemble du dataset)</li>
  <li>par variable</li>
</ul>

<p>
Cela me permet d’identifier précisément les variables responsables d’une dérive.
</p>

---

<h3 style="color: #48C9B0;">5. Monitoring métier</h3>

<p>
Je mesure les performances du modèle en comparant les prédictions avec les vérités terrain.
</p>

<ul>
  <li>precision</li>
  <li>recall</li>
  <li>F1 et F-beta</li>
  <li>ROC AUC</li>
  <li>PR AUC</li>
</ul>

<h4 style="color: #48C9B0;">Coût métier</h4>

<pre><code>coût = FN * coût_FN + FP * coût_FP</code></pre>

<p>
Cette métrique me permet de prendre en compte le fait que certaines erreurs sont plus coûteuses que d’autres dans le contexte du crédit.
</p>

---

<h3 style="color: #48C9B0;">6. Système d’alertes</h3>

<p>
Je génère automatiquement des alertes dans les cas suivants :
</p>

<ul>
  <li>baisse du recall</li>
  <li>latence trop élevée</li>
  <li>taux d’erreur important</li>
  <li>dérive des données</li>
</ul>

<p>
Ces alertes sont stockées dans une table dédiée et consultables via le dashboard.
</p>

---

<h3 style="color: #48C9B0;">7. Dashboard de monitoring</h3>

<p>
Le dashboard Streamlit me permet de piloter le système.
</p>

<p>
Je peux notamment :
</p>

<ul>
  <li>visualiser les métriques globales</li>
  <li>analyser les performances et la latence</li>
  <li>observer les dérives</li>
  <li>consulter les alertes</li>
  <li>inspecter le modèle actif</li>
</ul>

<p>
Cela me donne une vision rapide et claire de l’état du système.
</p>

---

<h3 style="color: #48C9B0;">8. Optimisation et performance</h3>

<p>
Pour améliorer les performances, j’ai mis en place plusieurs optimisations :
</p>

<ul>
  <li>mise en cache du modèle et du seuil</li>
  <li>mise en cache des données et des features</li>
  <li>réduction des transformations pandas</li>
  <li>conversion des données en <code>float32</code></li>
  <li>utilisation de ONNX Runtime</li>
  <li>profiling et benchmarks</li>
</ul>

<p>
Ces optimisations me permettent de réduire la latence et d’améliorer la scalabilité du système.
</p>

---

<h3 style="color: #48C9B0;">Conclusion</h3>

<p>
Mon système repose sur trois éléments principaux :
</p>

<ul>
  <li>la sécurité, avec un contrôle des accès et une gestion claire du modèle</li>
  <li>la traçabilité, grâce à une journalisation complète des prédictions</li>
  <li>le monitoring, avec des indicateurs techniques et métier</li>
</ul>

<p>
Ce travail me permet de simuler un environnement MLOps proche d’une mise en production réelle.
</p>