<h2 style="color: #48C9B0;">Analyse des décisions : sécurisation et monitoring du modèle</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Je présente les choix d’architecture et de conception mis en place pour sécuriser le modèle de scoring crédit, garantir la traçabilité complète des prédictions et assurer un monitoring continu de son comportement en production simulée.
</div>

---

<h3 style="color: #48C9B0;">1. Sécurisation du système</h3>

<p>
L’application est organisée selon une architecture en couches clairement séparées afin de limiter les risques, isoler les responsabilités et faciliter la maintenabilité :
</p>

<ul>
  <li>une API <strong>FastAPI</strong> comme point d’entrée unique</li>
  <li>des services métier pour la logique (prédiction, monitoring, dérive)</li>
  <li>une base <strong>PostgreSQL</strong> pour la persistance</li>
  <li>un dashboard <strong>Streamlit</strong> pour la visualisation</li>
</ul>

<p>
Le dashboard ne communique jamais directement avec la base de données. Toutes les interactions transitent par l’API, ce qui permet de centraliser les contrôles d’accès et de réduire la surface d’exposition.
</p>

<h4 style="color: #48C9B0;">Authentification API</h4>

<ul>
  <li>authentification par clé API via l’en-tête <code>X-API-Key</code></li>
  <li>protection des endpoints critiques (prédiction, monitoring)</li>
</ul>

<p>
Ce mécanisme, bien que simple, constitue une première couche de sécurité adaptée à un prototype. Il peut être étendu vers des solutions plus robustes (OAuth2, JWT).
</p>

<h4 style="color: #48C9B0;">Gestion du modèle actif</h4>

<ul>
  <li>utilisation d’une table <code>model_registry</code></li>
  <li>stockage du nom, de la version et du statut du modèle</li>
</ul>

<p>
Cette approche garantit la traçabilité du modèle en production et permet d’envisager des stratégies avancées comme le rollback, le versioning ou le déploiement multi-modèles.
</p>

<h4 style="color: #48C9B0;">Backend d’inférence configurable</h4>

<ul>
  <li><code>sklearn / joblib</code> pour la cohérence avec l’entraînement</li>
  <li><code>ONNX Runtime</code> pour optimiser les performances</li>
</ul>

<p>
Le backend est configurable via les variables d’environnement (<code>.env</code>), ce qui permet de modifier le moteur d’inférence sans impact sur le code applicatif.
</p>

<h4 style="color: #48C9B0;">Seuil métier externalisé</h4>

<ul>
  <li>seuil de décision stocké dans <code>threshold.json</code></li>
</ul>

<p>
Ce choix permet d’ajuster dynamiquement les décisions métier sans réentraîner le modèle, facilitant ainsi l’adaptation aux contraintes business.
</p>

---

<h3 style="color: #48C9B0;">2. Traçabilité des prédictions</h3>

<p>
Chaque prédiction est persistée dans la table <code>prediction_logs</code>, garantissant une traçabilité complète.
</p>

<ul>
  <li>identifiant unique <code>request_id</code></li>
  <li>identifiant client</li>
  <li>score et décision finale</li>
  <li>seuil utilisé</li>
  <li>modèle et version</li>
  <li>latence totale et temps d’inférence</li>
  <li>statut et erreurs éventuelles</li>
</ul>

<p>
Le <code>request_id</code> permet de suivre une requête de bout en bout dans le système (API → modèle → base → monitoring).
</p>

<h4 style="color: #48C9B0;">Snapshot des features</h4>

<p>
Les variables utilisées lors de la prédiction sont stockées dans une table dédiée.
</p>

<p>
Cela permet de reconstruire intégralement le contexte d’une décision, ce qui est essentiel pour :
</p>

<ul>
  <li>l’audit réglementaire</li>
  <li>le debug</li>
  <li>l’analyse post-mortem</li>
</ul>

---

<h3 style="color: #48C9B0;">3. Monitoring technique</h3>

<p>
Deux métriques principales sont suivies :
</p>

<ul>
  <li><code>latency_ms</code> : latence globale de l’API</li>
  <li><code>inference_latency_ms</code> : temps d’inférence du modèle</li>
</ul>

<h4 style="color: #48C9B0;">Latence API</h4>

<pre><code>requête reçue
+ validation
+ preprocessing
+ prédiction
+ logging
+ réponse HTTP</code></pre>

<h4 style="color: #48C9B0;">Temps d’inférence</h4>

<pre><code>predict_proba_with_backend(features)</code></pre>

<p>
Cette séparation permet d’isoler précisément les goulots d’étranglement :
</p>

<ul>
  <li>latence élevée + inférence faible → problème infrastructure/API</li>
  <li>inférence élevée → problème modèle/backend</li>
</ul>

<p>
👉 Cette approche est essentielle pour diagnostiquer efficacement les problèmes de performance en production.
</p>

---

<h3 style="color: #48C9B0;">4. Monitoring de dérive</h3>

<p>
La détection de dérive est réalisée avec Evidently en comparant les données de référence et les données de production.
</p>

<ul>
  <li>nom de la feature</li>
  <li>métrique calculée</li>
  <li>valeur mesurée</li>
  <li>seuil de référence</li>
  <li>indicateur de dérive</li>
</ul>

<p>
Deux niveaux d’analyse sont réalisés :
</p>

<ul>
  <li>global (dataset complet)</li>
  <li>par feature</li>
</ul>

<p>
Cela permet d’identifier précisément les variables responsables d’un drift et d’anticiper une dégradation du modèle.
</p>

---

<h3 style="color: #48C9B0;">5. Monitoring métier</h3>

<p>
Les performances sont suivies via des métriques adaptées au contexte du crédit :
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
Cette métrique permet d’intégrer les enjeux business, en tenant compte du fait que certaines erreurs (FN) sont beaucoup plus coûteuses que d’autres.
</p>

---

<h3 style="color: #48C9B0;">6. Système d’alertes</h3>

<p>
Un système d’alerting est mis en place pour détecter automatiquement les anomalies :
</p>

<ul>
  <li>baisse des performances (ex : recall)</li>
  <li>latence excessive</li>
  <li>augmentation du taux d’erreur</li>
  <li>dérive des données</li>
</ul>

<p>
Les alertes sont stockées en base et exposées via le dashboard.  
Elles peuvent être étendues vers des systèmes de notification (email, Slack) en environnement réel.
</p>

---

<h3 style="color: #48C9B0;">7. Dashboard de monitoring</h3>

<p>
Le dashboard Streamlit fournit une interface de pilotage du système.
</p>

<ul>
  <li>visualisation des métriques globales</li>
  <li>analyse des performances et de la latence</li>
  <li>suivi des dérives</li>
  <li>consultation des alertes</li>
  <li>inspection du modèle actif</li>
</ul>

<p>
Il permet une lecture rapide et opérationnelle de l’état du système.
</p>

---

<h3 style="color: #48C9B0;">8. Optimisation et performance</h3>

<p>
Plusieurs optimisations ont été mises en place :
</p>

<ul>
  <li>cache du modèle et du seuil</li>
  <li>cache des données et features</li>
  <li>réduction des opérations pandas</li>
  <li>conversion en <code>float32</code></li>
  <li>utilisation de ONNX Runtime</li>
  <li>profiling et benchmarks</li>
</ul>

<p>
Ces optimisations permettent de réduire la latence et d’améliorer la scalabilité du système.
</p>

---

<h3 style="color: #48C9B0;">Conclusion</h3>

<p>
Le système repose sur trois piliers fondamentaux :
</p>

<ul>
  <li><strong>sécurité</strong> : contrôle des accès et gestion du modèle</li>
  <li><strong>traçabilité</strong> : journalisation complète des prédictions</li>
  <li><strong>monitoring</strong> : suivi technique et métier</li>
</ul>

<p>
Chaque prédiction est traçable de bout en bout et le comportement du modèle est surveillé en continu.
</p>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
<strong>Conclusion MLOps</strong><br><br>
Un modèle en production ne se limite pas à prédire : il doit être sécurisé, traçable et monitoré pour garantir sa fiabilité dans le temps.
</div>