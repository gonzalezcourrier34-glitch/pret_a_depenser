<h2 style="color: #48C9B0;">Analyse des décisions : sécurisation et monitoring du modèle</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
  <strong>Objectif</strong><br><br>
  Je présente les choix d’architecture et de conception que j’ai mis en place afin de sécuriser le modèle de scoring crédit, garantir une traçabilité complète des prédictions et assurer un monitoring continu de son comportement en production simulée.
</div>

---

<h3 style="color: #48C9B0;">1. Sécurisation du système</h3>

<p>
J’ai structuré l’application selon une architecture en couches afin de séparer clairement les responsabilités, limiter les risques et faciliter la maintenabilité du système :
</p>

<ul>
  <li>une API <strong>FastAPI</strong> comme point d’entrée unique</li>
  <li>des services métier pour encapsuler la logique (prédiction, monitoring, dérive)</li>
  <li>une base <strong>PostgreSQL</strong> pour la persistance</li>
  <li>un dashboard <strong>Streamlit</strong> pour la visualisation</li>
</ul>

<p>
Le dashboard ne communique jamais directement avec la base de données. Toutes les interactions passent par l’API, ce qui permet de centraliser les contrôles et de réduire la surface d’exposition.
</p>

<h4 style="color: #48C9B0;">Authentification API</h4>

<ul>
  <li>authentification via une clé API (<code>X-API-Key</code>)</li>
  <li>protection des endpoints critiques (prédiction, monitoring)</li>
</ul>

<p>
Ce mécanisme constitue une première couche de sécurité adaptée à un prototype. Il est facilement extensible vers des solutions plus robustes comme OAuth2 ou JWT.
</p>

<h4 style="color: #48C9B0;">Gestion du modèle actif</h4>

<ul>
  <li>utilisation d’une table <code>model_registry</code></li>
  <li>stockage du nom, de la version et du statut du modèle</li>
</ul>

<p>
Cela me permet de tracer précisément quel modèle est utilisé en production et d’envisager des stratégies comme le rollback ou le versioning.
</p>

<h4 style="color: #48C9B0;">Backend d’inférence configurable</h4>

<ul>
  <li><code>sklearn</code> pour la cohérence avec l’entraînement</li>
  <li><code>ONNX Runtime</code> pour optimiser les performances</li>
</ul>

<p>
Le backend est configurable via les variables d’environnement, ce qui permet de changer le moteur d’inférence sans modifier le code applicatif.
</p>

<h4 style="color: #48C9B0;">Seuil métier externalisé</h4>

<ul>
  <li>stockage du seuil dans <code>threshold.json</code></li>
</ul>

<p>
Ce choix permet d’ajuster les décisions métier sans réentraîner le modèle, ce qui apporte une grande flexibilité opérationnelle.
</p>

---

<h3 style="color: #48C9B0;">2. Traçabilité des prédictions</h3>

<p>
Chaque prédiction est enregistrée dans la table <code>prediction_logs</code>, ce qui me permet d’assurer une traçabilité complète.
</p>

<ul>
  <li><code>request_id</code> unique</li>
  <li>identifiant client</li>
  <li>score et décision</li>
  <li>seuil utilisé</li>
  <li>modèle et version</li>
  <li>latence et temps d’inférence</li>
  <li>statut et erreurs</li>
</ul>

<p>
Le <code>request_id</code> est central : il me permet de suivre une requête de bout en bout dans tout le système.
</p>

<h4 style="color: #48C9B0;">Snapshot des features</h4>

<p>
Je stocke également les features utilisées lors de chaque prédiction dans une table dédiée.
</p>

<p>
Cela me permet de reconstruire entièrement le contexte d’une décision, ce qui est essentiel pour :
</p>

<ul>
  <li>l’audit</li>
  <li>le debug</li>
  <li>l’analyse post-mortem</li>
</ul>

---

<h3 style="color: #48C9B0;">3. Monitoring technique</h3>

<p>
Je distingue deux métriques principales :
</p>

<ul>
  <li><code>latency_ms</code> : latence globale API</li>
  <li><code>inference_latency_ms</code> : temps du modèle</li>
</ul>

<h4 style="color: #48C9B0;">Latence API</h4>

<pre><code>requête
→ validation
→ preprocessing
→ prédiction
→ logging
→ réponse</code></pre>

<h4 style="color: #48C9B0;">Temps d’inférence</h4>

<pre><code>predict_proba_with_backend(features)</code></pre>

<p>
Cette séparation me permet d’identifier rapidement les goulots d’étranglement :
</p>

<ul>
  <li>latence élevée + inférence faible → problème API/infrastructure</li>
  <li>inférence élevée → problème modèle</li>
</ul>

---

<h3 style="color: #48C9B0;">4. Monitoring de dérive</h3>

<p>
Je détecte la dérive des données avec Evidently en comparant un dataset de référence avec les données issues de la production simulée.
</p>

<ul>
  <li>feature analysée</li>
  <li>métrique statistique</li>
  <li>score calculé</li>
  <li>seuil</li>
  <li>détection de drift</li>
</ul>

<p>
L’analyse est réalisée à deux niveaux :
</p>

<ul>
  <li>global (dataset)</li>
  <li>par variable</li>
</ul>

<p>
Cela me permet d’identifier précisément les variables instables et d’anticiper une dégradation du modèle.
</p>

---

<h3 style="color: #48C9B0;">5. Monitoring métier</h3>

<p>
Je suis les performances avec des métriques adaptées au contexte crédit :
</p>

<ul>
  <li>precision</li>
  <li>recall</li>
  <li>F-beta</li>
  <li>ROC AUC</li>
  <li>PR AUC</li>
</ul>

<h4 style="color: #48C9B0;">Intégration du coût métier</h4>

<pre><code>coût = FN * coût_FN + FP * coût_FP</code></pre>

<p>
Cette métrique me permet d’aligner l’évaluation du modèle avec les enjeux business.
</p>

---

<h3 style="color: #48C9B0;">6. Système d’alertes</h3>

<p>
J’ai mis en place un système d’alerting basé sur plusieurs signaux :
</p>

<ul>
  <li>baisse des performances</li>
  <li>latence anormale</li>
  <li>augmentation des erreurs</li>
  <li>dérive des données</li>
</ul>

<p>
Les alertes sont stockées en base et consultables dans le dashboard.  
Ce système est extensible vers des notifications externes (email, Slack).
</p>

---

<h3 style="color: #48C9B0;">7. Dashboard de monitoring</h3>

<p>
Le dashboard Streamlit me permet de piloter le système de manière centralisée.
</p>

<ul>
  <li>vue globale des métriques</li>
  <li>analyse des performances</li>
  <li>suivi du drift</li>
  <li>consultation des alertes</li>
  <li>inspection du modèle actif</li>
</ul>

<p>
Il constitue une interface opérationnelle pour analyser rapidement l’état du système.
</p>

---

<h3 style="color: #48C9B0;">8. Optimisations mises en place</h3>

<ul>
  <li>cache du modèle et du seuil</li>
  <li>cache des données et features</li>
  <li>réduction des opérations coûteuses</li>
  <li>conversion en <code>float32</code></li>
  <li>utilisation de ONNX Runtime</li>
  <li>benchmark et profiling</li>
</ul>

<p>
Ces optimisations améliorent la latence et la scalabilité globale.
</p>

---

<h3 style="color: #48C9B0;">Conclusion</h3>

<p>
Le système repose sur trois piliers :
</p>

<ul>
  <li><strong>sécurité</strong></li>
  <li><strong>traçabilité</strong></li>
  <li><strong>monitoring</strong></li>
</ul>

<p>
Chaque prédiction est entièrement traçable et le comportement du modèle est surveillé en continu.
</p>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
<strong>Conclusion MLOps</strong><br><br>
Un modèle en production ne se limite pas à prédire.  
Il doit être sécurisé, traçable et monitoré afin de garantir sa fiabilité dans le temps.
</div>