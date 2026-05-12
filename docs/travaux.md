<h2 style="color: #48C9B0;">Analyse des décisions : sécurisation et monitoring du modèle</h2>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
<strong>Objectif</strong><br><br>
Je présente les choix d’architecture et de conception mis en place afin de sécuriser le modèle de scoring crédit, garantir une traçabilité complète des prédictions et assurer un monitoring continu du comportement du système en production simulée.
</div>

---

<h3 style="color: #48C9B0;">1. Architecture et sécurisation du système</h3>

<p>
L’application a été conçue selon une architecture en couches afin de séparer clairement les responsabilités, améliorer la maintenabilité et limiter les risques liés à l’exposition du système.
</p>

<ul>
<li>une API <strong>FastAPI</strong> comme point d’entrée unique</li>
<li>des services métier dédiés à la logique applicative</li>
<li>une base <strong>PostgreSQL</strong> pour la persistance</li>
<li>un dashboard <strong>Streamlit</strong> pour la visualisation</li>
</ul>

<p>
Le dashboard n’accède jamais directement à la base de données.  
Toutes les interactions passent exclusivement par l’API afin de centraliser les contrôles et réduire la surface d’exposition.
</p>

<h4 style="color: #48C9B0;">Authentification API</h4>

<ul>
<li>authentification via une clé API (<code>X-API-Key</code>)</li>
<li>protection des endpoints sensibles</li>
<li>contrôle des accès aux routes de monitoring et de prédiction</li>
</ul>

<p>
Cette approche constitue une première couche de sécurité adaptée à un prototype MLOps.  
Elle peut évoluer facilement vers des mécanismes plus robustes comme OAuth2 ou JWT.
</p>

<h4 style="color: #48C9B0;">Gestion du modèle actif</h4>

<ul>
<li>utilisation d’une table <code>model_registry</code></li>
<li>stockage du nom, de la version et du statut du modèle</li>
<li>identification du modèle actif utilisé en production</li>
</ul>

<p>
Cette gestion permet d’assurer une traçabilité complète du modèle déployé et prépare des stratégies avancées comme :
</p>

<ul>
<li>rollback de version</li>
<li>A/B testing</li>
<li>shadow deployment</li>
</ul>

<h4 style="color: #48C9B0;">Backend d’inférence configurable</h4>

<ul>
<li><code>sklearn</code> pour la cohérence avec l’entraînement</li>
<li><code>ONNX Runtime</code> pour optimiser les performances d’inférence</li>
</ul>

<p>
Le backend est configurable via les variables d’environnement, ce qui permet de changer le moteur d’inférence sans modifier le code applicatif.
</p>

<h4 style="color: #48C9B0;">Seuil métier externalisé</h4>

<ul>
<li>stockage du seuil dans <code>threshold.json</code></li>
<li>chargement dynamique au démarrage</li>
</ul>

<p>
Cette approche permet d’ajuster les décisions métier sans réentraîner le modèle.
</p>

---

<h3 style="color: #48C9B0;">2. Traçabilité des prédictions</h3>

<p>
Chaque prédiction est enregistrée dans la table <code>prediction_logs</code> afin d’assurer une traçabilité complète des appels API.
</p>

<h4 style="color: #48C9B0;">Informations journalisées</h4>

<ul>
<li><code>request_id</code> unique</li>
<li>identifiant client</li>
<li>score prédit</li>
<li>classe prédite</li>
<li>seuil utilisé</li>
<li>nom et version du modèle</li>
<li>backend d’inférence utilisé</li>
<li>latence API</li>
<li>temps d’inférence</li>
<li>statut et erreurs éventuelles</li>
</ul>

<p>
Le <code>request_id</code> joue un rôle central : il permet de suivre une requête de bout en bout dans tout le système.
</p>

<h4 style="color: #48C9B0;">Snapshot des features</h4>

<p>
Les features utilisées lors de chaque prédiction sont également sauvegardées dans une table dédiée.
</p>

<p>
Cela permet :
</p>

<ul>
<li>de reconstruire entièrement une décision</li>
<li>d’analyser un comportement anormal</li>
<li>de réaliser des audits techniques</li>
<li>de faciliter le debug et les analyses post-mortem</li>
</ul>

---

<h3 style="color: #48C9B0;">3. Monitoring technique</h3>

<p>
Le système distingue deux métriques principales :
</p>

<ul>
<li><code>latency_ms</code> : latence globale de l’API</li>
<li><code>inference_latency_ms</code> : temps d’inférence du modèle uniquement</li>
</ul>

<h4 style="color: #48C9B0;">Latence API</h4>

<pre><code>requête
→ validation
→ preprocessing
→ prédiction
→ logging
→ réponse HTTP</code></pre>

<h4 style="color: #48C9B0;">Temps d’inférence</h4>

<pre><code>predict_proba_with_backend(features)</code></pre>

<p>
Cette séparation permet d’identifier rapidement les goulots d’étranglement :
</p>

<ul>
<li>latence élevée + inférence faible → problème applicatif ou infrastructurel</li>
<li>inférence élevée → problème lié au modèle ou au backend</li>
</ul>

<h4 style="color: #48C9B0;">Métriques techniques suivies</h4>

<ul>
<li>latence moyenne</li>
<li>P95 / P99</li>
<li>temps d’inférence</li>
<li>taux d’erreur</li>
<li>volume de prédictions</li>
</ul>

---

<h3 style="color: #48C9B0;">4. Monitoring de dérive des données</h3>

<p>
La dérive des données est analysée avec <strong>Evidently</strong> en comparant un dataset de référence avec les données issues de la production simulée.
</p>

<h4 style="color: #48C9B0;">Informations analysées</h4>

<ul>
<li>feature concernée</li>
<li>métrique statistique utilisée</li>
<li>score de drift calculé</li>
<li>seuil de détection</li>
<li>détection de dérive</li>
</ul>

<p>
L’analyse est réalisée à deux niveaux :
</p>

<ul>
<li>niveau global du dataset</li>
<li>niveau individuel par feature</li>
</ul>

<p>
Cette approche permet d’identifier précisément les variables instables susceptibles de provoquer une dégradation progressive des performances du modèle.
</p>

---

<h3 style="color: #48C9B0;">5. Monitoring métier</h3>

<p>
Les performances du modèle sont suivies avec des métriques adaptées au contexte du scoring crédit.
</p>

<ul>
<li>precision</li>
<li>recall</li>
<li>F-beta</li>
<li>ROC AUC</li>
<li>PR AUC</li>
</ul>

<h4 style="color: #48C9B0;">Intégration du coût métier</h4>

<pre><code>coût = FN × coût_FN + FP × coût_FP</code></pre>

<p>
Cette métrique permet d’aligner l’évaluation technique du modèle avec les enjeux métier réels.
</p>

<p>
Dans un contexte bancaire, un faux négatif peut représenter un risque financier bien plus important qu’un faux positif.
</p>

---

<h3 style="color: #48C9B0;">6. Système d’alertes</h3>

<p>
Un système d’alerting a été mis en place afin de détecter automatiquement les comportements anormaux.
</p>

<h4 style="color: #48C9B0;">Déclencheurs surveillés</h4>

<ul>
<li>baisse des performances ML</li>
<li>augmentation de la latence</li>
<li>hausse du taux d’erreur</li>
<li>détection de drift</li>
</ul>

<p>
Les alertes sont stockées en base et consultables depuis le dashboard.
</p>

<p>
Cette architecture est facilement extensible vers :
</p>

<ul>
<li>notifications email</li>
<li>alertes Slack</li>
<li>webhooks</li>
</ul>

---

<h3 style="color: #48C9B0;">7. Dashboard de monitoring</h3>

<p>
Le dashboard Streamlit agit comme un poste de pilotage centralisé du système.
</p>

<h4 style="color: #48C9B0;">Fonctionnalités principales</h4>

<ul>
<li>vue globale des métriques</li>
<li>analyse des performances</li>
<li>suivi des dérives</li>
<li>consultation des alertes</li>
<li>historique des prédictions</li>
<li>inspection du modèle actif</li>
</ul>

<p>
Le dashboard permet d’analyser rapidement l’état global du système et de détecter les anomalies opérationnelles.
</p>

---

<h3 style="color: #48C9B0;">8. Optimisations mises en place</h3>

<ul>
<li>mise en cache du modèle et du seuil</li>
<li>cache des données et des features</li>
<li>réduction des opérations pandas coûteuses</li>
<li>conversion des données en <code>float32</code></li>
<li>utilisation de <strong>ONNX Runtime</strong></li>
<li>benchmark et profiling des performances</li>
</ul>

<p>
Ces optimisations améliorent la latence globale et la capacité du système à supporter un volume plus important de requêtes.
</p>

---

<h3 style="color: #48C9B0;">Conclusion</h3>

<p>
Le système repose sur trois piliers principaux :
</p>

<ul>
<li><strong>sécurité</strong></li>
<li><strong>traçabilité</strong></li>
<li><strong>monitoring</strong></li>
</ul>

<p>
Chaque prédiction est entièrement traçable et le comportement du modèle est surveillé en continu grâce à des métriques techniques, statistiques et métier.
</p>

<div style="border-left: 5px solid #48C9B0; background: #f8fdfc; padding: 14px 18px; margin: 18px 0;">
<strong>Conclusion MLOps</strong><br><br>
Un modèle en production ne se limite pas à produire une prédiction.  
Il doit être sécurisé, monitoré, traçable et observable afin de garantir sa fiabilité dans le temps.
</div>