# Rapport d’optimisation de la latence

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Objectif</strong><br><br>
L’objectif de cette analyse est d’optimiser la latence de prédiction du système de scoring crédit afin de garantir une utilisation fluide, fiable et exploitable dans un contexte d’API temps réel.
</div>

---

<h3 style="color:#48C9B0;">Contexte métier</h3>

<p>
Dans un système de scoring crédit, le temps de réponse influence directement l’expérience utilisateur et la capacité du système à être intégré dans des workflows temps réel.
</p>

<p>
Une faible latence permet notamment :
</p>

<ul>
<li>une validation rapide des demandes de crédit</li>
<li>une meilleure fluidité des interfaces utilisateur</li>
<li>une intégration plus simple avec des partenaires ou services externes</li>
<li>une réduction du temps d’attente côté front-end</li>
</ul>

<p>
Dans ce type d’usage, une latence inférieure à <strong>50 ms</strong> est généralement considérée comme très performante pour une API de machine learning.
</p>

---

<h3 style="color:#48C9B0;">Objectifs de l’analyse</h3>

<p>
Cette étude vise à :
</p>

<ul>
<li>mesurer précisément les performances réelles du système</li>
<li>identifier les goulots d’étranglement</li>
<li>évaluer l’impact du backend ONNX Runtime</li>
<li>distinguer la latence applicative du temps d’inférence pur</li>
<li>déterminer les principaux leviers d’optimisation</li>
</ul>

---

<h3 style="color:#48C9B0;">Conditions de benchmark</h3>

<ul>
<li>300 requêtes HTTP successives</li>
<li>modèle chargé en mémoire (<em>warm start</em>)</li>
<li>environnement local sous Docker</li>
<li>aucune contrainte réseau externe simulée</li>
<li>API FastAPI exécutée localement</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Important</strong><br><br>
Ces résultats sont obtenus dans un environnement contrôlé.  
Une validation complète en production nécessiterait des tests de charge complémentaires et des mesures sous trafic réel.
</div>

---

<h3 style="color:#48C9B0;">Périmètre des métriques</h3>

<p>
Plusieurs niveaux de mesure ont été utilisés afin d’éviter toute confusion entre les différents temps observés.
</p>

<ul>
<li>
<strong>Latence client end-to-end</strong> :
temps mesuré depuis le script de benchmark incluant :
<ul>
<li>appel HTTP</li>
<li>traitement serveur</li>
<li>sérialisation JSON</li>
<li>retour de réponse</li>
</ul>
</li>

<li>
<strong>Latence API interne</strong> :
temps mesuré côté serveur incluant :
<ul>
<li>validation FastAPI</li>
<li>préprocessing</li>
<li>inférence</li>
<li>logging</li>
<li>construction de la réponse</li>
</ul>
</li>

<li>
<strong>Temps d’inférence</strong> :
temps strictement passé dans le backend modèle :
<ul>
<li><code>predict_proba()</code></li>
<li>ou session ONNX Runtime</li>
</ul>
</li>
</ul>

<p>
Cette séparation permet d’identifier précisément les goulots d’étranglement du système.
</p>

---

<h3 style="color:#48C9B0;">Résultats comparatifs</h3>

| Backend | Latence moyenne | P95 | P99 |
|---|---:|---:|---:|
| sklearn / joblib | 40.59 ms | 62.32 ms | 112.70 ms |
| ONNX Runtime | 39.89 ms | 61.52 ms | 111.36 ms |

---

<h3 style="color:#48C9B0;">Analyse des gains</h3>

<ul>
<li><strong>Gain moyen :</strong> ~1.7 %</li>
<li><strong>Gain P95 :</strong> ~1.3 %</li>
<li><strong>Gain P99 :</strong> ~1.2 %</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Le gain apporté par ONNX Runtime est faible mais stable et reproductible.  
Cela valide techniquement l’optimisation, même si son impact reste limité dans le pipeline global.
</div>

<p>
Le modèle étant déjà relativement léger, le temps d’inférence représente une faible part de la latence totale.  
Dans des architectures plus lourdes ou avec des modèles plus complexes, l’impact de ONNX Runtime serait généralement plus important.
</p>

<p>
L’intérêt principal de cette optimisation réside également dans la capacité à rendre le backend d’inférence interchangeable sans modifier l’architecture applicative.
</p>

---

<h3 style="color:#48C9B0;">Optimisations mises en place</h3>

<h4 style="color:#48C9B0;">Optimisations structurelles</h4>

<ul>
<li>chargement unique du modèle au démarrage</li>
<li>mise en cache du modèle en mémoire</li>
<li>mise en cache du seuil métier</li>
<li>chargement unique des données et features</li>
<li>réduction des accès disque</li>
</ul>

<h4 style="color:#48C9B0;">Optimisations de performance</h4>

<ul>
<li>migration du backend sklearn vers ONNX Runtime</li>
<li>réduction des transformations pandas</li>
<li>conversion des features en <code>float32</code></li>
<li>réduction des logs applicatifs coûteux</li>
<li>indexation des tables PostgreSQL</li>
</ul>

---

<h3 style="color:#48C9B0;">Analyse du profiling</h3>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Observation principale</strong><br><br>
100 appels HTTP → 4.257 secondes  
Soit environ <strong>42 ms par requête</strong>
</div>

<h4 style="color:#48C9B0;">Répartition du temps observé</h4>

<ul>
<li>~94 % → attente de réponse HTTP (<code>socket.recv_into</code>)</li>
<li>~6 % → logique applicative (préprocessing + modèle)</li>
</ul>

<p>
Le profiling réalisé côté client montre que la majorité du temps est passée à attendre la réponse HTTP complète.
</p>

<p>
Cette attente inclut :
</p>

<ul>
<li>le traitement serveur</li>
<li>la sérialisation JSON</li>
<li>les échanges HTTP locaux</li>
<li>la construction de la réponse</li>
</ul>

---

<h3 style="color:#48C9B0;">Décomposition de la latence API</h3>

<pre><code>Latence totale API
│
├── transport HTTP
├── validation FastAPI
├── preprocessing features
├── inférence modèle
├── logging PostgreSQL
└── sérialisation JSON</code></pre>

<p>
Cette décomposition permet de distinguer les coûts réellement liés au modèle de ceux liés à l’architecture applicative.
</p>

---

<h3 style="color:#48C9B0;">Interprétation des résultats</h3>

<ul>
<li>Le système est performant côté calcul</li>
<li>Le modèle réalise l’inférence en quelques millisecondes</li>
<li>La majorité de la latence provient de la chaîne applicative</li>
</ul>

<p>
Les principaux coûts observés proviennent :
</p>

<ul>
<li>du framework HTTP</li>
<li>de la sérialisation JSON</li>
<li>du pipeline applicatif</li>
<li>du transport de données</li>
<li>du logging</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Dans ce benchmark, la latence observée dépend davantage de l’architecture globale du système que du modèle lui-même.
</div>

<p>
Cette analyse montre qu’une optimisation MLOps ne consiste pas uniquement à accélérer le modèle, mais à instrumenter et optimiser l’ensemble de la chaîne d’inférence.
</p>

---

<h3 style="color:#48C9B0;">Impact réel de ONNX Runtime</h3>

<ul>
<li>Temps total API ≈ 40 ms</li>
<li>Temps d’inférence modèle ≈ quelques millisecondes</li>
</ul>

<p>
Le gain ONNX reste donc mécaniquement dilué dans la latence globale du pipeline.
</p>

<p>
Cela confirme que le modèle n’est pas le principal facteur limitant dans cette architecture.
</p>

---

<h3 style="color:#48C9B0;">Optimisation du logging</h3>

| Métrique | Avant optimisation | Après optimisation |
|---|---:|---:|
| Latence moyenne API | 8.71 ms | 7.53 ms |
| P95 API | 11.26 ms | 9.22 ms |

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
La réduction des logs applicatifs excessifs a permis une amélioration d’environ <strong>18 %</strong> sur les latences élevées.
</div>

<p>
Cette optimisation illustre l’importance des traitements périphériques dans la performance globale d’une API ML.
</p>

---

<h3 style="color:#48C9B0;">Objectifs de performance (SLO)</h3>

<ul>
<li>P95 cible : &lt; 70 ms</li>
<li>Latence moyenne cible : &lt; 45 ms</li>
<li>Disponibilité cible : &gt; 99 %</li>
</ul>

<p>
Les objectifs de latence sont respectés dans cet environnement de benchmark local.
</p>

<p>
La disponibilité reste ici un objectif théorique : une validation réelle nécessiterait un monitoring longue durée et des tests de charge plus importants.
</p>

---

<h3 style="color:#48C9B0;">Conclusion</h3>

<ul>
<li>Latence stable autour de 40 ms</li>
<li>Système compatible avec un usage temps réel</li>
<li>Architecture suffisamment instrumentée pour analyser les performances</li>
<li>Monitoring capable d’identifier les goulots d’étranglement</li>
<li>Optimisation ONNX validée techniquement</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Conclusion clé :</strong><br><br>
Le modèle est déjà très rapide.  
Le principal levier d’optimisation se situe désormais dans l’architecture API, la sérialisation des échanges et l’infrastructure globale du système.
</div>

---

<h3 style="color:#48C9B0;">Pistes d’amélioration</h3>

<ul>
<li>mesurer séparément :
<ul>
<li><code>preprocessing_ms</code></li>
<li><code>inference_ms</code></li>
<li><code>postprocessing_ms</code></li>
</ul>
</li>

<li>mettre en place du batch inference</li>
<li>réduire les coûts de sérialisation (gRPC, msgpack)</li>
<li>réaliser des tests de charge intensifs</li>
<li>ajouter un monitoring système complet (CPU, RAM, throughput)</li>
<li>étudier l’autoscaling avec Kubernetes</li>
<li>instrumenter Prometheus + Grafana</li>
</ul>

---

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Synthèse finale</strong><br><br>
L’optimisation du modèle apporte un gain mesurable, mais l’analyse montre que l’essentiel de la latence provient de la chaîne applicative complète.  
Le travail réalisé a donc permis non seulement d’optimiser l’inférence, mais surtout de rendre le système observable, traçable et analysable dans une logique MLOps complète.
</div>