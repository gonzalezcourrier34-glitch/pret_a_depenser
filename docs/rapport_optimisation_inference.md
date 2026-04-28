<h1 style="text-align:center;">Rapport d’optimisation de la latence</h1>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Objectif</strong><br><br>
Je cherche à réduire la latence de prédiction du modèle de scoring crédit afin de garantir une utilisation fluide en production via une API temps réel.
</div>

---

<h3 style="color:#48C9B0;">Résultats comparatifs</h3>

<h4 style="color:#48C9B0;">Backend sklearn (baseline)</h4>

<ul>
<li><strong>Latence moyenne :</strong> 40.59 ms</li>
<li><strong>Médiane :</strong> 36.06 ms</li>
<li><strong>P95 :</strong> 62.32 ms</li>
<li><strong>P99 :</strong> 112.70 ms</li>
</ul>

<h4 style="color:#48C9B0;">Backend ONNX (optimisé)</h4>

<ul>
<li><strong>Latence moyenne :</strong> 39.89 ms</li>
<li><strong>Médiane :</strong> 35.73 ms</li>
<li><strong>P95 :</strong> 61.52 ms</li>
<li><strong>P99 :</strong> 111.36 ms</li>
</ul>

---

<h3 style="color:#48C9B0;">Gains observés</h3>

<ul>
<li><strong>Gain moyen :</strong> ~1.7 %</li>
<li><strong>Gain P95 :</strong> ~1.3 %</li>
<li><strong>Gain P99 :</strong> ~1.2 %</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Le gain apporté par ONNX est faible mais <strong>stable et reproductible</strong>, ce qui valide la cohérence de l’optimisation.
</div>

---

<h3 style="color:#48C9B0;">Optimisations réalisées</h3>

<h4 style="color:#48C9B0;">Optimisations structurelles</h4>

<ul>
<li>Mise en cache du modèle en mémoire</li>
<li>Chargement unique des features (suppression des I/O répétées)</li>
</ul>

<h4 style="color:#48C9B0;">Optimisations de performance</h4>

<ul>
<li>Création d’index en base de données</li>
<li>Migration du backend modèle :
    <ul>
        <li>sklearn → ONNX Runtime</li>
    </ul>
</li>
</ul>

---

<h3 style="color:#48C9B0;">Analyse du profiling</h3>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Observation clé</strong><br><br>
100 appels → 4.257 secondes<br>
Soit ≈ <strong>42 ms par requête</strong>
</div>

<h4 style="color:#48C9B0;">Répartition du temps</h4>

<ul>
<li>🔴 ~94 % → réseau / HTTP (<code>socket.recv_into</code>)</li>
<li>🟢 ~6 % → logique applicative (Python + chargement des données)</li>
</ul>

<h4 style="color:#48C9B0;">Extrait du profiling</h4>

<pre>
socket.recv_into → 3.997 s
requests / urllib3 → ~4.1 s
chargement features → 0.073 s
</pre>

---

<h3 style="color:#48C9B0;">Interprétation</h3>

<ul>
<li>Le système est <strong>très performant côté CPU</strong></li>
<li>La latence est majoritairement due à :
    <ul>
        <li>le transport HTTP</li>
        <li>la sérialisation JSON</li>
        <li>le traitement FastAPI</li>
    </ul>
</li>
<li>L’inférence du modèle représente une part <strong>minoritaire</strong> du temps total</li>
</ul>

---

<h3 style="color:#48C9B0;">Impact réel de ONNX</h3>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Le gain apporté par ONNX est limité car le temps global est dominé par l’API et non par le modèle.
</div>

<ul>
<li>Temps total API ≈ 40 ms</li>
<li>Temps d’inférence modèle ≈ quelques millisecondes</li>
</ul>

<p>Le gain ONNX est donc <strong>dilué dans le pipeline global</strong>.</p>

---

<h3 style="color:#48C9B0;">Analyse après optimisation du logging</h3>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Observation clé</strong><br><br>
La réduction des logs applicatifs a un impact direct sur la latence serveur.
</div>

<ul>
<li><strong>Total runs :</strong> 300</li>
<li><strong>Succès :</strong> 300</li>
<li><strong>Échecs :</strong> 0</li>
</ul>

<h4 style="color:#48C9B0;">Latence côté client</h4>

<ul>
<li>Moyenne : 39.38 ms</li>
<li>Médiane : 35.38 ms</li>
<li>P95 : 49.42 ms</li>
<li>P99 : 120.93 ms</li>
<li>Min : 29.18 ms</li>
<li>Max : 122.41 ms</li>
</ul>

<h4 style="color:#48C9B0;">Latence côté API</h4>

<ul>
<li>Moyenne : 7.53 ms</li>
<li>P95 : 9.22 ms</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Optimisation du logging applicatif</strong><br><br>
Après réduction des logs HTTP et suppression du bruit lié aux healthchecks pendant les benchmarks :
<ul>
<li>Latence moyenne : <strong>8.71 ms → 7.53 ms</strong></li>
<li>P95 : <strong>11.26 ms → 9.22 ms</strong></li>
</ul>
Soit une amélioration d’environ <strong>18 %</strong> sur la queue de distribution.
</div>

---

<h3 style="color:#48C9B0;">Conclusion</h3>

<ul>
<li>Latence globale stable autour de <strong>40 ms</strong></li>
<li>Système compatible avec une utilisation <strong>temps réel</strong></li>
<li>Architecture <strong>robuste et industrialisable</strong></li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Le principal levier d’optimisation n’est pas le modèle, mais l’architecture API et les choix d’implémentation autour.
</div>

---

<h3 style="color:#48C9B0;">Pistes d’amélioration</h3>

<ul>
<li>Mesurer séparément :
    <ul>
        <li><code>preprocessing_ms</code></li>
        <li><code>inference_ms</code></li>
        <li><code>total_ms</code></li>
    </ul>
</li>
<li>Mettre en place un profiling côté serveur</li>
<li>Implémenter du batch inference</li>
<li>Réduire le coût de sérialisation (gRPC, msgpack)</li>
</ul>

---

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
“Le passage à ONNX améliore légèrement la latence, mais le facteur dominant reste l’API. L’optimisation la plus impactante réside dans l’architecture globale du système.”
</div>