<h1 style="text-align:center;">Rapport d’optimisation de l’inférence</h1>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Objectif</strong><br><br>
Je cherche à réduire la latence de prédiction du modèle de scoring crédit afin de garantir une utilisation fluide en production via une API temps réel.
</div>

---

<h3 style="color:#48C9B0;">Résultats</h3>

<h4 style="color:#48C9B0;">🔹 Backend sklearn (baseline)</h4>

<ul>
<li><strong>Latence moyenne :</strong> 40.59 ms</li>
<li><strong>Médiane :</strong> 36.06 ms</li>
<li><strong>P95 :</strong> 62.32 ms</li>
<li><strong>P99 :</strong> 112.70 ms</li>
</ul>

<h4 style="color:#48C9B0;">🔹 Backend ONNX (optimisé)</h4>

<ul>
<li><strong>Latence moyenne :</strong> 39.89 ms</li>
<li><strong>Médiane :</strong> 35.73 ms</li>
<li><strong>P95 :</strong> 61.52 ms</li>
<li><strong>P99 :</strong> 111.36 ms</li>
</ul>

---

<h3 style="color:#48C9B0;">Gains</h3>

<ul>
<li><strong>Gain moyen :</strong> ~1.7 %</li>
<li><strong>Gain P95 :</strong> ~1.3 %</li>
<li><strong>Gain P99 :</strong> ~1.2 %</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Même si le gain est faible, il est <strong>stable et reproductible</strong>.
</div>

---

<h3 style="color:#48C9B0;">Optimisations réalisées</h3>

<p><strong>Celles effectuées dès le début :</strong></p>
<ul>
<li>Mise en cache du modèle en mémoire</li>
<li>Chargement unique des features (suppression des I/O répétées)</li>
</ul>

<p><strong>Celles pour l’optimisation :</strong></p>
<ul>
<li>Création d'index pour les tables de stockage d'information</li> 
<li>Passage du backend modèle :
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
<li>🟢 ~6 % → logique Python + chargement données</li>
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
<li>Le système est <strong>très efficace côté CPU</strong></li>
<li>La latence est dominée par :
    <ul>
        <li>le transport HTTP</li>
        <li>la sérialisation JSON</li>
        <li>le traitement FastAPI</li>
    </ul>
</li>
<li>L’inférence modèle représente une part <strong>minoritaire</strong></li>
</ul>

---

<h3 style="color:#48C9B0;">⚠️ Impact réel de ONNX</h3>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Le gain ONNX est limité car le temps total est dominé par l’API et non par le modèle.
</div>

<ul>
<li>Temps total API ≈ 40 ms</li>
<li>Temps modèle pur ≈ quelques ms</li>
</ul>

<p>Le gain ONNX est donc <strong>dilué dans le pipeline global</strong>.</p>

---

<h3 style="color:#48C9B0;">Conclusion</h3>

<ul>
<li>Latence stable autour de <strong>40 ms</strong></li>
<li>Système compatible <strong>temps réel</strong></li>
<li>Architecture robuste et industrialisable</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
Le passage à ONNX améliore légèrement les performances, mais le principal levier d’optimisation reste l’architecture API.
</div>

---

<h3 style="color:#48C9B0;">🔮 Pistes d’amélioration</h3>

<ul>
<li>Mesurer séparément :
    <ul>
        <li><code>preprocessing_ms</code></li>
        <li><code>inference_ms</code></li>
        <li><code>total_ms</code></li>
    </ul>
</li>
<li>Profiling côté serveur</li>
<li>Batch inference</li>
<li>Réduction du coût JSON (gRPC / msgpack)</li>
</ul>

---

<h3 style="color:#48C9B0;">🎤 Phrase clé (soutenance)</h3>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
“Le passage à ONNX améliore légèrement la latence, mais le facteur dominant reste l’API elle-même. L’optimisation la plus impactante n’est pas le modèle, mais l’architecture autour.”
</div>


amelioration et diminution des logs en utilisation normale

Total runs : 300
Succès : 300
Échecs : 0

Latence client moyenne : 39.38 ms
Médiane client : 35.38 ms
P95 client : 49.42 ms
P99 client : 120.93 ms
Min client : 29.18 ms
Max client : 122.41 ms

Latence API moyenne : 7.53 ms
P95 API : 9.22 ms

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Optimisation du logging applicatif</strong><br><br>
Après réduction des logs HTTP et suppression du bruit lié aux healthchecks pendant le benchmark, la latence serveur moyenne est passée de <strong>8.71 ms</strong> à <strong>7.53 ms</strong>.<br><br>
Le P95 est passé de <strong>11.26 ms</strong> à <strong>9.22 ms</strong>, soit une amélioration d’environ <strong>18 %</strong> sur la queue de distribution.
</div>