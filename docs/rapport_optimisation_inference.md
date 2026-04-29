<h1 style="text-align:center;">Rapport d’optimisation de la latence</h1>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; color:#000; padding:14px 18px; margin:18px 0;">
<strong>Objectif</strong><br><br>
Je cherche à optimiser la latence de prédiction de mon modèle de scoring crédit afin de garantir une utilisation fluide et fiable dans un contexte d’API temps réel.
</div>

---

<h3 style="color:#48C9B0;">Contexte métier</h3>

<p>
Dans un système de scoring crédit, la latence joue un rôle critique dans l’expérience utilisateur et la prise de décision.
Une réponse rapide permet :
</p>

<ul>
<li>une validation quasi instantanée des demandes de crédit</li>
<li>une expérience utilisateur fluide</li>
<li>une intégration dans des systèmes temps réel (front web, partenaires, APIs)</li>
</ul>

<p>
👉 Une latence inférieure à <strong>50 ms</strong> est considérée comme <strong>très performante</strong> pour ce type d’usage.
</p>

---

<h3 style="color:#48C9B0;">Conditions de benchmark</h3>

<ul>
<li>300 requêtes HTTP successives</li>
<li>modèle chargé en mémoire (warm start)</li>
<li>environnement local (Docker)</li>
<li>pas de contrainte réseau simulée</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; padding:14px 18px; margin:18px 0;">
<strong>Important</strong><br><br>
Ces résultats sont obtenus en conditions contrôlées. Une validation en production nécessiterait des tests de charge complémentaires.
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

<h3 style="color:#48C9B0;">Analyse des gains</h3>

<ul>
<li><strong>Gain moyen :</strong> ~1.7 %</li>
<li><strong>Gain P95 :</strong> ~1.3 %</li>
<li><strong>Gain P99 :</strong> ~1.2 %</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; padding:14px 18px; margin:18px 0;">
Le gain apporté par ONNX est faible mais <strong>stable et reproductible</strong>.  
Cela confirme la validité de l’optimisation, même si son impact reste limité dans le pipeline global.
</div>

---

<h3 style="color:#48C9B0;">Optimisations mises en place</h3>

<h4 style="color:#48C9B0;">Optimisations structurelles</h4>

<ul>
<li>Mise en cache du modèle en mémoire</li>
<li>Chargement unique des features</li>
<li>Réduction des accès disque</li>
</ul>

<h4 style="color:#48C9B0;">Optimisations de performance</h4>

<ul>
<li>Indexation de la base de données</li>
<li>Migration du backend : sklearn → ONNX Runtime</li>
</ul>

---

<h3 style="color:#48C9B0;">Analyse du profiling</h3>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; padding:14px 18px; margin:18px 0;">
<strong>Observation clé</strong><br><br>
100 appels → 4.257 secondes  
Soit ≈ <strong>42 ms par requête</strong>
</div>

<h4 style="color:#48C9B0;">Répartition du temps</h4>

<ul>
<li>🔴 ~94 % → réseau / HTTP (<code>socket.recv_into</code>)</li>
<li>🟢 ~6 % → logique applicative (préprocessing + modèle)</li>
</ul>

---

<h3 style="color:#48C9B0;">Interprétation</h3>

<ul>
<li>Le système est <strong>très performant côté calcul</strong></li>
<li>L’inférence modèle est <strong>très rapide</strong></li>
<li>La latence est dominée par :</li>
<ul>
<li>transport HTTP</li>
<li>sérialisation JSON</li>
<li>framework API</li>
</ul>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; padding:14px 18px; margin:18px 0;">
👉 En production, la latence est généralement dominée par l’infrastructure, et non par le modèle.
</div>

---

<h3 style="color:#48C9B0;">Impact réel de ONNX</h3>

<ul>
<li>Temps total API ≈ 40 ms</li>
<li>Temps modèle ≈ quelques ms</li>
</ul>

<p>
👉 Le gain ONNX est donc <strong>dilué dans le pipeline global</strong>.
</p>

---

<h3 style="color:#48C9B0;">Optimisation du logging</h3>

<ul>
<li><strong>Latence moyenne :</strong> 8.71 → 7.53 ms</li>
<li><strong>P95 :</strong> 11.26 → 9.22 ms</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; padding:14px 18px; margin:18px 0;">
Suppression des logs excessifs → amélioration d’environ <strong>18 %</strong> sur les latences élevées.
</div>

---

<h3 style="color:#48C9B0;">Objectifs de performance (SLO)</h3>

<ul>
<li>P95 < 50 ms</li>
<li>Moyenne < 40 ms</li>
<li>Disponibilité > 99 %</li>
</ul>

<p>
✔ Tous les objectifs sont respectés
</p>

---

<h3 style="color:#48C9B0;">Conclusion</h3>

<ul>
<li>Latence stable autour de <strong>40 ms</strong></li>
<li>Système compatible <strong>temps réel</strong></li>
<li>Performance conforme aux standards industriels</li>
</ul>

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; padding:14px 18px; margin:18px 0;">
<strong>Conclusion clé :</strong><br><br>
Le modèle est déjà très rapide.  
Le principal levier d’optimisation se situe dans l’architecture API et l’infrastructure.
</div>

---

<h3 style="color:#48C9B0;">Pistes d’amélioration</h3>

<ul>
<li>Mesurer finement :
    <ul>
        <li><code>preprocessing_ms</code></li>
        <li><code>inference_ms</code></li>
        <li><code>postprocessing_ms</code></li>
    </ul>
</li>
<li>Batch inference</li>
<li>Réduction de la sérialisation (gRPC, msgpack)</li>
<li>Load testing</li>
<li>Autoscaling (Kubernetes)</li>
</ul>

---

<div style="border-left: 6px solid #48C9B0; background:#f8fdfc; padding:14px 18px; margin:18px 0;">
“L’optimisation du modèle apporte un gain mesurable, mais l’impact principal sur la latence provient de l’architecture globale du système.”
</div>