"""
Script de réconciliation ground truth ↔ prediction_logs

Objectif
--------
Associer les request_id aux lignes de ground_truth_labels
en se basant sur client_id.

- Si plusieurs requêtes existent pour un client :
    -> on duplique la ligne ground truth
- Si une seule requête :
    -> on met à jour directement
"""

from app.core.db import SessionLocal
from app.model.model_SQLalchemy import GroundTruthLabel, PredictionLog


def reconcile_fast(db):
    print("🚀 Mode rapide activé")

    # 1. Charger toutes les prédictions UNE SEULE FOIS
    preds = db.query(PredictionLog).all()

    # 2. Index par client_id
    preds_by_client = {}
    for p in preds:
        if p.client_id not in preds_by_client:
            preds_by_client[p.client_id] = []
        preds_by_client[p.client_id].append(p)

    # 3. Charger les ground truth
    gt_rows = (
        db.query(GroundTruthLabel)
        .filter(GroundTruthLabel.request_id.is_(None))
        .all()
    )

    print(f"GT à traiter : {len(gt_rows)}")

    inserted = 0
    updated = 0

    for gt in gt_rows:
        client_id = gt.client_id

        preds = preds_by_client.get(client_id, [])

        if not preds:
            continue

        if len(preds) == 1:
            gt.request_id = preds[0].request_id
            updated += 1
        else:
            for pred in preds:
                new_gt = GroundTruthLabel(
                    request_id=pred.request_id,
                    client_id=client_id,
                    **{
                        col: getattr(gt, col)
                        for col in gt.__table__.columns.keys()
                        if col not in ["id", "request_id", "client_id"]
                    }
                )
                db.add(new_gt)
                inserted += 1

            db.delete(gt)

    db.commit()

    print("✅ Terminé")
    print(f"Updated : {updated}")
    print(f"Inserted: {inserted}")