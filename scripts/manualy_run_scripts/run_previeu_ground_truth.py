from __future__ import annotations

from collections import defaultdict
import pandas as pd

print("1. Import du script OK")

from app.core.db import SessionLocal
from app.model.model_SQLalchemy import GroundTruthLabel, PredictionLog

print("2. Imports app OK")


def preview_ground_truth_associations(limit: int = 50) -> None:
    print("3. Entrée dans preview_ground_truth_associations")

    db = SessionLocal()
    print("4. Session DB ouverte")

    try:
        gt_rows = (
            db.query(GroundTruthLabel)
            .filter(GroundTruthLabel.request_id.is_(None))
            .all()
        )
        print(f"5. Ground truth sans request_id : {len(gt_rows)}")

        pred_rows = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.client_id.is_not(None),
                PredictionLog.request_id.is_not(None),
            )
            .all()
        )
        print(f"6. Prediction logs exploitables : {len(pred_rows)}")

        preds_by_client = defaultdict(list)
        for pred in pred_rows:
            preds_by_client[pred.client_id].append(pred)

        preview_rows = []

        for gt in gt_rows:
            client_id = getattr(gt, "client_id", None)
            gt_id = getattr(gt, "id", None)

            if client_id is None:
                preview_rows.append(
                    {
                        "ground_truth_id": gt_id,
                        "client_id": None,
                        "nb_predictions_found": 0,
                        "association_status": "client_id_absent",
                        "prediction_request_ids": None,
                    }
                )
                continue

            matched_preds = preds_by_client.get(client_id, [])

            if not matched_preds:
                preview_rows.append(
                    {
                        "ground_truth_id": gt_id,
                        "client_id": client_id,
                        "nb_predictions_found": 0,
                        "association_status": "aucune_prediction_trouvee",
                        "prediction_request_ids": None,
                    }
                )
                continue

            request_ids = [getattr(pred, "request_id", None) for pred in matched_preds]
            request_ids = [rid for rid in request_ids if rid is not None]

            preview_rows.append(
                {
                    "ground_truth_id": gt_id,
                    "client_id": client_id,
                    "nb_predictions_found": len(request_ids),
                    "association_status": (
                        "association_simple"
                        if len(request_ids) == 1
                        else "plusieurs_predictions"
                    ),
                    "prediction_request_ids": " | ".join(map(str, request_ids)),
                }
            )

        preview_df = pd.DataFrame(preview_rows)
        print(f"7. DataFrame preview construit : {len(preview_df)} lignes")

        if preview_df.empty:
            print("Aucune ligne ground truth sans request_id trouvée.")
            return

        print("\n=== APERÇU DES ASSOCIATIONS ===")
        print(preview_df.head(limit).to_string(index=False))

        print("\n=== RÉSUMÉ ===")
        print(f"Total ground truth sans request_id : {len(preview_df)}")
        print(preview_df["association_status"].value_counts(dropna=False).to_string())

        preview_df.to_csv("preview_ground_truth_associations.csv", index=False)
        print("\n8. CSV exporté : preview_ground_truth_associations.csv")

    except Exception as exc:
        print(f"ERREUR DANS LE SCRIPT : {exc}")
        raise

    finally:
        db.close()
        print("9. Session DB fermée")


if __name__ == "__main__":
    print("0. Script lancé via __main__")
    preview_ground_truth_associations(limit=100)