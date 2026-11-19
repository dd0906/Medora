"""Work out which features the active model leans on most for each traffic type.

Usage: python -m scripts.explain_model [--per-class 500]

For a sample of test records the model predicted as each type, this averages how
much each feature pushed the model towards that answer, using the same method as
each alert's explanation. It describes the model's behaviour, not what causes
attacks, and it is saved beside the model's evaluation so the dashboard can show it.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from detection.classifier import Detector, ModelLoadError
from detection.config import ACTIVE_MODEL_PATH, CLASSES, EVALUATION_DIR, LABEL_COLUMN, PROCESSED_DIR, SEED

logger = logging.getLogger("medora.explain_model")

TOP_FEATURES = 8
OUTPUT_NAME = "feature_effects.json"


def summarize_effects(
    detector: Detector, records: pd.DataFrame, per_class: int, seed: int = SEED
) -> dict[str, Any]:
    """Average each feature's push towards the predicted type, for each type."""
    batch = detector.detect(records.drop(columns=[LABEL_COLUMN], errors="ignore"))
    predicted = pd.Series({detection.row: detection.predicted_class for detection in batch.detections})

    types = {}
    for class_name in CLASSES:
        rows = predicted[predicted == class_name].index
        if len(rows) == 0:
            continue
        chosen = pd.Series(rows).sample(n=min(per_class, len(rows)), random_state=seed)
        effects = detector.contributions(records.loc[chosen, list(detector.feature_columns)], class_name)
        if effects is None:
            return {}
        average = effects.mean(axis=0)
        ranked = np.argsort(average)[::-1][:TOP_FEATURES]
        types[class_name] = {
            "records": int(len(chosen)),
            "features": [
                {"feature": detector.feature_columns[index], "average_effect": float(average[index])}
                for index in ranked
                if average[index] > 0
            ],
        }
    return types


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Summarize which features the active model relies on.")
    parser.add_argument("--per-class", type=int, default=500, help="Test records to sample for each type.")
    args = parser.parse_args()

    try:
        detector = Detector(ACTIVE_MODEL_PATH)
        records = pd.read_csv(PROCESSED_DIR / "test_processed.csv")
    except (ModelLoadError, OSError) as error:
        sys.exit(f"Couldn't load the model or test data: {error}")

    run_id = detector.metadata.get("evaluation_run")
    if not run_id or not (EVALUATION_DIR / run_id).is_dir():
        sys.exit("The active model has no evaluation folder to save into.")

    types = summarize_effects(detector, records, args.per_class)
    if not types:
        sys.exit("This model type has no per-feature breakdown.")
    result = {
        "model_version": detector.version,
        "method": "Tree path contributions, averaged over test records predicted as each type",
        "split": "test",
        "seed": SEED,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "types": types,
    }
    path = EVALUATION_DIR / run_id / OUTPUT_NAME
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for class_name, summary in types.items():
        top = ", ".join(item["feature"] for item in summary["features"][:3])
        logger.info("%s (%d records): %s", class_name, summary["records"], top)
    logger.info("Saved: %s", path)


if __name__ == "__main__":
    main()
