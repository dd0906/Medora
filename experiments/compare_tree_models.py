
from pathlib import Path
import time

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_processed.csv"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Benign",
    "ARP_Spoofing",
    "MQTT_DDoS_Connect_Flood",
    "Recon_Port_Scan",
]

SEED = 42


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="label")
    y = df["label"].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    # Detect constant columns using the training fold only.
    informative = X_train.nunique(dropna=False) > 1
    excluded = X_train.columns[~informative].tolist()
    kept = X_train.columns[informative].tolist()

    X_train = X_train[kept].astype("float32")
    X_val = X_val[kept].astype("float32")

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Training data contains non-finite values.")
    if not np.isfinite(X_val.to_numpy()).all():
        raise ValueError("Validation data contains non-finite values.")

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features retained: {len(kept)}")
    print(f"Constant features excluded: {excluded}")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=None,
            random_state=SEED,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=None,
            random_state=SEED,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            objective="multiclass",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight=None,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    summaries = []
    class_reports = []

    for name, model in models.items():
        print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")

        started = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = model.predict(X_val)
        prediction_seconds = time.perf_counter() - started

        report = classification_report(
            y_val,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        print(classification_report(
            y_val,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        ))

        benign = (y_val == "Benign").to_numpy()
        false_alarm_rate = (predictions[benign] != "Benign").mean()

        summaries.append({
            "model": name,
            "accuracy": accuracy_score(y_val, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "macro_f1": f1_score(
                y_val, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            ),
            "arp_precision": report["ARP_Spoofing"]["precision"],
            "arp_recall": report["ARP_Spoofing"]["recall"],
            "arp_f1": report["ARP_Spoofing"]["f1-score"],
            "benign_false_alarm_rate": false_alarm_rate,
            "training_seconds": training_seconds,
            "prediction_ms_per_1000_rows": (
                prediction_seconds / len(X_val) * 1_000_000
            ),
        })

        for class_name in CLASSES:
            class_reports.append({
                "model": name,
                "class": class_name,
                **report[class_name],
            })

        matrix = confusion_matrix(
            y_val, predictions, labels=CLASSES
        )
        pd.DataFrame(
            matrix,
            index=[f"actual_{c}" for c in CLASSES],
            columns=[f"predicted_{c}" for c in CLASSES],
        ).to_csv(
            RESULTS_DIR / f"{name.lower()}_validation_confusion.csv"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        RESULTS_DIR / "tree_models_validation_comparison.csv",
        index=False,
    )
    pd.DataFrame(class_reports).to_csv(
        RESULTS_DIR / "tree_models_class_reports.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("TREE MODEL COMPARISON")
    print("=" * 75)
    print(summary_df.round(4).to_string(index=False))
    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
