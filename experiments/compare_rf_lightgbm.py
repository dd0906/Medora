
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_processed.csv"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"
MODEL_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Benign",
    "ARP_Spoofing",
    "MQTT_DDoS_Connect_Flood",
    "Recon_Port_Scan",
]

SEED = 42


def main():
    print("=" * 65)
    print("MEDORA - RANDOM FOREST VS LIGHTGBM")
    print("=" * 65)

    print("\nLoading training data...")
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("The label column is missing.")

    X = df.drop(columns=["label"])
    y = df["label"].astype(str)

    if set(y.unique()) != set(CLASSES):
        raise ValueError("Unexpected labels found in the dataset.")

    # Same stratified validation split used in earlier comparisons.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    # Remove constant features using the training fold only.
    informative = X_train.nunique(dropna=False) > 1
    feature_columns = X_train.columns[informative].tolist()
    excluded = X_train.columns[~informative].tolist()

    X_train = X_train[feature_columns].astype("float32")
    X_val = X_val[feature_columns].astype("float32")

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Training data contains non-finite values.")

    if not np.isfinite(X_val.to_numpy()).all():
        raise ValueError("Validation data contains non-finite values.")

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features retained: {len(feature_columns)}")
    print(f"Excluded constant features: {excluded}")

    models = {
        "RandomForest": RandomForestClassifier(
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
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight=None,
            importance_type="gain",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    summaries = []
    class_reports = []

    for name, model in models.items():
        print("\n" + "=" * 65)
        print(f"TRAINING {name}")
        print("=" * 65)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X_val)
        prediction_seconds = time.perf_counter() - start

        report = classification_report(
            y_val,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        print("\nClassification report:")
        print(classification_report(
            y_val,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        ))

        benign_mask = (y_val == "Benign").to_numpy()
        benign_false_alarm_rate = (
            predictions[benign_mask] != "Benign"
        ).mean()

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_val, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "macro_precision": precision_score(
                y_val, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            ),
            "macro_recall": recall_score(
                y_val, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            ),
            "macro_f1": f1_score(
                y_val, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_val, predictions,
                labels=CLASSES,
                average="weighted",
                zero_division=0,
            ),
            "arp_precision": report["ARP_Spoofing"]["precision"],
            "arp_recall": report["ARP_Spoofing"]["recall"],
            "arp_f1": report["ARP_Spoofing"]["f1-score"],
            "benign_false_alarm_rate": benign_false_alarm_rate,
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "prediction_ms_per_1000_rows": (
                prediction_seconds / len(X_val) * 1000 * 1000
            ),
        }

        summaries.append(metrics)

        for class_name in CLASSES:
            class_reports.append({
                "model": name,
                "class": class_name,
                **report[class_name],
            })

        cm = confusion_matrix(
            y_val, predictions, labels=CLASSES
        )

        pd.DataFrame(
            cm,
            index=[f"actual_{c}" for c in CLASSES],
            columns=[f"predicted_{c}" for c in CLASSES],
        ).to_csv(
            RESULTS_DIR / f"{name.lower()}_validation_confusion.csv"
        )

        importance = pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance.to_csv(
            RESULTS_DIR / f"{name.lower()}_feature_importance.csv",
            index=False,
        )

        # Save LightGBM as a separate candidate; do not overwrite RF.
        if name == "LightGBM":
            joblib.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns,
                    "class_order": CLASSES,
                },
                MODEL_DIR / "medora_lightgbm_candidate.joblib",
            )

    summary_df = pd.DataFrame(summaries)
    reports_df = pd.DataFrame(class_reports)

    summary_path = (
        RESULTS_DIR / "rf_lightgbm_validation_comparison.csv"
    )
    report_path = RESULTS_DIR / "rf_lightgbm_class_reports.csv"

    summary_df.to_csv(summary_path, index=False)
    reports_df.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("VALIDATION COMPARISON")
    print("=" * 65)

    display_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "arp_precision",
        "arp_recall",
        "arp_f1",
        "benign_false_alarm_rate",
        "training_seconds",
        "prediction_ms_per_1000_rows",
    ]

    print(summary_df[display_columns].round(4).to_string(index=False))

    print("\nSaved summary:", summary_path)
    print("Saved class reports:", report_path)
    print("\nComparison complete.")


if __name__ == "__main__":
    main()
