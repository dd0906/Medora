from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "medora_rf.joblib"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Benign",
    "ARP_Spoofing",
    "MQTT_DDoS_Connect_Flood",
    "Recon_Port_Scan",
]


def main():
    print("Loading datasets...")
    train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
    test_df = pd.read_csv(DATA_DIR / "test_processed.csv")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Existing Random Forest model not found. "
            "Run train_model.py first."
        )

    feature_columns = [
        c for c in train_df.columns if c != "label"
    ]

    if feature_columns != [
        c for c in test_df.columns if c != "label"
    ]:
        raise ValueError("Train/test feature columns differ.")

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Testing data contains non-finite values.")

    # Load the original balanced model.
    saved = joblib.load(MODEL_PATH)
    balanced_model = saved["model"]

    # train_model.py now saves an unweighted model. Without this check, a rerun
    # would overwrite balanced_test_report.csv with mislabelled results.
    if balanced_model.get_params()["class_weight"] is None:
        raise RuntimeError(
            "medora_rf.joblib is no longer class-weighted, so it cannot stand "
            "in for the Balanced model here. Use scripts/run_protocol.py."
        )

    if saved["feature_columns"] != feature_columns:
        raise ValueError("Saved model feature order does not match.")

    # Train a comparable model without class weighting.
    print("Training unweighted Random Forest...")
    unweighted_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )
    unweighted_model.fit(X_train, y_train)

    models = {
        "Balanced": balanced_model,
        "Unweighted": unweighted_model,
    }

    all_metrics = []

    for name, model in models.items():
        print("\n" + "=" * 65)
        print(f"{name.upper()} MODEL — SUPPLIED TEST SET")
        print("=" * 65)

        predictions = model.predict(X_test)

        report = classification_report(
            y_test,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        print(classification_report(
            y_test,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        ))

        benign_mask = y_test == "Benign"
        false_alarm_rate = (
            (predictions[benign_mask] != "Benign").mean()
        )

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            ),
            "arp_precision": report["ARP_Spoofing"]["precision"],
            "arp_recall": report["ARP_Spoofing"]["recall"],
            "arp_f1": report["ARP_Spoofing"]["f1-score"],
            "benign_false_alarm_rate": false_alarm_rate,
        }

        all_metrics.append(metrics)

        pd.DataFrame(report).transpose().to_csv(
            RESULTS_DIR / f"{name.lower()}_test_report.csv"
        )

    comparison = pd.DataFrame(all_metrics)
    comparison.to_csv(
        RESULTS_DIR / "test_model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("COMPARISON SUMMARY")
    print("=" * 65)
    print(comparison.to_string(index=False))
    print("\nSaved comparison to:")
    print(RESULTS_DIR / "test_model_comparison.csv")


if __name__ == "__main__":
    main()