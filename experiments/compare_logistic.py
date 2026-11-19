
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    X = df.drop(columns=["label"])
    y = df["label"].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    # Determine constant features using training data only.
    informative = X_train.nunique(dropna=False) > 1
    kept_features = X_train.columns[informative].tolist()
    excluded = X_train.columns[~informative].tolist()

    X_train = X_train[kept_features].astype("float64")
    X_val = X_val[kept_features].astype("float64")

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Training data contains non-finite values.")
    if not np.isfinite(X_val.to_numpy()).all():
        raise ValueError("Validation data contains non-finite values.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight=None,
            random_state=SEED,
        )),
    ])

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features retained: {len(kept_features)}")
    print(f"Excluded constant features: {excluded}")

    print("\nTraining Logistic Regression...")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started

    predictions = model.predict(X_val)

    report = classification_report(
        y_val,
        predictions,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_val,
        predictions,
        labels=CLASSES,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    ))

    benign = (y_val == "Benign").to_numpy()
    false_alarm_rate = (
        predictions[benign] != "Benign"
    ).mean()

    metrics = {
        "model": "LogisticRegression",
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
    }

    output_path = RESULTS_DIR / "logistic_validation_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)

    print("\nSUMMARY")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
