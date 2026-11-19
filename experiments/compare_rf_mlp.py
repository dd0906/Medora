from pathlib import Path
import gc

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label"])
    y = df["label"]

    # Use the same stratified validation split for both models.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    del X, y, df
    gc.collect()

    # Identify constant columns using training data only.
    informative = X_train.nunique(dropna=False) > 1
    dropped = X_train.columns[~informative].tolist()
    feature_columns = X_train.columns[informative].tolist()

    X_train = X_train[feature_columns].astype("float32")
    X_val = X_val[feature_columns].astype("float32")

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features retained: {len(feature_columns)}")
    print(f"Constant features excluded: {dropped or 'None'}")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                batch_size=2048,
                learning_rate_init=0.001,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=42,
            )),
        ]),
    }

    results = []
    class_reports = []

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name}")
        print("=" * 60)

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

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

        benign_mask = y_val == "Benign"
        false_alarm_rate = (
            (predictions[benign_mask] != "Benign").mean()
        )

        results.append({
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
        })

        for class_name in CLASSES:
            class_reports.append({
                "model": name,
                "class": class_name,
                **report[class_name],
            })

        del model
        gc.collect()

    results_df = pd.DataFrame(results)
    reports_df = pd.DataFrame(class_reports)

    results_path = RESULTS_DIR / "rf_mlp_validation_comparison.csv"
    reports_path = RESULTS_DIR / "rf_mlp_class_reports.csv"

    results_df.to_csv(results_path, index=False)
    reports_df.to_csv(reports_path, index=False)

    print("\n" + "=" * 60)
    print("VALIDATION COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print(f"\nSaved summary: {results_path}")
    print(f"Saved class reports: {reports_path}")
    print("\nCOMPARISON COMPLETE")


if __name__ == "__main__":
    main()