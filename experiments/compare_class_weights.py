from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
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


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    features = [c for c in df.columns if c != "label"]
    X = df[features]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    settings = {
        "unweighted": None,
        "balanced_subsample": "balanced_subsample",
    }

    results = []

    for name, weight in settings.items():
        print("\n" + "=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=weight,
            random_state=42,
            n_jobs=-1,
        )

        print("Training model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        print("\nClassification report:")
        print(
            classification_report(
                y_val,
                predictions,
                labels=CLASSES,
                target_names=CLASSES,
                digits=4,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_val, predictions, labels=CLASSES
        )
        print("Confusion matrix (actual rows, predicted columns):")
        print(pd.DataFrame(
            matrix, index=CLASSES, columns=CLASSES
        ).to_string())

        report = classification_report(
            y_val,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        benign_mask = y_val == "Benign"
        benign_false_alarm_rate = (
            (predictions[benign_mask] != "Benign").mean()
        )

        results.append({
            "model": name,
            "macro_f1": f1_score(
                y_val, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            ),
            "arp_precision": report["ARP_Spoofing"]["precision"],
            "arp_recall": report["ARP_Spoofing"]["recall"],
            "arp_f1": report["ARP_Spoofing"]["f1-score"],
            "benign_false_alarm_rate": benign_false_alarm_rate,
        })

    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "class_weight_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()