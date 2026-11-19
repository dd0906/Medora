from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_processed.csv"
TEST_PATH = DATA_DIR / "test_processed.csv"

CLASSES = [
    "Benign",
    "ARP_Spoofing",
    "MQTT_DDoS_Connect_Flood",
    "Recon_Port_Scan",
]


def main():
    print("=" * 60)
    print("MEDORA - RANDOM FOREST BASELINE")
    print("=" * 60)

    # Load the prepared datasets.
    print("\nLoading training dataset...")
    train_df = pd.read_csv(TRAIN_PATH)

    print("Loading testing dataset...")
    test_df = pd.read_csv(TEST_PATH)

    if "label" not in train_df or "label" not in test_df:
        raise ValueError("The label column is missing.")

    feature_columns = [
        col for col in train_df.columns if col != "label"
    ]

    test_features = [
        col for col in test_df.columns if col != "label"
    ]

    if feature_columns != test_features:
        raise ValueError("Training and testing features do not match.")

    X_train = train_df[feature_columns]
    y_train = train_df["label"]

    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    # Confirm that all expected classes are present.
    if set(y_train.unique()) != set(CLASSES):
        raise ValueError("Unexpected training labels.")

    if set(y_test.unique()) != set(CLASSES):
        raise ValueError("Unexpected testing labels.")

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Features:      {len(feature_columns)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().reindex(CLASSES).to_string())

    # Create and train the baseline classifier.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate on the separate testing dataset.
    print("\nEvaluating on test data...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_test, predictions,
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    print(report)

    # Save the evaluation reports.
    report_dict = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    matrix = confusion_matrix(
        y_test, predictions, labels=CLASSES
    )

    pd.DataFrame(
        matrix,
        index=[f"actual_{c}" for c in CLASSES],
        columns=[f"predicted_{c}" for c in CLASSES],
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "feature_count": len(feature_columns),
    }

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "metrics.csv", index=False
    )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False
    )

    # Save the model together with its expected feature order.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "class_order": CLASSES,
        },
        MODEL_DIR / "medora_rf.joblib",
    )

    print("\nSUMMARY")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1-score:    {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nTOP 10 FEATURES")
    print(importance.head(10).to_string(index=False))

    print("\nModel saved to:", MODEL_DIR / "medora_rf.joblib")
    print("Reports saved to:", RESULTS_DIR)
    print("\nBASELINE COMPLETE")


if __name__ == "__main__":
    main()