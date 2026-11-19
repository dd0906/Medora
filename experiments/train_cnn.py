from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Medora paths and configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_processed.csv"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"
MODEL_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "medora_cnn_checkpoint.pt"
CHECKPOINT_TEMP_PATH = MODEL_DIR / "medora_cnn_checkpoint.tmp"
FINAL_MODEL_PATH = MODEL_DIR / "medora_cnn.pt"
PREPROCESSING_PATH = MODEL_DIR / "medora_cnn_preprocessing.joblib"

CLASSES = [
    "Benign",
    "ARP_Spoofing",
    "MQTT_DDoS_Connect_Flood",
    "Recon_Port_Scan",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}

SEED = 42
BATCH_SIZE = 2048
MAX_EPOCHS = 12
PATIENCE = 3
CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------------
class MedoraCNN(nn.Module):
    def __init__(self, number_of_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, number_of_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def predict_in_batches(model, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict class IDs in batches to limit memory use."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(features), 4096):
            batch_array = features[start : start + 4096]
            batch = torch.from_numpy(batch_array).unsqueeze(1).to(device)
            logits = model(batch)
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    if not predictions:
        return np.empty(0, dtype=np.int64)

    return np.concatenate(predictions)


def save_checkpoint_atomically(checkpoint: dict) -> None:
    """Write a checkpoint to a temporary file, then replace the previous one."""
    torch.save(checkpoint, CHECKPOINT_TEMP_PATH)
    CHECKPOINT_TEMP_PATH.replace(CHECKPOINT_PATH)


def main() -> None:
    # Seed all relevant random number generators for repeatability.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("MEDORA - 1D CNN WITH CHECKPOINT / RESUME")
    print("=" * 65)
    print("Device:", device)
    print("Dataset:", DATA_PATH)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Processed training data was not found: {DATA_PATH}\n"
            "Run experiments/prepare_dataset.py first."
        )

    # Load the prepared training data. The supplied test set is not used here.
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise ValueError("The processed dataset does not contain a 'label' column.")

    feature_names_all = [column for column in df.columns if column != "label"]
    mapped_labels = df["label"].map(CLASS_TO_ID)
    if mapped_labels.isna().any():
        unknown = sorted(df.loc[mapped_labels.isna(), "label"].astype(str).unique())
        raise ValueError(f"Unexpected class labels: {unknown}")

    y_all = mapped_labels.to_numpy(dtype=np.int64)
    indices = np.arange(len(df))

    # Match the stratified 80/20 validation split used in previous experiments.
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y_all,
    )

    X_train_frame = df.iloc[train_idx][feature_names_all]
    X_val_frame = df.iloc[val_idx][feature_names_all]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    # Detect constants on the training fold only, not on validation data.
    variable_features = X_train_frame.nunique(dropna=False) > 1
    feature_columns = X_train_frame.columns[variable_features].tolist()
    excluded_features = X_train_frame.columns[~variable_features].tolist()

    X_train_raw = X_train_frame[feature_columns].to_numpy(dtype=np.float32)
    X_val_raw = X_val_frame[feature_columns].to_numpy(dtype=np.float32)

    del df, X_train_frame, X_val_frame, mapped_labels

    if not np.isfinite(X_train_raw).all() or not np.isfinite(X_val_raw).all():
        raise ValueError("Non-finite values were found in the dataset.")

    # Fit the scaler on training rows only to avoid validation leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    del X_train_raw, X_val_raw

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features retained: {len(feature_columns)}")
    print(f"Excluded constant features: {excluded_features or 'None'}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).unsqueeze(1),
        torch.from_numpy(y_train),
    )

    # A separate generator allows the shuffled DataLoader order to be resumed.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    model = MedoraCNN(number_of_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1

    # Resume only if a compatible checkpoint exists.
    if CHECKPOINT_PATH.exists():
        print("\nCheckpoint found. Attempting to resume...")
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(
                "Checkpoint version is incompatible. Move or delete "
                f"{CHECKPOINT_PATH} to start a fresh run."
            )
        if checkpoint.get("class_order") != CLASSES:
            raise ValueError(
                "Checkpoint classes do not match this script. "
                "Move the old checkpoint before starting a new run."
            )
        if checkpoint.get("feature_columns") != feature_columns:
            raise ValueError(
                "Checkpoint feature columns do not match the current data. "
                "Move the old checkpoint before starting a new run."
            )
        if checkpoint.get("seed") != SEED or checkpoint.get("batch_size") != BATCH_SIZE:
            raise ValueError(
                "Checkpoint split/training settings changed. Move the old "
                "checkpoint before starting a fresh run."
            )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_f1 = float(checkpoint["best_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state"]
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        loader_generator.set_state(checkpoint["loader_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])

        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

        print(f"Resuming after completed epoch {checkpoint['epoch']}.")
        print(f"Best validation macro F1 so far: {best_f1:.4f} (epoch {best_epoch})")
    else:
        print("\nNo checkpoint found. Starting a fresh run.")

    if start_epoch > MAX_EPOCHS:
        print(
            f"Checkpoint already completed {MAX_EPOCHS} epochs; "
            "finalising the saved best model."
        )

    training_started = time.perf_counter()
    last_epoch_completed = start_epoch - 1

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        last_epoch_completed = epoch
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        predictions = predict_in_batches(model, X_val, device)
        val_f1 = f1_score(
            y_val,
            predictions,
            labels=list(range(len(CLASSES))),
            average="macro",
            zero_division=0,
        )
        val_accuracy = accuracy_score(y_val, predictions)
        mean_loss = total_loss / len(train_dataset)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={mean_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f} | "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint_data = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "stale_epochs": stale_epochs,
            "feature_columns": feature_columns,
            "class_order": CLASSES,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "loader_rng_state": loader_generator.get_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }
        save_checkpoint_atomically(checkpoint_data)
        print(f"Checkpoint saved after epoch {epoch}.")

        if stale_epochs >= PATIENCE:
            print("Early stopping: validation macro F1 did not improve.")
            break

    training_seconds = time.perf_counter() - training_started

    if best_state is None:
        raise RuntimeError("No best model state was available for evaluation.")

    # Always evaluate the best saved model, including after resuming.
    model.load_state_dict(best_state)
    best_predictions = predict_in_batches(model, X_val, device)

    report_text = classification_report(
        y_val,
        best_predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_val,
        best_predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    benign_mask = y_val == CLASS_TO_ID["Benign"]
    benign_false_alarm_rate = float(
        (best_predictions[benign_mask] != CLASS_TO_ID["Benign"]).mean()
    )

    metrics = {
        "model": "CNN",
        "accuracy": accuracy_score(y_val, best_predictions),
        "balanced_accuracy": balanced_accuracy_score(y_val, best_predictions),
        "macro_f1": best_f1,
        "arp_precision": report_dict["ARP_Spoofing"]["precision"],
        "arp_recall": report_dict["ARP_Spoofing"]["recall"],
        "arp_f1": report_dict["ARP_Spoofing"]["f1-score"],
        "benign_false_alarm_rate": benign_false_alarm_rate,
        "best_epoch": best_epoch,
        "last_epoch_completed": last_epoch_completed,
        "epochs_run_this_session": max(0, last_epoch_completed - start_epoch + 1),
        "training_seconds_this_session": training_seconds,
    }

    print("\nCLASSIFICATION REPORT (BEST VALIDATION CHECKPOINT)")
    print(report_text)

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "cnn_validation_metrics.csv", index=False
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / "cnn_validation_report.csv"
    )

    matrix = confusion_matrix(
        y_val,
        best_predictions,
        labels=list(range(len(CLASSES))),
    )
    pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in CLASSES],
        columns=[f"predicted_{name}" for name in CLASSES],
    ).to_csv(RESULTS_DIR / "cnn_confusion_matrix.csv")

    torch.save(
        {
            "state_dict": best_state,
            "number_of_classes": len(CLASSES),
            "feature_count": len(feature_columns),
            "class_order": CLASSES,
            "feature_columns": feature_columns,
            "best_epoch": best_epoch,
            "validation_macro_f1": best_f1,
        },
        FINAL_MODEL_PATH,
    )

    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "class_order": CLASSES,
        },
        PREPROCESSING_PATH,
    )

    print("\nSUMMARY")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nSaved final model:", FINAL_MODEL_PATH)
    print("Saved preprocessing:", PREPROCESSING_PATH)
    print("Saved results:", RESULTS_DIR)

    # Delete checkpoint only after final outputs have been written.
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    if CHECKPOINT_TEMP_PATH.exists():
        CHECKPOINT_TEMP_PATH.unlink()

    print("Checkpoint removed after successful completion.")
    print("CNN experiment complete.")


if __name__ == "__main__":
    main()
