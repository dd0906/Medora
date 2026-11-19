"""Loading of the processed dataset splits as validated model inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from detection.config import CLASSES, LABEL_COLUMN, PROCESSED_DIR, SPLITS
from detection.feature_validator import validate_features


def load_split(split: str, data_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, labels) for one split, refusing anything unexpected.

    The processed files were verified as value-identical to the raw CSVs, so a
    rejected row or unknown class here means the file changed and is an error.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split!r}")

    frame = pd.read_csv(data_dir / f"{split}_processed.csv")
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"The {split} split has no {LABEL_COLUMN!r} column.")

    result = validate_features(frame)
    if result.rejected:
        raise ValueError(f"{len(result.rejected)} invalid rows in the {split} split.")

    labels = frame[LABEL_COLUMN].astype(str)
    if set(labels.unique()) != set(CLASSES):
        raise ValueError(f"Unexpected classes in the {split} split: {sorted(labels.unique())}")

    return result.features, labels


def feature_hashes(features: pd.DataFrame) -> np.ndarray:
    """Hash each row's feature vector, for finding exact duplicates across data."""
    return pd.util.hash_pandas_object(features, index=False).to_numpy()
