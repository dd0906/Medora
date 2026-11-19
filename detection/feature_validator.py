"""Schema and value validation shared by training, evaluation, and inference."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from detection.config import FEATURE_COLUMNS, LABEL_COLUMN

# scikit-learn's tree models convert input to float32, where anything larger becomes infinity.
LARGEST_USABLE_VALUE = float(np.finfo(np.float32).max)


class SchemaError(ValueError):
    """The input's columns cannot be mapped onto the expected feature schema."""


@dataclass(frozen=True)
class ValidationResult:
    """Valid rows in model order, plus the rows that were refused and why."""

    features: pd.DataFrame
    rejected: dict[int, str]


def validate_features(
    frame: pd.DataFrame,
    expected: Sequence[str] = FEATURE_COLUMNS,
    ignore: Sequence[str] = (LABEL_COLUMN,),
) -> ValidationResult:
    """Check columns and values, returning float64 features in `expected` order.

    A wrong schema raises SchemaError because no row could be trusted. A bad
    value only rejects its own row. Rows are never imputed: the training
    pipeline does not impute, so a filled-in value would be one the model was
    never evaluated on. `frame` is not modified and row labels are preserved.
    """
    columns = list(frame.columns)
    if len(columns) != len(set(columns)):
        raise SchemaError("Duplicate column names in input.")

    missing = [c for c in expected if c not in columns]
    unexpected = [c for c in columns if c not in expected and c not in ignore]
    if missing or unexpected:
        raise SchemaError(f"Schema mismatch: missing={missing}, unexpected={unexpected}")

    numeric = frame[list(expected)].apply(pd.to_numeric, errors="coerce").astype("float64")
    values = numeric.to_numpy()
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(values) & (np.abs(values) <= LARGEST_USABLE_VALUE)

    rejected = {}
    for position in np.flatnonzero(~finite.all(axis=1)):
        bad_columns = [c for c, ok in zip(expected, finite[position]) if not ok]
        rejected[frame.index[position]] = f"Missing, non-finite, or too large values in: {bad_columns}"

    return ValidationResult(numeric[finite.all(axis=1)], rejected)
