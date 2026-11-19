import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from detection.preprocessing import feature_hashes
from scripts.run_protocol import PRIMARY_TOLERANCE, Candidate, select_model


def candidates():
    return [
        Candidate("random_forest", None),
        Candidate("lightgbm", None),
        Candidate("rf_with_iat", None, selectable=False),
    ]


def summary(**scores):
    return pd.DataFrame({"macro_f1": scores})


def test_primary_model_kept_when_within_tolerance():
    scores = summary(random_forest=0.97, lightgbm=0.97 + PRIMARY_TOLERANCE / 2, rf_with_iat=0.5)
    assert select_model(scores, candidates())["selected"] == "random_forest"


def test_clearly_better_candidate_replaces_primary():
    scores = summary(random_forest=0.90, lightgbm=0.97, rf_with_iat=0.5)
    assert select_model(scores, candidates())["selected"] == "lightgbm"


def test_diagnostic_candidates_cannot_be_selected():
    scores = summary(random_forest=0.90, lightgbm=0.91, rf_with_iat=0.99)
    assert select_model(scores, candidates())["selected"] != "rf_with_iat"


def test_selection_records_that_test_data_was_not_used():
    scores = summary(random_forest=0.90, lightgbm=0.91, rf_with_iat=0.5)
    assert select_model(scores, candidates())["test_split_used"] is False


def test_duplicate_vectors_share_a_hash_and_never_cross_folds():
    rng = np.random.default_rng(0)
    unique = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    features = pd.concat([unique, unique.iloc[:10]], ignore_index=True)
    labels = np.array(["x", "y"] * 30)
    groups = feature_hashes(features)

    assert len(set(groups)) == 50
    for train_idx, val_idx in StratifiedGroupKFold(n_splits=3).split(features, labels, groups):
        assert not set(groups[train_idx]) & set(groups[val_idx])
