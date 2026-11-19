import numpy as np
import pandas as pd

from detection.config import ACTIVE_MODEL_PATH, CLASSES
from detection.preprocessing import feature_hashes
from scripts.improve_arp import FORWARD_FOLDS, apply_multiplier, forward_splits, stable_features, with_ratios


def labelled_rows(per_class=100, seed=0):
    rng = np.random.default_rng(seed)
    labels = pd.Series(np.repeat(CLASSES, per_class))
    features = pd.DataFrame({"a": rng.normal(size=len(labels)), "b": rng.normal(size=len(labels))})
    return features, labels


def test_validation_rows_always_come_after_fitting_rows_in_each_class():
    features, labels = labelled_rows()
    for fit, validation in forward_splits(labels, feature_hashes(features)):
        for name in CLASSES:
            positions = np.flatnonzero(labels.to_numpy() == name)
            fit_rows = np.intersect1d(fit, positions)
            validation_rows = np.intersect1d(validation, positions)
            assert len(validation_rows) > 0
            assert fit_rows.max() < validation_rows.min()


def test_fold_sizes_follow_the_configured_fractions():
    features, labels = labelled_rows()
    splits = forward_splits(labels, feature_hashes(features))
    for (fit, validation), (fit_end, validation_end) in zip(splits, FORWARD_FOLDS):
        assert len(fit) == round(400 * fit_end)
        assert len(validation) == round(400 * (validation_end - fit_end))


def test_validation_copies_of_fitting_rows_are_removed():
    features, labels = labelled_rows()
    features.iloc[-1] = features.iloc[0]
    _, validation = forward_splits(labels, feature_hashes(features))[-1]
    assert len(features) - 1 not in validation


def test_multiplier_of_one_keeps_the_top_class():
    probabilities = np.array([[0.6, 0.3, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]])
    assert list(apply_multiplier(probabilities, 1.0)) == ["Benign", "ARP_Spoofing"]


def test_multiplier_changes_only_arp_decisions():
    probabilities = np.array([[0.6, 0.3, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.8, 0.05]])
    assert list(apply_multiplier(probabilities, 3.0)) == ["ARP_Spoofing", "ARP_Spoofing", "MQTT_DDoS_Connect_Flood"]
    assert list(apply_multiplier(probabilities, 0.1)) == ["Benign", "Benign", "MQTT_DDoS_Connect_Flood"]


def test_multiplier_does_not_modify_its_input():
    probabilities = np.array([[0.6, 0.3, 0.05, 0.05]])
    apply_multiplier(probabilities, 3.0)
    assert probabilities[0, 1] == 0.3


def test_ratios_handle_zero_denominators():
    columns = ["Header_Length", "Tot sum", "syn_count", "ack_count", "rst_count", "fin_count", "Std", "AVG", "Max"]
    frame = pd.DataFrame([dict.fromkeys(columns, 0.0)])
    assert np.isfinite(with_ratios(frame).to_numpy()).all()


def test_stable_features_drops_a_feature_that_shifts_over_time():
    labels = pd.Series(["Benign"] * 200 + ["ARP_Spoofing"] * 200)
    rng = np.random.default_rng(0)
    features = pd.DataFrame({
        "stable": rng.normal(size=400),
        "shifting": np.concatenate([np.arange(200.0), np.arange(200.0)]),
    })
    assert stable_features(0.2)(features, labels) == ["stable"]


def test_rejected_candidate_is_not_the_active_model():
    assert "20261102-194943" not in ACTIVE_MODEL_PATH.name
