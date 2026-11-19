import numpy as np
import pandas as pd

from scripts.external_check import FOLDER_CLASSES, benign_drift, load_external, summarize


def write_folder(root, folder, rows):
    path = root / folder
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "part.csv", index=False)


def test_labels_come_from_folders_and_bad_rows_are_refused(tmp_path):
    write_folder(tmp_path, "Benign_Final", {"a": [1.0, np.inf], "b": [2.0, 3.0]})
    write_folder(tmp_path, "MITM-ArpSpoofing", {"a": [4.0], "b": [5.0]})
    write_folder(tmp_path, "Recon-PortScan", {"a": [6.0], "b": [7.0]})

    features, labels, rejected = load_external(tmp_path, ["a", "b"])

    assert list(labels) == list(FOLDER_CLASSES.values())
    assert features["a"].tolist() == [1.0, 4.0, 6.0]
    assert rejected == {"Benign_Final": 1, "MITM-ArpSpoofing": 0, "Recon-PortScan": 0}


def test_drift_separates_matching_and_mismatched_features():
    rng = np.random.default_rng(0)
    source = pd.DataFrame({"same": rng.normal(size=500), "shifted": rng.normal(size=500)})
    target = pd.DataFrame({"same": rng.normal(size=500), "shifted": rng.normal(size=500) + 100})

    drift = benign_drift(source, target)

    assert drift["same"] < 0.2
    assert drift["shifted"] == 1.0


def test_summary_excludes_classes_absent_from_the_data():
    truth = pd.Series(["Benign", "ARP_Spoofing", "Recon_Port_Scan"])
    summary = summarize(truth, truth.to_numpy())

    assert "MQTT_DDoS_Connect_Flood" not in summary["per_class"]
    assert summary["macro_f1_present_classes"] == 1.0
