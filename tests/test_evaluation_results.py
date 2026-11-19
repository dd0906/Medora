import json

import pytest

from backend.evaluation_results import load_evaluation
from detection.config import EVALUATION_DIR


def write_run(root, run_id="run1"):
    run = root / run_id
    run.mkdir()
    (run / "selection.json").write_text(json.dumps({"selected": "rf", "rule": "rule", "test_split_used": False}))
    (run / "test_metrics.json").write_text(json.dumps({"rf": {"accuracy": 0.9}, "other": {"accuracy": 0.1}}))
    (run / "test_comparison.csv").write_text("model,accuracy,macro_f1\nrf,0.9,0.8\n")


def test_reads_the_selected_models_metrics(tmp_path):
    write_run(tmp_path)
    result = load_evaluation(tmp_path, "run1")
    assert result["available"] is True
    assert result["test_metrics"] == {"accuracy": 0.9}
    assert result["comparison"] == [{"model": "rf", "accuracy": 0.9, "macro_f1": 0.8}]
    assert result["arp_attempt"] is None
    assert result["outside_data"] is None


@pytest.mark.parametrize("run_id", [None, "absent"])
def test_missing_run_is_reported_without_a_path(tmp_path, run_id):
    result = load_evaluation(tmp_path, run_id)
    assert result["available"] is False
    assert str(tmp_path) not in result["reason"]


def test_damaged_files_are_reported_not_raised(tmp_path):
    write_run(tmp_path)
    (tmp_path / "run1" / "selection.json").write_text("{not json")
    assert load_evaluation(tmp_path, "run1")["available"] is False


def test_real_saved_results_match_the_documented_scores():
    if not (EVALUATION_DIR / "20261101-173518").is_dir():
        pytest.skip("Saved evaluation runs are not available.")
    result = load_evaluation(EVALUATION_DIR, "20261101-173518")
    assert result["test_metrics"]["macro_f1"] == pytest.approx(0.8942, abs=1e-4)
    assert result["arp_attempt"]["arp_f1"] == pytest.approx(0.5941, abs=1e-4)
    assert result["outside_data"]["always_benign_accuracy"] == pytest.approx(0.7380, abs=1e-4)
