"""Reads saved evaluation runs, so results shown anywhere come from real run files."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("medora.evaluation_results")

COMPARISON_COLUMNS = (
    "model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "arp_precision", "arp_recall",
    "arp_f1", "benign_false_alarm_rate", "cv_macro_f1", "cv_arp_f1", "cv_benign_false_alarm_rate",
    "single_row_milliseconds",
)


def load_evaluation(evaluation_dir: Path, run_id: str | None) -> dict[str, Any]:
    """Collect the active model's evaluation and the follow-up checks, if saved.

    `run_id` comes from the model's own metadata, never from a request.
    """
    if not run_id:
        return {"available": False, "reason": "This model has no linked evaluation run."}

    run_dir = evaluation_dir / run_id
    try:
        selection = _read_json(run_dir / "selection.json")
        metrics = _read_json(run_dir / "test_metrics.json")
        comparison = pd.read_csv(run_dir / "test_comparison.csv")
        selected = selection["selected"]
        active = metrics[selected]
    except (OSError, KeyError, ValueError):
        logger.exception("Could not read evaluation run %s", run_id)
        return {"available": False, "reason": "The evaluation files for this model couldn't be read."}

    columns = [column for column in COMPARISON_COLUMNS if column in comparison.columns]
    return {
        "available": True,
        "run_id": run_id,
        "selected": selected,
        "selection_rule": selection.get("rule"),
        "test_split_used_for_selection": selection.get("test_split_used"),
        "test_metrics": active,
        "comparison": comparison[columns].to_dict(orient="records"),
        "arp_attempt": _arp_attempt(evaluation_dir),
        "outside_data": _outside_data(evaluation_dir),
        "arp_rules": _arp_rules(evaluation_dir),
        "device_baselines": _device_baselines(evaluation_dir),
        "feature_effects": _feature_effects(run_dir),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest(evaluation_dir: Path, pattern: str) -> Path | None:
    matches = sorted(evaluation_dir.glob(pattern))
    return matches[-1] if matches else None


def _arp_attempt(evaluation_dir: Path) -> dict[str, Any] | None:
    final = _latest(evaluation_dir, "arp_final_*")
    if final is None:
        return None
    try:
        metrics = _read_json(final / "test_metrics.json")
        arp = metrics["per_class"]["ARP_Spoofing"]
        return {
            "candidate": metrics["candidate"],
            "arp_f1": arp["f1"],
            "arp_precision": arp["precision"],
            "arp_recall": arp["recall"],
            "macro_f1": metrics["macro_f1"],
            "benign_false_alarm_rate": metrics["benign_false_alarm_rate"],
        }
    except (OSError, KeyError, ValueError):
        logger.exception("Could not read ARP attempt results")
        return None


def _arp_rules(evaluation_dir: Path) -> list[dict[str, Any]] | None:
    """Latest saved result for each group of labelled captures."""
    latest: dict[str, Path] = {}
    for run in sorted(evaluation_dir.glob("arp_rules_*")):
        latest[run.name.rsplit("_", 1)[0]] = run
    if not latest:
        return None

    rows = []
    try:
        for run in latest.values():
            results = _read_json(run / "results.json")
            for capture in results["captures"]:
                rows.append({
                    "group": results["group"],
                    "capture": capture["capture"],
                    "label": capture["label"],
                    "flagged": capture["flagged"],
                    "primary_findings": capture["primary_findings"],
                    "hours": capture["duration_seconds"] / 3600,
                    "seconds_to_first_primary_finding": capture["seconds_to_first_primary_finding"],
                })
    except (OSError, KeyError, ValueError):
        logger.exception("Could not read ARP rule results")
        return None
    return rows


def _device_baselines(evaluation_dir: Path) -> list[dict[str, Any]] | None:
    """Held-out results from the latest final and other-attack runs for each network."""
    latest: dict[str, Path] = {}
    for pattern in ("baseline_final_*", "baseline_other_attacks_*"):
        for run in sorted(evaluation_dir.glob(pattern)):
            latest[run.name.rsplit("_", 1)[0]] = run
    if not latest:
        return None

    rows = []
    try:
        for run in latest.values():
            results = _read_json(run / "results.json")
            for capture in results["captures"]:
                # Other-attack runs score held-out captures only, so they have no list.
                if capture["capture"] not in results.get("held_out_captures", [capture["capture"]]):
                    continue
                rows.append({
                    "network": results["network"],
                    "capture": capture["capture"],
                    "label": capture["label"],
                    "hours": capture["hours"],
                    "findings": capture["findings"],
                    "findings_per_hour": capture["findings_per_hour"],
                    "devices_flagged": len(capture["devices_flagged"]),
                    "largest_rise": _largest_rise(capture["details"]),
                })
    except (OSError, KeyError, ValueError):
        logger.exception("Could not read device baseline results")
        return None
    return rows


def _feature_effects(run_dir: Path) -> dict[str, Any] | None:
    """Which features the model leans on for each type, if scripts/explain_model.py has run."""
    path = run_dir / "feature_effects.json"
    if not path.is_file():
        return None
    try:
        results = _read_json(path)
        return {"method": results["method"], "types": results["types"]}
    except (OSError, KeyError, ValueError):
        logger.exception("Could not read feature effects")
        return None


def _largest_rise(findings: list[dict[str, Any]]) -> float | None:
    """How many times over its own limit any device went in this capture."""
    rises = [item["observed"] / item["limit"] for finding in findings for item in finding["evidence"]]
    return max(rises) if rises else None


def _outside_data(evaluation_dir: Path) -> dict[str, Any] | None:
    run = _latest(evaluation_dir, "external_ciciot2023_*")
    if run is None:
        return None
    try:
        results = _read_json(run / "results.json")
        checks = []
        for name, label in (
            ("all_shared_features_model", "Shared features"),
            ("comparable_features_model", "Most comparable features"),
        ):
            for dataset, dataset_label in (("ciciomt2024_test", "CICIoMT2024 test"), ("ciciot2023", "CICIoT2023")):
                values = results[name][dataset]
                checks.append({
                    "model": label,
                    "feature_count": results[name]["feature_count"],
                    "scored_on": dataset_label,
                    "accuracy": values["accuracy"],
                    "macro_f1": values["macro_f1_present_classes"],
                    "attack_detection_rate": values["attack_detection_rate"],
                    "benign_false_alarm_rate": values["benign_false_alarm_rate"],
                    "arp_recall": values["per_class"]["ARP_Spoofing"]["recall"],
                })
        benign = results["all_shared_features_model"]["ciciot2023"]["per_class"]["Benign"]["support"]
        rows = results["all_shared_features_model"]["ciciot2023"]["rows"]
        return {
            "missing_model_features": results["missing_model_features"],
            "always_benign_accuracy": benign / rows,
            "checks": checks,
        }
    except (OSError, KeyError, ValueError):
        logger.exception("Could not read outside-data results")
        return None
