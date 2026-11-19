"""Evaluate the ARP rules on labelled capture files, one group at a time.

Usage: python -m scripts.evaluate_arp_rules ciciomt_train [--config rules.json]

Labels apply to whole captures, so this measures whether a capture was flagged,
how fast, and how many findings it produced, not per-packet accuracy. Tune on
ciciomt_train only, and score ciciomt_test once after thresholds are fixed.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from detection.arp_rules import load_rule_config
from detection.config import DATA_DIR, EVALUATION_DIR
from detection.pcap_reader import UnsupportedCapture
from scripts.analyze_pcap import analyze

logger = logging.getLogger("medora.evaluate_arp_rules")

PCAP_DIR = DATA_DIR / "external" / "pcaps"
ATTACK = "arp_spoofing"
BENIGN = "benign"
CAPTURE_GROUPS = {
    "ciciomt_train": {"ARP_Spoofing_train.pcap": ATTACK, "Benign_train.pcap": BENIGN},
    "ciciomt_test": {"ARP_Spoofing_test.pcap": ATTACK, "Benign_test.pcap": BENIGN},
    "ciciot2023": {
        "MITM-ArpSpoofing.pcap": ATTACK,
        "MITM-ArpSpoofing1.pcap": ATTACK,
        "BenignTraffic1.pcap": BENIGN,
        "BenignTraffic3.pcap": BENIGN,
    },
    # Kept unread until the flapping-only rule was proposed, to test it on new data.
    "ciciot2023_validation": {"BenignTraffic.pcap": BENIGN, "BenignTraffic2.pcap": BENIGN},
}


def find_capture(name: str, directory: Path = PCAP_DIR) -> Path:
    """Find a capture by exact filename anywhere under the captures folder."""
    matches = sorted(directory.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} under {directory}, found {len(matches)}.")
    return matches[0]


def summarize_group(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare findings between attack and benign captures in one group."""
    attack = [report for report in reports if report["label"] == ATTACK]
    benign = [report for report in reports if report["label"] == BENIGN]
    benign_hours = sum(report["duration_seconds"] for report in benign) / 3600
    benign_primary = sum(report["primary_findings"] for report in benign)
    return {
        "attack_captures_flagged": sum(report["flagged"] for report in attack),
        "attack_captures": len(attack),
        "benign_captures_flagged": sum(report["flagged"] for report in benign),
        "benign_captures": len(benign),
        "benign_hours": benign_hours,
        "benign_primary_findings": benign_primary,
        "benign_primary_findings_per_hour": benign_primary / benign_hours if benign_hours else None,
        "benign_supporting_findings": sum(len(report["findings"]) for report in benign) - benign_primary,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the ARP rules on labelled captures.")
    parser.add_argument("group", choices=sorted(CAPTURE_GROUPS))
    parser.add_argument("--config", help="JSON file overriding rule thresholds.")
    args = parser.parse_args()

    config = load_rule_config(args.config)
    reports = []
    try:
        for name, label in CAPTURE_GROUPS[args.group].items():
            logger.info("Reading %s", name)
            reports.append({"label": label, **analyze(find_capture(name), config)})
    except (OSError, UnsupportedCapture) as error:
        sys.exit(f"Couldn't read a capture: {error}")

    run_dir = EVALUATION_DIR / f"arp_rules_{args.group}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    result = {
        "group": args.group,
        "thresholds": asdict(config),
        "summary": summarize_group(reports),
        "captures": reports,
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    brief = {
        "summary": result["summary"],
        "captures": [
            {key: report[key] for key in (
                "capture", "label", "frames", "arp_frames", "malformed_arp_frames", "truncated",
                "duration_seconds", "findings_by_rule", "primary_findings", "flagged",
                "seconds_to_first_primary_finding", "seconds_to_first_finding",
            )}
            for report in reports
        ],
    }
    print(json.dumps(brief, indent=2))
    logger.info("Saved: %s", run_dir)


if __name__ == "__main__":
    main()
