"""Tune and evaluate per-device baselines on labelled capture files.

Usage:
    python -m scripts.evaluate_baselines tune ciciomt
    python -m scripts.evaluate_baselines development ciciomt --settings experiments/evaluation/<run>/selected.json
    python -m scripts.evaluate_baselines final ciciomt --settings experiments/evaluation/<run>/selected.json
    python -m scripts.evaluate_baselines corroborate ciciomt
    python -m scripts.evaluate_baselines other-attacks ciciomt

The tune stage reads benign training captures only: it fits on the earlier 70%,
then picks the most sensitive setting that stays within a false alarm budget on the
later 30%. No attack traffic is involved in the choice. The final stage refits on
all benign training traffic and scores the other captures, which should happen once
per network. Attack labels cover whole captures, not devices, so this reports how
often findings occur, not per-device accuracy. The corroborate stage looks for
evidence the baselines don't use: a device claiming an address over ARP while the
address's usual holder is still claiming it. The other-attacks stage scores attack
types no setting or limit was chosen on, using the saved baseline as it stands.
"""

import argparse
import itertools
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from detection.arp_rules import UNSPECIFIED_IP
from detection.config import EVALUATION_DIR
from detection.device_baseline import WINDOW_SECONDS, BaselineConfig, DeviceBaseline, capture_windows
from detection.pcap_reader import CaptureStats, UnsupportedCapture, read_arp_packets
from scripts.evaluate_arp_rules import ATTACK, BENIGN, find_capture

logger = logging.getLogger("medora.evaluate_baselines")

TRAIN_SHARE = 0.7
# The budget and pass marks were set before any held-out capture was read.
MAX_FINDINGS_PER_HOUR = 1.0
PASS_MAX_BENIGN_FINDINGS_PER_HOUR = 2.0
PASS_MIN_RATE_RATIO = 5.0
# Attack types the baseline wasn't evaluated on when its settings and marks were fixed.
OTHER_ATTACKS = {
    "ciciomt": {
        "Recon-Port_Scan_test.pcap": "recon_port_scan",
        "MQTT-DDoS-Connect_Flood_test.pcap": "mqtt_connect_flood",
    },
}
SEARCH_SPACE = {
    "quantile": (0.99, 0.999, 1.0),
    "multiplier": (1.0, 1.5, 2.0, 3.0),
    "min_features": (1, 2, 3),
    "min_windows": (1, 2, 3),
}

NETWORKS = {
    "ciciomt": {
        "train": ("Benign_train.pcap",),
        "development": {"ARP_Spoofing_train.pcap": ATTACK},
        "held_out": {"Benign_test.pcap": BENIGN, "ARP_Spoofing_test.pcap": ATTACK},
    },
    "ciciot2023": {
        "train": ("BenignTraffic.pcap", "BenignTraffic1.pcap"),
        "development": {},
        "held_out": {
            "BenignTraffic2.pcap": BENIGN,
            "BenignTraffic3.pcap": BENIGN,
            "MITM-ArpSpoofing.pcap": ATTACK,
            "MITM-ArpSpoofing1.pcap": ATTACK,
        },
    },
}


def read_capture(name: str) -> tuple[pd.DataFrame, CaptureStats]:
    logger.info("Reading %s", name)
    stats = CaptureStats()
    windows = capture_windows(find_capture(name), stats)
    return windows, stats


def join_captures(captures: list[tuple[pd.DataFrame, CaptureStats]]) -> tuple[pd.DataFrame, int]:
    """Number windows on after each other, so captures read as one long timeline."""
    parts = []
    offset = 0
    for windows, stats in captures:
        parts.append(windows.assign(window=windows["window"] + offset))
        offset += int(stats.duration_seconds // WINDOW_SECONDS) + 1
    return pd.concat(parts, ignore_index=True), offset


def search_settings(windows: pd.DataFrame, total_windows: int) -> dict[str, Any]:
    """Pick the setting flagging the most windows while staying within the budget."""
    cutoff = int(total_windows * TRAIN_SHARE)
    fit_rows = windows[windows["window"] < cutoff]
    check_rows = windows[windows["window"] >= cutoff]
    check_hours = (total_windows - cutoff) * WINDOW_SECONDS / 3600

    trials = []
    for values in itertools.product(*SEARCH_SPACE.values()):
        config = BaselineConfig(**dict(zip(SEARCH_SPACE, values)))
        findings = DeviceBaseline.fit(fit_rows, cutoff, config).score(check_rows)
        trials.append({
            **dict(zip(SEARCH_SPACE, values)),
            "findings": len(findings),
            "unknown_device_findings": sum(finding["kind"] == "unknown_device" for finding in findings),
            "findings_per_hour": len(findings) / check_hours,
            "flagged_windows": sum(finding["windows"] for finding in findings),
        })
    allowed = [trial for trial in trials if trial["findings_per_hour"] <= MAX_FINDINGS_PER_HOUR]
    if not allowed:
        raise ValueError("No setting met the false alarm budget.")
    # Among equals, prefer the lowest limits, so quiet training traffic doesn't blunt the baseline.
    best = max(
        allowed,
        key=lambda t: (t["flagged_windows"], -t["min_features"], -t["min_windows"], -t["quantile"], -t["multiplier"]),
    )
    return {"selected": best, "fit_windows": cutoff, "check_hours": check_hours, "trials": trials}


def capture_report(name: str, label: str, baseline: DeviceBaseline) -> dict[str, Any]:
    windows, stats = read_capture(name)
    findings = baseline.score(windows)
    hours = stats.duration_seconds / 3600
    return {
        "capture": name,
        "label": label,
        "frames": stats.frames,
        "truncated": stats.truncated,
        "hours": hours,
        "devices": int(windows["device"].nunique()),
        "findings": len(findings),
        "findings_per_hour": len(findings) / hours if hours else None,
        "flagged_windows_per_hour": sum(f["windows"] for f in findings) / hours if hours else None,
        "devices_flagged": sorted({finding["device"] for finding in findings}),
        "details": findings,
    }


def compare_rates(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare finding rates across whole captures against the marks set beforehand."""
    def rate(label: str) -> float | None:
        chosen = [report for report in reports if report["label"] == label]
        hours = sum(report["hours"] for report in chosen)
        return sum(report["findings"] for report in chosen) / hours if hours else None

    benign, attack = rate(BENIGN), rate(ATTACK)
    ratio = None
    if benign is not None and attack is not None and benign > 0:
        ratio = attack / benign
    separated = ratio is not None and ratio >= PASS_MIN_RATE_RATIO
    if benign == 0 and attack:
        separated = True
    return {
        "benign_findings_per_hour": benign,
        "attack_findings_per_hour": attack,
        "attack_to_benign_ratio": ratio,
        "benign_rate_passed": benign is not None and benign <= PASS_MAX_BENIGN_FINDINGS_PER_HOUR,
        "separation_passed": separated,
    }


def save(name: str, result: dict[str, Any]) -> Path:
    run_dir = EVALUATION_DIR / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return run_dir


def tune(network: str) -> None:
    windows, total = join_captures([read_capture(name) for name in NETWORKS[network]["train"]])
    search = search_settings(windows, total)
    run_dir = save(f"baseline_tune_{network}", {"network": network, "budget": MAX_FINDINGS_PER_HOUR, **search})
    (run_dir / "selected.json").write_text(json.dumps(search["selected"], indent=2), encoding="utf-8")
    print(json.dumps({key: search[key] for key in ("selected", "fit_windows", "check_hours")}, indent=2))
    logger.info("Saved: %s", run_dir)


def final(network: str, settings_path: Path, include_held_out: bool) -> None:
    selected = json.loads(settings_path.read_text(encoding="utf-8"))
    config = BaselineConfig(**{key: selected[key] for key in SEARCH_SPACE})
    windows, total = join_captures([read_capture(name) for name in NETWORKS[network]["train"]])
    baseline = DeviceBaseline.fit(windows, total, config)

    captures = dict(NETWORKS[network]["development"])
    if include_held_out:
        captures.update(NETWORKS[network]["held_out"])
    reports = [capture_report(name, label, baseline) for name, label in captures.items()]

    stage = "final" if include_held_out else "development"
    held_out = [r for r in reports if r["capture"] in NETWORKS[network]["held_out"]]
    result = {
        "network": network,
        "stage": stage,
        "settings": asdict(config),
        "baseline_devices": len(baseline.devices),
        "held_out_captures": sorted(NETWORKS[network]["held_out"]),
        "comparison": compare_rates(held_out) if include_held_out else None,
        "captures": reports,
    }
    run_dir = save(f"baseline_{stage}_{network}", result)
    baseline.save(run_dir / "baseline.json")

    brief = {
        "comparison": result["comparison"],
        "captures": [{key: value for key, value in report.items() if key != "details"} for report in reports],
    }
    print(json.dumps(brief, indent=2))
    logger.info("Saved: %s", run_dir)


def address_claims(names: Iterable[str]) -> dict[str, Counter]:
    """Count, for each IP, how many ARP frames each MAC sent claiming it."""
    claims: dict[str, Counter] = defaultdict(Counter)
    for name in names:
        for packet in read_arp_packets(find_capture(name)):
            if packet.sender_ip != UNSPECIFIED_IP:
                claims[packet.sender_ip][packet.sender_mac] += 1
    return claims


def contested_addresses(usual: dict[str, Counter], seen: dict[str, Counter]) -> list[dict[str, Any]]:
    """Find newcomers claiming an IP while a device known to hold it is also claiming it."""
    contested = []
    for ip, macs in seen.items():
        holders = [mac for mac in macs if mac in usual.get(ip, ())]
        newcomers = [mac for mac in macs if mac not in usual.get(ip, ())]
        if holders and newcomers:
            contested.append({"ip": ip, "usual_holders": holders, "newcomers": newcomers})
    return contested


def device_ranking(findings: list[dict[str, Any]]) -> list[str]:
    """Devices with the most flagged windows first."""
    windows: Counter = Counter()
    for finding in findings:
        windows[finding["device"]] += finding["windows"]
    return [device for device, _ in sorted(windows.items(), key=lambda item: (-item[1], item[0]))]


def corroborate(network: str) -> None:
    matches = sorted(EVALUATION_DIR.glob(f"baseline_final_{network}_*"))
    if not matches:
        raise ValueError("Run the final stage first.")
    final_results = json.loads((matches[-1] / "results.json").read_text(encoding="utf-8"))
    usual = address_claims(NETWORKS[network]["train"])

    reports = []
    for capture in final_results["captures"]:
        logger.info("Reading %s", capture["capture"])
        contested = contested_addresses(usual, address_claims([capture["capture"]]))
        ranking = device_ranking(capture["details"])
        newcomers = sorted({mac for item in contested for mac in item["newcomers"]})
        reports.append({
            "capture": capture["capture"],
            "label": capture["label"],
            "contested_addresses": contested,
            "devices_ranked_by_flagged_windows": ranking,
            "newcomer_ranks": {mac: ranking.index(mac) + 1 if mac in ranking else None for mac in newcomers},
        })
    run_dir = save(f"baseline_corroborate_{network}", {"network": network, "final_run": matches[-1].name, "captures": reports})
    print(json.dumps(reports, indent=2))
    logger.info("Saved: %s", run_dir)


def score_other_attacks(network: str) -> None:
    matches = sorted(EVALUATION_DIR.glob(f"baseline_final_{network}_*"))
    if network not in OTHER_ATTACKS or not matches:
        raise ValueError("This needs a network with other attacks listed, and a final run.")
    final_results = json.loads((matches[-1] / "results.json").read_text(encoding="utf-8"))
    benign_rate = final_results["comparison"]["benign_findings_per_hour"]
    baseline = DeviceBaseline.load(matches[-1] / "baseline.json")

    reports = [capture_report(name, label, baseline) for name, label in OTHER_ATTACKS[network].items()]
    for report in reports:
        report["attack_to_benign_ratio"] = report["findings_per_hour"] / benign_rate
        report["separation_passed"] = report["attack_to_benign_ratio"] >= PASS_MIN_RATE_RATIO
        report["devices_ranked_by_flagged_windows"] = device_ranking(report["details"])
    result = {
        "network": network,
        "final_run": matches[-1].name,
        "benign_findings_per_hour": benign_rate,
        "captures": reports,
    }
    run_dir = save(f"baseline_other_attacks_{network}", result)
    print(json.dumps([{key: value for key, value in report.items() if key != "details"} for report in reports], indent=2))
    logger.info("Saved: %s", run_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Tune and evaluate per-device baselines.")
    parser.add_argument("stage", choices=("tune", "development", "final", "corroborate", "other-attacks"))
    parser.add_argument("network", choices=sorted(NETWORKS))
    parser.add_argument("--settings", type=Path, help="selected.json from the tune stage.")
    args = parser.parse_args()

    try:
        if args.stage == "tune":
            tune(args.network)
        elif args.stage == "corroborate":
            corroborate(args.network)
        elif args.stage == "other-attacks":
            score_other_attacks(args.network)
        else:
            if args.settings is None:
                parser.error(f"The {args.stage} stage needs --settings.")
            final(args.network, args.settings, include_held_out=args.stage == "final")
    except (OSError, UnsupportedCapture, ValueError) as error:
        sys.exit(f"Couldn't finish: {error}")


if __name__ == "__main__":
    main()
