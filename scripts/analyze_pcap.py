"""Check a capture file for ARP spoofing behaviour.

Usage: python -m scripts.analyze_pcap capture.pcap [--config rules.json] [--send]

Reads the file only, and captures no network traffic. With --send, findings go to
the Medora API as alerts. Findings are leads to investigate, not confirmed attacks.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from detection.arp_rules import ArpMonitor, ArpRuleConfig, load_rule_config, primary_findings, summarize_findings
from detection.config import API_KEY, API_URL, MAX_BATCH_RECORDS
from detection.pcap_reader import CaptureStats, UnsupportedCapture, read_arp_packets


def analyze(path: Path, config: ArpRuleConfig | None = None) -> dict[str, Any]:
    """Run the rules over one capture and return a JSON-ready report."""
    stats = CaptureStats()
    monitor = ArpMonitor(config)
    monitor.process(read_arp_packets(path, stats))
    primary = primary_findings(monitor.findings)
    return {
        "capture": path.name,
        "frames": stats.frames,
        "arp_frames": stats.arp_frames,
        "malformed_arp_frames": stats.malformed_arp_frames,
        "truncated": stats.truncated,
        "duration_seconds": stats.duration_seconds,
        "findings_by_rule": summarize_findings(monitor.findings),
        "primary_findings": len(primary),
        "flagged": bool(primary),
        "seconds_to_first_primary_finding": (
            min(f.first_seen for f in primary) - stats.first_timestamp if primary else None
        ),
        "seconds_to_first_finding": (
            min(f.first_seen for f in monitor.findings) - stats.first_timestamp if monitor.findings else None
        ),
        "findings": [asdict(finding) for finding in monitor.findings],
    }


def send_findings(client: httpx.Client, report: dict[str, Any], api_key: str | None = API_KEY) -> dict[str, int]:
    """Send a report's findings to the API, in batches it accepts."""
    headers = {"X-API-Key": api_key} if api_key else {}
    totals = {"created": 0, "duplicates": 0}
    findings = report["findings"]
    for start in range(0, len(findings), MAX_BATCH_RECORDS):
        response = client.post(
            "/api/v1/rule-findings",
            json={"capture": report["capture"], "findings": findings[start:start + MAX_BATCH_RECORDS]},
            headers=headers,
        )
        response.raise_for_status()
        for key, value in response.json().items():
            totals[key] += value
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a capture file for ARP spoofing behaviour.")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--config", help="JSON file overriding rule thresholds.")
    parser.add_argument("--send", action="store_true", help="Send findings to the Medora API as alerts.")
    parser.add_argument("--api-url", default=API_URL)
    args = parser.parse_args()

    try:
        report = analyze(args.capture, load_rule_config(args.config))
    except (OSError, UnsupportedCapture) as error:
        sys.exit(f"Couldn't read the capture: {error}")
    print(json.dumps(report, indent=2))

    if args.send:
        try:
            with httpx.Client(base_url=args.api_url, timeout=60) as client:
                totals = send_findings(client, report)
        except httpx.HTTPError as error:
            sys.exit(f"Couldn't send findings to the API: {error}")
        print(f"Sent to the API: {totals['created']} new alerts, {totals['duplicates']} already stored.")


if __name__ == "__main__":
    main()
