"""Learn what's normal for each device, then check other captures against it.

Usage:
    python -m scripts.device_baselines learn benign.pcap [more.pcap] --output baseline.json
    python -m scripts.device_baselines check capture.pcap --baseline baseline.json [--send]

Reads files only, and captures no network traffic. Learn only from traffic you
believe is benign, because anything in it becomes normal. Findings are leads to
investigate, not confirmed attacks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from detection.config import API_KEY, API_URL, MAX_BATCH_RECORDS
from detection.device_baseline import BaselineConfig, DeviceBaseline, capture_windows
from detection.pcap_reader import CaptureStats, UnsupportedCapture
from scripts.evaluate_baselines import SEARCH_SPACE, join_captures, search_settings

# Under this, a baseline can't have seen much of a device's normal variety.
MIN_LEARNING_WINDOWS = 60


def learn(paths: list[Path]) -> DeviceBaseline:
    """Choose settings on the benign captures, then learn from all of them."""
    captures: list[tuple[pd.DataFrame, CaptureStats]] = []
    for path in paths:
        stats = CaptureStats()
        captures.append((capture_windows(path, stats), stats))
    windows, total = join_captures(captures)
    if total < MIN_LEARNING_WINDOWS:
        raise ValueError(f"Learning needs at least {MIN_LEARNING_WINDOWS} minutes of traffic, and this has {total}.")
    selected = search_settings(windows, total)["selected"]
    return DeviceBaseline.fit(windows, total, BaselineConfig(**{key: selected[key] for key in SEARCH_SPACE}))


def check(path: Path, baseline: DeviceBaseline) -> dict[str, Any]:
    """Compare one capture with a baseline and return a JSON-ready report."""
    stats = CaptureStats()
    windows = capture_windows(path, stats)
    findings = baseline.score(windows, stats.first_timestamp or 0.0)
    return {
        "capture": path.name,
        "frames": stats.frames,
        "truncated": stats.truncated,
        "duration_seconds": stats.duration_seconds,
        "devices": int(windows["device"].nunique()),
        "findings": findings,
    }


def send_findings(client: httpx.Client, report: dict[str, Any], api_key: str | None = API_KEY) -> dict[str, int]:
    """Send a report's findings to the API, in batches it accepts."""
    headers = {"X-API-Key": api_key} if api_key else {}
    totals = {"created": 0, "duplicates": 0}
    findings = report["findings"]
    for start in range(0, len(findings), MAX_BATCH_RECORDS):
        response = client.post(
            "/api/v1/baseline-findings",
            json={"capture": report["capture"], "findings": findings[start:start + MAX_BATCH_RECORDS]},
            headers=headers,
        )
        response.raise_for_status()
        for key, value in response.json().items():
            totals[key] += value
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn device baselines and check captures against them.")
    commands = parser.add_subparsers(dest="command", required=True)
    learner = commands.add_parser("learn", help="Learn a baseline from benign captures.")
    learner.add_argument("captures", type=Path, nargs="+")
    learner.add_argument("--output", type=Path, required=True)
    checker = commands.add_parser("check", help="Check a capture against a baseline.")
    checker.add_argument("capture", type=Path)
    checker.add_argument("--baseline", type=Path, required=True)
    checker.add_argument("--send", action="store_true", help="Send findings to the Medora API as alerts.")
    checker.add_argument("--api-url", default=API_URL)
    args = parser.parse_args()

    try:
        if args.command == "learn":
            baseline = learn(args.captures)
            baseline.save(args.output)
            print(f"Learned {len(baseline.devices)} devices from {baseline.total_windows} minutes: {args.output}")
            return
        report = check(args.capture, DeviceBaseline.load(args.baseline))
    except (OSError, UnsupportedCapture, ValueError, KeyError, TypeError) as error:
        sys.exit(f"Couldn't finish: {error}")
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
