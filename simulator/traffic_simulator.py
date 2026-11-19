"""Replay labelled dataset records through the running Medora API.

This is a dataset replay, not live monitoring: records come from the processed
CSVs, timestamps are replay time, and no packets are captured or sent.

Usage: python -m simulator.traffic_simulator --limit 5000 --rate 500
"""

import argparse
import json
import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import httpx
import pandas as pd

from detection.config import API_KEY, EVALUATION_DIR, LABEL_COLUMN, MAX_BATCH_RECORDS, MODEL_FEATURE_COLUMNS, PROCESSED_DIR, SEED

logger = logging.getLogger("medora.simulator")

DEFAULT_API_URL = "http://127.0.0.1:8000"


def load_records(split: str, limit: int | None, seed: int) -> pd.DataFrame:
    """Load a shuffled sample of a dataset split. Only known splits can be read."""
    frame = pd.read_csv(PROCESSED_DIR / f"{split}_processed.csv")
    if limit is not None and limit < len(frame):
        frame = frame.sample(n=limit, random_state=seed)
    else:
        frame = frame.sample(frac=1, random_state=seed)
    return frame.reset_index(drop=True)


def build_batches(frame: pd.DataFrame, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield API records. The label is sent only as ground_truth for later scoring."""
    features = frame[list(MODEL_FEATURE_COLUMNS)].to_dict(orient="records")
    labels = frame[LABEL_COLUMN].tolist()
    for start in range(0, len(features), batch_size):
        yield [
            {"features": row, "ground_truth": label}
            for row, label in zip(features[start:start + batch_size], labels[start:start + batch_size])
        ]


def summarize(sent: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare predictions with the known labels of the replayed records."""
    truth = [record["ground_truth"] for record in sent]
    is_attack_true = [label != "Benign" for label in truth]

    severity_counts: dict[str, Counter] = defaultdict(Counter)
    for result, label, attack in zip(results, truth, is_attack_true):
        if result["alert_id"]:
            severity_counts[result["severity"]]["attack" if attack else "benign"] += 1

    return {
        "records_scored": len(results),
        "correct": sum(r["predicted_class"] == label for r, label in zip(results, truth)),
        "predicted_by_class": dict(Counter(r["predicted_class"] for r in results)),
        "actual_by_class": dict(Counter(truth)),
        "new_alerts": sum(r["new_alert"] for r in results),
        "alerted_events_by_severity": {
            severity: {
                "true_attack": counts["attack"],
                "actually_benign": counts["benign"],
                "share_true_attack": counts["attack"] / sum(counts.values()),
            }
            for severity, counts in severity_counts.items()
        },
    }


def replay(
    client: httpx.Client,
    frame: pd.DataFrame,
    batch_size: int,
    rate: float | None,
) -> dict[str, Any]:
    """Send records at about `rate` per second and return a labelled summary."""
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    sent, results = [], []
    rejected = 0
    started = time.perf_counter()

    for number, batch in enumerate(build_batches(frame, batch_size), start=1):
        response = client.post(
            "/api/v1/detect/batch",
            json={"records": batch, "source": "replay"},
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()

        scored = {result["index"] for result in body["results"]}
        sent += [record for index, record in enumerate(batch) if index in scored]
        results += body["results"]
        rejected += len(body["rejected"])

        if number == 1 or number % 10 == 0:
            logger.info("Replayed %d of %d records", len(sent) + rejected, len(frame))

        if rate:
            ahead = (len(sent) + rejected) / rate - (time.perf_counter() - started)
            if ahead > 0:
                time.sleep(ahead)

    summary = summarize(sent, results)
    summary["records_rejected"] = rejected
    summary["seconds"] = time.perf_counter() - started
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay dataset records through the Medora API.")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--limit", type=int, default=5000, help="Records to replay. Use 0 for all.")
    parser.add_argument("--rate", type=float, default=500.0, help="Records per second. Use 0 for no limit.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= MAX_BATCH_RECORDS:
        parser.error(f"--batch-size must be between 1 and {MAX_BATCH_RECORDS}.")
    if args.limit < 0 or args.rate < 0 or math.isinf(args.rate):
        parser.error("--limit and --rate cannot be negative.")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    logger.info("DATASET REPLAY: recorded %s records, not live network monitoring.", args.split)
    frame = load_records(args.split, args.limit or None, args.seed)

    with httpx.Client(base_url=args.api_url, timeout=120) as client:
        client.get("/api/v1/health").raise_for_status()
        summary = replay(client, frame, args.batch_size, args.rate or None)

    summary = {"mode": "dataset replay", "split": args.split, "seed": args.seed, **summary}
    output_dir = EVALUATION_DIR / "replays"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"replay_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    logger.info("Saved summary: %s", path)


if __name__ == "__main__":
    main()
