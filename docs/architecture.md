# Architecture

## Layout

| Path | Responsibility |
|---|---|
| `detection/config.py` | Paths, classes, feature schema, and limits; environment overrides |
| `detection/feature_validator.py` | Schema and value checks shared by training and inference |
| `detection/preprocessing.py` | Loads dataset splits through the validator |
| `detection/classifier.py` | Loads a saved model, scores records, and explains predictions |
| `detection/risk_scoring.py` | Configurable risk formula and severity bands |
| `detection/alerts.py` | Alert explanation text |
| `detection/evaluation.py` | Metrics used for every model |
| `backend/service.py` | Runs one record through the full chain; alert grouping and status rules |
| `backend/database.py` | SQLite schema and queries |
| `backend/schemas.py`, `backend/main.py` | REST API |
| `simulator/traffic_simulator.py` | Dataset replay client for the command line |
| `detection/pcap_reader.py` | Reads ARP frames from capture files, treating them as untrusted |
| `detection/arp_rules.py` | Rule-based ARP spoofing checks |
| `scripts/analyze_pcap.py`, `scripts/evaluate_arp_rules.py` | Check one capture, or evaluate the rules on labelled captures |
| `detection/device_baseline.py` | Per-minute device summaries from capture headers, learning limits, and scoring |
| `scripts/device_baselines.py`, `scripts/evaluate_baselines.py` | Learn and check baselines, or tune and evaluate them on labelled captures |
| `detection/feature_descriptions.py` | Plain names and meanings for dataset features, used in explanations |
| `scripts/explain_model.py` | Averages feature effects per attack type, saved beside the model's evaluation |
| `backend/replay.py` | Runs one background replay at a time for the API |
| `backend/evaluation_results.py` | Reads saved evaluation runs for the API |
| `dashboard/app.py`, `dashboard/app_pages/` | Streamlit navigation and the four screens |
| `dashboard/api_client.py` | API client, which is the dashboard's only data source |
| `dashboard/views.py`, `dashboard/charts.py` | Display tables and charts |
| `scripts/run_protocol.py` | Evaluation protocol and versioned model output |
| `experiments/` | Earlier one-off experiments, kept as the historical record |

## Request flow

1. The API checks the request shape: batch size, IP format, and known fields.
2. `validate_features` checks the columns and puts them in model order. Known dataset columns the model doesn't use, such as `IAT`, are ignored. Any other wrong schema fails the request. A bad value rejects only that row, and rows are never imputed because the model was never evaluated on imputed values.
3. `Detector` returns the predicted class and per-class probabilities.
4. `assess_risk` calculates the score from the class, confidence, and recent repeats.
5. If the score reaches `alert_min_score`, the detection joins a matching active alert or opens a new one.
6. The event and any alert changes are committed in one transaction per request.

Detection has no dependency on the API, so a command-line tool can reuse it directly.

## Dashboard

The dashboard reads and changes data only through the REST API, so everything it shows comes from the database or saved evaluation runs. When the API is down, each screen says how to start it. Overview refreshes every 2 seconds, and the sidebar labels every screen as dataset replay and detections as predictions.

Time charts use the time Medora processed each record, because the dataset has no timestamps. Alerts show known labels only for replayed data, and network fields stay empty when the data has none.

## Risk scoring

```
score = base[class] * (floor + (1 - floor) * confidence)
        + min(repeat_cap, repeat_points * recent_repeats)
```

| Setting | Default | Meaning |
|---|---|---|
| Base score | ARP 70, MQTT flood 75, port scan 45, benign 0 | Judgement of potential impact in a clinical network |
| `confidence_floor` | 0.5 | Confidence can reduce a score by at most half |
| `repeat_points`, `repeat_cap` | 2, 20 | Bonus for matching events in the window |
| `repeat_window_seconds` | 60 | Window for counting repeats |
| Severity thresholds | 20, 40, 60, 80 | Low, Medium, High, Critical |

Confidence only scales the score, so a very confident port scan still ranks below a moderately confident flood. To change any setting, point `MEDORA_RISK_CONFIG` at a JSON file containing the keys to override.

Without source addresses, repeats are counted per class, so sustained replayed traffic quickly reaches the repeat cap. Most replayed attack events therefore score at base + 20.

## Alerts

- **Grouping:** a detection joins an active alert with the same source, class, and `src_ip` if that alert was seen in the last 300 seconds. Replayed and API traffic never merge.
- **Risk:** an alert's score and severity only rise while it's active.
- **Status:** `open` can move to `acknowledged`, `resolved`, or `false_positive`. `acknowledged` can move to `resolved` or `false_positive`. The last two are final, and a later detection opens a new alert.
- **Explanation:** alerts state that they are predictions and list the features that most raised the predicted class's probability for the record that opened the alert.

Feature effects come from Random Forest decision paths. Each split's change in class probability is credited to its feature, and a test confirms the effects add up to the model's output. They describe the model's reasoning, not what caused the traffic. Other model types return no feature list.

## Explainability (XAI)

Medora uses explainable AI in two places, both built on the tree path contributions above:

- **Local explanations:** each alert shows a one-sentence reason and a chart of the features that pushed the model towards its answer, in percentage points of probability.
- **Global explanations:** `scripts/explain_model.py` averages the contributions over 500 test records predicted as each type and saves them as `feature_effects.json` in the model's evaluation folder. The Model performance screen shows the top features for each attack type.

Path contributions are computed directly from the trained trees, so no extra library such as SHAP is needed. Feature names are shown in plain language using `detection/feature_descriptions.py`. The datasets don't define their features, so these names come from the column names, and statistics whose names don't describe them are labelled only as packet-size statistics.

## Capture findings

`analyze_pcap --send` reads a capture on the client side and posts only validated findings to the API, so the API never receives or opens files. Each finding becomes an alert with source `capture` and detector `arp_rules`. Its ID comes from the capture name, rule, address, MACs, and first time seen, so sending a capture again adds no duplicates.

Rule alerts don't have a confidence value, because rules don't produce a probability. Their risk scores are fixed judgements capped at Medium: 55 for `binding_flapping`, 45 for `unsolicited_reply_burst`, 30 for `ip_mac_conflict`, and 20 for `mac_claims_many_addresses`. They don't count as detection events, so traffic statistics still describe model predictions only.

## Device baselines

`detection/device_baseline.py` reads only frame headers, so payloads are never parsed or stored. For each source MAC and minute, it counts frames and bytes sent, frames received, MACs contacted, source and destination IPs, destination ports, new TCP connections, and ARP requests and replies.

A device's limit for each measure is `max(quantile x multiplier, minimum)`, computed across the whole learning period with silent minutes counted as zeros. The minimums stop tiny rises on a nearly silent device from counting. Only rises are flagged, so a device going quiet isn't reported. Enough measures must rise, for enough consecutive minutes, to make a finding, and a device missing from the baseline becomes an `unknown_device` finding once it sends 10 frames in a minute. Findings disclose when a device had under 30 active minutes of history.

Baselines are saved as plain JSON, unlike models, so loading one can't run code. Each minute tracks at most 5,000 addresses, so a capture with invented addresses can't grow memory without limit. Addresses that only received frames aren't treated as devices, because anyone can send to any MAC.

`check --send` follows the same path as ARP findings: the client reads the capture, and the API only receives validated findings. These alerts have detector `device_baseline`, no confidence value, and a fixed score: 35 for `behaviour_change` and 25 for `unknown_device`, both Low. `DELETE /rule-findings` removes them along with other capture alerts.

## Security decisions

- The API takes JSON only, with no file uploads or caller-supplied paths. Replay can only choose a known dataset split.
- Both servers are set up to listen on `127.0.0.1` only. The README's commands and `.streamlit/config.toml` set this.
- SQL values are always parameterized, and column names are fixed in code.
- Batch size and field values are limited and validated.
- Errors and model info don't reveal file locations.
- Model files are unpickled when loaded, so `MEDORA_MODEL_PATH` must only point to trusted files.
- Only the features cited by an alert's explanation are stored, not every event's full feature vector.
