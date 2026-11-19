"""Central configuration for Medora.

Paths can be overridden with environment variables so nothing outside this
module needs to know where the project lives on a given machine.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("MEDORA_DATA_DIR", PROJECT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(os.environ.get("MEDORA_MODEL_DIR", PROJECT_DIR / "models"))
EVALUATION_DIR = Path(
    os.environ.get("MEDORA_EVALUATION_DIR", PROJECT_DIR / "experiments" / "evaluation")
)

ACTIVE_MODEL_PATH = Path(
    os.environ.get(
        "MEDORA_MODEL_PATH",
        MODEL_DIR / "medora_random_forest_20261101-173518.joblib",
    )
)
DATABASE_PATH = Path(os.environ.get("MEDORA_DATABASE_PATH", DATA_DIR / "medora.db"))
RISK_CONFIG_PATH = os.environ.get("MEDORA_RISK_CONFIG")
# Optional shared secret for write endpoints. Unset means local-only use.
API_KEY = os.environ.get("MEDORA_API_KEY")
MAX_BATCH_RECORDS = 1000
API_URL = os.environ.get("MEDORA_API_URL", "http://127.0.0.1:8000")

DATASET_ID = "CICIoMT2024 (WiFi/MQTT extracted-feature CSVs, 4-class subset)"
LABEL_COLUMN = "label"
BENIGN_CLASS = "Benign"
SPLITS = ("train", "test")
SEED = 42

# Reporting order for classes. A fitted model's `classes_` is sorted
# alphabetically and differs from this, so probability columns must always be
# mapped through `model.classes_`, never by position against this tuple.
CLASSES = (
    "Benign",
    "ARP_Spoofing",
    "MQTT_DDoS_Connect_Flood",
    "Recon_Port_Scan",
)

# The exact 45-column schema, in the order used by the source CSVs and by the
# saved models. Spellings such as "Magnitue" come from the dataset.
FEATURE_COLUMNS = (
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count",
    "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet",
    "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv",
    "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
)

# Dataset columns models must not learn from. IAT behaves like a capture-session
# clock rather than a traffic property (see docs/results.md), so a model using it
# would partly recognize the capture instead of the attack.
EXCLUDED_FEATURES = ("IAT",)
MODEL_FEATURE_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in EXCLUDED_FEATURES)
