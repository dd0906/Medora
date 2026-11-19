"""Per-device behaviour baselines learned from capture files.

Each device, identified by source MAC, is summarized once per time window. A baseline
records how high each measure normally goes for that device, so a quiet sensor and a
busy gateway are judged against themselves. Only rises are flagged, and a finding
means a device behaved unlike its own history, not that it's been compromised.
"""

import json
import logging
import struct
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from detection.pcap_reader import (
    ARP_REPLY,
    ARP_REQUEST,
    ETHERNET_HEADER_LENGTH,
    ETHERTYPE_ARP,
    ETHERTYPE_VLAN,
    VLAN_TAG_LENGTH,
    CaptureStats,
    format_mac,
    read_frames,
)

logger = logging.getLogger("medora.baseline")

ETHERTYPE_IPV4 = 0x0800
PROTOCOL_TCP = 6
PROTOCOL_UDP = 17
TCP_SYN = 0x02
TCP_ACK = 0x10
WINDOW_SECONDS = 60
# A hostile capture could invent endless addresses, so each window tracks a limited number.
MAX_DEVICES_PER_WINDOW = 5000

# The lowest limit for each measure, so tiny rises on a nearly silent device don't count.
# These are judgements about what is too small to matter in one minute.
MINIMUM_LIMITS = {
    "frames_sent": 60,
    "bytes_sent": 60_000,
    "frames_received": 60,
    "peer_macs": 5,
    "source_ips": 5,
    "destination_ips": 5,
    "destination_ports": 5,
    "tcp_syns": 10,
    "arp_requests": 10,
    "arp_replies": 5,
}
FEATURES = tuple(MINIMUM_LIMITS)
KIND_BEHAVIOUR = "behaviour_change"
KIND_UNKNOWN = "unknown_device"


@dataclass(frozen=True)
class BaselineConfig:
    """Set these from benign training captures, never from attack or test captures."""

    quantile: float = 0.99
    multiplier: float = 3.0
    # How many measures must rise in one window for that window to count.
    min_features: int = 1
    # How many counted windows in a row make a finding.
    min_windows: int = 1
    # An unknown device must send this much in a window to count, which ignores stray frames.
    unknown_device_min_frames: int = 10
    # Fewer active windows than this is too little history to trust, which findings disclose.
    reliable_active_windows: int = 30

    def __post_init__(self) -> None:
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile must be above 0 and at most 1.")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1.")
        if not 1 <= self.min_features <= len(FEATURES):
            raise ValueError(f"min_features must be from 1 to {len(FEATURES)}.")
        if self.min_windows < 1 or self.unknown_device_min_frames < 1 or self.reliable_active_windows < 1:
            raise ValueError("Window and frame counts must be positive.")


@dataclass
class _Window:
    frames_sent: int = 0
    bytes_sent: int = 0
    frames_received: int = 0
    tcp_syns: int = 0
    arp_requests: int = 0
    arp_replies: int = 0
    peer_macs: set = field(default_factory=set)
    source_ips: set = field(default_factory=set)
    destination_ips: set = field(default_factory=set)
    destination_ports: set = field(default_factory=set)

    def row(self, device: str, window: int) -> dict[str, Any]:
        return {
            "device": device,
            "window": window,
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "frames_received": self.frames_received,
            "peer_macs": len(self.peer_macs),
            "source_ips": len(self.source_ips),
            "destination_ips": len(self.destination_ips),
            "destination_ports": len(self.destination_ports),
            "tcp_syns": self.tcp_syns,
            "arp_requests": self.arp_requests,
            "arp_replies": self.arp_replies,
        }


def _is_individual(mac: bytes) -> bool:
    """Broadcast and multicast addresses have the lowest bit of the first byte set."""
    return not mac[0] & 1


def _add_frame(sent: _Window, frame: bytes) -> None:
    """Count one frame's headers. Payloads are never read, and short headers are skipped."""
    sent.frames_sent += 1
    sent.bytes_sent += len(frame)
    offset = ETHERNET_HEADER_LENGTH
    ether_type = struct.unpack_from("!H", frame, 12)[0]
    if ether_type == ETHERTYPE_VLAN:
        if len(frame) < offset + VLAN_TAG_LENGTH:
            return
        ether_type = struct.unpack_from("!H", frame, 16)[0]
        offset += VLAN_TAG_LENGTH

    if ether_type == ETHERTYPE_ARP:
        if len(frame) >= offset + 8:
            opcode = struct.unpack_from("!H", frame, offset + 6)[0]
            if opcode == ARP_REQUEST:
                sent.arp_requests += 1
            elif opcode == ARP_REPLY:
                sent.arp_replies += 1
        return

    if ether_type != ETHERTYPE_IPV4 or len(frame) < offset + 20 or frame[offset] >> 4 != 4:
        return
    sent.source_ips.add(frame[offset + 12:offset + 16])
    sent.destination_ips.add(frame[offset + 16:offset + 20])

    # Later fragments have no port header.
    if struct.unpack_from("!H", frame, offset + 6)[0] & 0x1FFF:
        return
    protocol = frame[offset + 9]
    transport = offset + (frame[offset] & 0x0F) * 4
    if protocol not in (PROTOCOL_TCP, PROTOCOL_UDP) or len(frame) < transport + 4:
        return
    sent.destination_ports.add(struct.unpack_from("!H", frame, transport + 2)[0])
    if protocol == PROTOCOL_TCP and len(frame) >= transport + 14:
        flags = frame[transport + 13]
        if flags & TCP_SYN and not flags & TCP_ACK:
            sent.tcp_syns += 1


def summarize_frames(
    frames: Iterable[tuple[float, bytes]],
    window_seconds: int = WINDOW_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Yield one row per active device per window. Feed frames in capture order."""
    start: float | None = None
    current = 0
    windows: dict[bytes, _Window] = {}
    dropped = 0

    for timestamp, frame in frames:
        if len(frame) < ETHERNET_HEADER_LENGTH:
            continue
        if start is None:
            start = timestamp
        # A timestamp running backwards stays in the current window.
        position = max(int((timestamp - start) // window_seconds), current)
        if position != current:
            for device, window in windows.items():
                yield window.row(format_mac(device), current)
            windows = {}
            current = position

        source, destination = frame[6:12], frame[:6]
        for mac in (source, destination):
            if _is_individual(mac) and mac not in windows:
                if len(windows) >= MAX_DEVICES_PER_WINDOW:
                    dropped += 1
                    continue
                windows[mac] = _Window()

        if source in windows:
            _add_frame(windows[source], frame)
            windows[source].peer_macs.add(destination)
        if destination in windows:
            windows[destination].frames_received += 1

    for device, window in windows.items():
        yield window.row(format_mac(device), current)
    if dropped:
        logger.warning("Ignored %d addresses over the limit of %d per window", dropped, MAX_DEVICES_PER_WINDOW)


def capture_windows(path: Path, stats: CaptureStats | None = None) -> pd.DataFrame:
    """Read a capture into a table of device windows."""
    columns = ["device", "window", *FEATURES]
    return pd.DataFrame(summarize_frames(read_frames(path, stats)), columns=columns)


def _devices_that_sent(windows: pd.DataFrame) -> pd.DataFrame:
    """An address that only received frames may not exist, since anyone can send to any MAC."""
    senders = windows.loc[windows["frames_sent"] > 0, "device"].unique()
    return windows[windows["device"].isin(senders)]


@dataclass(frozen=True)
class DeviceBaseline:
    """What's normal for each device, stored as plain JSON, so loading one can't run code."""

    window_seconds: int
    total_windows: int
    config: BaselineConfig
    devices: dict[str, dict[str, Any]]

    @classmethod
    def fit(
        cls,
        windows: pd.DataFrame,
        total_windows: int,
        config: BaselineConfig | None = None,
        window_seconds: int = WINDOW_SECONDS,
    ) -> "DeviceBaseline":
        """Learn from benign windows. Windows where a device was silent count as zeros."""
        config = config or BaselineConfig()
        if total_windows < 1:
            raise ValueError("total_windows must be positive.")
        devices = {}
        for device, rows in _devices_that_sent(windows).groupby("device"):
            active = len(rows)
            idle = max(total_windows - active, 0)
            limits = {}
            typical = {}
            for feature in FEATURES:
                values = np.concatenate([rows[feature].to_numpy(dtype=float), np.zeros(idle)])
                limits[feature] = max(
                    float(np.quantile(values, config.quantile)) * config.multiplier,
                    float(MINIMUM_LIMITS[feature]),
                )
                typical[feature] = float(rows[feature].median())
            devices[device] = {"active_windows": active, "limits": limits, "typical": typical}
        return cls(window_seconds, total_windows, config, devices)

    def to_json(self) -> dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "total_windows": self.total_windows,
            "config": asdict(self.config),
            "devices": self.devices,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DeviceBaseline":
        return cls(data["window_seconds"], data["total_windows"], BaselineConfig(**data["config"]), data["devices"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DeviceBaseline":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, windows: pd.DataFrame, capture_start: float = 0.0) -> list[dict[str, Any]]:
        """Compare new windows with the baseline, and return findings in time order."""
        findings = []
        for name, rows in _devices_that_sent(windows).groupby("device"):
            rows = rows.sort_values("window")
            device = self.devices.get(name)
            if device is None:
                findings.extend(self._unknown_device(name, rows, capture_start))
            else:
                findings.extend(self._changed_behaviour(name, device, rows, capture_start))
        return sorted(findings, key=lambda finding: (finding["first_seen"], finding["device"]))

    def _unknown_device(self, name: str, rows: pd.DataFrame, capture_start: float) -> list[dict[str, Any]]:
        busy = rows[rows["frames_sent"] >= self.config.unknown_device_min_frames]
        if busy.empty:
            return []
        return [{
            "kind": KIND_UNKNOWN,
            "device": name,
            "first_seen": capture_start + int(busy["window"].iloc[0]) * self.window_seconds,
            "last_seen": capture_start + (int(busy["window"].iloc[-1]) + 1) * self.window_seconds,
            "windows": len(busy),
            "evidence": [],
            "baseline_active_windows": 0,
            "limited_history": True,
            "detail": f"{name} isn't in the baseline and sent {int(busy['frames_sent'].sum())} frames.",
        }]

    def _changed_behaviour(
        self, name: str, device: dict[str, Any], rows: pd.DataFrame, capture_start: float
    ) -> list[dict[str, Any]]:
        findings = []
        run: list[tuple[int, list[dict[str, Any]]]] = []
        for row in rows.itertuples(index=False):
            risen = [
                {
                    "feature": feature,
                    "observed": float(getattr(row, feature)),
                    "limit": device["limits"][feature],
                    "typical": device["typical"][feature],
                }
                for feature in FEATURES
                if getattr(row, feature) > device["limits"][feature]
            ]
            if len(risen) < self.config.min_features:
                continue
            if run and row.window != run[-1][0] + 1:
                if len(run) >= self.config.min_windows:
                    findings.append(self._behaviour_finding(name, device, run, capture_start))
                run = []
            run.append((row.window, risen))
        if len(run) >= self.config.min_windows:
            findings.append(self._behaviour_finding(name, device, run, capture_start))
        return findings

    def _behaviour_finding(
        self, name: str, device: dict[str, Any], run: list[tuple[int, list[dict[str, Any]]]], capture_start: float
    ) -> dict[str, Any]:
        # Keep each measure's highest point, so the evidence stays short on a long finding.
        peaks: dict[str, dict[str, Any]] = {}
        for _, risen in run:
            for item in risen:
                if item["feature"] not in peaks or item["observed"] > peaks[item["feature"]]["observed"]:
                    peaks[item["feature"]] = item
        evidence = sorted(peaks.values(), key=lambda item: -item["observed"] / item["limit"])
        names = ", ".join(item["feature"] for item in evidence)
        return {
            "kind": KIND_BEHAVIOUR,
            "device": name,
            "first_seen": capture_start + run[0][0] * self.window_seconds,
            "last_seen": capture_start + (run[-1][0] + 1) * self.window_seconds,
            "windows": len(run),
            "evidence": evidence,
            "baseline_active_windows": device["active_windows"],
            "limited_history": device["active_windows"] < self.config.reliable_active_windows,
            "detail": f"{name} went above its own normal limits for {len(run)} windows in a row: {names}.",
        }
