"""Rule-based ARP spoofing detection on ARP packets.

The rules look at behaviour spoofing depends on: more than one MAC claiming an IP.
Legitimate events can trigger them too, such as failover or a device changing
network card, so a finding is something to investigate, not a confirmed attack.
"""

import json
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from detection.pcap_reader import ARP_REPLY, ARP_REQUEST, ArpPacket

UNSPECIFIED_IP = "0.0.0.0"
ZERO_MAC = "00:00:00:00:00:00"
BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"

# Chosen on the CICIoMT2024 training captures, where only these rules stayed silent
# on benign traffic. The others fired there too: one-off conflicts, and a machine
# hosting many addresses. They remain as supporting evidence only.
PRIMARY_RULES = ("binding_flapping", "unsolicited_reply_burst")


@dataclass(frozen=True)
class ArpRuleConfig:
    """Rule thresholds. Set them from training captures, never from test captures."""

    # Two MACs claiming one IP within this time is a conflict, not a device change.
    conflict_window_seconds: float = 60.0
    flap_window_seconds: float = 60.0
    flap_min_changes: int = 3
    unsolicited_window_seconds: float = 10.0
    unsolicited_min_replies: int = 5
    # A reply counts as solicited if a request for that IP came this recently.
    request_timeout_seconds: float = 5.0
    multi_ip_window_seconds: float = 60.0
    multi_ip_min_addresses: int = 3
    # Repeats of one finding within this time are counted, not reported again.
    cooldown_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass
class ArpFinding:
    rule: str
    first_seen: float
    last_seen: float
    ip: str | None
    macs: list[str]
    count: int
    detail: str


@dataclass
class _IpState:
    current_mac: str
    last_seen_by_mac: dict[str, float] = field(default_factory=dict)
    changes: deque = field(default_factory=deque)


class ArpMonitor:
    """Tracks IP-to-MAC claims over time. Feed packets in timestamp order."""

    def __init__(self, config: ArpRuleConfig | None = None) -> None:
        self.config = config or ArpRuleConfig()
        self.findings: list[ArpFinding] = []
        self.packets_seen = 0
        self._ips: dict[str, _IpState] = {}
        self._recent_requests: dict[str, float] = {}
        self._unsolicited: dict[str, deque] = defaultdict(deque)
        self._ips_claimed_by_mac: dict[str, dict[str, float]] = defaultdict(dict)
        self._open_findings: dict[tuple, ArpFinding] = {}
        self._last_prune = 0.0

    def process(self, packets: Iterable[ArpPacket]) -> list[ArpFinding]:
        for packet in packets:
            self.observe(packet)
        return self.findings

    def observe(self, packet: ArpPacket) -> None:
        self.packets_seen += 1
        now = packet.timestamp
        self._prune(now)

        if packet.opcode == ARP_REQUEST:
            self._recent_requests[packet.target_ip] = now

        # A probe has no sender address, so it claims nothing.
        if packet.sender_ip == UNSPECIFIED_IP or packet.sender_mac in (ZERO_MAC, BROADCAST_MAC):
            return

        self._check_binding(packet, now)
        self._check_many_addresses(packet, now)
        if packet.opcode == ARP_REPLY:
            self._check_unsolicited(packet, now)

    def _check_binding(self, packet: ArpPacket, now: float) -> None:
        state = self._ips.get(packet.sender_ip)
        if state is None:
            self._ips[packet.sender_ip] = _IpState(packet.sender_mac, {packet.sender_mac: now})
            return

        previous_seen = {
            mac: seen for mac, seen in state.last_seen_by_mac.items()
            if mac != packet.sender_mac and now - seen <= self.config.conflict_window_seconds
        }
        state.last_seen_by_mac[packet.sender_mac] = now

        if previous_seen:
            macs = sorted({packet.sender_mac, *previous_seen})
            self._record(
                "ip_mac_conflict",
                (packet.sender_ip, tuple(macs)),
                now,
                packet.sender_ip,
                macs,
                f"{len(macs)} MAC addresses claimed {packet.sender_ip} within "
                f"{self.config.conflict_window_seconds:g} seconds.",
            )

        if state.current_mac != packet.sender_mac:
            state.current_mac = packet.sender_mac
            state.changes.append(now)
            while state.changes and now - state.changes[0] > self.config.flap_window_seconds:
                state.changes.popleft()
            if len(state.changes) >= self.config.flap_min_changes:
                macs = sorted(state.last_seen_by_mac)
                self._record(
                    "binding_flapping",
                    (packet.sender_ip,),
                    now,
                    packet.sender_ip,
                    macs,
                    f"The MAC for {packet.sender_ip} changed {len(state.changes)} times within "
                    f"{self.config.flap_window_seconds:g} seconds.",
                )

    def _check_unsolicited(self, packet: ArpPacket, now: float) -> None:
        requested_at = self._recent_requests.get(packet.sender_ip)
        if requested_at is not None and now - requested_at <= self.config.request_timeout_seconds:
            return
        replies = self._unsolicited[packet.sender_mac]
        replies.append(now)
        while replies and now - replies[0] > self.config.unsolicited_window_seconds:
            replies.popleft()
        if len(replies) >= self.config.unsolicited_min_replies:
            self._record(
                "unsolicited_reply_burst",
                (packet.sender_mac,),
                now,
                None,
                [packet.sender_mac],
                f"{packet.sender_mac} sent {len(replies)} replies nobody requested within "
                f"{self.config.unsolicited_window_seconds:g} seconds.",
            )

    def _check_many_addresses(self, packet: ArpPacket, now: float) -> None:
        claimed = self._ips_claimed_by_mac[packet.sender_mac]
        claimed[packet.sender_ip] = now
        for ip in [ip for ip, seen in claimed.items() if now - seen > self.config.multi_ip_window_seconds]:
            del claimed[ip]
        if len(claimed) >= self.config.multi_ip_min_addresses:
            self._record(
                "mac_claims_many_addresses",
                (packet.sender_mac,),
                now,
                None,
                [packet.sender_mac],
                f"{packet.sender_mac} claimed {len(claimed)} IP addresses within "
                f"{self.config.multi_ip_window_seconds:g} seconds. Routers doing proxy ARP can also do this.",
            )

    def _record(self, rule: str, key: tuple, now: float, ip: str | None, macs: list[str], detail: str) -> None:
        full_key = (rule, *key)
        finding = self._open_findings.get(full_key)
        if finding and now - finding.last_seen <= self.config.cooldown_seconds:
            finding.last_seen = now
            finding.count += 1
            finding.detail = detail
            return
        finding = ArpFinding(rule, now, now, ip, macs, 1, detail)
        self._open_findings[full_key] = finding
        self.findings.append(finding)

    def _prune(self, now: float) -> None:
        """Drop expired state, so a long or hostile capture can't grow memory forever."""
        longest = max(
            self.config.conflict_window_seconds,
            self.config.flap_window_seconds,
            self.config.unsolicited_window_seconds,
            self.config.multi_ip_window_seconds,
            self.config.cooldown_seconds,
            self.config.request_timeout_seconds,
        )
        if now - self._last_prune < longest:
            return
        self._last_prune = now
        cutoff = now - longest

        for ip, state in list(self._ips.items()):
            state.last_seen_by_mac = {m: s for m, s in state.last_seen_by_mac.items() if s >= cutoff}
            if not state.last_seen_by_mac:
                del self._ips[ip]
        self._recent_requests = {ip: seen for ip, seen in self._recent_requests.items() if seen >= cutoff}
        for mac, replies in list(self._unsolicited.items()):
            if not replies or replies[-1] < cutoff:
                del self._unsolicited[mac]
        for mac, claimed in list(self._ips_claimed_by_mac.items()):
            if not any(seen >= cutoff for seen in claimed.values()):
                del self._ips_claimed_by_mac[mac]
        self._open_findings = {k: f for k, f in self._open_findings.items() if f.last_seen >= cutoff}


def load_rule_config(path: str | Path | None) -> ArpRuleConfig:
    if path is None:
        return ArpRuleConfig()
    return ArpRuleConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def primary_findings(findings: list[ArpFinding]) -> list[ArpFinding]:
    """Findings from the rules strong enough to flag a capture on their own."""
    return [finding for finding in findings if finding.rule in PRIMARY_RULES]


def summarize_findings(findings: list[ArpFinding]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        counts[finding.rule] += 1
    return dict(counts)
