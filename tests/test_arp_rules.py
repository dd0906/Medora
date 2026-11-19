import socket
import struct

import dpkt
import pytest

from detection.arp_rules import ArpMonitor, ArpRuleConfig, load_rule_config, primary_findings, summarize_findings
from detection.pcap_reader import (
    ARP_REPLY,
    ARP_REQUEST,
    ArpPacket,
    CaptureStats,
    UnsupportedCapture,
    format_mac,
    parse_arp_frame,
    read_arp_packets,
)
from scripts.analyze_pcap import analyze
from scripts.evaluate_arp_rules import ATTACK, BENIGN, find_capture, summarize_group

VICTIM = "aa:aa:aa:aa:aa:aa"
GATEWAY = "bb:bb:bb:bb:bb:bb"
ATTACKER = "ee:ee:ee:ee:ee:ee"
GATEWAY_IP = "10.0.0.1"
VICTIM_IP = "10.0.0.5"


def packet(timestamp, sender_mac, sender_ip, target_ip=VICTIM_IP, opcode=ARP_REPLY):
    return ArpPacket(timestamp, opcode, sender_mac, sender_mac, sender_ip, "00:00:00:00:00:00", target_ip)


def rules_fired(packets, config=None):
    return summarize_findings(ArpMonitor(config).process(packets))


def normal_exchange(start, count=5):
    """A device repeatedly asks for the gateway, and the gateway answers."""
    packets = []
    for i in range(count):
        time = start + i * 30
        packets.append(packet(time, VICTIM, VICTIM_IP, GATEWAY_IP, ARP_REQUEST))
        packets.append(packet(time + 0.01, GATEWAY, GATEWAY_IP, VICTIM_IP, ARP_REPLY))
    return packets


# Rules


def test_normal_traffic_raises_nothing():
    assert rules_fired(normal_exchange(0)) == {}


def test_spoofed_gateway_is_detected_as_conflict():
    packets = [*normal_exchange(0, 2), packet(40, ATTACKER, GATEWAY_IP)]
    findings = ArpMonitor().process(packets)
    conflict = next(f for f in findings if f.rule == "ip_mac_conflict")
    assert conflict.ip == GATEWAY_IP
    assert conflict.macs == sorted([ATTACKER, GATEWAY])


def test_device_changing_mac_after_a_long_gap_is_not_a_conflict():
    packets = [packet(0, GATEWAY, GATEWAY_IP), packet(1000, ATTACKER, GATEWAY_IP)]
    assert "ip_mac_conflict" not in rules_fired(packets)


def test_repeated_mac_changes_are_flapping():
    packets = []
    for i in range(6):
        packets.append(packet(i * 5, GATEWAY if i % 2 else ATTACKER, GATEWAY_IP))
    assert "binding_flapping" in rules_fired(packets)


def test_a_single_change_is_not_flapping():
    packets = [packet(0, GATEWAY, GATEWAY_IP), packet(5, ATTACKER, GATEWAY_IP)]
    assert "binding_flapping" not in rules_fired(packets)


def test_burst_of_unrequested_replies_is_detected():
    packets = [packet(i * 0.5, ATTACKER, GATEWAY_IP) for i in range(6)]
    assert "unsolicited_reply_burst" in rules_fired(packets)


def test_requested_replies_are_not_unsolicited():
    packets = []
    for i in range(6):
        packets.append(packet(i, VICTIM, VICTIM_IP, GATEWAY_IP, ARP_REQUEST))
        packets.append(packet(i + 0.01, GATEWAY, GATEWAY_IP, VICTIM_IP, ARP_REPLY))
    assert "unsolicited_reply_burst" not in rules_fired(packets)


def test_late_reply_counts_as_unsolicited():
    packets = []
    for i in range(6):
        packets.append(packet(i * 1.0, VICTIM, VICTIM_IP, GATEWAY_IP, ARP_REQUEST))
    packets = [packets[0], *[packet(6 + i * 0.5, GATEWAY, GATEWAY_IP) for i in range(6)]]
    assert "unsolicited_reply_burst" in rules_fired(packets)


def test_mac_claiming_many_addresses_is_detected():
    packets = [packet(i, ATTACKER, f"10.0.0.{i + 1}") for i in range(3)]
    assert "mac_claims_many_addresses" in rules_fired(packets)


def test_address_probes_claim_nothing():
    packets = [packet(i, ATTACKER, "0.0.0.0", GATEWAY_IP, ARP_REQUEST) for i in range(10)]
    assert rules_fired(packets) == {}


def test_repeats_are_grouped_during_cooldown_and_reported_again_after():
    monitor = ArpMonitor()
    packets = [packet(0, GATEWAY, GATEWAY_IP)]
    packets += [packet(1 + i, ATTACKER, GATEWAY_IP) for i in range(4)]
    monitor.process(packets)
    conflicts = [f for f in monitor.findings if f.rule == "ip_mac_conflict"]
    assert len(conflicts) == 1
    assert conflicts[0].count == 4

    monitor.process([packet(5, GATEWAY, GATEWAY_IP), packet(6, GATEWAY, GATEWAY_IP)])
    assert [f.count for f in monitor.findings if f.rule == "ip_mac_conflict"] == [6]

    monitor.process([packet(200, GATEWAY, GATEWAY_IP), packet(201, ATTACKER, GATEWAY_IP)])
    assert len([f for f in monitor.findings if f.rule == "ip_mac_conflict"]) == 2


def test_old_state_is_dropped_from_memory():
    monitor = ArpMonitor()
    monitor.process(packet(i * 0.001, f"02:00:00:00:{i >> 8:02x}:{i & 255:02x}", f"10.1.{i >> 8}.{i & 255}") for i in range(5000))
    monitor.observe(packet(100_000, VICTIM, VICTIM_IP))
    assert len(monitor._ips) == 1
    assert len(monitor._ips_claimed_by_mac) == 1


def test_thresholds_are_configurable(tmp_path):
    packets = [packet(i, ATTACKER, f"10.0.0.{i + 1}") for i in range(3)]
    path = tmp_path / "rules.json"
    path.write_text('{"multi_ip_min_addresses": 10}', encoding="utf-8")
    assert "mac_claims_many_addresses" not in rules_fired(packets, load_rule_config(path))


def test_invalid_thresholds_are_refused():
    with pytest.raises(ValueError):
        ArpRuleConfig(flap_min_changes=0)


# Capture files


def ethernet_arp(sender_mac, sender_ip, opcode=ARP_REPLY, target_ip=VICTIM_IP):
    mac = bytes.fromhex(sender_mac.replace(":", ""))
    arp = dpkt.arp.ARP(
        op=opcode, sha=mac, spa=socket.inet_aton(sender_ip), tha=b"\0" * 6, tpa=socket.inet_aton(target_ip)
    )
    return dpkt.ethernet.Ethernet(src=mac, dst=b"\xff" * 6, type=dpkt.ethernet.ETH_TYPE_ARP, data=arp)


def write_capture(path, frames, link_type=dpkt.pcap.DLT_EN10MB):
    with open(path, "wb") as handle:
        writer = dpkt.pcap.Writer(handle, linktype=link_type)
        for timestamp, frame in frames:
            writer.writepkt(bytes(frame), ts=timestamp)


def test_reads_arp_frames_and_ignores_other_traffic(tmp_path):
    other = dpkt.ethernet.Ethernet(src=b"\x01" * 6, dst=b"\x02" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=b"x" * 40)
    path = tmp_path / "a.pcap"
    write_capture(path, [(1.0, ethernet_arp(GATEWAY, GATEWAY_IP)), (2.0, other), (3.0, ethernet_arp(VICTIM, VICTIM_IP))])

    stats = CaptureStats()
    packets = list(read_arp_packets(path, stats))

    assert [p.sender_ip for p in packets] == [GATEWAY_IP, VICTIM_IP]
    assert packets[0].sender_mac == GATEWAY
    assert (stats.frames, stats.arp_frames, stats.malformed_arp_frames) == (3, 2, 0)
    assert stats.duration_seconds == 2.0


def test_vlan_tagged_arp_is_read(tmp_path):
    plain = bytes(ethernet_arp(GATEWAY, GATEWAY_IP))
    tagged = plain[:12] + struct.pack("!HH", 0x8100, 7) + plain[12:]
    path = tmp_path / "vlan.pcap"
    write_capture(path, [(1.0, tagged)])
    assert [p.sender_ip for p in read_arp_packets(path)] == [GATEWAY_IP]


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"\x00" * 5,
        b"\x00" * 12 + b"\x08\x06",
        b"\x00" * 12 + b"\x08\x06" + b"\x00" * 3,
        b"\x00" * 12 + b"\x81\x00",
    ],
)
def test_malformed_frames_are_counted_not_fatal(tmp_path, frame):
    path = tmp_path / "bad.pcap"
    write_capture(path, [(1.0, frame), (2.0, ethernet_arp(GATEWAY, GATEWAY_IP))])
    assert [p.sender_ip for p in read_arp_packets(path)] == [GATEWAY_IP]


def test_non_ipv4_arp_is_skipped():
    raw = bytes(ethernet_arp(GATEWAY, GATEWAY_IP))
    wrong_protocol = raw[:16] + b"\x86\xdd" + raw[18:]
    assert parse_arp_frame(wrong_protocol, 1.0) is None


def test_unknown_opcode_is_skipped():
    raw = bytes(ethernet_arp(GATEWAY, GATEWAY_IP))
    assert parse_arp_frame(raw[:20] + b"\x00\x09" + raw[22:], 1.0) is None


def test_truncated_capture_keeps_what_was_read(tmp_path):
    path = tmp_path / "cut.pcap"
    write_capture(path, [(1.0, ethernet_arp(GATEWAY, GATEWAY_IP)), (2.0, ethernet_arp(VICTIM, VICTIM_IP))])
    path.write_bytes(path.read_bytes()[:-10])

    stats = CaptureStats()
    packets = list(read_arp_packets(path, stats))

    assert [p.sender_ip for p in packets] == [GATEWAY_IP]


def test_non_capture_file_is_refused(tmp_path):
    path = tmp_path / "notes.pcap"
    path.write_bytes(b"this is not a capture file at all")
    with pytest.raises(UnsupportedCapture):
        list(read_arp_packets(path))


def test_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.pcap"
    path.write_bytes(b"")
    with pytest.raises(UnsupportedCapture):
        list(read_arp_packets(path))


def test_non_ethernet_capture_is_refused(tmp_path):
    path = tmp_path / "wifi.pcap"
    write_capture(path, [(1.0, b"\x00" * 30)], link_type=dpkt.pcap.DLT_IEEE802_11)
    with pytest.raises(UnsupportedCapture, match="Ethernet"):
        list(read_arp_packets(path))


def test_mac_formatting():
    assert format_mac(bytes.fromhex("0a1b2c3d4e5f")) == "0a:1b:2c:3d:4e:5f"


def test_analyze_reports_a_spoofed_capture_end_to_end(tmp_path):
    frames = [(float(i), ethernet_arp(GATEWAY, GATEWAY_IP)) for i in range(2)]
    frames.append((10.0, ethernet_arp(ATTACKER, GATEWAY_IP)))
    path = tmp_path / "spoof.pcap"
    write_capture(path, frames)

    report = analyze(path)

    assert report["capture"] == "spoof.pcap"
    assert report["arp_frames"] == 3
    assert report["findings_by_rule"]["ip_mac_conflict"] == 1
    assert report["seconds_to_first_finding"] == 10.0
    assert report["findings"][0]["ip"] == GATEWAY_IP


def test_analyze_reports_a_clean_capture(tmp_path):
    path = tmp_path / "clean.pcap"
    write_capture(path, [(0.0, ethernet_arp(VICTIM, VICTIM_IP, ARP_REQUEST, GATEWAY_IP))])
    report = analyze(path)
    assert report["findings"] == []
    assert report["seconds_to_first_finding"] is None


def test_single_conflict_is_supporting_evidence_only():
    packets = [packet(0, GATEWAY, GATEWAY_IP), packet(5, ATTACKER, GATEWAY_IP)]
    findings = ArpMonitor().process(packets)
    assert summarize_findings(findings) == {"ip_mac_conflict": 1}
    assert primary_findings(findings) == []


def test_many_addresses_on_one_mac_is_supporting_evidence_only():
    findings = ArpMonitor().process(packet(i * 30, ATTACKER, f"10.0.0.{i + 1}", opcode=ARP_REQUEST) for i in range(3))
    assert summarize_findings(findings) == {"mac_claims_many_addresses": 1}
    assert primary_findings(findings) == []


def test_sustained_spoofing_flags_a_capture(tmp_path):
    frames = [(float(i * 5), ethernet_arp(GATEWAY if i % 2 else ATTACKER, GATEWAY_IP)) for i in range(8)]
    path = tmp_path / "spoof.pcap"
    write_capture(path, frames)
    report = analyze(path)
    assert report["flagged"] is True
    assert report["primary_findings"] >= 1


def test_supporting_findings_alone_do_not_flag_a_capture(tmp_path):
    path = tmp_path / "one_off.pcap"
    write_capture(path, [(0.0, ethernet_arp(GATEWAY, GATEWAY_IP)), (5.0, ethernet_arp(ATTACKER, GATEWAY_IP))])
    report = analyze(path)
    assert report["findings"]
    assert report["flagged"] is False
    assert report["seconds_to_first_primary_finding"] is None


def test_group_summary_counts_flagged_captures_and_benign_rate():
    def report(label, flagged, primary, total, seconds):
        return {"label": label, "flagged": flagged, "primary_findings": primary, "findings": [0] * total, "duration_seconds": seconds}

    summary = summarize_group([
        report(ATTACK, True, 4, 6, 100),
        report(BENIGN, False, 0, 5, 3600),
        report(BENIGN, True, 2, 2, 3600),
    ])
    assert (summary["attack_captures_flagged"], summary["attack_captures"]) == (1, 1)
    assert (summary["benign_captures_flagged"], summary["benign_captures"]) == (1, 2)
    assert summary["benign_primary_findings_per_hour"] == 1.0
    assert summary["benign_supporting_findings"] == 5


def test_find_capture_needs_exactly_one_match(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.pcap").write_bytes(b"")
    assert find_capture("x.pcap", tmp_path) == tmp_path / "a" / "x.pcap"
    (tmp_path / "b" / "x.pcap").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        find_capture("x.pcap", tmp_path)
    with pytest.raises(FileNotFoundError):
        find_capture("absent.pcap", tmp_path)
