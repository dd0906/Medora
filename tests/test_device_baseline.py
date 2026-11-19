import socket
import struct

import dpkt
import pandas as pd
import pytest

from detection import device_baseline
from detection.device_baseline import (
    FEATURES,
    KIND_BEHAVIOUR,
    KIND_UNKNOWN,
    MINIMUM_LIMITS,
    BaselineConfig,
    DeviceBaseline,
    capture_windows,
    summarize_frames,
)
from detection.pcap_reader import CaptureStats, UnsupportedCapture
from scripts.evaluate_arp_rules import ATTACK, BENIGN
from scripts.evaluate_baselines import MAX_FINDINGS_PER_HOUR, compare_rates, join_captures, search_settings

SENSOR = "aa:aa:aa:aa:aa:01"
GATEWAY = "aa:aa:aa:aa:aa:02"
STRANGER = "ee:ee:ee:ee:ee:01"
BROADCAST = "ff:ff:ff:ff:ff:ff"
STRICT = BaselineConfig(quantile=1.0, multiplier=1.0, min_features=1, min_windows=1)


def raw_mac(mac):
    return bytes.fromhex(mac.replace(":", ""))


def ethernet(src, dst, ether_type, payload=b""):
    return raw_mac(dst) + raw_mac(src) + struct.pack("!H", ether_type) + payload


def ipv4(src_ip, dst_ip, protocol=0, transport=b"", fragment=0):
    header = struct.pack(
        "!BBHHHBBH4s4s", 0x45, 0, 20 + len(transport), 0, fragment, 64, protocol, 0,
        socket.inet_aton(src_ip), socket.inet_aton(dst_ip),
    )
    return header + transport


def tcp(dst_port, flags):
    return struct.pack("!HHIIBBHHH", 40000, dst_port, 0, 0, 0x50, flags, 0, 0, 0)


def udp(dst_port):
    return struct.pack("!HHHH", 40000, dst_port, 8, 0)


def arp(opcode):
    return struct.pack("!HHBBH", 1, 0x0800, 6, 4, opcode) + bytes(20)


def rows(frames):
    return {(row["device"], row["window"]): row for row in summarize_frames(frames)}


def window(device, position, **values):
    return {"device": device, "window": position, **dict.fromkeys(FEATURES, 0), **values}


def table(*items):
    return pd.DataFrame(list(items), columns=["device", "window", *FEATURES])


# Summarizing frames


def test_counts_what_a_device_sent_and_received():
    frame = ethernet(SENSOR, GATEWAY, 0x0800, ipv4("10.0.0.5", "10.0.0.1", 17, udp(53)))
    result = rows([(0.0, frame), (1.0, frame)])
    sensor = result[(SENSOR, 0)]
    assert sensor["frames_sent"] == 2
    assert sensor["bytes_sent"] == 2 * len(frame)
    assert (sensor["source_ips"], sensor["destination_ips"], sensor["destination_ports"]) == (1, 1, 1)
    assert sensor["peer_macs"] == 1
    assert result[(GATEWAY, 0)]["frames_received"] == 2
    assert result[(GATEWAY, 0)]["frames_sent"] == 0


def test_a_new_syn_counts_but_a_syn_ack_doesnt():
    syn = ethernet(SENSOR, GATEWAY, 0x0800, ipv4("10.0.0.5", "10.0.0.1", 6, tcp(80, 0x02)))
    syn_ack = ethernet(SENSOR, GATEWAY, 0x0800, ipv4("10.0.0.5", "10.0.0.1", 6, tcp(81, 0x12)))
    sensor = rows([(0.0, syn), (0.1, syn_ack)])[(SENSOR, 0)]
    assert sensor["tcp_syns"] == 1
    assert sensor["destination_ports"] == 2


def test_counts_arp_by_type():
    frames = [
        (0.0, ethernet(SENSOR, BROADCAST, 0x0806, arp(1))),
        (0.1, ethernet(SENSOR, GATEWAY, 0x0806, arp(2))),
        (0.2, ethernet(SENSOR, GATEWAY, 0x0806, arp(2))),
    ]
    sensor = rows(frames)[(SENSOR, 0)]
    assert (sensor["arp_requests"], sensor["arp_replies"]) == (1, 2)


def test_reads_past_a_vlan_tag():
    tag = struct.pack("!HH", 5, 0x0800)
    frame = ethernet(SENSOR, GATEWAY, 0x8100, tag + ipv4("10.0.0.5", "10.0.0.1", 17, udp(53)))
    assert rows([(0.0, frame)])[(SENSOR, 0)]["destination_ports"] == 1


def test_splits_traffic_into_windows():
    frame = ethernet(SENSOR, GATEWAY, 0x0800)
    result = rows([(100.0, frame), (159.0, frame), (160.0, frame), (400.0, frame)])
    assert result[(SENSOR, 0)]["frames_sent"] == 2
    assert result[(SENSOR, 1)]["frames_sent"] == 1
    assert result[(SENSOR, 5)]["frames_sent"] == 1


def test_a_timestamp_running_backwards_stays_in_the_current_window():
    frame = ethernet(SENSOR, GATEWAY, 0x0800)
    result = rows([(0.0, frame), (70.0, frame), (5.0, frame)])
    assert result[(SENSOR, 1)]["frames_sent"] == 2


def test_multicast_and_broadcast_sources_are_not_devices():
    frame = ethernet(BROADCAST, GATEWAY, 0x0800)
    result = rows([(0.0, frame)])
    assert (BROADCAST, 0) not in result
    assert result[(GATEWAY, 0)]["frames_received"] == 1


@pytest.mark.parametrize(
    "payload",
    [b"", b"\x45", ipv4("10.0.0.5", "10.0.0.1", 6)[:19], ipv4("10.0.0.5", "10.0.0.1", 6), b"\x60" + bytes(30)],
)
def test_damaged_headers_are_counted_as_frames_and_dont_crash(payload):
    frame = ethernet(SENSOR, GATEWAY, 0x0800, payload)
    assert rows([(0.0, frame)])[(SENSOR, 0)]["frames_sent"] == 1


def test_short_arp_and_vlan_frames_dont_crash():
    frames = [(0.0, ethernet(SENSOR, GATEWAY, 0x0806, b"\x00")), (0.1, ethernet(SENSOR, GATEWAY, 0x8100, b"\x00"))]
    assert rows(frames)[(SENSOR, 0)]["frames_sent"] == 2


def test_frames_too_short_for_ethernet_are_skipped():
    assert rows([(0.0, b"\x00" * 5)]) == {}


def test_later_fragments_have_no_ports():
    frame = ethernet(SENSOR, GATEWAY, 0x0800, ipv4("10.0.0.5", "10.0.0.1", 17, udp(53), fragment=100))
    assert rows([(0.0, frame)])[(SENSOR, 0)]["destination_ports"] == 0


def test_invented_addresses_cant_grow_memory_without_limit(monkeypatch):
    monkeypatch.setattr(device_baseline, "MAX_DEVICES_PER_WINDOW", 10)
    frames = [(0.0, ethernet(f"02:00:00:00:{i >> 8:02x}:{i & 255:02x}", BROADCAST, 0x0800)) for i in range(100)]
    assert len(rows(frames)) == 10


def test_reads_a_capture_file_and_records_totals(tmp_path):
    path = tmp_path / "small.pcap"
    with open(path, "wb") as handle:
        writer = dpkt.pcap.Writer(handle)
        writer.writepkt(ethernet(SENSOR, GATEWAY, 0x0800), ts=10.0)
        writer.writepkt(ethernet(SENSOR, GATEWAY, 0x0800), ts=20.0)
    stats = CaptureStats()
    windows = capture_windows(path, stats)
    assert stats.frames == 2
    assert stats.duration_seconds == 10.0
    assert windows.loc[windows["device"] == SENSOR, "frames_sent"].iloc[0] == 2


def test_rejects_a_file_that_isnt_a_capture(tmp_path):
    path = tmp_path / "fake.pcap"
    path.write_bytes(b"not a capture at all, just text")
    with pytest.raises(UnsupportedCapture):
        capture_windows(path)


# Fitting


def test_each_device_gets_its_own_limit():
    windows = table(window(SENSOR, 0, frames_sent=100), window(GATEWAY, 0, frames_sent=5000))
    baseline = DeviceBaseline.fit(windows, 1, STRICT)
    assert baseline.devices[SENSOR]["limits"]["frames_sent"] == 100
    assert baseline.devices[GATEWAY]["limits"]["frames_sent"] == 5000


def test_silent_windows_count_towards_the_limit():
    windows = table(window(SENSOR, 0, frames_sent=1000))
    config = BaselineConfig(quantile=0.5, multiplier=1.0, min_features=1, min_windows=1)
    baseline = DeviceBaseline.fit(windows, 100, config)
    assert baseline.devices[SENSOR]["limits"]["frames_sent"] == MINIMUM_LIMITS["frames_sent"]
    assert baseline.devices[SENSOR]["active_windows"] == 1


def test_multiplier_raises_the_limit_and_the_minimum_still_applies():
    windows = table(window(SENSOR, 0, frames_sent=1000, tcp_syns=1))
    config = BaselineConfig(quantile=1.0, multiplier=2.0, min_features=1, min_windows=1)
    limits = DeviceBaseline.fit(windows, 1, config).devices[SENSOR]["limits"]
    assert limits["frames_sent"] == 2000
    assert limits["tcp_syns"] == MINIMUM_LIMITS["tcp_syns"]


def test_an_address_that_only_received_isnt_learned():
    windows = table(window(SENSOR, 0, frames_sent=5), window(STRANGER, 0, frames_received=9))
    assert set(DeviceBaseline.fit(windows, 1, STRICT).devices) == {SENSOR}


def test_fit_needs_a_positive_window_count():
    with pytest.raises(ValueError):
        DeviceBaseline.fit(table(window(SENSOR, 0, frames_sent=5)), 0)


@pytest.mark.parametrize(
    "settings",
    [
        {"quantile": 0},
        {"quantile": 1.5},
        {"multiplier": 0.5},
        {"min_features": 0},
        {"min_features": len(FEATURES) + 1},
        {"min_windows": 0},
        {"unknown_device_min_frames": 0},
        {"reliable_active_windows": 0},
    ],
)
def test_invalid_settings_are_refused(settings):
    with pytest.raises(ValueError):
        BaselineConfig(**settings)


# Scoring


@pytest.fixture
def baseline():
    history = [window(SENSOR, i, frames_sent=10) for i in range(50)]
    return DeviceBaseline.fit(table(*history), 50, STRICT)


def test_normal_behaviour_raises_nothing(baseline):
    assert baseline.score(table(window(SENSOR, 0, frames_sent=10))) == []


def test_a_rise_above_the_devices_limit_is_a_finding_with_evidence(baseline):
    finding = baseline.score(table(window(SENSOR, 3, frames_sent=500)), capture_start=1000.0)[0]
    assert finding["kind"] == KIND_BEHAVIOUR
    assert finding["device"] == SENSOR
    assert (finding["first_seen"], finding["last_seen"]) == (1180.0, 1240.0)
    evidence = finding["evidence"][0]
    assert evidence["feature"] == "frames_sent"
    assert evidence["observed"] == 500
    assert evidence["limit"] == MINIMUM_LIMITS["frames_sent"]
    assert evidence["typical"] == 10


def test_consecutive_windows_join_and_a_gap_splits_them(baseline):
    windows = table(*[window(SENSOR, i, frames_sent=500) for i in (0, 1, 5)])
    findings = baseline.score(windows)
    assert [finding["windows"] for finding in findings] == [2, 1]


def test_min_windows_ignores_a_short_burst():
    config = BaselineConfig(quantile=1.0, multiplier=1.0, min_features=1, min_windows=2)
    fitted = DeviceBaseline.fit(table(window(SENSOR, 0, frames_sent=10)), 1, config)
    assert fitted.score(table(window(SENSOR, 0, frames_sent=500))) == []
    windows = table(window(SENSOR, 0, frames_sent=500), window(SENSOR, 1, frames_sent=500))
    assert len(fitted.score(windows)) == 1


def test_min_features_needs_measures_to_agree():
    config = BaselineConfig(quantile=1.0, multiplier=1.0, min_features=2, min_windows=1)
    fitted = DeviceBaseline.fit(table(window(SENSOR, 0, frames_sent=10)), 1, config)
    assert fitted.score(table(window(SENSOR, 0, frames_sent=500))) == []
    assert len(fitted.score(table(window(SENSOR, 0, frames_sent=500, bytes_sent=900_000)))) == 1


def test_evidence_keeps_each_measures_peak(baseline):
    windows = table(window(SENSOR, 0, frames_sent=500), window(SENSOR, 1, frames_sent=900))
    evidence = baseline.score(windows)[0]["evidence"]
    assert [item["observed"] for item in evidence] == [900]


def test_a_quiet_devices_small_rise_is_ignored():
    assert DeviceBaseline.fit(table(window(SENSOR, 0, frames_sent=1)), 1, STRICT).score(
        table(window(SENSOR, 0, frames_sent=30))
    ) == []


def test_only_rises_are_flagged(baseline):
    assert baseline.score(table(window(SENSOR, 0, frames_sent=1))) == []


def test_unknown_device_is_reported_once_with_no_invented_evidence(baseline):
    windows = table(window(STRANGER, 2, frames_sent=50), window(STRANGER, 3, frames_sent=50))
    findings = baseline.score(windows)
    assert len(findings) == 1
    assert findings[0]["kind"] == KIND_UNKNOWN
    assert findings[0]["evidence"] == []
    assert findings[0]["windows"] == 2
    assert findings[0]["limited_history"] is True


def test_stray_frames_from_an_unknown_device_are_ignored(baseline):
    assert baseline.score(table(window(STRANGER, 0, frames_sent=2))) == []


def test_an_unknown_address_that_only_received_isnt_reported(baseline):
    assert baseline.score(table(window(STRANGER, 0, frames_received=500))) == []


def test_limited_history_is_disclosed():
    config = BaselineConfig(quantile=1.0, multiplier=1.0, min_features=1, min_windows=1, reliable_active_windows=30)
    history = [window(SENSOR, i, frames_sent=1) for i in range(3)]
    fitted = DeviceBaseline.fit(table(*history), 3, config)
    assert fitted.score(table(window(SENSOR, 0, frames_sent=500)))[0]["limited_history"] is True


def test_baseline_survives_being_saved_and_loaded(tmp_path, baseline):
    path = tmp_path / "nested" / "baseline.json"
    baseline.save(path)
    loaded = DeviceBaseline.load(path)
    assert loaded == baseline
    assert len(loaded.score(table(window(SENSOR, 0, frames_sent=500)))) == 1


# Evaluation script


def test_join_captures_numbers_windows_on_after_each_other():
    first_stats = CaptureStats(first_timestamp=0.0, last_timestamp=119.0)
    second_stats = CaptureStats(first_timestamp=500.0, last_timestamp=510.0)
    joined, total = join_captures([
        (table(window(SENSOR, 1, frames_sent=1)), first_stats),
        (table(window(SENSOR, 0, frames_sent=2)), second_stats),
    ])
    assert total == 3
    assert list(joined["window"]) == [1, 2]


def test_search_keeps_to_the_budget_and_uses_only_what_it_was_given():
    history = [window(SENSOR, i, frames_sent=10) for i in range(70)]
    later = [window(SENSOR, 70 + i, frames_sent=10) for i in range(30)]
    later[10]["frames_sent"] = 800
    search = search_settings(table(*history, *later), 100)
    assert search["fit_windows"] == 70
    assert search["selected"]["findings_per_hour"] <= MAX_FINDINGS_PER_HOUR


def test_search_fails_clearly_when_nothing_meets_the_budget():
    history = [window(SENSOR, i, frames_sent=10) for i in range(70)]
    later = [window(f"02:00:00:00:00:{i:02x}", 70 + i, frames_sent=50) for i in range(30)]
    with pytest.raises(ValueError):
        search_settings(table(*history, *later), 100)


def report(label, findings, hours):
    return {"label": label, "findings": findings, "hours": hours}


def test_compare_rates_uses_totals_across_captures():
    result = compare_rates([report(BENIGN, 1, 1.0), report(BENIGN, 1, 3.0), report(ATTACK, 5, 1.0)])
    assert result["benign_findings_per_hour"] == 0.5
    assert result["attack_to_benign_ratio"] == 10
    assert result["benign_rate_passed"] and result["separation_passed"]


def test_compare_rates_fails_when_attack_rate_is_close_to_benign():
    result = compare_rates([report(BENIGN, 3, 1.0), report(ATTACK, 4, 1.0)])
    assert not result["benign_rate_passed"]
    assert not result["separation_passed"]


def test_compare_rates_handles_no_benign_findings():
    result = compare_rates([report(BENIGN, 0, 1.0), report(ATTACK, 2, 1.0)])
    assert result["attack_to_benign_ratio"] is None
    assert result["separation_passed"]
