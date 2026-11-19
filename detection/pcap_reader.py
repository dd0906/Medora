"""Reads ARP traffic from capture files. Capture contents are treated as untrusted.

Only the ARP fields are read, so payloads are never parsed or stored. Malformed
frames are counted and skipped, so a damaged capture can't stop the analysis.
"""

import logging
import socket
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import dpkt

logger = logging.getLogger("medora.pcap")

DLT_ETHERNET = 1
ETHERTYPE_ARP = 0x0806
ETHERTYPE_VLAN = 0x8100
ETHERNET_HEADER_LENGTH = 14
VLAN_TAG_LENGTH = 4
ARP_REQUEST = 1
ARP_REPLY = 2
ARP_IPV4_OVER_ETHERNET = struct.Struct("!HHBBH6s4s6s4s")


class UnsupportedCapture(ValueError):
    """The file isn't a readable capture of Ethernet traffic."""


@dataclass(frozen=True)
class ArpPacket:
    timestamp: float
    opcode: int
    frame_src_mac: str
    sender_mac: str
    sender_ip: str
    target_mac: str
    target_ip: str


@dataclass
class CaptureStats:
    frames: int = 0
    arp_frames: int = 0
    malformed_arp_frames: int = 0
    truncated: bool = False
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    @property
    def duration_seconds(self) -> float:
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0.0
        return self.last_timestamp - self.first_timestamp


def format_mac(raw: bytes) -> str:
    return ":".join(f"{byte:02x}" for byte in raw)


def parse_arp_frame(frame: bytes, timestamp: float) -> ArpPacket | None:
    """Parse one Ethernet frame known to hold ARP. Returns None if it's unusable."""
    offset = ETHERNET_HEADER_LENGTH
    if len(frame) >= offset and struct.unpack_from("!H", frame, 12)[0] == ETHERTYPE_VLAN:
        offset += VLAN_TAG_LENGTH
    try:
        hardware, protocol, hardware_length, protocol_length, opcode, sha, spa, tha, tpa = (
            ARP_IPV4_OVER_ETHERNET.unpack_from(frame, offset)
        )
    except struct.error:
        return None
    if (hardware, protocol, hardware_length, protocol_length) != (1, 0x0800, 6, 4):
        return None
    if opcode not in (ARP_REQUEST, ARP_REPLY):
        return None
    return ArpPacket(
        timestamp=timestamp,
        opcode=opcode,
        frame_src_mac=format_mac(frame[6:12]),
        sender_mac=format_mac(sha),
        sender_ip=socket.inet_ntoa(spa),
        target_mac=format_mac(tha),
        target_ip=socket.inet_ntoa(tpa),
    )


def _is_arp(frame: bytes) -> bool:
    """Check the EtherType directly, which is far faster than parsing every frame."""
    if len(frame) < ETHERNET_HEADER_LENGTH:
        return False
    ether_type = struct.unpack_from("!H", frame, 12)[0]
    if ether_type == ETHERTYPE_VLAN:
        if len(frame) < ETHERNET_HEADER_LENGTH + VLAN_TAG_LENGTH:
            return False
        ether_type = struct.unpack_from("!H", frame, 16)[0]
    return ether_type == ETHERTYPE_ARP


def read_frames(path: Path, stats: CaptureStats | None = None) -> Iterator[tuple[float, bytes]]:
    """Yield (timestamp, frame) in file order, recording totals in `stats` as it goes."""
    stats = stats if stats is not None else CaptureStats()
    with open(path, "rb") as handle:
        try:
            reader = dpkt.pcap.UniversalReader(handle)
        except (ValueError, dpkt.dpkt.Error, struct.error) as error:
            raise UnsupportedCapture("The file isn't a readable pcap or pcapng capture.") from error

        link_type = reader.datalink()
        if link_type != DLT_ETHERNET:
            raise UnsupportedCapture(f"Only Ethernet captures are supported (link type {link_type}).")

        packets = iter(reader)
        while True:
            try:
                timestamp, frame = next(packets)
            except StopIteration:
                return
            except (ValueError, dpkt.dpkt.Error, struct.error, EOFError):
                # dpkt signals a capture cut off mid-record this way. Keep what was read.
                stats.truncated = True
                logger.warning("Capture ended unexpectedly after %d frames", stats.frames)
                return

            timestamp = float(timestamp)
            stats.frames += 1
            if stats.first_timestamp is None:
                stats.first_timestamp = timestamp
            stats.last_timestamp = timestamp
            yield timestamp, frame


def read_arp_packets(path: Path, stats: CaptureStats | None = None) -> Iterator[ArpPacket]:
    """Yield ARP packets in file order, recording totals in `stats` as it goes."""
    stats = stats if stats is not None else CaptureStats()
    for timestamp, frame in read_frames(path, stats):
        if not _is_arp(frame):
            continue
        stats.arp_frames += 1
        packet = parse_arp_frame(frame, timestamp)
        if packet is None:
            stats.malformed_arp_frames += 1
            continue
        yield packet
