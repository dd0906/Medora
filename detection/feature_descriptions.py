"""Plain names for the dataset's features, for explanations people can read.

The CICIoMT2024 and CICIoT2023 pages list these columns with statistics only, and
define none of them. So these come from the column names and standard TCP/IP
meanings. Where a name doesn't say what a statistic measures, it is described
only as a packet-size statistic rather than guessed.
"""

FEATURE_NAMES = {
    "Header_Length": "Header length",
    "Protocol Type": "Protocol number",
    "Duration": "Duration field",
    "Rate": "Packet rate",
    "Srate": "Sending rate",
    "Drate": "Receiving rate",
    "fin_flag_number": "FIN flags",
    "syn_flag_number": "SYN flags",
    "rst_flag_number": "RST flags",
    "psh_flag_number": "PSH flags",
    "ack_flag_number": "ACK flags",
    "ece_flag_number": "ECE flags",
    "cwr_flag_number": "CWR flags",
    "ack_count": "ACK count",
    "syn_count": "SYN count",
    "fin_count": "FIN count",
    "rst_count": "RST count",
    "HTTP": "Uses HTTP",
    "HTTPS": "Uses HTTPS",
    "DNS": "Uses DNS",
    "Telnet": "Uses Telnet",
    "SMTP": "Uses SMTP",
    "SSH": "Uses SSH",
    "IRC": "Uses IRC",
    "TCP": "Uses TCP",
    "UDP": "Uses UDP",
    "DHCP": "Uses DHCP",
    "ARP": "Uses ARP",
    "ICMP": "Uses ICMP",
    "IGMP": "Uses IGMP",
    "IPv": "Uses IP",
    "LLC": "Uses LLC",
    "Tot sum": "Total packet size",
    "Min": "Smallest packet",
    "Max": "Largest packet",
    "AVG": "Average packet size",
    "Std": "Packet size spread",
    "Tot size": "Packet size",
    "IAT": "Time between packets",
    "Number": "Number of packets",
    "Magnitue": "Size magnitude",
    "Radius": "Size radius",
    "Covariance": "Size covariance",
    "Variance": "Size variance",
    "Weight": "Size weight",
}

FEATURE_MEANINGS = {
    "Header_Length": "Length of the packet headers.",
    "Protocol Type": "Number identifying the network protocol.",
    "Duration": "Named duration in the dataset, though its values look like a time-to-live.",
    "Rate": "How fast packets are sent.",
    "Srate": "Packet rate in the sending direction.",
    "Drate": "Packet rate in the receiving direction.",
    "fin_flag_number": "Packets that close a connection.",
    "syn_flag_number": "Packets that open a connection.",
    "rst_flag_number": "Packets that abruptly reset a connection.",
    "psh_flag_number": "Packets asking for data to be passed on straight away.",
    "ack_flag_number": "Packets acknowledging received data.",
    "ece_flag_number": "Packets signalling network congestion.",
    "cwr_flag_number": "Packets responding to network congestion.",
    "ack_count": "How many acknowledgement packets were seen.",
    "syn_count": "How many connection-opening packets were seen.",
    "fin_count": "How many connection-closing packets were seen.",
    "rst_count": "How many reset packets were seen.",
    "Tot sum": "Combined size of the packets.",
    "Min": "Size of the smallest packet.",
    "Max": "Size of the largest packet.",
    "AVG": "Average packet size.",
    "Std": "How much packet sizes vary.",
    "Tot size": "Packet size.",
    "IAT": "Time between packets. Medora's model doesn't use it.",
    "Number": "How many packets were counted.",
}
PACKET_SIZE_STATISTICS = ("Magnitue", "Radius", "Covariance", "Variance", "Weight")


def feature_name(feature: str) -> str:
    return FEATURE_NAMES.get(feature, feature)


def feature_meaning(feature: str) -> str:
    if feature in FEATURE_MEANINGS:
        return FEATURE_MEANINGS[feature]
    if feature in PACKET_SIZE_STATISTICS:
        return f"A packet-size statistic the dataset calls \"{feature}\". It isn't defined further."
    if FEATURE_NAMES.get(feature, "").startswith("Uses "):
        return f"Whether the traffic uses {FEATURE_NAMES[feature][5:]}."
    return "Not described by the dataset."
