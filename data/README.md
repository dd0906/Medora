# Dataset setup

Medora uses a four-class subset of **CICIoMT2024**, from the Canadian Institute for Cybersecurity. The data files aren't included in this repository. Check the dataset's license and citation requirements on its official page before using or redistributing it.

## Citations

If you use these datasets, cite them as their pages request:

- S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi and A. A. Ghorbani, "CICIoMT2024: Attack Vectors in Healthcare devices-A Multi-Protocol Dataset for Assessing IoMT Device Security," Internet of Things, v. 28, December 2024. [Dataset page](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html).
- E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," Sensors, 2023. [Dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

## Files

Download the Wi-Fi and MQTT extracted-feature CSVs and place these eight files in `data/raw/`:

| Class | Training file | Test file |
|---|---|---|
| Benign | `Benign_train.pcap.csv` | `Benign_test.pcap.csv` |
| ARP Spoofing | `ARP_Spoofing_train.pcap.csv` | `ARP_Spoofing_test.pcap.csv` |
| MQTT DDoS Connect Flood | `MQTT-DDoS-Connect_Flood_train.pcap.csv` | `MQTT-DDoS-Connect_Flood_test.pcap.csv` |
| Recon Port Scan | `Recon-Port_Scan_train.pcap.csv` | `Recon-Port_Scan_test.pcap.csv` |

Then build the processed files:

```powershell
python experiments\prepare_dataset.py
```

This creates `train_processed.csv`, `test_processed.csv`, and `label_distribution.csv` in `data/processed/`. Raw files are never modified.

## Optional: outside-data check

`scripts/external_check.py` needs these CICIoT2023 CSV folders in `data/external/ciciot2023/`: `Benign_Final`, `MITM-ArpSpoofing`, and `Recon-PortScan`. CICIoT2023 comes from the same institute, and its download page also asks you to fill in a form.

## Optional: ARP rule evaluation

`scripts/evaluate_arp_rules.py` looks for capture files by exact name anywhere under `data/external/pcaps/`:

- **CICIoMT2024 (`WiFi_and_MQTT/attacks/PCAP`):** `ARP_Spoofing_train.pcap`, `Benign_train.pcap`, `ARP_Spoofing_test.pcap`, and `Benign_test.pcap`.
- **CICIoT2023 (`PCAP`):** `MITM-ArpSpoofing.pcap`, `MITM-ArpSpoofing1.pcap`, and `BenignTraffic.pcap` plus `BenignTraffic1.pcap` to `BenignTraffic3.pcap`.

Device addresses from these captures aren't published here. Results these scripts save contain them, and `.gitignore` keeps those folders out of the repository.

## Expected contents

| Class | Training records | Test records |
|---|---|---|
| Benign | 192,732 | 37,607 |
| ARP Spoofing | 16,047 | 1,744 |
| MQTT DDoS Connect Flood | 173,036 | 41,916 |
| Recon Port Scan | 83,981 | 22,622 |
| **Total** | **465,796** | **103,889** |

Each record has 45 numeric features and no missing or infinite values. The files contain no IP addresses, device IDs, or timestamps. Models use 44 of the features. See the [results](../docs/results.md) for why `IAT` is excluded and for other known issues with the data.
