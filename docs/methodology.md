# Evaluation methodology

Implemented in `scripts/run_protocol.py`. Each run writes its configuration, dataset hashes, package versions, metrics, confusion matrices, and figures to its own folder under `experiments/evaluation/`.

## Protocol

1. **Cross-validate on training data only.** Every candidate uses the same 5 stratified folds. Exact duplicate feature vectors are grouped, so a copy can't appear on both sides of a fold.
2. **Select by a fixed rule.** The highest mean macro F1 wins, but Random Forest stays unless beaten by more than 0.005. The decision is written to `selection.json` before the test file is loaded.
3. **Evaluate on the supplied test split.** Every candidate is refit on all training data and scored once.
4. **Save the selected model** as a new versioned file with metadata. Earlier models are never overwritten.

## Preprocessing

The processed files add a label taken from each source filename and otherwise leave values unchanged. The audit confirmed they match the raw files value for value. No records are dropped or imputed. Scaling for Logistic Regression and the MLP is part of each model's pipeline, so it's fitted only on training rows in each fold.

All candidates receive the same 44 features: every dataset column except `IAT`, which is excluded for the reasons in the [results](results.md). The `rf_with_iat` diagnostic keeps it to show how much scores rely on it, and can't be selected. Earlier experiments dropped the constant `Drate` column instead, so their results aren't directly comparable.

## Hyperparameters

Hyperparameters were carried over from the earlier experiments and weren't tuned in this protocol.

## Class imbalance

ARP Spoofing is 3.4% of training records. The protocol compares `class_weight=None` with `balanced_subsample`. SMOTE and Balanced Random Forest weren't tried. ARP is weak on both precision and recall, and class weighting gained recall at a large cost in precision, so another recall-boosting method wasn't the priority. This is a reasoned omission, not an experimental result.

## Forward validation

`scripts/improve_arp.py` validates without the test split by fitting on earlier records of each class and validating on later ones: 40% to predict the next 20%, 60% to predict the next 20%, and 80% to predict the last 20%. Copies of fitting rows are removed from validation. This assumes file order is chronological. Large `IAT` readings rise with file order 78% to 99.8% of the time, depending on class, which supports the assumption without proving it.

It estimated the current model's test scores much better than random folds, so future model changes should use it. It isn't a guarantee: the one candidate it favoured still did worse on the test split. See the [results](results.md).

## Known limits on the test split

The supplied test split was already used before this protocol: an earlier experiment compared weighted and unweighted Random Forests on it, and `train_model.py` was switched to unweighted minutes later. Cross-validation on training data separately prefers the unweighted model, so the choice stands, but the test split isn't fully untouched. Comparing `IAT` values across the splits also informed the decision to remove that feature. That decision lowered the test scores, so it doesn't inflate them.

## What can't be validated with this data

- **Held-out device:** the CSVs have no device identifiers.
- **Cross-dataset:** CICIoT2023's CSVs use different feature calculations, so they can't validate the active model. A fair test would need the same extractor run on both datasets' PCAPs.
- **Timestamp-based validation:** there are no timestamps. `IAT` suggests file order is chronological, but that is unconfirmed.

So these results don't establish device-level or real-world generalization.

## Device baseline evaluation

`scripts/evaluate_baselines.py` follows this order, with the budget and pass marks fixed in the script before any held-out capture was read:

1. **Tune on benign traffic only:** fit on the earlier 70% of benign training traffic, then choose the setting flagging the most minutes on the later 30% while staying within 1 finding per hour. No attack traffic takes part. This is repeated separately for each network.
2. **Look at one training attack capture:** CICIoMT2024's `ARP_Spoofing_train.pcap` was read as a development check. Nothing was changed because of it.
3. **Score held-out captures once:** refit on all benign training traffic, then score the rest. To pass, benign captures must produce at most 2 findings per hour, and attack captures at least 5 times the benign rate.
4. **Look for evidence the baselines don't use:** report any device claiming an address over ARP while that address's usual holder is still claiming it.

5. **Try other attack types:** the saved CICIoMT2024 baseline scored `Recon-Port_Scan_test.pcap` and `MQTT-DDoS-Connect_Flood_test.pcap` once, unchanged. Before reading them, the mark was set at 5 times the held-out benign rate, with the scan expected to raise ports or new connections, and the flood to raise frames.

One tie-break was changed after these runs: equal settings now prefer the lowest limits. No selection here was tied, so no result changed. The final stage was rerun to add the list of held-out captures to its saved results, and it produced identical findings.

Labels cover whole captures, not devices, so this measures how often findings occur, not per-device accuracy. Neither dataset documents which device attacked.

## Metrics

Metrics are accuracy, balanced accuracy, macro and weighted F1, macro precision and recall, per-class precision, recall, and F1, one-vs-rest ROC-AUC and PR-AUC, confusion matrices, batch and single-row inference time, and three operational rates:

- **Benign false alarm rate:** share of benign records flagged as any attack.
- **Attack detection rate:** share of attack records flagged as any attack, even with the wrong attack type.
- **Alert precision:** share of flagged records that really were attacks.
