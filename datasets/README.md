# Datasets

The raw datasets are public and are **not** redistributed here. This page records what the study used and how to regenerate the train/validation/test splits.

> **Status:** the split files themselves (`train/val/test.parquet`) have not been added to this repository yet. They are generated locally by `docs/dataset_study_code/prepare.py` and were not saved with the code. Until they are added, regenerate them as described below.

## Sources

| Dataset | Raw file used | Source |
|---|---|---|
| BATADAL | `BATADAL_dataset03.csv`, `BATADAL_dataset04.csv` | BATtle of the Attack Detection ALgorithms (batadal.net) |
| Edge-IIoTset | `DNN-EdgeIIoT-dataset.csv` | Ferrag et al., Edge-IIoTset |
| X-IIoTID | `X-IIoTID dataset.csv` | Al-Hawawreh et al., X-IIoTID |
| TON_IoT-Network | `train_test_network.csv` | UNSW Canberra TON_IoT |
| TON_IoT-Modbus | `Train_Test_IoT_Modbus.csv` | UNSW Canberra TON_IoT |

Check each dataset's own licence and terms before redistributing any processed rows.

## Protocol (identical for all datasets)

Drop identifiers (IPs, timestamps, counters, raw option bytes), de-duplicate after identifier removal, cap at 250,000 rows (stratified by attack type, at least 1,000 per class), split 70/15/15 stratified. BATADAL is split chronologically. Seed 42.

## Split sizes (train / validation / test)

| Dataset | Train | Validation | Test |
|---|---|---|---|
| BATADAL | 11,353 | 624 | 961 |
| Edge-IIoTset | 4,561 | 976 | 976 |
| X-IIoTID | 174,995 | 37,499 | 37,499 |
| TON_IoT-Network | 73,305 | 15,708 | 15,709 |
| TON_IoT-Modbus | 6,207 | 1,330 | 1,331 |

Per-dataset metadata (class balance and cleaning counts) is in [`docs/results/meta/`](../docs/results/meta/).

## Merged model training subset

From `docs/results/final_model_card.json`:

| Dataset | Training rows | Validation rows |
|---|---|---|
| Edge-IIoTset | 4,561 | 976 |
| TON_IoT-Network | 19,999 | 3,001 |
| X-IIoTID | 20,000 | 3,000 |

## Regenerating the splits

1. Download the raw files above into `<project>/data/raw/` and follow the layout in [`docs/README.md`](../docs/README.md#re-running-the-study).
2. Run `python dataset_study/prepare.py`. Splits are cached as `dataset_study/cache/<dataset>/{train,val,test}.parquet`.
3. Check that the row counts match the table above.
