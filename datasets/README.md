# Datasets

The raw datasets are public and are **not** redistributed here. This page records what the study used and how to regenerate the train/validation/test splits.

## Files in this repository

Each folder holds `train.parquet`, `val.parquet`, `test.parquet` and `meta.json` (cleaning counts and class balance) for one dataset: `BATADAL/`, `Edge-IIoTset/`, `X-IIoTID/`, `TON_IoT-Network/`, `TON_IoT-Modbus/` (about 25 MB in total). `SHA256SUMS.txt` lists a checksum for every parquet file (`cd datasets && sha256sum -c SHA256SUMS.txt`).

Columns: the cleaned features plus `__y` (0 = normal, 1 = attack), `__type` (attack type), `__multi` and, for some datasets, helper columns such as `__txt_http`, `__time` and `__part`.

**How these were produced.** The original split files were lost, so they were regenerated from the public raw datasets with `docs/dataset_study_code/prepare.py` (seed 42). Row counts match the tables below exactly for BATADAL, Edge-IIoTset, X-IIoTID and TON_IoT-Network. **TON_IoT-Modbus does not reproduce** (see below).

**Licences.** These are processed subsets of the public datasets and remain under their original licences (for example Edge-IIoTset is CC BY-NC-SA 4.0: non-commercial, attribution, share-alike). They are included for coursework and reproducibility; cite the original dataset papers.

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

**TON_IoT-Modbus discrepancy.** The Modbus split in this repository has **9,883 / 2,118 / 2,118** rows (14,119 unique rows after de-duplication of 51,106), not the 6,207 / 1,330 / 1,331 in the table above. The only copy of the raw file available for regeneration was a third-party GitHub copy (3.3 MB), not the original UNSW file, so it is probably not the file used for the original numbers. The original Modbus split sizes and the 71.5% de-duplication figure were therefore not reproduced here. The merged model does not use Modbus data.

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
3. Check the row counts against the table above (all but Modbus should match).
