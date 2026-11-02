# Dataset study

This folder documents the study used to choose training data for the detector in this app. It compares five public ICS/IIoT intrusion datasets, checks for leakage, tests how well models transfer between datasets, and records the merged model that came out of it.

## What is here

| Path | Contents |
|---|---|
| `report.html` | Full report with all tables and charts embedded. Download it and open it in a browser; GitHub shows HTML files as source. |
| `report.md` | The same report as Markdown. GitHub renders it, and its charts load from `plots/`. |
| `recommendation.md` | The summary, recommendation and limitations sections on their own. |
| `plots/` | Confusion matrices, ROC/PR curves, cross-dataset heatmap and merge chart (PNG). |
| `results/` | The JSON results the report is built from: per-dataset model metrics, sanity checks, BATADAL leave-one-window-out, cross-dataset and zero-shot results, the Step 0 data profile, per-dataset split metadata (`results/meta/`) and the final model's metadata (`results/final_model_card.json`). |
| `dataset_study_code/` | The study code and its `requirements.txt`. |

An online copy of the report is at https://claude.ai/artifact/3NmMfAZRCW13CrgQhSKriA. That page is private unless its owner shares it, so the link may not open for you.

## Headline results

The final model merges **Edge-IIoTset, TON_IoT-Network and X-IIoTID**. It is a LightGBM classifier on all-MiniLM-L6-v2 embeddings of short event descriptions, because the three datasets share almost no columns. BATADAL and TON_IoT-Modbus were left out: after removing duplicate rows, every model scored at chance on them.

F1 for the attack class on held-out test rows that were never used for training or tuning:

| Test set | F1 |
|---|---|
| All three, pooled | **0.986** |
| Edge-IIoTset | 0.993 |
| TON_IoT-Network | 0.994 |
| X-IIoTID | 0.968 |

## Limits

- **These are in-testbed scores.** Test rows come from the same captures as the training rows. Expect much lower accuracy on traffic from another plant or network.
- **Transfer between datasets is poor.** A model trained on one dataset often scores near chance on another (cross-dataset ROC-AUC as low as 0.27–0.59 for X-IIoTID). Only Edge-IIoTset and TON_IoT-Network transfer reasonably to each other.
- **Near chance on handwritten sentences.** On the app's own example sentences the model's ROC-AUC is 0.55. The app therefore uses it only for structured events, and keeps free text as an experimental, low-confidence mode.
- **Edge-IIoTset is small after de-duplication.** Its 2.2 million rows contain only 6,513 unique feature vectors under the study's field policy, and its test set has 976 rows. Three attack types have 5 or fewer unique packets and could not be tested.
- **The ATT&CK mapping is not verified.** The mapping from dataset attack labels to MITRE ATT&CK for ICS techniques was assigned by the study author and has not been checked by the dataset publishers or an independent analyst. The technique shown next to each verdict in the app is an approximate text-similarity match, not a classification.

## Datasets (not included)

The study uses five public datasets. None of their files are in this repository; download them from their publishers:

- **BATADAL**: Battle of the Attack Detection Algorithms, water distribution SCADA data (`BATADAL_dataset03.csv`, `BATADAL_dataset04.csv`)
- **Edge-IIoTset**: Edge-IIoT security dataset (`DNN-EdgeIIoT-dataset.csv`)
- **X-IIoTID**: Industrial IoT intrusion dataset (`X-IIoTID dataset.csv`)
- **TON_IoT**: UNSW Canberra TON_IoT "Train_Test" datasets. The study uses the network flows (`train_test_network.csv`) and the Modbus telemetry (`Train_Test_IoT_Modbus.csv`).

## Re-running the study

The code expects this layout next to the repository, which is not included here:

```
<project>/
  data/raw/                  original dataset files
  data/extracted/<name>/     each zip extracted into its own folder
  dataset_study/             the files from dataset_study_code/
  models/                    the final model is written here
```

Then run `pip install -r dataset_study/requirements.txt` and `python dataset_study/run_all.py`. Every stage saves its output and skips finished work, so the command can be re-run after an interruption. `final_model.py` builds the merged model and `export_samples.py` refreshes the app's sample events. Random seeds are fixed (42).
