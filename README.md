# Leakage-Aware, Explainable Intrusion Detection for ICS

Course project (NLP). Team: Rhythm Jain, Shourya Chopra, Mini Aggarwal. Department of Computer Science & Engineering, Bennett University, Greater Noida, India.

This repository implements and evaluates the idea proposed (as a concept, without a prototype or results) in:

> S. Ann, S. Cho, H. Kim, "A Preliminary Study on an Intrusion Detection Method using Large Language Models in Industrial Control Systems," in Proc. 15th Int. Conf. Ubiquitous and Future Networks (ICUFN), 2024, pp. 600-602. DOI: [10.1109/ICUFN61752.2024.10625633](https://doi.org/10.1109/ICUFN61752.2024.10625633)

Our paper: **"Leakage-Aware, Explainable Intrusion Detection for Industrial Control Systems Using Sentence Embeddings and MITRE ATT&CK for ICS"** ([PDF](papers/ICS_IDS_IEEE_ResearchPaper.pdf)). Slides: [presentation/](presentation/).

## What we did

- Evaluated five public ICS/IIoT datasets under one leakage-aware protocol: BATADAL, Edge-IIoTset, X-IIoTID, TON_IoT-Network and TON_IoT-Modbus.
- Preprocessing: drop identifiers (IPs, timestamps, counters, raw option bytes), de-duplicate after identifier removal, cap at 250,000 rows (stratified by attack type, at least 1,000 per class), split 70/15/15 stratified (BATADAL is split chronologically), seed 42.
- Trained four supervised models per dataset (logistic regression, random forest, LightGBM, MLP) and a zero-shot ATT&CK similarity method (all-MiniLM-L6-v2 embeddings against 97 ATT&CK for ICS technique texts).
- Built a merged detector: each event is rendered as text, embedded with MiniLM (384 dimensions) and classified by one LightGBM. It is trained on Edge-IIoTset, TON_IoT-Network and X-IIoTID. BATADAL and TON_IoT-Modbus are excluded because no learnable signal remains after de-duplication.
- Served it as a FastAPI web app with a structured-event mode (merged detector) and an experimental free-text mode (zero-shot).

## Key results

| Result | Value |
|---|---|
| Rows removed by de-duplication | Edge-IIoTset 99.7% (2,219,201 raw rows to 6,513 unique), TON_IoT-Modbus 71.5%, TON_IoT-Network 50.4%, X-IIoTID 10.5%, BATADAL 0% |
| TON_IoT-Modbus LightGBM accuracy, before to after de-duplication | 0.9715 to 0.5409 |
| LightGBM attack F1 | Edge-IIoTset 0.995, X-IIoTID 0.996, TON_IoT-Network 0.998 (BATADAL and Modbus: no learnable signal) |
| Zero-shot ATT&CK matching, ROC-AUC | 0.43-0.51 on four datasets, 0.74 on TON_IoT-Network (does not detect attacks) |
| Merged model, held-out test | pooled F1 0.986 (ROC-AUC 0.998, FPR 0.020); Edge-IIoTset 0.993, TON_IoT-Network 0.994, X-IIoTID 0.968 |
| Merged model on 271 handwritten attack descriptions | ROC-AUC 0.55 (near chance) |
| Web app on 60 held-out sample events | 58/60 correct (misses: one MITM, one fuzzing) |

Full tables, plots and per-dataset results are in [docs/](docs/README.md).

## Limitations

- Scores are in-testbed: each dataset was recorded in one lab, and cross-dataset transfer is poor.
- The model is near chance on free-text attack descriptions; free-text mode is experimental.
- ATT&CK technique matches are approximate explanations, not ground truth.
- No zero-day claim is made; evaluation on unseen attack types was not done.
- Single seed (42).
- BATADAL and TON_IoT-Modbus gave no reliable signal under this protocol.

## Repository map

| Path | Purpose |
|---|---|
| `app.py`, `static/index.html` | FastAPI web app and its frontend |
| `event_template.py`, `event_schema.json` | Turn a structured event into the exact text used in training |
| `samples/heldout_events.json` | 60 held-out events for trying the app |
| `techniques.json`, `procedure_examples.json`, `build_dataset.py` | MITRE ATT&CK for ICS data (97 techniques, 271 procedure examples) |
| `ics_llm_ids.py`, `results.json` | First prototype (zero-shot only), see below |
| `docs/` | Dataset study: report, plots, results and the study code |
| `datasets/` | Train/validation/test split information, see [datasets/README.md](datasets/README.md) |
| `models/` | Trained merged detector (see "Model" below) |
| `papers/` | Our paper and the base paper |
| `presentation/` | Project slides |
| `outputs/` | Early milestone report |

## Setup and run

```bash
pip install -r requirements.txt
```

### Model

The app needs `classifier.joblib` and `model_card.json` in a directory, by default `models/ics_ids_merged_v1/`. Set `ICS_MODEL_DIR` to use another location. `model_card.json` is also in `docs/results/final_model_card.json`. See the notes in `models/README.md` for where to get the classifier file.

### Run the web app

The first run downloads `all-MiniLM-L6-v2` from the Hugging Face Hub and caches it. After that, set `HF_HUB_OFFLINE=1` to skip the Hub check:

```bash
# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python app.py

# bash
HF_HUB_OFFLINE=1 python app.py
```

Open http://127.0.0.1:8000. Interactive API docs are at `/docs`. `POST /api/analyze_event` takes a structured event (merged detector); `POST /api/analyze` takes free text (experimental zero-shot).

## Re-running the study

See "Re-running the study" in [docs/README.md](docs/README.md). The raw datasets are not in this repository; see [datasets/README.md](datasets/README.md) for sources.

## First prototype (zero-shot only)

The first version of this project (`ics_llm_ids.py`, `results.json`) only did zero-shot matching of event text to the 97 ATT&CK technique descriptions with all-MiniLM-L6-v2. On the 271 MITRE procedure examples it reached top-1 accuracy 0.321 and top-3 accuracy 0.480 for 97-way technique classification. `results.json` comes from that prototype, not from the final merged model. The later study showed this approach does not detect attacks in real traffic (see Key results).

To rerun it: `python build_dataset.py` (pulls MITRE ATT&CK for ICS data), then `python ics_llm_ids.py`.
