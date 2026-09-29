# LLM-Based Intrusion Detection for ICS — Implementation

Implements and **evaluates** the approach the base paper only *describes*:

> Yoo et al., "A Preliminary Study on an Intrusion Detection Method using
> Large Language Models in Industrial Control Systems," ICUFN 2024.
> DOI: 10.1109/ICUFN61752.2024.10625633

The base paper gathers MITRE ATT&CK ICS data, structures it, and trains an
LLM — but its abstract/results/conclusion report **no experimental
results**. This project runs the missing evaluation.

## Setup (run once, needs internet)

```bash
pip install -r requirements.txt
python build_dataset.py      # pulls the official MITRE ATT&CK ICS data
python ics_llm_ids.py         # downloads the language model (first run only,
                               # then cached — runs fully offline after that)
```

Run `ics_llm_ids.py` once so the model is cached locally
(`~/.cache/huggingface`). After that, set `HF_HUB_OFFLINE=1` to skip the
Hugging Face Hub freshness check entirely and avoid demo failures on
flaky wifi:

```bash
# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python app.py

# bash
HF_HUB_OFFLINE=1 python app.py
```

## Live prototype — web UI (`app.py`)

Same classifier, served as a local web app with a dashboard-style UI —
type an event in the browser instead of a terminal:

```bash
# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python app.py

# bash
HF_HUB_OFFLINE=1 python app.py
```

Then open **http://127.0.0.1:8000**. The model and knowledge base load
once at startup (leave the process running for the whole demo), and
every query after that is sub-second. Shows a colored ATTACK/BENIGN
badge, similarity score vs. threshold, a grounded explanation +
mitigation suggestion (templated from the matched technique's own MITRE
description — no fabricated content), the top-3 matched techniques with
tactic + description, and a scrollable history of every query on the
page. The backend also exposes `POST /api/analyze` directly (JSON in,
JSON out) and interactive API docs at `/docs`.

## What it does

1. **Reference set** (`techniques.json`) — 97 real ICS attack techniques
   from the official MITRE ATT&CK ICS matrix (name + description each).
2. **Evaluation set** (`procedure_examples.json`) — 271 real descriptions
   of how specific malware/intrusion-sets/campaigns used a specific
   technique, pulled from MITRE's own "uses" relationships. These stand
   in for observed intrusion events.
3. **Classifier** — a pretrained transformer language model
   (`all-MiniLM-L6-v2`) embeds both sets; each event is classified by
   nearest-neighbor cosine similarity to a technique description. This is
   **zero-shot / in-context matching** — no technique-specific training —
   directly testing the base paper's own claim that LLMs can "identify
   emerging threats even in cases where they haven't been explicitly
   trained on them."
4. **Task A** — 97-way technique classification: top-1 / top-3 accuracy,
   macro-F1, per-technique breakdown.
5. **Task B** — benign vs. attack detection, using a small author-written
   synthetic set of normal ICS operational log lines (clearly marked in
   code, not from MITRE) to test the detector's false-positive behavior.

## Results (real run, no synthetic labels — see `results.json`)

| Metric | Value |
|---|---|
| Task A top-1 accuracy (97-way) | 0.321 |
| Task A top-3 accuracy | 0.480 |
| Task A macro F1 | 0.265 |
| Task B benign/attack detection accuracy | 0.745 |
| Task B attack recall | 203/271 (0.749) |
| Task B false positive rate (benign flagged as attack) | 5/15 (0.33) |

Chance level for Task A is ~1% (1-in-97), so 32%/48% top-1/top-3 accuracy
with zero attack-specific training is a meaningful signal, not a solved
problem — plenty of techniques are confused with semantically similar
neighbors (e.g. "Loss of View" vs. "Manipulate I/O Image"). This is
reported honestly rather than inflated.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Pulls + structures the real MITRE ATT&CK ICS data |
| `techniques.json` | Reference set (already built — 97 techniques) |
| `procedure_examples.json` | Evaluation set (already built — 271 real examples) |
| `ics_llm_ids.py` | Embeds, classifies, evaluates, prints + saves results |
| `app.py` | FastAPI web app — same classifier behind a browser UI + JSON API |
| `static/index.html` | Frontend for `app.py` |
| `results.json` | Produced after running `ics_llm_ids.py` |

## Mapping to Milestone 1 claims

- **Novelty**: base paper stops at data prep/training setup with zero
  evaluation. This is the first empirical test of that exact pipeline.
- **Differs from existing work**: classifies to the *specific ATT&CK
  technique* (actionable for an analyst), not just a binary anomaly flag
  like traditional signature-based ICS IDS.
- **Zero-shot design**: no attack-specific training data required —
  matches the paper's generalization claim and means the same code
  extends to new ATT&CK techniques MITRE adds later, with no retraining.
