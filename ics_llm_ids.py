"""
LLM-Based Intrusion Detection for Industrial Control Systems (ICS)
====================================================================
Implements and evaluates the approach proposed (but never empirically
tested) in the base paper:

  "A Preliminary Study on an Intrusion Detection Method using Large
   Language Models in Industrial Control Systems"
   (Yoo et al., ICUFN 2024, DOI: 10.1109/ICUFN61752.2024.10625633)

The base paper describes gathering MITRE ATT&CK ICS data, organizing
it into a structured learning format, and training an LLM -- but
reports NO experimental results. This script closes that gap:

  1. Pulls the REAL, official MITRE ATT&CK ICS knowledge base
     (technique names + descriptions) as the structured reference
     set -- this is the "ATT&CK ICS matrix" the base paper refers to.
  2. Pulls REAL adversary "procedure examples" (how specific malware /
     intrusion sets / campaigns actually used each technique) as the
     evaluation set -- these stand in for observed intrusion events.
  3. Uses a transformer language model to embed both sets and
     classifies each event via semantic similarity (zero-shot,
     in-context matching -- no technique-specific training, directly
     testing the paper's own claim that LLMs can "identify emerging
     threats even without explicit training on them").
  4. Adds a small set of synthetic benign ICS operational log
     descriptions (author-written, clearly marked) to test
     benign-vs-attack separation -- the actual "intrusion detection"
     half of the task, not just technique tagging.
  5. Reports accuracy, top-3 accuracy, macro-F1, per-technique
     breakdown, and a benign/attack confusion matrix.
"""

import json
import re
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODEL_NAME = "all-MiniLM-L6-v2"          # small pretrained transformer LM
MIN_EXAMPLES_PER_TECHNIQUE = 1            # keep every technique with >=1 real example
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Load the structured ATT&CK ICS reference data (built from the official
#    MITRE STIX feed: github.com/mitre-attack/attack-stix-data)
# ---------------------------------------------------------------------------

techniques = json.load(open("techniques.json"))          # {id: {name, description}}
procedure_examples = json.load(open("procedure_examples.json"))  # [{text, technique_id, technique_name}]

tech_ids = list(techniques.keys())
tech_names = [techniques[t]["name"] for t in tech_ids]
tech_texts = [f'{techniques[t]["name"]}. {techniques[t]["description"]}' for t in tech_ids]

# keep only test examples whose technique still exists in our reference set
eval_samples = [s for s in procedure_examples if s["technique_id"] in techniques]

print(f"Reference techniques (ATT&CK ICS matrix): {len(tech_ids)}")
print(f"Real-world evaluation examples (procedure examples): {len(eval_samples)}")

# ---------------------------------------------------------------------------
# 2. Synthetic benign ICS operational log descriptions
#    (author-written baseline -- NOT from MITRE data -- used only to test
#     benign-vs-attack separation)
# ---------------------------------------------------------------------------

benign_samples = [
    "PLC #12 responded to a scheduled Modbus read request for register block 40001-40010 with normal values.",
    "HMI performed its routine 5-second polling cycle of the tank level sensor with no anomalies detected.",
    "Scheduled nightly configuration backup of the SCADA historian completed successfully with no errors.",
    "Engineer logged into the engineering workstation using their assigned credentials during normal working hours.",
    "Watchdog heartbeat signal from the RTU was received on schedule, confirming the device is online.",
    "Firmware version check on the PLC ran as part of the weekly maintenance window and reported the current version.",
    "Historian database performed its routine hourly batch write of process trend data with no discrepancies.",
    "Operator acknowledged a low-priority alarm on the HMI following standard operating procedure.",
    "Field technician performed a scheduled calibration of the pressure transmitter as part of preventive maintenance.",
    "SCADA server sent a routine status poll to all connected field devices and received expected responses.",
    "Network switch logged a normal link-up event after a scheduled maintenance reboot of an RTU.",
    "Batch process controller executed a pre-programmed recipe change at the scheduled production shift start.",
    "Authorized vendor remote-support session was opened using the pre-approved change ticket and VPN account.",
    "Router recorded a routine NTP time synchronization with the plant's internal time server.",
    "PLC program was updated during a planned maintenance window using the documented change-control process.",
]

# ---------------------------------------------------------------------------
# 3. Embed everything with a pretrained transformer language model
# ---------------------------------------------------------------------------

print(f"\nLoading language model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

tech_emb = model.encode(tech_texts, normalize_embeddings=True, show_progress_bar=False)
eval_emb = model.encode([s["text"] for s in eval_samples], normalize_embeddings=True, show_progress_bar=False)
benign_emb = model.encode(benign_samples, normalize_embeddings=True, show_progress_bar=False)

# ---------------------------------------------------------------------------
# 4. Task A -- zero-shot technique classification (multiclass, 97-way)
# ---------------------------------------------------------------------------

sims = eval_emb @ tech_emb.T                      # cosine similarity (both normalized)
top1_idx = sims.argmax(axis=1)
top3_idx = np.argsort(-sims, axis=1)[:, :3]

y_true = [s["technique_id"] for s in eval_samples]
y_pred_top1 = [tech_ids[i] for i in top1_idx]

top1_acc = accuracy_score(y_true, y_pred_top1)
top3_hits = sum(y_true[i] in [tech_ids[j] for j in top3_idx[i]] for i in range(len(y_true)))
top3_acc = top3_hits / len(y_true)
macro_f1 = f1_score(y_true, y_pred_top1, average="macro", zero_division=0)

print("\n" + "=" * 70)
print("TASK A: Zero-shot ATT&CK ICS technique classification (97 classes)")
print("=" * 70)
print(f"  Evaluation samples : {len(eval_samples)}")
print(f"  Top-1 accuracy     : {top1_acc:.3f}")
print(f"  Top-3 accuracy     : {top3_acc:.3f}")
print(f"  Macro F1 (top-1)   : {macro_f1:.3f}")

# per-technique breakdown for techniques with the most real-world examples
counts = Counter(y_true)
print("\n  Per-technique breakdown (techniques with >=4 real examples):")
print(f"  {'Technique':45s} {'n':>3s} {'Top-1 acc':>10s}")
for tid, n in counts.most_common():
    if n < 4:
        continue
    idxs = [i for i, t in enumerate(y_true) if t == tid]
    acc = np.mean([y_pred_top1[i] == tid for i in idxs])
    print(f"  {techniques[tid]['name'][:45]:45s} {n:>3d} {acc:>10.3f}")

# a handful of concrete example predictions for the demo
print("\n  Sample predictions:")
for i in list(range(0, len(eval_samples), max(1, len(eval_samples) // 6)))[:6]:
    correct = "OK" if y_pred_top1[i] == y_true[i] else "X "
    print(f"   [{correct}] true={techniques[y_true[i]]['name']!r:35s} pred={techniques[y_pred_top1[i]]['name']!r}")
    print(f"        text: {eval_samples[i]['text'][:110]}...")

# ---------------------------------------------------------------------------
# 5. Task B -- benign vs. attack separation (the actual "intrusion
#    detection" half of the pipeline)
# ---------------------------------------------------------------------------

attack_max_sim = sims.max(axis=1)
benign_max_sim = (benign_emb @ tech_emb.T).max(axis=1)

# calibrate a threshold at the midpoint between the two distributions' means
threshold = (attack_max_sim.mean() + benign_max_sim.mean()) / 2

pred_labels = ["Attack" if s >= threshold else "Benign" for s in attack_max_sim] + \
              ["Attack" if s >= threshold else "Benign" for s in benign_max_sim]
true_labels = ["Attack"] * len(attack_max_sim) + ["Benign"] * len(benign_max_sim)

det_acc = accuracy_score(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels, labels=["Benign", "Attack"])

print("\n" + "=" * 70)
print("TASK B: Benign vs. Attack detection")
print("=" * 70)
print(f"  Calibrated similarity threshold: {threshold:.3f}")
print(f"  Detection accuracy             : {det_acc:.3f}")
print(f"  Confusion matrix [rows=true, cols=pred] (order: Benign, Attack)")
print(f"    {cm}")
print(f"  Mean max-similarity | attack samples: {attack_max_sim.mean():.3f}  benign samples: {benign_max_sim.mean():.3f}")

# ---------------------------------------------------------------------------
# 6. Save results for the report
# ---------------------------------------------------------------------------

results = {
    "model": MODEL_NAME,
    "reference_techniques": len(tech_ids),
    "eval_samples": len(eval_samples),
    "task_a_top1_accuracy": top1_acc,
    "task_a_top3_accuracy": top3_acc,
    "task_a_macro_f1": macro_f1,
    "task_b_threshold": float(threshold),
    "task_b_detection_accuracy": det_acc,
    "task_b_confusion_matrix": cm.tolist(),
}
json.dump(results, open("results.json", "w"), indent=2)
print("\nSaved results.json")
