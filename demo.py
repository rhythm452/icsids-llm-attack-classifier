"""
Interactive / scriptable prototype for the LLM-based ICS intrusion
detector. Loads the same reference set and model as ics_llm_ids.py,
then classifies ad-hoc event descriptions (typed interactively or
passed as CLI arguments) instead of only running the fixed evaluation
set.

Usage:
    python demo.py
        interactive mode -- type an event description, get a
        benign/attack call + top-3 ATT&CK ICS technique matches

    python demo.py "Engineer workstation issued an unscheduled write
    to a safety instrumented system register outside maintenance hours"
        one-shot mode -- classify the text given as an argument

Requires techniques.json (run build_dataset.py first) and the
calibrated threshold from results.json (run ics_llm_ids.py first).
"""

import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_reference():
    techniques = json.load(open("techniques.json"))
    tech_ids = list(techniques.keys())
    tech_texts = [f'{techniques[t]["name"]}. {techniques[t]["description"]}' for t in tech_ids]
    return techniques, tech_ids, tech_texts


def load_threshold(default=0.42):
    try:
        results = json.load(open("results.json"))
        return results["task_b_threshold"]
    except FileNotFoundError:
        print("(no results.json found -- run ics_llm_ids.py first for a "
              f"calibrated threshold; using default {default})", file=sys.stderr)
        return default


def classify(text, model, tech_ids, tech_names, tech_emb, threshold):
    emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = tech_emb @ emb
    top3_idx = np.argsort(-sims)[:3]
    max_sim = sims[top3_idx[0]]
    verdict = "ATTACK" if max_sim >= threshold else "BENIGN"
    matches = [(tech_ids[i], tech_names[i], float(sims[i])) for i in top3_idx]
    return verdict, max_sim, matches


def print_result(text, verdict, max_sim, matches):
    print(f"\nEvent : {text}")
    print(f"Verdict: {verdict}  (max similarity {max_sim:.3f})")
    print("Top-3 ATT&CK ICS technique matches:")
    for tid, name, score in matches:
        print(f"  {score:.3f}  {tid}  {name}")


def main():
    techniques, tech_ids, tech_texts = load_reference()
    tech_names = [techniques[t]["name"] for t in tech_ids]
    threshold = load_threshold()

    print(f"Loading language model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    tech_emb = model.encode(tech_texts, normalize_embeddings=True, show_progress_bar=False)

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        verdict, max_sim, matches = classify(text, model, tech_ids, tech_names, tech_emb, threshold)
        print_result(text, verdict, max_sim, matches)
        return

    print(f"Calibrated threshold: {threshold:.3f}")
    print("Type an ICS event description and press Enter (blank line or Ctrl+C to quit).\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        verdict, max_sim, matches = classify(text, model, tech_ids, tech_names, tech_emb, threshold)
        print_result(text, verdict, max_sim, matches)
        print()


if __name__ == "__main__":
    main()
