"""
FastAPI web app for the LLM-based ICS intrusion detector.

Wraps the SAME zero-shot classification logic as demo.py / ics_llm_ids.py
(embed event text with all-MiniLM-L6-v2, cosine-similarity against the 97
real MITRE ATT&CK ICS techniques, threshold calibrated in results.json) --
no changes to that logic, just a JSON API + a static frontend on top of it.

Run:
    python app.py
    -> open http://127.0.0.1:8000

The model and technique knowledge base are loaded once at import time
(module scope), so every request after the first is fast.
"""

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Load knowledge base + model ONCE at startup
# ---------------------------------------------------------------------------

def load_reference():
    techniques = json.load(open(os.path.join(BASE_DIR, "techniques.json")))
    tech_ids = list(techniques.keys())
    tech_texts = [f'{techniques[t]["name"]}. {techniques[t]["description"]}' for t in tech_ids]
    return techniques, tech_ids, tech_texts


def load_threshold(default=0.42):
    try:
        results = json.load(open(os.path.join(BASE_DIR, "results.json")))
        return results["task_b_threshold"]
    except FileNotFoundError:
        return default


print(f"Loading knowledge base and language model ({MODEL_NAME}) ...")
techniques, tech_ids, tech_texts = load_reference()
THRESHOLD = load_threshold()
model = SentenceTransformer(MODEL_NAME)
tech_emb = model.encode(tech_texts, normalize_embeddings=True, show_progress_bar=False)
print(f"Ready. {len(tech_ids)} techniques loaded, threshold={THRESHOLD:.3f}")


# ---------------------------------------------------------------------------
# Detection + explanation logic (same math as demo.py, plus a templated,
# non-hallucinated explanation/mitigation grounded in the matched
# technique's own MITRE description)
# ---------------------------------------------------------------------------

def classify(text: str):
    emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = tech_emb @ emb
    top3_idx = np.argsort(-sims)[:3]
    max_sim = float(sims[top3_idx[0]])
    verdict = "ATTACK" if max_sim >= THRESHOLD else "BENIGN"

    matches = []
    for i in top3_idx:
        tid = tech_ids[i]
        t = techniques[tid]
        matches.append({
            "id": tid,
            "name": t["name"],
            "tactic": t.get("tactic", "Unknown"),
            "description": t["description"],
            "similarity": float(sims[i]),
        })
    return verdict, max_sim, matches


def explain(verdict: str, max_sim: float, matches: list):
    top = matches[0]
    if verdict == "ATTACK":
        explanation = (
            f"This event's description is semantically closest (similarity {max_sim:.3f}, "
            f"above the {THRESHOLD:.3f} detection threshold) to the MITRE ATT&CK ICS technique "
            f"\"{top['name']}\" ({top['id']}, tactic: {top['tactic']}). MITRE describes this "
            f"technique as: {top['description'][:280].rstrip()}"
            f"{'...' if len(top['description']) > 280 else ''}"
        )
        mitigation = (
            f"Review MITRE ATT&CK mitigations associated with {top['id']} ({top['name']}) and the "
            f"\"{top['tactic']}\" tactic. As a general first response: isolate the affected "
            "asset/segment, verify the action against the authorized change/maintenance log, and "
            "escalate to the ICS security team for confirmation before taking further action."
        )
    else:
        explanation = (
            f"This event's description does not closely match any ATT&CK ICS technique "
            f"(highest similarity {max_sim:.3f}, closest was \"{top['name']}\" at "
            f"{THRESHOLD - max_sim:.3f} below the {THRESHOLD:.3f} detection threshold), so it is "
            "classified as routine/benign operational activity."
        )
        mitigation = (
            "No action required. Continue standard monitoring; benign classification does not "
            "guarantee an event is risk-free, only that it does not resemble a known ATT&CK ICS "
            "technique in this reference set."
        )
    return explanation, mitigation


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="ICS LLM Intrusion Detection")


class AnalyzeRequest(BaseModel):
    event_text: str


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    text = req.event_text.strip()
    if not text:
        return {"error": "event_text is empty"}
    verdict, max_sim, matches = classify(text)
    explanation, mitigation = explain(verdict, max_sim, matches)
    return {
        "event_text": text,
        "verdict": verdict,
        "similarity": max_sim,
        "threshold": THRESHOLD,
        "matches": matches,
        "explanation": explanation,
        "mitigation": mitigation,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "techniques_loaded": len(tech_ids), "threshold": THRESHOLD}


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
