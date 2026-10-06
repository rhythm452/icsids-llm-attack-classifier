"""
FastAPI web app for the ICS intrusion detector.

Two modes:
  * Structured event (detector) -- POST /api/analyze_event
      The event's fields are rendered into the description template the detector was
      trained on (event_template.py), embedded with all-MiniLM-L6-v2, and classified by
      the supervised model in models/ics_ids_merged_v1 (trained on Edge-IIoTset,
      TON_IoT-Network and X-IIoTID). Verdict = attack probability >= 0.5.
  * Free text (experimental, low-confidence) -- POST /api/analyze
      The original zero-shot similarity method from ics_llm_ids.py (cosine similarity to
      the 97 MITRE ATT&CK ICS techniques, threshold from results.json). It is NOT a
      reliable detector; the detector's probability is reported alongside for reference
      only, since the detector was not trained on prose.

In both modes the top-3 ATT&CK ICS technique matches (text similarity) are returned as
an approximate explanation, not as the basis of the structured verdict.

Run:
    python app.py
    -> open http://127.0.0.1:8000

The language model, technique knowledge base and detector are loaded once at startup.
Set ICS_MODEL_DIR to load a different detector directory.
"""

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from event_template import SCHEMA, describe_event

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("ICS_MODEL_DIR",
                           os.path.join(BASE_DIR, "..", "models", "ics_ids_merged_v1"))
SAMPLES_PATH = os.path.join(BASE_DIR, "samples", "heldout_events.json")


# ---------------------------------------------------------------------------
# Load knowledge base + models ONCE at startup
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


def load_detector(model_dir):
    card = json.load(open(os.path.join(model_dir, "model_card.json"), encoding="utf-8"))
    if card["embedding_model"].split("/")[-1] != MODEL_NAME:
        raise RuntimeError(f"detector expects {card['embedding_model']}, app loads {MODEL_NAME}")
    clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
    return card, clf


def load_samples():
    try:
        return json.load(open(SAMPLES_PATH, encoding="utf-8"))["samples"]
    except FileNotFoundError:
        return []


print(f"Loading knowledge base, language model ({MODEL_NAME}) and detector ...")
techniques, tech_ids, tech_texts = load_reference()
THRESHOLD = load_threshold()                      # similarity threshold (free-text mode)
model = SentenceTransformer(MODEL_NAME)
tech_emb = model.encode(tech_texts, normalize_embeddings=True, show_progress_bar=False)
CARD, detector = load_detector(MODEL_DIR)
DETECTOR_THRESHOLD = CARD["decision_threshold"]   # attack probability threshold (0.5)
SAMPLES = load_samples()
MODEL_INFO = {
    "name": CARD["name"], "version": CARD.get("version", "unknown"), "trained": CARD.get("trained"),
    "classifier": CARD["selected"], "embedding_model": CARD["embedding_model"],
    "training_datasets": CARD["datasets"], "decision_threshold": DETECTOR_THRESHOLD,
}
print(f"Ready. {len(tech_ids)} techniques, similarity threshold={THRESHOLD:.3f}, "
      f"detector={CARD['name']} v{MODEL_INFO['version']} ({CARD['selected']}), {len(SAMPLES)} samples")


# ---------------------------------------------------------------------------
# Detection + explanation logic
# ---------------------------------------------------------------------------

def embed(text: str):
    return model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]


def technique_matches(emb, k=3):
    sims = tech_emb @ emb
    top_idx = np.argsort(-sims)[:k]
    matches = []
    for i in top_idx:
        tid = tech_ids[i]
        t = techniques[tid]
        matches.append({
            "id": tid,
            "name": t["name"],
            "tactic": t.get("tactic", "Unknown"),
            "description": t["description"],
            "similarity": float(sims[i]),
        })
    return matches


def attack_probability(emb):
    return float(detector.predict_proba(emb.reshape(1, -1))[0, 1])


def classify(text: str):
    """Free-text mode: the original zero-shot similarity verdict (unchanged math)."""
    emb = embed(text)
    matches = technique_matches(emb)
    max_sim = matches[0]["similarity"]
    verdict = "ATTACK" if max_sim >= THRESHOLD else "BENIGN"
    return verdict, max_sim, matches, attack_probability(emb)


def classify_structured(text: str):
    """Structured mode: verdict from the supervised detector's attack probability."""
    emb = embed(text)
    prob = attack_probability(emb)
    verdict = "ATTACK" if prob >= DETECTOR_THRESHOLD else "BENIGN"
    return verdict, prob, technique_matches(emb)


def _technique_sentence(top):
    return (f"The closest MITRE ATT&CK ICS technique by text similarity is \"{top['name']}\" "
            f"({top['id']}, tactic: {top['tactic']}, similarity {top['similarity']:.3f}). This match is an "
            "approximate explanation only: it is not used for the verdict, and in the dataset study it "
            "agreed with analyst-assigned techniques for under 1% of attack events.")


def explain_structured(verdict: str, prob: float, matches: list):
    top = matches[0]
    if verdict == "ATTACK":
        explanation = (
            f"The detector ({CARD['name']}, a {CARD['selected']} classifier on {MODEL_NAME} embeddings, "
            f"trained on {', '.join(CARD['datasets'])}) estimates an attack probability of {prob:.3f}, "
            f"at or above its {DETECTOR_THRESHOLD:.2f} decision threshold. " + _technique_sentence(top))
        mitigation = (
            "Treat as a suspected intrusion: isolate or rate-limit the affected asset/segment, check the "
            "event against authorized change and maintenance records, and escalate to the ICS security team. "
            f"Use the ATT&CK mitigations for {top['name']} only as a starting point, since the technique "
            "match is approximate.")
    else:
        explanation = (
            f"The detector ({CARD['name']}) estimates an attack probability of {prob:.3f}, below its "
            f"{DETECTOR_THRESHOLD:.2f} decision threshold, so the event is classified as normal. "
            + _technique_sentence(top))
        mitigation = (
            "No action required. Continue standard monitoring; on held-out test data the detector misses "
            "about 1-3% of attacks, so a benign verdict is not a guarantee.")
    return explanation, mitigation


def explain(verdict: str, max_sim: float, matches: list):
    top = matches[0]
    warning = ("EXPERIMENTAL, LOW-CONFIDENCE: this free-text verdict uses text similarity only and is not "
               "a reliable detector. ")
    if verdict == "ATTACK":
        explanation = warning + (
            f"The description is semantically closest (similarity {max_sim:.3f}, above the {THRESHOLD:.3f} "
            f"similarity threshold) to the MITRE ATT&CK ICS technique \"{top['name']}\" ({top['id']}, "
            f"tactic: {top['tactic']}). This technique match is approximate. MITRE describes it as: "
            f"{top['description'][:280].rstrip()}{'...' if len(top['description']) > 280 else ''}")
        mitigation = (
            f"Verify before acting: review MITRE ATT&CK mitigations for {top['id']} ({top['name']}), check "
            "the action against the authorized change/maintenance log, and escalate to the ICS security "
            "team for confirmation.")
    else:
        explanation = warning + (
            f"The description does not closely match any ATT&CK ICS technique (highest similarity "
            f"{max_sim:.3f}, closest was \"{top['name']}\", {THRESHOLD - max_sim:.3f} below the "
            f"{THRESHOLD:.3f} threshold).")
        mitigation = (
            "No action taken on this basis. A low similarity score does not mean the event is safe; this "
            "mode is for exploration only.")
    return explanation, mitigation


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="ICS Intrusion Detection")


class AnalyzeRequest(BaseModel):
    event_text: str


class EventRequest(BaseModel):
    source: str
    fields: dict = {}
    http_request: str | None = None


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    text = req.event_text.strip()
    if not text:
        return {"error": "event_text is empty"}
    verdict, max_sim, matches, prob = classify(text)
    explanation, mitigation = explain(verdict, max_sim, matches)
    return {
        "mode": "free_text",
        "verdict_source": "similarity (experimental, low-confidence)",
        "event_text": text,
        "verdict": verdict,
        "similarity": max_sim,
        "threshold": THRESHOLD,
        "attack_probability": prob,
        "attack_probability_note": "detector probability for reference only; the detector was not trained "
                                   "on free-text prose and is near chance on it",
        "model": MODEL_INFO,
        "matches": matches,
        "explanation": explanation,
        "mitigation": mitigation,
    }


@app.post("/api/analyze_event")
def analyze_event(req: EventRequest):
    try:
        text, unknown = describe_event(req.source, req.fields, req.http_request)
    except ValueError as e:
        return {"error": str(e)}
    if len(unknown) == len(req.fields):
        # No recognised field: the description would be "all fields zero", which the
        # detector never saw as a real event, so its verdict would be meaningless.
        return {"error": f"no recognised fields for {req.source}; expected field names like "
                         f"{(SCHEMA[req.source]['categorical'] + SCHEMA[req.source]['numeric'])[:5]}",
                "unknown_fields_ignored": unknown}
    verdict, prob, matches = classify_structured(text)
    explanation, mitigation = explain_structured(verdict, prob, matches)
    return {
        "mode": "structured",
        "verdict_source": "detector",
        "source": req.source,
        "event_text": text,
        "verdict": verdict,
        "attack_probability": prob,
        "threshold": DETECTOR_THRESHOLD,
        "model": MODEL_INFO,
        "matches": matches,
        "explanation": explanation,
        "mitigation": mitigation,
        "unknown_fields_ignored": unknown,
    }


@app.get("/api/samples")
def samples():
    return {"note": "Held-out test-split events (never used for training or tuning).", "samples": SAMPLES}


@app.get("/api/health")
def health():
    tests = CARD.get("test_per_dataset", {})
    return {
        "status": "ok",
        "techniques_loaded": len(tech_ids),
        "threshold": THRESHOLD,
        "similarity_threshold": THRESHOLD,
        "detector": {**MODEL_INFO,
                     "test_f1": {k: round(v["f1"], 4) for k, v in tests.items()},
                     "sources": sorted(SCHEMA)},
        "samples_loaded": len(SAMPLES),
    }


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
