"""Run the 60 held-out sample events through the app and report accuracy.

Expected result with the released model: 58/60 correct (the two misses are one MITM and one
fuzzing event). Needs the model in models/ics_ids_merged_v1/ (see models/README.md).

    python scripts/check_samples.py
"""

import json
import os
import sys

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as ics_app  # noqa: E402  (loads the model on import)


def main():
    samples = json.load(open(os.path.join(ROOT, "samples", "heldout_events.json"),
                             encoding="utf-8"))["samples"]
    client = TestClient(ics_app.app)
    correct, misses = 0, []
    for s in samples:
        r = client.post("/api/analyze_event", json={
            "source": s["source"], "fields": s["fields"], "http_request": s.get("http_request")})
        got = r.json()["verdict"]
        if got == s["true_label"]:
            correct += 1
        else:
            misses.append((s["id"], s["true_type"], got))
    print(f"{correct}/{len(samples)} correct")
    for m in misses:
        print("miss:", *m)
    return 0 if correct == 58 and len(samples) == 60 else 1


if __name__ == "__main__":
    sys.exit(main())
