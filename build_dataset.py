"""
Builds techniques.json and procedure_examples.json from the OFFICIAL MITRE
ATT&CK ICS knowledge base (STIX 2.1 bundle, maintained by MITRE at
https://github.com/mitre-attack/attack-stix-data).

This is the "structured learning format based on the ATT&CK ICS matrix"
step described in the base paper's Methods section -- except here it is
actually implemented against real MITRE data rather than only described.

  techniques.json         -> {technique_id: {name, description}}
                              the reference "matrix": 97 non-deprecated
                              ICS attack-pattern (technique) objects.

  procedure_examples.json -> [{text, technique_id, technique_name}, ...]
                              271 REAL descriptions of how a specific
                              malware family / intrusion set / campaign
                              used a specific technique in the wild,
                              extracted from STIX "uses" relationships.
                              These stand in for observed intrusion
                              events that a detector must classify.

Run once (requires internet -- this fetches ~4MB from GitHub):
    python build_dataset.py
"""

import json
import re
import urllib.request

STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack.json"

def fetch_stix():
    print(f"Downloading official MITRE ATT&CK ICS data from {STIX_URL} ...")
    with urllib.request.urlopen(STIX_URL) as resp:
        return json.load(resp)


def clean_text(text: str) -> str:
    """Strip markdown links [Name](url) -> Name and (Citation: ...) refs."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\(Citation:[^)]*\)", "", text).strip()
    return text


def main():
    data = fetch_stix()
    objects = data["objects"]

    techniques = {
        o["id"]: o
        for o in objects
        if o["type"] == "attack-pattern"
        and not o.get("x_mitre_deprecated")
        and not o.get("revoked")
    }

    relationships = [
        o for o in objects
        if o["type"] == "relationship" and o.get("relationship_type") == "uses"
    ]

    samples = []
    for r in relationships:
        target = r.get("target_ref", "")
        if target in techniques and r.get("description"):
            text = clean_text(r["description"])
            if len(text) > 20:
                samples.append({
                    "text": text,
                    "technique_id": target,
                    "technique_name": techniques[target]["name"],
                })

    tech_out = {
        tid: {"name": t["name"], "description": t["description"]}
        for tid, t in techniques.items()
    }

    json.dump(tech_out, open("techniques.json", "w"), indent=2)
    json.dump(samples, open("procedure_examples.json", "w"), indent=2)

    print(f"Wrote techniques.json          ({len(tech_out)} techniques)")
    print(f"Wrote procedure_examples.json  ({len(samples)} real-world labeled examples)")


if __name__ == "__main__":
    main()
