import json
import os

import pytest

from event_template import SCHEMA, describe_event

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = json.load(open(os.path.join(ROOT, "samples", "heldout_events.json"),
                         encoding="utf-8"))["samples"]


def test_schema_matches_json_file():
    on_disk = json.load(open(os.path.join(ROOT, "event_schema.json"), encoding="utf-8"))
    assert set(on_disk) == set(SCHEMA)


def test_sample_count():
    assert len(SAMPLES) == 60


@pytest.mark.parametrize("sample", SAMPLES, ids=[s["id"] for s in SAMPLES])
def test_samples_render(sample):
    assert sample["source"] in SCHEMA
    text, unknown = describe_event(sample["source"], sample["fields"], sample.get("http_request"))
    assert text.strip()
    assert not unknown, f"sample uses fields not in the schema: {unknown}"


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        describe_event("NotADataset", {"a": 1})


def test_model_card_embedding_model():
    card = json.load(open(os.path.join(ROOT, "docs", "results", "final_model_card.json"),
                          encoding="utf-8"))
    assert card["embedding_model"].split("/")[-1] == "all-MiniLM-L6-v2"
    assert card["decision_threshold"] == 0.5
