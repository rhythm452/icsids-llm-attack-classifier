# Model

`ics_ids_merged_v1/` holds the trained merged detector used by the web app:

- `classifier.joblib`: LightGBM (31 leaves) trained on all-MiniLM-L6-v2 sentence embeddings of event descriptions from Edge-IIoTset, TON_IoT-Network and X-IIoTID (4,561 / 19,999 / 20,000 training rows).
- `model_card.json`: training setup, candidates, per-dataset test metrics and the 271-description domain-shift check.

Retrain with `dataset_study/final_model.py` in the study code (not part of this repository). The model card in [`docs/results/final_model_card.json`](../docs/results/final_model_card.json) is from an earlier run of the same script: metrics are identical; it differs in timing fields, the embedding model name string, and it includes single-dataset baselines that the rerun did not have available.
