## Summary

- **Merge Edge-IIoTset + TON_IoT-Network, and add X-IIoTID for attack coverage.** These are the only datasets with a learnable signal after de-duplication. Edge-IIoTset and TON_IoT-Network transfer to each other and lose nothing when merged. X-IIoTID is learnable on its own but does not transfer, and merging lowers its own F1.
- **Exclude BATADAL and TON_IoT-Modbus from the merge.** After de-duplication, every supervised model scores at chance on them (test ROC-AUC 0.09–0.54).
- **Use LightGBM as the final model.** It has the best or tied-best attack F1 on every learnable dataset.
- **The project's zero-shot similarity method does not detect attacks in any of the five datasets.** Its ROC-AUC is 0.43–0.51 on four datasets and 0.74 on TON_IoT-Network. At the deployed threshold (0.416) it flags either 0% of rows or 83–100% of them.
- **Most published-style near-100% scores in these datasets come from duplicated rows.** TON_IoT-Modbus drops from 0.97 to 0.54 accuracy when duplicates are removed. The three network datasets stay above 99% after de-duplication, with no single-feature leak. But their patterns transfer poorly to other datasets, so those scores mostly reflect each testbed, not general attack detection.

All numbers below come from the runs in this study (seed 42). The judgement calls are mine and are marked as recommendations.

<!-- split -->

## 6. Recommendation

### Datasets to merge

| Dataset | Decision | Evidence |
|---|---|---|
| **Edge-IIoTset** | **Merge** | De-duplicated test: LightGBM F1 0.995, with no single-feature leak (best single-feature AUC 0.87; dropping the top feature changes nothing). Embedding space: Edge-IIoTset → TON_IoT-Network F1 0.89 (AUC 0.85), and TON_IoT-Network → Edge-IIoTset AUC 0.93. Merging with TON_IoT-Network keeps both: Edge 0.992 vs 0.995 single, TON 0.932 vs 0.928 single. |
| **TON_IoT-Network** | **Merge** | De-duplicated test: LightGBM F1 0.998, with no single-feature leak (best single-feature AUC 0.84). It's the only dataset that clearly improves in the all-5 merge (0.969 vs 0.928 single). Row-level transfer from X-IIoTID reaches AUC 0.82. |
| **X-IIoTID** | **Merge, with a known cost** | Learnable in-domain: LightGBM F1 0.996, multiclass macro-F1 0.994. But the embedding-space probe trained on it scores AUC 0.27–0.59 on the other datasets, and TON_IoT-Network → X-IIoTID scores only AUC 0.55 on shared flow features. Merging lowers its own embedding F1 from 0.87 to 0.74–0.83, though AUC stays ≥0.91. A row-level merge with TON_IoT-Network on the 10 shared flow features costs nothing (0.985 merged vs 0.985 single). I recommend including it because it adds attack stages no other dataset has: exfiltration, C&C, tampering/false-data injection, Modbus register reading, and lateral movement. |
| **BATADAL** | **Exclude from the merge; handle separately** | No row-level signal. Chronological test: every model has F1 ≤ 0.05, and LogReg/MLP ROC-AUC is 0.09/0.17, meaning the patterns are inverted between attack windows. Leave-one-window-out mean F1 is at most 0.20 (RandomForest AUC 0.79 ± 0.17). Zero-shot AUC is 0.48. Its attacks are process manipulations that need a time-series or residual anomaly detector trained on normal operation, not a per-row classifier. Keep it as an evaluation set for a separate process-anomaly module. |
| **TON_IoT-Modbus** | **Exclude** | After de-duplication, all four models are at chance (ROC-AUC 0.48–0.54, multiclass macro-F1 ≤ 0.17). The four register values carry no label signal. The 0.97 no-dedup accuracy comes entirely from duplicates: 93.6% of test rows also appear in train. |
| TON_IoT IoT devices (appendix) | Exclude | Garage_Door and Motion_Light collapse to 16 unique rows, and Fridge and Thermostat are at chance. GPS_Tracker and Weather score 0.93–0.97, but with 2–3 sensor values per row that more likely reflects when the capture was made than the attack itself. I didn't verify this. |

### Model for the final system
**LightGBM.** It has the best or tied-best attack F1 on all three learnable datasets (Edge-IIoTset 0.995, X-IIoTID 0.996, TON_IoT-Network 0.998) and the best or tied-best multiclass macro-F1 on all four multiclass datasets. It also trains 3–8× faster than RandomForest or the MLP on X-IIoTID, and inference costs about 9–20 ms per 1,000 rows on CPU. RandomForest is a close second. LogReg and the MLP are 0.4–2.6 F1 points lower.

### What this means for the web app
The web app takes free-text events, so it can't use native dataset columns. Two concrete options for Step 6:
1. **Supervised detector on the same embeddings (recommended).** Train a class-weighted classifier on `all-MiniLM-L6-v2` embeddings of the merged datasets. In-domain probe F1 is 0.99 on Edge-IIoTset, 0.93 on TON_IoT-Network and 0.87 on X-IIoTID, compared with 0.00–0.87 for zero-shot at the deployed threshold. The attack/benign verdict then comes from the classifier, and the nearest ATT&CK technique is kept only as an explanation, clearly labelled as approximate.
2. **Native LightGBM per data source**, for structured flow or packet input. This gives the best scores, but the app would need structured input fields, not free text.

**Do not keep the similarity threshold as the detector.** On these datasets attack and normal rows have nearly identical mean similarity to the technique texts: X-IIoTID 0.348 vs 0.352, Modbus 0.264 vs 0.264, BATADAL 0.156 vs 0.156. The top technique matches my analyst mapping for at most 0.4% of attack rows. On network data it is almost always "Commonly Used Port" or "Standard Application Layer Protocol", which reflects the port wording in the description, not the attack.

## 7. Limitations

- **The headline in-domain scores do not show real-world performance.** Edge-IIoTset, X-IIoTID and TON_IoT-Network all score above 99% after de-duplication. That passed the leakage checks (overlap 0–2.8%, no feature with single-feature AUC above 0.87, ablations unchanged). But the cross-dataset results show the learned patterns are mostly testbed-specific. Expect much lower scores on traffic from a different plant.
- **Edge-IIoTset is far smaller than its file size suggests.** 2,219,201 rows contain only 6,513 unique feature vectors under the agreed field policy. DDoS_UDP, DDoS_ICMP and Port_Scanning reduce to 1, 3 and 5 unique packets. They stay in training but can't be tested: their multiclass F1 of 0 means "no test rows", not "missed". The test set is only 976 rows.
- **Source-data defects.** All DDoS_UDP and MITM rows in Edge-IIoTset are column-misaligned in the published CSV. Their timestamp, IP and port columns were dropped or neutralised, and excluding them from the test set doesn't change the score (F1 0.995 vs 0.995). TON_IoT IoT files encode the label through formatting (Garage_Door `true`/`false` vs `1`/`0`, and a leading space in Modbus `time`). Both were normalised or dropped.
- **BATADAL labels are noisy.** `-999` was treated as normal, although some dataset04 attacks were deliberately left unlabelled. The test set has only 2 attack windows (80 hours). The leave-one-window-out folds train on data from after the test period.
- **Embedding experiments use samples:** 5,000 training rows and 2,000 test rows per dataset (fewer where a test split is smaller), all-MiniLM-L6-v2, and a linear probe. The event descriptions use my template: up to 20 non-zero fields, plus HTTP method and URI with the host removed. A different template or a larger language model could change these results.
- **The zero-shot reproduction uses the web app's exact model, technique texts and threshold.** The threshold recomputes to 0.4158, matching `results.json`. But the web app was designed for prose event descriptions, not table rows turned into text, which are much less like the MITRE descriptions. Its failure here applies to that input form.
- **The ATT&CK mapping is mine, not the dataset authors'.** Several IT attacks (SQLi, XSS, password guessing) have only loose ICS equivalents. `techniques.json` lacks T0855 Unauthorized Command Message and T0856 Spoof Reporting Message, the most relevant techniques for BATADAL and Modbus injection.
- **Hyperparameter search was light** (2–3 configurations per model), and the decision threshold was fixed at 0.5. Multiclass runs reuse the binary configuration. X-IIoTID was capped at 250,000 of its 734,479 unique rows, sampled by attack type.
- **Not yet tested:** temporal models for BATADAL, testing on a dataset outside these five, and the final merged model itself (that's Step 6, after your decision).
