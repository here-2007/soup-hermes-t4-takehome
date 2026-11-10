# Evidence map

All `run/` files below retain the hashes recorded in the original 215-entry [index](../run/evidence-index.json). The [CPU review](../tools/review.py) verifies that index, the 53 data-preparation entries and the saved score calculations. An internally consistent record can still describe a failed model.

In the curated email attachment, run `python tools/review.py --email-bundle`. It checks the 205 included original run files and seven preparation entries, then reproduces the same 450 test scores and three bootstrap comparisons. `EMAIL_MANIFEST.json` lists the ten original run files and 46 preparation entries left in the full archive. The original indexes are included unchanged; the email review reports those omissions explicitly.

| Claim or requirement | Primary evidence |
|---|---|
| Frozen recipe, identities and thresholds | [run_policy.json](../run/run_policy.json), [training plan](../run/evidence/training-plan.json), [freeze record](../run/evidence/policy-frozen-before-test.json) |
| Estimate recorded before optimization | [memory-estimate-before.json](../run/evidence/memory-estimate-before.json); compare its timestamp with training logs |
| Allocated/reserved/device/host peaks | [memory-reconciliation.json](../run/evidence/memory-reconciliation.json), [stage reports](../run/evidence/stages) |
| Raw device and host measurements | [nvidia-smi.raw.log](../run/evidence/nvidia-smi.raw.log), [host-samples.jsonl](../run/evidence/host-samples.jsonl), initial/final nvidia-smi files in the same directory |
| Actual training and active adapter | [training-42.json](../run/evidence/training-42.json), [training-43.json](../run/evidence/training-43.json), [training-44.json](../run/evidence/training-44.json) |
| Full command status and raw timestamped logs | [commands.jsonl](../run/evidence/commands.jsonl), [events.jsonl](../run/evidence/events.jsonl), [logs](../run/evidence/logs) |
| Selected seed and validation-only rule | [selection.json](../run/evidence/selection.json), [validation scores](../run/evidence/scores) |
| B/M/Q exact scores and paired uncertainty | [verdict.json](../run/verdict.json), [all saved predictions](../run/evidence/predictions), [references](../run/evidence/references.jsonl) |
| Standalone M/Q identity and actual NF4 | [identity-M.json](../run/evidence/identity-M.json), [identity-Q.json](../run/evidence/identity-Q.json), [artifact-M.json](../run/evidence/artifact-M.json), [artifact-Q.json](../run/evidence/artifact-Q.json) |
| Weight/logprob fidelity and wrong-export controls | [fidelity-and-export-controls.json](../run/evidence/fidelity-and-export-controls.json), [raw probe tensors in the full repository](https://github.com/here-2007/soup-hermes-t4-takehome/tree/main/run/evidence/probes) |
| Deliberate data/scorer failures | [scorer-and-data-controls.json](../run/evidence/scorer-and-data-controls.json), [silent-failure-coverage.json](../run/evidence/silent-failure-coverage.json), matching fixture logs/stage reports |
| Soup's own decisions | [soup-ship-M.json](../run/evidence/soup-ship-M.json), [soup-ship-Q.json](../run/evidence/soup-ship-Q.json); inputs in the same directory |
| Eight general suites, raw generations | [general](../run/evidence/general) |
| Dataset origin, grouping, exclusions and masks | [data audit](../data-preparation/audit/data-audit.json), [split manifest](../data-preparation/audit/split-manifest.json), [preparation manifest](../data-preparation/manifest.json) |
| Exact completed implementation | [original verifier](../run/verify_artifacts.py), [executed training notebook](../notebooks/02_train_and_verify_soup_t4.executed.ipynb), [executed preparation notebook](../notebooks/01_prepare_hermes_for_soup.executed.ipynb) |
| Final human-readable deliverable | [two-page PDF](../report.pdf), [report source](../REPORT.md), [reflection and AI use](reflection-and-ai-use.md) |

The automated draft and placeholder in `run/REPORT.md` and `run/REFLECTION_AND_AI_USE.md` remain unchanged. They are historical evidence, not the completed submission report.
