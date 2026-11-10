# Reproduction and artifact access

## Verify the saved evidence on CPU

Install `requirements-review.txt` and run `python tools/review.py` from the repository root. This verifies all indexed original run files and data-preparation files, recomputes strict scores for 450 test generations, repeats the three 10,000-resample paired bootstrap calculations, replays the decision and compares the raw GPU-0 trace peak with the stage summaries. It does not load weights or repeat model inference.

To write a new CPU-review result without changing the original records:

```bash
python tools/review.py --output /tmp/soup-evidence-review.json
```

`analysis/review.json` is the captured CPU review. The model release decision remains DON'T SHIP even when this evidence audit returns PASS.

## Exact environment and source pins

| Item | Recorded identity |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Base revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Dataset | `NousResearch/hermes-function-calling-v1` |
| Dataset revision | `dae3e1d28cfbcf4b915c04ea1e072030529b4bda` |
| Prepared Kaggle dataset | `pernavjain/tool-calling`; exact files are also in `run/data/` |
| Soup | `0.77.2`, commit `4cff43bf967e96a484381205e5125e2fc39227f1` |
| Recorded runtime | Python `3.13.16`, PyTorch `2.11.0+cu130` |
| Training libraries | Transformers `5.20.2`, PEFT `0.21.2`, TRL `0.29.1`, bitsandbytes `0.50.2`, Accelerate `1.15.0` |
| Hardware | Tesla T4, compute capability 7.5; one visible GPU, 15,636,037,632 bytes usable |

The completed notebook pins the training stack, keeps compatible Kaggle PyTorch, removes an incompatible optional TorchAO installation and records the environment. The full dependency listing is in the original `pip-freeze` log. Report-build dependencies are separate from the GPU environment.

## Download the original model artifacts

The versioned backup is [Kaggle model version 1](https://www.kaggle.com/models/pernavjain/soup-experiment/PyTorch/experiment-artifacts/1). The exact handle is:

```text
pernavjain/soup-experiment/pytorch/experiment-artifacts/1
```

With Kaggle's CLI installed and any authentication required by Kaggle configured:

```bash
kaggle models instances versions download \
  pernavjain/soup-experiment/pytorch/experiment-artifacts/1 \
  -p downloads --untar
```

The uploaded tree contains `20261110T044014005029Z/artifacts/`: base B, adapters for all three seeds, their initialized controls, M (`merged-fp16`), Q (`merged-nf4`), quantized-base control Q0 and Trainer checkpoints. The selected adapter is `adapter-42`. Model file hashes and sizes are recorded in `run/evidence/base-identity.json`, `artifact-M.json`, `artifact-Q.json` and each seed's training record.

Weights are excluded from Git and the submission ZIP. They are not needed for the CPU score replay. The original GPU verifier requires the weights and the pinned training stack for new model-owning checks. Recorded GPU UUIDs, absolute paths and cache hashes belong to the original runtime; restoring files on another machine does not recreate that runtime automatically.

## Reproduce the GPU pipeline

Run `01_prepare_hermes_for_soup.executed.ipynb` with Internet enabled to reproduce preparation. Its original prepared outputs and audit are already included under `data-preparation/`. Attach the four files `train.jsonl`, `validation.jsonl`, `test.jsonl` and `soup-data.yaml` for the training notebook.

Run `02_train_and_verify_soup_t4.executed.ipynb` in a fresh Kaggle T4 session. It creates a new run directory, performs the measured memory gate, trains three fresh adapters, exports M/Q/Q0, reloads the artifacts, evaluates them and produces a verdict. Skip its final auxiliary backup-upload cell, which targets the author's Kaggle account. The model-owning implementation is embedded in the notebook and is also preserved as `run/verify_artifacts.py`.

Do not overwrite the original `run/` to make a cached stage accept different hardware or settings. A new execution, changed export or additional training is a new experiment. Exact bitwise replication across GPU/runtime versions is not claimed. After development informed by this test set, use a fresh held-out set for the next release decision.

## Build the report and submission ZIP

```bash
python -m pip install -r requirements-report.txt
python tools/build_report.py
python tools/package_submission.py --output ../dist/soup-ai-engineer-submission.zip
```

The report source is `REPORT.md`. Its explicit page break separates the two report pages. The package builder writes a SHA-256 manifest, excludes Git metadata/caches/model downloads and checks every archived file. The original run and preparation manifests remain intact inside the final package.
