# Tool-call gains did not survive NF4

**Pernav Jain | Soup AI Engineer take-home | 10 November 2026**  
Run `20261110T044014005029Z` | **DON'T SHIP**

The dense merge improves exact tool-call correctness from 50.0% to 60.0%. The saved NF4 model scores 49.3%, loses 22 previously correct turns and gains six. Its general tool-call score also regresses. The recorded merge-equivalence gate remains unresolved. Training and export completed, but this 4-bit artifact fails the release criteria.

## 1. Experiment and memory

Qwen2.5-1.5B-Instruct fits the model-size requirement and supports a tractable English task: emit ordered JSON tool calls. Hermes supplies tool schemas and reference calls. The prepared split contains 1,600 train, 150 validation and 150 test turns, with tool/request families separated. Every row fits 1,024 tokens; assistant content and EOT receive loss. Structural audits cover all 1,900 rows. References were not exhaustively adjudicated for semantic correctness.

Soup `0.77.2` at commit `4cff43bf967e` trains a frozen FP16 base with LoRA on seven projections per layer: rank 8, alpha 16, dropout 0.05; AdamW, LR `1e-4`, cosine decay, 5% warmup, batch 1 and accumulation 8. Each of seeds 42/43/44 completes two epochs, or 400 steps. Epoch validation loss selects the checkpoint; validation exact correctness selects seed 42. Test data select neither. One T4 is exposed from Kaggle's two-card allocation; usable capacity is 14.562 GiB.

The preserved pre-training estimate contains the following floors and a **12 GiB planning envelope**:

| Component | Calculation | GiB |
|---|---|---:|
| Frozen weights | 1,543,714,304 x 2 bytes | 2.875 |
| LoRA, gradients, Adam moments | 9,232,384 x (4 + 4 + 8) bytes | 0.138 |
| One full logits tensor | 1 x 1,024 x 151,936 x 2 / 4 bytes | 0.290 / 0.580 |
| NF4 storage floor | 1,310,195,712 x 0.5 + 233,373,696 x 2 bytes | 1.045 |

Checkpointed activations, loss workspaces, allocator/context overhead and export staging sit outside these floors. A retained FP16 hidden state per layer alone is about 0.082 GiB; this reconstruction is a lower bound, not a full activation forecast. The original activation allowance was coarse.

| Measured phase | GPU allocated peak | Sampled device peak | Host RSS peak |
|---|---:|---:|---:|
| Training, maximum across seeds | 4.855 GiB | 5.569 GiB | 3.884 GiB |
| Dense merge | 0.034 GiB | 0.138 GiB | 6.436 GiB |
| Dense staging and NF4 save | 1.089 GiB | 1.247 GiB | 6.132 GiB |

Global allocator reservation peaks at 5.430 GiB, leaving 8.993 GiB of measured headroom. The allocated peak exceeds the base-plus-state floor by 1.842 GiB; that residual includes activations and workspaces. Two dense copies would require 5.751 GiB of host storage, consistent with the order of the export RSS, but tensor-level host attribution was not measured. CUDA counters and 0.2-second-target device/RSS samples describe different quantities; samples can miss spikes and independent peaks cannot be added. M/Q weight files occupy 2.875/1.065 GiB: 2.70x compression, limited by floating-point embeddings and quantization metadata.

## 2. Evidence that the adapter trained

For seed 42, all 392 LoRA tensors change; they contain **9,232,384 scalar weights**. Optimizer membership matches trainable tensors. There are 399 nonzero-LR optimizer steps, one zero-LR warmup step and no AMP skips; the first materialized update is `5.01e-6`. Selected frozen projection hashes remain unchanged. Disabling the adapter restores the base's probe log probabilities exactly; enabling it changes them by up to 18.47. These probes cover four validation rows, 416 reference tokens and 16 distribution positions.

The initial zero-effect adapter is rejected by Soup's adapter check. Substituting the base for M fails the weight test; substituting quantized base Q0 for Q fails identity checks. Cold M/Q reloads block reads from the base/adapter folders. Q contains 196 NF4 linear modules, nested quantization state and FP16 compute. These checks establish an active, changed adapter and real exports; they do not establish desirable behavior or universal equivalence. The training loss of 0.7271 to 0.0023 alone would establish neither.

<!-- PAGE BREAK -->

## 3. Held-out artifact comparison

Complete correctness requires the reference's tool names, JSON argument values, types, call count and order. Invalid or length-stopped responses fail. All 150 test turns remain in the denominator. B is the base, M the dense merge and Q the saved NF4 model.

| Artifact | Correct turns | Exact correctness | Parseable | Assistant NLL |
|---|---:|---:|---:|---:|
| B | 75 / 150 | 50.00% | 94.00% | 0.64338 |
| M | 90 / 150 | 60.00% | 98.00% | 0.04658 |
| Q | 74 / 150 | 49.33% | 94.67% | 0.05657 |

Paired bootstrap over 150 held-out groups, 10,000 resamples: **M-B +10.00 pp [2.00, 18.00]**; **Q-M -10.67 pp [-17.33, -4.00]**; **Q-B -0.67 pp [-7.33, 6.00]** (95% intervals). The eight bundled general suites include a tool-call regression of 70.0% to 57.5% to 55.0% for B/M/Q. No new critical failure appears on six fixed challenge turns; this small set supports no broad safety claim. Across seeds, validation exact correctness is 56.67%, 54.67% and 54.67%; the final test evaluates only selected seed 42.

## 4. Silent failures and check coverage

| Boundary | Observed check and result |
|---|---|
| Data and scoring | Strict parser/schema fixtures reject wrong arguments, duplicate keys, type confusion and extra/reordered calls. Soup format validation accepts the malformed-call fixture; it does not inspect that payload's semantics. |
| Tokenizer/template | Exact response/EOT masks and padding checks pass on every row. Soup doctor rejects the length-boundary fixture (exit 2) and accepts the missing-tools fixture (exit 0). Format validation accepts the double-template fixture (exit 0). |
| Training | Fingerprints, actual updates, adapter on/off probes and frozen-weight hashes corroborate training. Soup catches the initialized no-op adapter (exit 2); a falling loss is insufficient evidence. |
| Merge | Four sampled matrices equal FP16-rounded W + scaled BA, with zero residual. However, A/M maximum reference-logprob error is 0.03867, above the frozen 0.02000 limit; mean KL is 1.08e-6. This is an unresolved numerical-equivalence failure, not proof of a wrong export. |
| Quantization | Q is a standalone NF4 checkpoint, but loses 10.67 pp against M. Quantization metadata and valid JSON do not guarantee retained task behavior. |
| Inference/release | Cold-load identity/device checks pass. Soup `ship --evidence` returns DON'T SHIP for both M and Q using saved measured scores; this is offline decision replay, not another model evaluation. |

The holdout distribution differs from training: Glaive supplies 1,081/1,600 training turns, while primary Hermes supplies 148/150 test turns. Exact-reference grading can reject alternative valid calls. Four projection checks and four probe rows cannot establish whole-model merge equivalence. Eight small general suites and one selected-seed test leave substantial uncertainty.

## 5. Verdict and required changes

**DON'T SHIP.** Six frozen gates fail: merge fidelity, minimum Q/B gain, positive gain interval, maximum Q/M loss, its confidence bound and general-suite retention. Soup independently applies its decision rules to the supplied measured scores and also rejects both candidates.

First investigate A/M differences with matched precision and kernels and a wider validation probe set; retain the failed 0.02 gate rather than relax it after seeing the result. Then inspect the 22 M-correct/Q-wrong turns and compare revised export settings on validation. Address general tool-call retention with better development coverage and data balance. Any tuned successor needs a fresh held-out evaluation before release; the current test has been inspected.

## Reflection and AI use

The surprise was how much exact correctness fell after NF4 despite the small NLL increase from 0.0466 to 0.0566. I remain concerned about semantic reference quality and the limited merge probes. I set the fine-tuning settings in the Soup configuration, accepted the three-seed evaluation and rejected a LoRA-versus-QLoRA training comparison because it was outside the requested pipeline. AI assisted the notebooks, instrumentation, debugging and report drafting; the submission's CPU review checks its measurements against the preserved run. My next step would be to investigate export fidelity and quantization errors before further training.

Evidence: `run/verdict.json`, `run/evidence/{training-42,fidelity-and-export-controls,memory-estimate-before}.json`; full mapping in `docs/evidence-map.md`. Repository: https://github.com/here-2007/soup-hermes-t4-takehome. Exact weights: https://www.kaggle.com/models/pernavjain/soup-experiment/PyTorch/experiment-artifacts/1.
