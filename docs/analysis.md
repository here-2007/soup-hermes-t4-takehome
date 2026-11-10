# Analysis of the completed run

This document expands the two-page report. Measurements come from the original run; the CPU replay adds consistency checks, transition counts and descriptive summaries. It does not add a new GPU experiment or change the frozen release policy.

## Dataset and split tradeoffs

Two files from pinned `NousResearch/hermes-function-calling-v1` were considered: 1,893 primary single-turn rows and 5,209 Glaive rows. The preparation audit records 4,237 strict candidates, 4,218 unique schema-valid rows and 4,004 rows within 1,024 tokens. The grouped split selects 1,600/150/150 rows without cutting a reference answer to fit. At 512 tokens, the retained pool could not satisfy the chosen grouped split.

| Split | Primary Hermes | Glaive | Single-call turns | Multiple-call turns | Median tokens |
|---|---:|---:|---:|---:|---:|
| Train | 519 | 1,081 | 1,333 | 267 | 424 |
| Validation | 149 | 1 | 69 | 81 | 753 |
| Test | 148 | 2 | 86 | 64 | 732 |

Grouping connected source IDs, normalized requests, tool names and structural schema families prevents those known families crossing splits. It also produces a source and length shift: training is 67.6% Glaive, while test is 98.7% primary Hermes and has longer examples. This is a limitation of the sampling design. The result supports the reported held-out distribution; it does not estimate performance on a balanced deployment population.

Training receives 63,502 shifted supervised tokens per epoch, despite 1,600 examples. With two epochs, each seed records 127,004 supervised tokens and 1,630,902 nonpadding tokens. Response-only masking makes example count a poor proxy for the amount of supervised signal.

The audit validates structure, not the truth or completeness of every reference. It provides a semantic-review sample; the saved records do not establish exhaustive human adjudication. A strict ordered-call metric can penalize alternative valid responses, redundant defaults, or a different ordering of independent calls. These outcomes are called exact-reference correctness throughout the submission.

## Training and checkpoint selection

| Seed | Epoch-1 validation loss | Epoch-2 validation loss | Validation exact turns | Selected epoch |
|---|---:|---:|---:|---:|
| 42 | 0.0450430 | 0.0434547 | 85/150 | 2 |
| 43 | 0.0474378 | 0.0435663 | 82/150 | 2 |
| 44 | 0.0451205 | 0.0445921 | 82/150 | 2 |

Each seed starts from the pinned base and fresh adapter/optimizer. Validation exact correctness has mean 55.33% and sample standard deviation 1.15 percentage points across three seeds. That small sample describes variability; it is not a precise variance estimate. The final test applies only to seed 42, selected by validation.

The actual optimizer has exactly the trainable tensor IDs. All seeds complete 400 optimizer callbacks and record 399 nonzero-LR steps, no AMP skips, finite adapters and changed parameter fingerprints. The zero-LR first warmup step is not counted as a useful update. Exported adapter tensors are checked against the lowest-validation-loss epoch checkpoint.

For seed 42, all 392 trainable tensors change. Selected base projection hashes remain unchanged and adapter-disabled probes equal the pristine base. Adapter-enabled probes differ. This supports mechanical training and active adapter use. Useful learning is a separate claim, tested by held-out generations and general-suite retention.

## Memory calculation and measurement limits

The frozen model has 1,543,714,304 unique base weights. FP16 storage is 2.875 GiB. Rank-8 LoRA on q/k/v/o/gate/up/down contributes 9,232,384 weights. FP32 adapter values, gradients and two Adam moments require 16 bytes per scalar, or 0.138 GiB. Adam's materialized CUDA tensors occupy 73,859,072 bytes, with 1,568 bytes of CPU step counters.

At sequence length 1,024 and vocabulary size 151,936, one complete logits tensor costs 0.290 GiB in FP16 or 0.580 GiB in FP32. Its loss-related copies and workspace lifetimes matter. A single retained FP16 `[sequence, hidden]` boundary across 28 layers is `28 x 1024 x 1536 x 2 = 0.082 GiB`. This is an architectural reconstruction, not an observed decomposition or a complete activation estimate. Checkpoint recomputation, MLP/attention intermediates and backward workspaces add to it.

The pre-training JSON records explicit state/logit floors and a conservative 12 GiB envelope. It does not provide a precise activation or host-staging forecast. The observed 4.855 GiB allocated peak leaves 1.842 GiB above the 3.013 GiB base-plus-adapter/state floor. That difference is an aggregate residual; it cannot identify individual activation tensors.

Peak reserved memory is 5.430 GiB and peak sampled device use is 5.569 GiB. The 14.562 GiB reported T4 capacity leaves 8.993 GiB against the more conservative of reservation and sampled device use. The raw trace contains 60,824 physical-GPU-0 samples. Peak RSS and device usage can occur at different moments; peak-to-peak subtraction does not isolate CUDA context or allocator fragmentation. Sampling has a 0.2-second target plus subprocess overhead and can miss brief transients.

Dense merge and NF4 conversion peak at 6.436 and 6.132 GiB host RSS. Two dense copies would cost 5.751 GiB, but the run does not attribute the sampled RSS to specific buffers or mapped pages. Export memory must be considered on the host as well as the GPU.

Q contains 196 quantized linear modules with nested NF4 state and FP16 compute. Embeddings remain floating point. Its 1,143,327,335-byte weights are 2.70x smaller than M's 3,087,466,808-byte weights. A naive fourfold reduction assumes every parameter becomes four-bit storage and ignores quantization metadata.

## Where the task gain was lost

| Transition | Both correct | Lost correct | Gained correct | Both incorrect |
|---|---:|---:|---:|---:|
| B to M | 63 | 12 | 27 | 48 |
| M to Q | 68 | 22 | 6 | 54 |

The dense model improves single-call exact correctness from 40/86 to 57/86, while multiple-call correctness falls from 35/64 to 33/64. Q scores 43/86 and 31/64 respectively. The aggregate dense improvement therefore does not establish improved multi-call behavior. Full-length and call-count slices remain visible in the original verdict.

Q's teacher-forced assistant NLL rises only from 0.04658 to 0.05657 compared with M, while exact correctness falls 10.67 percentage points. NLL averages reference-token probabilities; exact correctness requires every relevant tool choice, argument and call to match in an autoregressively generated turn. Small probability changes can alter the decoded sequence. This mechanism is plausible; the saved data do not isolate the causal contribution of any specific quantized layer.

`analysis/review.json` includes the first three M-correct/Q-wrong turns in frozen test order. They are descriptive examples selected after evaluation, not independent validation cases. The full 150-turn predictions remain available to avoid relying on a few examples.

## What the merge failure means

The numerical gate uses four validation rows, 416 reference tokens and 16 distribution positions. The initialized-adapter and repeated-base floors are zero. The frozen A/M maximum reference-logprob tolerance is 0.02000; the observed maximum is 0.03867495. Mean reference-logprob error is 0.00042847 and mean KL is 0.00000108. Four checked q/v matrices from the first and last layers exactly equal FP16-rounded `W + (alpha/r) BA`.

These results do not identify a corrupt checkpoint. Different FP16 operation ordering between an active adapter and a merged matrix is one possible explanation. Precision, kernel choice and broader layer/probe coverage should be tested before attributing the discrepancy to the merge implementation. The threshold remains failed in this submission. Even if that discrepancy were resolved, the NF4 task loss and general tool-call regression independently block release.

## Soup coverage and release rules

The tested format validator accepts a malformed tool payload and double-template fixture. The tested data doctor accepts a missing-tools fixture and rejects the deliberate length-boundary case with exit 2. These observed results describe these commands and fixtures, not every Soup check or a claim that Soup promises semantic validation.

The project adds exact tokenizer masks, reserved-token checks, strict JSON/schema/call scoring, update evidence, wrong-export controls and standalone reload checks. The initialized no-op adapter is rejected by Soup's adapter check. Soup `ship --evidence` consumes the measured score files offline: M is rejected for a general regression; Q is rejected for failing to beat the base. Its general results also show the Q regression. The full project verdict additionally enforces confidence, fidelity, artifact identity and measured memory headroom.

| Failed gate | Frozen requirement | Observed result |
|---|---|---|
| Merge numerical fidelity | Maximum A/M reference-logprob error <= 0.02000 | 0.03867495; four sampled weight checks and KL limit pass |
| Q task gain | Q-B >= +5 pp | -0.67 pp |
| Gain uncertainty | Lower 95% Q-B bound > 0 | -7.33 pp |
| Quantization loss | Q-M >= -2 pp | -10.67 pp |
| Quantization uncertainty | Lower 95% Q-M bound >= -5 pp | -17.33 pp |
| General-suite retention | No suite loses more than 5 pp | Tool-call suite loses 15 pp for Q; 12.5 pp for M |

The resource gate requires at least 1 GiB of measured headroom and passes at 8.993 GiB. The pre-flight doctor/dry-run and split reports establish configuration and structural checks; their successful execution does not establish held-out behavior. All 17 other required integrity checks in the recorded verdict pass. The five task/retention gates above and the merge gate remain failed.

A changed model or export needs validation-led development and fresh final holdouts. More epochs, higher LoRA rank, a different quantizer or mixed-precision exclusions are hypotheses to evaluate, not fixes demonstrated by this run.
