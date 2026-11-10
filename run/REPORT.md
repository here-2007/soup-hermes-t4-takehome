SOUP / HERMES / QWEN2.5-1.5B
Measured experiment report - DON'T SHIP

Base: Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4c
Soup: 4cff43bf967e; prepared Kaggle dataset: pernavjain/tool-calling
Data: 1600 training /150 validation /150 test; exact file hashes in run_policy.json.
Training: frozen FP16 base, rank8 all seven projections, batch1, accumulation8, two epochs.
Recipe: LR1e-4, cosine decay, 5% warmup; export the lowest validation-loss epoch.
Seed selection uses validation complete-turn correctness, then NLL, then seed.

TASK RESULTS - complete ordered turns; every held-out turn remains in denominator
B: 50.000% complete; 94.000% parse; assistant NLL=0.64338; n=150
M: 60.000% complete; 98.000% parse; assistant NLL=0.04658; n=150
Q: 49.333% complete; 94.667% parse; assistant NLL=0.05657; n=150

PAIRED GROUP BOOTSTRAP - 10000 seeded resamples
Q_minus_B: -0.667%;95% CI [-7.333%,+6.000%];groups=150
Q_minus_M: -10.667%;95% CI [-17.333%,-4.000%];groups=150
M_minus_B: +10.000%;95% CI [+2.000%,+18.000%];groups=150

RESOURCE RECONCILIATION
usable_device_bytes: 14.562 GiB
max_allocated_bytes: 4.855 GiB
max_reserved_bytes: 5.430 GiB
max_device_used_bytes_sampled: 5.569 GiB
headroom_bytes: 8.993 GiB
Allocator counters belong to the tensor-owning child. Device/RSS samples can miss spikes.
The original estimate precedes optimization. Activations, workspaces and context add to the floor.

ARTIFACTS: adapter, merged-fp16, merged-nf4 retained under artifacts/.
Cold M/Q loads use local files, block base/adapter reads and reject CPU/meta fallback.

---

GATES, CONTROLS AND LIMITATIONS

RELEASE GATES
PASS: required_stage_reports_pass
PASS: policy_and_test_unchanged
PASS: artifact_hashes_bound
PASS: updated_and_active_adapter
PASS: selected_frozen_weights_unchanged
PASS: disabled_base_agreement
BLOCK: merge_fidelity
PASS: scorer_data_controls
PASS: base_for_merge_control
PASS: q0_for_q_control
PASS: Soup_noop_adapter_control
PASS: portable_M_Q
PASS: real_saved_nf4
PASS: no_new_critical_challenge_failures
PASS: one_gib_tested_headroom
PASS: three_seed_development_complete
PASS: Soup_environment_bounds_ok
PASS: Soup_fixture_coverage_resolved
BLOCK: task_gain_ge_5pp
BLOCK: task_gain_ci_positive
BLOCK: quant_point_loss_le_2pp
BLOCK: quant_ci_lower_ge_minus5pp
BLOCK: general_suite_regression_le_5pp

GENERAL REGRESSION - absolute suite scores B /M /Q
mini_mmlu: 0.850 / 0.850 / 0.825
mini_common_sense: 0.825 / 0.825 / 0.850
mini_instruction: 1.000 / 1.000 / 0.958
mini_arithmetic: 0.750 / 0.825 / 0.875
mini_tool_call: 0.700 / 0.575 / 0.550
mini_format_json: 0.950 / 0.950 / 1.000
mini_safety: 1.000 / 1.000 / 1.000
mini_over_refusal: 0.325 / 0.450 / 0.375

LIMITS AND REFLECTION
Four validation distribution probes and four weight projections do not prove full-distribution equivalence.
Prepared holdouts are source-shifted toward primary Hermes; source labels are absent from the four-file export.
References are structurally audited; semantic correctness and alternative valid calls are not automatically adjudicated.
Bundled general suites and six challenge turns have finite coverage; this is not an arbitrary-tool safety claim.
Author reflection: PENDING - add your own observed surprise, concern and next step before submission.
AI disclosure: AI assisted notebook/code design; this report uses saved run outputs. Author should describe their checks and edits.
This is an automated report draft. Inspect the final two-page PDF and complete reflection before submission.
