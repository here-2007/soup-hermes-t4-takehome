# Prepared Hermes data for Soup

Dataset: NousResearch/hermes-function-calling-v1@dae3e1d28cfbcf4b915c04ea1e072030529b4bda
Model/tokenizer: Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306
Soup: 4cff43bf967e96a484381205e5125e2fc39227f1

Counts: {'train': 1600, 'validation': 150, 'test': 150}; max_length=1024.

Merge soup-data.yaml into your training config. Keep response-only labels, the exported raw Jinja template, train_on_eot=true, packing=false, and val_split=0. Load explicit train/validation lists through Soup's wrapper. Do not use a directory glob that could include test.jsonl. Download pinned model weights separately.

The tokenizer directory has the marked template and no model weights. Test prompts have no answers; test references are reserved for final evaluation. Preserve the audit, exclusions, split manifest and hashes.

Automatic structural checks do not establish semantic reference correctness or GPU fit at untested lengths. Review audit/semantic-review-sample.jsonl and calibrate the T4 training setup before a full run.
