# Evaluation utilities

These are research tools, not a validated benchmark or an automatically promoted classifier. No real evaluation corpus is distributed.

- `evaluate.py` evaluates supplied cached class scores with explicit configuration, grouping checks, coverage and uncertainty reporting. Inspect its `--help` and input validation before constructing a manifest.
- `run_window_comparison.py` runs declared development window comparisons and writes reproducibility receipts. Its cache and engine must be supplied explicitly. Development results do not count as an independent test.
- `score_candidates.py` and `region_candidate.py` support development-only aggregation experiments. They do not silently replace the default scanner.
- `masked_pooling.py` implements attention-mask-aware mean pooling for tensor inputs. Its synthetic tests check padding invariance and edge cases. It is not a complete FAID model or retrieval pipeline; compatible reference embeddings/indexes and independent end-to-end evaluation would be required to assess a detector change.

Run the synthetic suite with `python -m unittest discover -s evaluation -p 'test_*.py'` from the repository root. These tests require installed dependencies but no model weights.

Keep writer/source/prompt/generator groups separate across splits. Freeze the independent test set before tuning and keep writing-development material out of it. Report human false-positive rates, mixed recall, coverage/abstentions and calibration separately, with sample counts and uncertainty. Retain failed experiments. Never transfer accuracy claims across changed code, configuration, model assets or runtime without matching evidence.
