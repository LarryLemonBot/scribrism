# Agent workflow

This is an experimental local detector, not an authorship certification service. Read README.md and PRIVACY.md before running or modifying it.

## Start

1. Inspect Git status. Preserve unrelated changes.
2. Run `./scribrism --help` and `./scribrism identity`. Identity reports capabilities and dependency problems without downloading weights.
3. Reuse an existing compatible environment/cache through `SCRIBRISM_PYTHON` and `SCRIBRISM_MODEL_CACHE` when available. Ask before a new paid service or large download if it was not authorized. Model setup is explicit: `./scribrism setup --download`.
4. Run both unittest discovery commands in README.md. Do not use real private documents as test fixtures.

## Make a bounded change

State the problem, expected behavior, proposed test and failure criteria before changing scoring. Keep changes on a branch. Preserve failed experiments and their fingerprints in private research storage when they contain non-public data.

Separate software correctness from detection performance. A passing synthetic pooling test proves a tensor property; it does not prove better document classification. Any model, tokenizer, windowing, extraction, threshold, aggregation or dependency change needs matching evaluation before reusing an accuracy claim.

For performance evaluation, keep writer, source, prompt and generator-model groups disjoint across training, development, calibration and untouched test sets. Reserve an independent final test; do not repeatedly tune against it. Writing-development examples and rewritten assignments must not enter that test set. Report human false positives, mixed recall, per-class recall, coverage/abstention, calibration and uncertainty separately. Never lower thresholds merely to obtain a requested label.

## Finish

Run checks appropriate to the change, then the synthetic regression suite. Read the diff and check tracked files and commit author/committer metadata for private information. Public PRs may include synthetic fixtures and aggregate results with public provenance, never personal documents, paths, model caches or private reports. Describe exactly what was tested and what remains unknown. Keep notices and source attribution intact.

A CLI call returning a score is not a benchmark. Do not claim an agent participated, a scan ran, a test passed or a model improved without the corresponding evidence. Ask contributors to return the exact commit, environment and command results so their work can be reproduced.
