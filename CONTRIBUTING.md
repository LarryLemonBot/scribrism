# Contributing

Use a focused branch and a small pull request describing the problem, resulting behavior and verification. Run the two test-discovery commands in README.md. Include synthetic regression cases for correctness fixes that could otherwise recur.

Do not commit downloaded checkpoints, environments, inputs, extracted text, scan reports, private evaluation records, credentials or machine-specific settings. Keep sensitive material outside this checkout. Use invented examples with no real people, organizations or document fragments.

A detector-performance proposal must define its hypothesis and failure criteria before evaluation. Preserve grouped writer/source/prompt/model holdouts, distinguish development findings from independent test results, and report false-positive rates, mixed recall, coverage/abstentions and calibration separately. Document dataset rights and provenance. Never publish held-out examples simply to make a result reproducible; publish a safe manifest or aggregate evidence when the data cannot be shared.

Keep the scoring implementation, configuration, checkpoint and runtime fingerprints with every result. Changed code invalidates inherited claims until the matching implementation is evaluated. Promotion requires independent evidence for the stated operating conditions, with no unexplained regression in human false positives or coverage. This initial release remains experimental and unvalidated.

Before submitting, inspect the full diff and staged file list. Check commit author and committer names and email addresses. If your configured identity is personal, choose an appropriate project-scoped identity before committing. Do not rewrite other contributors' history without agreement.

By contributing code, you agree to provide it under the repository's Apache-2.0 license and preserve applicable third-party notices.
