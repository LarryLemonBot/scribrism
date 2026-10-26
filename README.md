# Scribrism

**Local text-origin signals with inspectable evidence.** Scribrism combines “scribe” and “prism”: it examines writing through several model signals and keeps the evidence behind each result.

Scribrism is an experimental Python CLI for scanning text, Markdown, DOCX, text-based PDF and PPTX text/notes. It runs a pinned four-class model on the CPU, covers every input token with overlapping windows, and saves an HTML report plus reproducibility receipts. Inference uses local model files; the explicit setup command downloads the checkpoint once.

**This release has no independently validated detection-accuracy claim.** Its scores are uncalibrated model outputs, not proof of authorship or percentages of words written by AI. A result must not be used alone to accuse a writer or make an educational or employment decision.

## Quick start

Use Python 3.12 with a virtual environment. Model weights and dependencies are not bundled.

```sh
git clone https://github.com/LarryLemonBot/scribrism.git
cd scribrism
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
./scribrism --help
./scribrism identity
./scribrism setup --download
./scribrism scan document.txt --out outputs/example
./scribrism verify document.txt --report outputs/example/report.json
```

Each scan needs a new output directory and 80–30,000 extracted words. Unsupported inputs, missing or changed model assets, dependency mismatches and incomplete coverage fail with an explicit error instead of an origin score. Scanned PDFs need a text layer; there is no OCR. DOCX/PDF/PPTX extraction may omit visual meaning, equations or non-text objects: check the extraction notes and `extracted.txt`.

The launcher uses `.venv/bin/python` when present. To reuse an existing compatible installation, set `SCRIBRISM_PYTHON` to its Python executable. Set `SCRIBRISM_MODEL_CACHE` to a cache root containing the pinned revision directory and its verified `manifest.local.json`. These overrides avoid copying environments or weights. `identity` does not load or download a model, and does not certify that model assets are present.

The exact dependency lock targets Python 3.12 or later. The official `torch==2.8.0+cpu` Linux build is also accepted and recorded as a distinct runtime. Compatibility acceptance is not evidence that platforms produce identical scores.

## What the report means

The model emits `human`, `ai`, `ai_edited` and `humanized` signals. The decision policy combines `ai` and `humanized` for its AI candidate; `ai_edited` supports a mixed candidate. A mixed label is a model classification, not evidence that multiple people or tools wrote a document. The default configuration is 256-token windows with 128-token stride. Configuration and code hashes appear in every successful report.

Outputs include:

- `report.html`: readable result, limitations and extraction details.
- `report.json`: model, code, configuration, runtime and input fingerprints.
- `extracted.txt`: the text actually scanned.
- `receipt.json`: hashes of those three saved artifacts.

`verify` checks that the saved artifacts, input and current scoring implementation still match. It does not independently rerun inference, authenticate the writer, or make unsigned receipts tamper-proof against someone who can rewrite every file.

## Privacy

Scans run locally after setup. **Generated reports contain the input text and may contain absolute file paths and error details.** Keep them private unless you have reviewed and authorized their contents. Do not attach real documents or scan outputs to public issues, pull requests or CI jobs. The repository ships code, synthetic tests and public upstream notices; no user documents, model weights or evaluation corpora.

See [PRIVACY.md](PRIVACY.md) for release and contribution checks.

## Development and agent access

```sh
.venv/bin/python -m unittest discover -p 'test_*.py'
.venv/bin/python -m unittest discover -s evaluation -p 'test_*.py'
```

The synthetic software suite needs installed dependencies but no downloaded weights. Passing it establishes software behavior, not detector accuracy. See [AGENTS.md](AGENTS.md) for a compact workflow usable by coding agents, [CONTRIBUTING.md](CONTRIBUTING.md) for review expectations and [evaluation/README.md](evaluation/README.md) for research boundaries.

The repository includes development-only evaluation utilities and a tested masked-mean pooling primitive. It does **not** bundle a working FAID retrieval index, a span detector, a Jev service or an MCP server. Agents can invoke the CLI directly from a checked-out repository; no privileged access is needed.

## Model and licensing

The default checkpoint is [`wasitaigeneratedcom/ai-text-detector-small`](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615), pinned to revision `f1795c86806e6838d4afa33d0b1427f8430c9615`. Downloads are verified against the pinned manifest. A changed checkpoint requires a deliberate profile change and new evaluation.

Scribrism code is licensed under [Apache-2.0](LICENSE). Preserve [NOTICE](NOTICE) and the upstream notices in [licenses/](licenses/). [THIRD_PARTY.md](THIRD_PARTY.md) records model lineage, source pins and license-evidence limitations. No benchmark accuracy is inherited from upstream or an earlier implementation.
