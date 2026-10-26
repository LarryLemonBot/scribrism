# Third-party provenance

Scribrism's original code and modifications are distributed under the Apache
License 2.0 in [LICENSE](LICENSE). Third-party notices retain their own scope.
This repository does not contain model weights, tokenizer payloads, research
corpora, reference-vector databases, or installed dependency packages.

## Inference architecture adapter

The local four-class adapter implements the architecture illustrated in the
[`wasitaigeneratedcom/ai-text-detector-small` model card](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/blob/f1795c86806e6838d4afa33d0b1427f8430c9615/README.md):
a transformer backbone, attention-masked mean pooling and a linear output head.
That pinned card declares Apache-2.0. Its upstream
[NOTICE](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/blob/f1795c86806e6838d4afa33d0b1427f8430c9615/NOTICE)
credits Desklib and Microsoft under MIT. The pinned model tree supplies no
standalone Apache LICENSE file. Scribrism includes the full standard Apache-2.0
text and preserves the publisher's notice verbatim in
[licenses/wasitaigenerated-model-NOTICE.txt](licenses/wasitaigenerated-model-NOTICE.txt).

The four-class model's declared lineage includes
[`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01/blob/5fdea974cd4287c61674951ec78803aa274e2fb7/README.md).
The pinned model card declares MIT, but its tree contains no standalone LICENSE
and its card supplies no copyright line. The Desklib attribution in the
four-class publisher's NOTICE is preserved with the standard MIT permission
and warranty text in [licenses/MIT-Desklib.txt](licenses/MIT-Desklib.txt).
This is a reconstruction from those explicit declarations, not a claim that
Desklib published that separate license file.

The adapter is a modified implementation: it uses local pinned assets,
explicit safe tensor loading, overlapping-window coverage and local validation
and reporting. It does not reproduce the publisher's complete serving or
benchmark pipeline. These changes are also identified in four_class.py.

## Model lineage

Both model cards identify [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
as backbone lineage. No Microsoft model implementation or weights are vendored.
The Microsoft notice is nevertheless retained with the upstream attributions.
[licenses/MIT-Microsoft-DeBERTa.txt](licenses/MIT-Microsoft-DeBERTa.txt) reproduces
the Microsoft project's
[MIT license](https://github.com/microsoft/DeBERTa/blob/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098/LICENSE).
That code-repository revision identifies the license text; it is not asserted
to be the training revision of the external detector checkpoint.

Model identifiers and hashes are references to separately obtained artifacts.
The Apache license for Scribrism does not replace the licenses attached to
those artifacts. The model publisher's attribution and license declarations
are recorded here; they are not an independent audit of training-data rights.

## FAID research reference

The research-only masked pooling primitive is newly written code. It was
developed after identifying discarded out-of-place masking in the published
[FAID embedding implementation](https://github.com/ngocminhta/FAID/blob/a6551e639c36ba84f4948b2005aed3a12974033c/algorithm/src/text_embedding.py).
No FAID implementation, model, index, label mapping or corpus is vendored.
The source reference is acknowledged in [NOTICE](NOTICE), and its
[MIT license](https://github.com/ngocminhta/FAID/blob/a6551e639c36ba84f4948b2005aed3a12974033c/LICENSE.md)
is retained verbatim in [licenses/MIT-FAID.txt](licenses/MIT-FAID.txt) for
provenance. That attribution does not claim the new primitive reproduces FAID
or inherits its reported performance.

## Installed dependencies and research data

PyTorch, Transformers, Safetensors and other packages listed in the dependency
files are installed separately and retain their own licenses. This source
release does not vendor those packages. A binary, container or bundled release
would need a separate inventory of the dependency artifacts it actually ships.

Only synthetic fixtures should accompany this source release. References to a
research corpus or paper do not grant redistribution rights to its texts.
Public model and research names identify provenance and do not imply
endorsement or trademark rights.
