"""Bind an evaluation to the implementation it actually measured.

This identity is a reproducibility checksum, not a signature or proof that an
evaluation was conducted honestly. Evaluation writers must retain data lineage
and predictions separately. No binding is created automatically by a scan.
"""
import hashlib
import json
from pathlib import Path


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def scoring_identity(root, profile, runtime):
    root = Path(root)
    files = ['service.py', 'review.py', 'report_html.py', 'pptx_reader.py', 'validation_binding.py', 'model_store.py',
             'requirements.lock', profile['config_file']]
    if profile['backend'] == 'four_class':
        files.append('four_class.py')
    components = {}
    for name in files:
        with (root / name).open('rb') as stream:
            components[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    payload = {
        'schema_version': 1,
        'backend': profile['backend'],
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'model_manifest_sha256': profile['model_manifest_sha256'],
        'components': components,
        'runtime': runtime,
    }
    return {'sha256': canonical_sha(payload), 'payload': payload}


def check_binding(binding, result, result_sha256, identity):
    """A result and a separate binding must both identify the measured engine."""
    reasons = []
    if binding.get('schema_version') != 1:
        reasons.append('Unsupported or missing evaluation binding schema.')
    if binding.get('scoring_identity_sha256') != identity['sha256']:
        reasons.append('Evaluation was not bound to this scoring implementation and runtime.')
    if binding.get('validation_result_sha256') != result_sha256:
        reasons.append('Evaluation result does not match its binding.')
    evaluated = result.get('evaluation', {})
    if evaluated.get('scoring_identity_sha256') != identity['sha256']:
        reasons.append('Result does not identify this evaluated scoring implementation.')
    dataset_sha = evaluated.get('dataset_manifest_sha256')
    if (not isinstance(dataset_sha, str) or len(dataset_sha) != 64 or
            any(c not in '0123456789abcdef' for c in dataset_sha)):
        reasons.append('Result has no valid dataset-manifest digest.')
    if binding.get('dataset_manifest_sha256') != dataset_sha:
        reasons.append('Evaluation dataset identity does not match its binding.')
    return reasons
