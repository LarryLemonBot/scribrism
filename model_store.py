"""Locate and verify an explicit local checkpoint cache; never fetch assets."""
import os
from pathlib import Path
import review


def model_root(product):
    value = os.environ.get('SCRIBRISM_MODEL_CACHE')
    return Path(value).expanduser().absolute() if value else Path(product) / 'models'


def verify_model(product, profile):
    folder = model_root(product) / profile['model_revision']
    manifest_path = folder / 'manifest.local.json'
    if review.file_sha(manifest_path) != profile['model_manifest_sha256']:
        raise ValueError('Pinned model manifest changed.')
    manifest = review.read_json(manifest_path)
    if (manifest.get('model_id') != profile['model_id'] or
            manifest.get('revision') != profile['model_revision']):
        raise ValueError('Model identity mismatch.')
    files = manifest.get('files', {})
    if not files or 'model.safetensors' not in files:
        raise ValueError('Model manifest has no weight file.')
    for name, record in files.items():
        path = (folder / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder.resolve()):
            raise ValueError('Model manifest path escapes checkpoint directory.')
        if review.file_sha(path) != record['sha256']:
            raise ValueError('Model or tokenizer changed: ' + name)
    return manifest
