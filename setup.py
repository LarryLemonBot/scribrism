"""Explicit pinned-checkpoint download; no download occurs on module import."""
import argparse
import json
from pathlib import Path
from model_store import model_root, verify_model
import review

ROOT = Path(__file__).resolve().parent


def install():
    p = review.read_json(ROOT / 'profile.json')
    cache = model_root(ROOT)
    folder = cache / p['model_revision']
    if (folder / 'manifest.local.json').exists():
        verify_model(ROOT, p)
        return {'status': 'verified_existing', 'cache': str(folder), 'network_used': False}
    if cache.is_symlink() or folder.is_symlink():
        raise ValueError('Refusing to download through a cache symlink.')
    from huggingface_hub import snapshot_download
    snapshot_download(p['model_id'], revision=p['model_revision'], local_dir=folder,
                      allow_patterns=['*.json', '*.safetensors', 'spm.model', 'README.md', 'NOTICE'])
    manifest = {'model_id': p['model_id'], 'revision': p['model_revision'], 'files': {}}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != 'manifest.local.json':
            manifest['files'][path.name] = {'sha256': review.file_sha(path), 'bytes': path.stat().st_size}
    encoded = json.dumps(manifest, indent=2) + '\n'
    if review.sha_bytes(encoded.encode()) != p['model_manifest_sha256']:
        raise ValueError('Downloaded files differ from the pinned manifest. Profile was not changed.')
    (folder / 'manifest.local.json').write_text(encoded, encoding='utf-8')
    verify_model(ROOT, p)
    return {'status': 'downloaded_and_verified', 'cache': str(folder), 'network_used': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(install()))
