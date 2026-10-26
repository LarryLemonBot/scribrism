"""Scribrism: experimental local text-origin signals with reproducible receipts."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version as package_version, PackageNotFoundError
import json
from pathlib import Path
import platform
import sys
import time

import review
from model_store import model_root, verify_model
from pptx_reader import extract_pptx
from report_html import render
from validation_binding import scoring_identity, check_binding

ROOT = Path(__file__).resolve().parent
VERSION = '0.1.0'
SCHEMA = 1
SCORE_SEMANTICS = {
    'human': 'Raw model output for human-written text.',
    'ai': 'Raw model output for direct AI-generated text.',
    'ai_edited': 'Raw model output for human text edited with AI; mapped to a mixed candidate.',
    'humanized': 'Raw model output for paraphrased AI text; combined with ai for the AI candidate.',
    'limits': 'Uncalibrated signals, not authorship probabilities or percentages of words written by AI.',
}


def profile():
    return review.read_json(ROOT / 'profile.json')


def extract(path):
    return extract_pptx(path) if Path(path).suffix.lower() == '.pptx' else review.extract(path)


def runtime_versions(strict=True):
    packages, errors = {}, []
    for line in (ROOT / 'requirements.lock').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        try:
            actual = package_version(name)
        except PackageNotFoundError:
            actual = None
        packages[name] = actual
        # The official Linux CPU wheel has this explicit local version suffix.
        # Record it fully; this exception does not claim numerical equivalence.
        cpu_build = name == 'torch' and expected == '2.8.0' and actual == '2.8.0+cpu'
        if actual != expected and not cpu_build:
            errors.append(f'{name}: expected {expected}, found {actual or "missing"}')
    if not packages:
        errors.append('Dependency lock is empty.')
    if sys.version_info < (3, 12):
        errors.append('Python 3.12 or later is required by the pinned dependency set.')
    if strict and errors:
        raise ValueError('Runtime unavailable: ' + '; '.join(errors))
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'platform': platform.platform(), 'machine': platform.machine(),
            'packages': packages, 'dependency_errors': errors, 'inference_device': 'cpu'}


def artifacts(p):
    files = ['service.py', 'review.py', 'report_html.py', 'pptx_reader.py', 'model_store.py',
             'validation_binding.py', 'four_class.py', 'profile.json', 'requirements.lock', p['config_file']]
    if p.get('validation_binding_file'):
        files.append(p['validation_binding_file'])
    return {name: review.file_sha(ROOT / name) for name in files}


def validation(p, runtime):
    identity = scoring_identity(ROOT, p, runtime)
    record = {'status': 'experimental_unvalidated', 'applicable_to_current_scoring': False,
              'binding_reasons': ['No evaluation is bound to this exported implementation and runtime.'],
              'scoring_identity': identity, 'metrics': {}, 'historical_metrics': {}}
    # A clean clone has no evaluation result, historical metrics or hidden dependency.
    if not p.get('validation_result_path'):
        return record
    result_path = ROOT / p['validation_result_path']
    result_sha = review.file_sha(result_path)
    if result_sha != p['validation_result_sha256']:
        raise ValueError('Validation record changed.')
    result = review.read_json(result_path)
    reasons = ['No evaluation binding has been recorded.']
    if p.get('validation_binding_file'):
        reasons = check_binding(review.read_json(ROOT / p['validation_binding_file']), result, result_sha, identity)
    record.update(binding_reasons=reasons, applicable_to_current_scoring=not reasons,
                  result_path=str(result_path), result_sha256=result_sha)
    if not reasons:
        record.update(status=p.get('validation_status', 'experimental_unvalidated'), metrics=result.get('metrics', {}))
    else:
        record['historical_metrics'] = result.get('metrics', {})
    return record


def make_engine(p):
    if p['backend'] != 'four_class':
        raise ValueError('Only the pinned four-class backend is included.')
    from four_class import Engine
    return Engine(review.read_json(ROOT / p['config_file']))


def scan(path, out):
    path, out = Path(path).expanduser().resolve(), Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    r = {'schema_version': SCHEMA, 'product': 'scribrism', 'product_version': VERSION,
         'created_at': datetime.now(timezone.utc).isoformat(), 'input': {'path': str(path), 'sha256': None},
         'scan_status': 'failed', 'validation_status': 'experimental_unvalidated',
         'authorship_certified': False, 'submission_approved': False,
         'network_used_for_inference': False, 'model_id': 'unavailable', 'score_semantics': SCORE_SEMANTICS}
    text, started = '', time.monotonic()
    try:
        p = profile()
        r.update(model_id=p['model_id'], model_revision=p['model_revision'],
                 model_manifest_sha256=p['model_manifest_sha256'], artifacts=artifacts(p),
                 decision_config=review.read_json(ROOT / p['config_file']))
        r['input']['sha256'] = review.file_sha(path)
        text, r['extraction_notes'] = extract(path)
        r['text_sha256'] = review.sha_bytes(text.encode())
        if not 80 <= len(text.split()) <= 30000:
            raise ValueError('Outside 80–30,000 word policy; no origin score issued.')
        r['runtime'] = runtime_versions()
        manifest = verify_model(ROOT, p)
        r['validation'] = validation(p, r['runtime'])
        r['validation_status'] = r['validation']['status']
        engine = make_engine(p)
        if engine.model_sha != manifest['files']['model.safetensors']['sha256']:
            raise ValueError('Loaded model identity differs from verified checkpoint.')
        r['result'] = engine.score(text)
        if r['result'].get('token_coverage') != 1.:
            raise ValueError('Scan coverage is incomplete.')
        r['model_sha256'] = engine.model_sha
        verify_model(ROOT, p)
        if (review.file_sha(path) != r['input']['sha256'] or artifacts(p) != r['artifacts'] or
                runtime_versions() != r['runtime']):
            raise ValueError('Input, code, configuration or runtime changed during scanning.')
        r['scan_status'] = 'complete'
        page = render(r, text)  # Rendering failure must not publish a complete result.
    except Exception as exc:
        r.update(scan_status='failed', validation_status='experimental_unvalidated',
                 error=str(exc), error_type=type(exc).__name__)
        r.pop('result', None)
        r.pop('validation', None)
        page = render(r, text)
    r['elapsed_seconds'] = round(time.monotonic() - started, 3)
    review.save_json(out / 'report.json', r)
    (out / 'extracted.txt').write_text(text, encoding='utf-8')
    (out / 'report.html').write_text(page, encoding='utf-8')
    review.save_json(out / 'receipt.json', {name: review.file_sha(out / name)
        for name in ('report.json', 'report.html', 'extracted.txt')})
    return r


def verify(path, report_path):
    reasons = []
    try:
        report_path = Path(report_path).expanduser().resolve()
        r = review.read_json(report_path)
        receipt = review.read_json(report_path.parent / 'receipt.json')
        for name in ('report.json', 'report.html', 'extracted.txt'):
            if review.file_sha(report_path.parent / name) != receipt[name]:
                reasons.append('Saved artifact changed: ' + name)
        if r.get('product') != 'scribrism' or r.get('schema_version') != SCHEMA:
            reasons.append('Unsupported report identity or schema.')
        if r.get('scan_status') != 'complete':
            reasons.append('Scan incomplete.')
        if not reasons:
            p = profile()
            if artifacts(p) != r['artifacts'] or runtime_versions() != r['runtime']:
                reasons.append('Code, configuration or runtime changed.')
            verify_model(ROOT, p)
            if r['model_manifest_sha256'] != p['model_manifest_sha256']:
                reasons.append('Model identity changed.')
            if validation(p, r['runtime']) != r['validation']:
                reasons.append('Validation binding changed.')
            if review.file_sha(path) != r['input']['sha256']:
                reasons.append('Input changed.')
            text, _ = extract(path)
            if review.sha_bytes(text.encode()) != r['text_sha256']:
                reasons.append('Extraction changed.')
            if r.get('result', {}).get('token_coverage') != 1.:
                reasons.append('Coverage incomplete.')
    except Exception as exc:
        reasons.append('Verification unavailable: ' + str(exc))
    return {'fresh_scan': not reasons, 'reasons': reasons,
            'authorship_certified': False, 'submission_approved': False}


def identity():
    p, runtime = profile(), runtime_versions(strict=False)
    return {'product': 'scribrism', 'version': VERSION,
            'scoring_identity': scoring_identity(ROOT, p, runtime),
            'model_cache': str(model_root(ROOT)), 'runtime_ready': not runtime['dependency_errors'],
            'model_assets_verified': False,
            'capabilities': {'commands': ['scan', 'verify', 'identity', 'setup'],
                'formats': ['txt', 'md', 'docx', 'pdf-text', 'pptx-text-and-notes'],
                'inference': 'local CPU; explicit prior model setup required',
                'evaluation': 'cached-score evaluation and development-only research runners',
                'jev': 'not included', 'span_detector': 'not included',
                'remote_server_or_mcp': 'not included', 'trained_accuracy_claim': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('scan'); p.add_argument('file', type=Path); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify'); p.add_argument('file', type=Path); p.add_argument('--report', type=Path, required=True)
    sub.add_parser('identity', help='Inspect code/runtime identity without loading or fetching a model.')
    p = sub.add_parser('setup', help='Explicitly download the pinned public model or verify an existing cache.')
    p.add_argument('--download', action='store_true', required=True)
    args = parser.parse_args()
    try:
        if args.action == 'identity':
            print(json.dumps(identity(), indent=2)); return 0
        if args.action == 'setup':
            from setup import install
            print(json.dumps(install())); return 0
        if args.action == 'scan':
            r = scan(args.file, args.out)
            print(json.dumps({'scan_status': r['scan_status'], 'label': r.get('result', {}).get('label'),
                              'report': str(args.out / 'report.html'), 'error': r.get('error')}))
            return 0 if r['scan_status'] == 'complete' else 2
        r = verify(args.file, args.report); print(json.dumps(r)); return 0 if r['fresh_scan'] else 2
    except Exception as exc:
        print(json.dumps({'error': str(exc), 'error_type': type(exc).__name__})); return 2


if __name__ == '__main__':
    raise SystemExit(main())
