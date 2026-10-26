import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import four_class
import model_store
import review
import service
import setup


class PortableServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'clone with spaces'
        self.root.mkdir()
        for name in ['service.py', 'review.py', 'report_html.py', 'pptx_reader.py', 'four_class.py',
                     'model_store.py', 'validation_binding.py', 'requirements.lock', 'profile.json',
                     'config-four-class.json', 'scribrism', 'setup.py']:
            shutil.copyfile(service.ROOT / name, self.root / name)
        os.chmod(self.root / 'scribrism', 0o755)
        p = patch.object(service, 'ROOT', self.root); p.start(); self.addCleanup(p.stop)
        p = patch.dict(os.environ, {}, clear=False); p.start(); self.addCleanup(p.stop)
        os.environ.pop('SCRIBRISM_MODEL_CACHE', None)
        self.doc = self.root / 'sample.txt'
        self.doc.write_text('Synthetic fixture for software checks. ' * 30)
        self.out = self.root / 'report'

    def fake_assets(self):
        p = service.profile()
        folder = self.root / 'models' / p['model_revision']
        folder.mkdir(parents=True)
        (folder / 'model.safetensors').write_bytes(b'not real model weights')
        manifest = {'model_id': p['model_id'], 'revision': p['model_revision'], 'files': {
            'model.safetensors': {'sha256': review.file_sha(folder / 'model.safetensors'),
                                 'bytes': (folder / 'model.safetensors').stat().st_size}}}
        (folder / 'manifest.local.json').write_text(json.dumps(manifest))
        p['model_manifest_sha256'] = review.file_sha(folder / 'manifest.local.json')
        (self.root / 'profile.json').write_text(json.dumps(p))
        return manifest['files']['model.safetensors']['sha256']

    def fake_scan(self, effect=None):
        digest = self.fake_assets()
        def score(text):
            if effect:
                effect()
            return {'label': 'uncertain', 'token_coverage': 1., 'windows': []}
        fake = types.SimpleNamespace(model_sha=digest, score=score)
        with patch.object(service, 'make_engine', return_value=fake):
            return service.scan(self.doc, self.out)

    def test_help_and_identity_work_from_clean_clone_without_weights(self):
        env = dict(os.environ, SCRIBRISM_PYTHON=sys.executable)
        for args in (['--help'], ['identity']):
            completed = subprocess.run([str(self.root / 'scribrism'), *args],
                cwd=self.root.parent, env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            if args == ['identity']:
                payload = json.loads(completed.stdout)
                self.assertFalse(payload['model_assets_verified'])
                self.assertFalse(payload['capabilities']['trained_accuracy_claim'])
                self.assertEqual(payload['capabilities']['jev'], 'not included')
        self.assertFalse((self.root / 'models').exists())

    def test_identity_reports_missing_dependencies_without_loading_engine(self):
        with patch.object(service, 'package_version', side_effect=service.PackageNotFoundError), \
                patch.object(service, 'make_engine', side_effect=AssertionError('No model')):
            result = service.identity()
        self.assertFalse(result['runtime_ready'])
        self.assertTrue(result['scoring_identity']['payload']['runtime']['dependency_errors'])

    def test_known_linux_cpu_build_is_recorded_not_normalized_away(self):
        actual = service.package_version
        with patch.object(service, 'package_version', side_effect=lambda name: '2.8.0+cpu' if name == 'torch' else actual(name)):
            runtime = service.runtime_versions()
        self.assertEqual(runtime['packages']['torch'], '2.8.0+cpu')
        with patch.object(service, 'package_version', return_value='unknown-build'):
            with self.assertRaisesRegex(ValueError, 'Runtime unavailable'):
                service.runtime_versions()

    def test_missing_assets_retain_failed_receipts_without_engine_or_download(self):
        with patch.object(service, 'make_engine', side_effect=AssertionError('No model')) as engine:
            r = service.scan(self.doc, self.out)
        engine.assert_not_called()
        self.assertEqual(r['scan_status'], 'failed')
        self.assertNotIn('result', r)
        for name, expected in review.read_json(self.out / 'receipt.json').items():
            self.assertEqual(review.file_sha(self.out / name), expected)
        self.assertFalse(service.verify(self.doc, self.out / 'report.json')['fresh_scan'])

    def test_fake_ready_scan_verifies_as_unvalidated_without_private_results(self):
        r = self.fake_scan()
        self.assertEqual(r['scan_status'], 'complete')
        self.assertEqual(r['validation_status'], 'experimental_unvalidated')
        self.assertEqual(r['validation']['historical_metrics'], {})
        with patch.object(service, 'make_engine', side_effect=AssertionError('Verification does not load model')):
            verified = service.verify(self.doc, self.out / 'report.json')
        self.assertTrue(verified['fresh_scan'], verified)
        self.assertFalse(verified['submission_approved'])
        self.assertFalse(verified['authorship_certified'])

    def test_html_tampering_is_detected(self):
        self.fake_scan()
        (self.out / 'report.html').write_text('changed')
        verified = service.verify(self.doc, self.out / 'report.json')
        self.assertFalse(verified['fresh_scan'])
        self.assertIn('Saved artifact changed: report.html', verified['reasons'])

    def test_input_deleted_during_scan_retains_failure_without_stale_score(self):
        r = self.fake_scan(self.doc.unlink)
        self.assertEqual(r['scan_status'], 'failed')
        self.assertNotIn('result', r)
        self.assertTrue((self.out / 'receipt.json').exists())

    def test_missing_input_produces_failed_report(self):
        self.doc.unlink()
        r = service.scan(self.doc, self.out)
        self.assertEqual(r['scan_status'], 'failed')
        self.assertTrue((self.out / 'receipt.json').exists())

    def test_missing_or_malformed_report_returns_structured_failure(self):
        self.assertFalse(service.verify(self.doc, self.root / 'missing.json')['fresh_scan'])
        bad = self.root / 'bad.json'; bad.write_text('[]')
        self.assertFalse(service.verify(self.doc, bad)['fresh_scan'])

    def test_existing_cache_is_verified_without_downloading(self):
        self.fake_assets()
        cache = self.root / 'models'
        os.environ['SCRIBRISM_MODEL_CACHE'] = str(cache)
        with patch.object(setup, 'ROOT', self.root), patch('huggingface_hub.snapshot_download', side_effect=AssertionError('No download')):
            result = setup.install()
        self.assertEqual(result['status'], 'verified_existing')
        self.assertFalse(result['network_used'])

    def test_download_refuses_symlink_cache_without_a_valid_manifest(self):
        outside = self.root / 'shared'; outside.mkdir()
        link = self.root / 'models'; link.symlink_to(outside, target_is_directory=True)
        with patch.object(setup, 'ROOT', self.root), patch('huggingface_hub.snapshot_download', side_effect=AssertionError('No download')):
            with self.assertRaisesRegex(ValueError, 'symlink'):
                setup.install()

    def test_setup_requires_explicit_download_flag(self):
        result = subprocess.run([sys.executable, str(self.root / 'service.py'), 'setup'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / 'models').exists())

    def test_no_personal_absolute_paths_or_legacy_services_in_exported_core(self):
        forbidden = ('/Users/', '/home/')
        for name in ('service.py', 'review.py', 'four_class.py', 'model_store.py', 'profile.json', 'report_html.py'):
            content = (self.root / name).read_text()
            for token in forbidden:
                self.assertNotIn(token, content, name)
        self.assertNotIn('validation_result_path', service.profile())


class ScoringAndExtractionTests(unittest.TestCase):
    def test_known_label_and_region_policy_is_preserved(self):
        cfg = {'minimum_score': .7, 'region_override': True, 'region_confidence': .6, 'mixed_min_fraction': .1}
        h, a, edited, para = [1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]
        cases = [([h], 'human'), ([a], 'ai'), ([edited], 'mixed'), ([para], 'ai'),
                 ([[.25]*4], 'uncertain'), ([h]*9+[a], 'mixed'), ([h]*99+[a], 'human')]
        for vectors, expected in cases:
            self.assertEqual(four_class.classify(vectors, cfg)['label'], expected)

    def test_txt_and_docx_extraction_preserve_authored_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); txt = root / 'a.txt'; txt.write_text('First paragraph.\n\nSecond paragraph.')
            self.assertEqual(review.extract(txt)[0], txt.read_text())
            doc = root / 'a.docx'
            xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>one</w:t><w:tab/><w:t>two</w:t></w:r></w:p></w:body></w:document>'
            with ZipFile(doc, 'w') as z:
                z.writestr('word/document.xml', xml)
            self.assertEqual(review.extract(doc)[0], 'one\ttwo')

    def test_html_escapes_document_markup(self):
        r = {'input': {'path': 'test.txt', 'sha256': 'fixture'}, 'scan_status': 'failed',
             'validation_status': 'experimental_unvalidated', 'created_at': 'fixture', 'model_id': 'fixture'}
        rendered = service.render(r, '<script>alert(1)</script>')
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)


if __name__ == '__main__':
    unittest.main()
