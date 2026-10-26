import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import run_window_comparison as runner


class WindowComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache_env = patch.dict('os.environ', {}, clear=False)
        self.cache_env.start()
        self.addCleanup(self.cache_env.stop)
        import os
        os.environ.pop('SCRIBRISM_MODEL_CACHE', None)
        self.manifest = {"purpose": "development_only", "samples": [], "limitations": ["Synthetic development fixtures."]}
        for identifier, label in [("human-1", "human"), ("ai-1", "ai"), ("mixed-1", "mixed")]:
            path = self.root / (identifier + ".txt")
            path.write_text(identifier + " words " * 90)
            self.manifest["samples"].append({
                "id": identifier, "label": label, "split": "development", "consumed_for_development": True,
                "text_path": path.name, "text_sha256": runner.sha(path), "independence_group": identifier,
                "groups": {"source": "source-" + identifier, "prompt": "prompt-" + identifier,
                           "writer": "writer-" + identifier, "model": None if label == "human" else "fixture-model"},
                "not_applicable_groups": ["model"] if label == "human" else [],
            })
        self.manifest_path = self.root / "samples.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.engine_dir = self.root / "engine"
        self.engine_dir.mkdir()
        self.config = {"window_tokens": 256, "minimum_score": .7, "region_override": True,
                       "region_confidence": .6, "mixed_min_fraction": .1}
        (self.engine_dir / "config-four-class.json").write_text(json.dumps(self.config))
        (self.engine_dir / "requirements.lock").write_text("fixture-package==1.2\n")
        for file in ("four_class.py", "review.py"):
            (self.engine_dir / file).write_text("# fixture\n")
        folder = self.engine_dir / "models" / "revision"
        folder.mkdir(parents=True)
        files = {}
        for name in ("model.safetensors", "tokenizer.json"):
            (folder / name).write_text(name)
            files[name] = {"sha256": runner.sha(folder / name)}
        model_manifest = {"model_id": "fixture-model", "revision": "revision", "files": files}
        (folder / "manifest.local.json").write_text(json.dumps(model_manifest))
        self.profile = {"backend": "four_class", "config_file": "config-four-class.json",
                        "model_id": "fixture-model", "model_revision": "revision",
                        "model_manifest_sha256": runner.sha(folder / "manifest.local.json")}
        (self.engine_dir / "profile.json").write_text(json.dumps(self.profile))
        self.valid_result = {"label": "uncertain", "token_coverage": 1., "tokens": 2,
                             "token_distributions": [[.25, .25, .25, .25], [.25, .25, .25, .25]],
                             "windows": [{"token_start": 0, "token_end": 2, "scores": [.25, .25, .25, .25]}]}
        self.fake = SimpleNamespace(model_sha=files["model.safetensors"]["sha256"], config=None,
                                    score=lambda text: copy.deepcopy(self.valid_result))
        self.module = SimpleNamespace(MODEL_ID="fixture-model", REVISION="revision", Engine=lambda config: self.fake)

    def audit(self, manifest=None):
        return runner.audit_manifest(manifest or self.manifest, self.root)

    def run_fixture(self, windows=None, out_name="out"):
        with patch.object(runner, "load_engine_module", return_value=self.module), \
                patch.object(runner, "package_version", return_value="1.2"), contextlib.redirect_stdout(io.StringIO()):
            return runner.run_comparison(self.manifest_path, self.engine_dir, self.root / out_name, windows or [256, 768])

    def test_dataset_summary_uses_actual_mixed_cases_and_prompt_counts(self):
        summary = self.audit()
        self.assertEqual(summary["class_counts"], {"human": 1, "ai": 1, "mixed": 1})
        self.assertEqual(summary["declared_group_counts"]["prompt"], 3)
        self.assertEqual(summary["declared_group_counts_by_class"]["mixed"]["source"], 1)
        self.assertEqual(summary["manifest_limitations"], self.manifest["limitations"])
        self.assertEqual(summary["declared_independence_groups"], 3)

    def test_unknown_groups_stay_distinct_from_not_applicable(self):
        self.manifest["samples"][0]["groups"]["writer"] = None
        summary = self.audit()
        self.assertEqual(summary["unknown_group_counts"]["writer"], 1)
        self.assertEqual(summary["unknown_group_counts"]["model"], 0)
        self.assertEqual(summary["not_applicable_group_counts"]["model"], 1)

    def test_multisource_and_repeated_group_metadata_is_counted(self):
        self.manifest["samples"][2]["groups"]["source"] = ["source-human-1", "source-ai-1"]
        self.manifest["samples"][2]["independence_group"] = "human-1"
        summary = self.audit()
        self.assertEqual(summary["declared_group_counts"]["source"], 2)
        self.assertEqual(summary["declared_group_counts_by_class"]["mixed"]["source"], 2)
        self.assertEqual(summary["repeated_independence_groups"], {"human-1": 2})
        self.assertEqual(summary["repeated_declared_groups"]["source"]["source-ai-1"], 2)

    def test_empty_or_non_development_manifests_are_rejected(self):
        for manifest in ({}, [], {"purpose": "development_only", "samples": []}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                runner.audit_manifest(manifest, self.root)
        for field, value in (("split", "test"), ("consumed_for_development", 1), ("label", "ai_edited")):
            manifest = copy.deepcopy(self.manifest)
            manifest["samples"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(manifest)

    def test_duplicate_ids_exact_texts_and_conflicting_labels_are_rejected(self):
        for change, message in (("id", "unique"), ("same", "Duplicate exact"), ("conflict", "Conflicting labels")):
            manifest = copy.deepcopy(self.manifest)
            first, second = manifest["samples"][:2]
            if change == "id":
                second["id"] = first["id"]
            else:
                second.update(text_sha256=first["text_sha256"], text_path=first["text_path"])
                if change == "same":
                    second["label"] = first["label"]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.audit(manifest)

    def test_bad_or_changed_sample_hashes_are_rejected(self):
        for value in (None, "wrong", "0" * 64):
            manifest = copy.deepcopy(self.manifest)
            manifest["samples"][0]["text_sha256"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.audit(manifest)

    def test_invalid_and_contradictory_group_metadata_is_rejected(self):
        for groups, na in (({"writer": ["twice", "twice"]}, []), ({"writer": "known"}, ["writer"]),
                           ({"unexpected": "x"}, []), ({"writer": [""]}, []), ({}, ["unknown-field"])):
            manifest = copy.deepcopy(self.manifest)
            manifest["samples"][0].update(groups=groups, not_applicable_groups=na)
            with self.subTest(groups=groups, na=na), self.assertRaises(ValueError):
                self.audit(manifest)

    def test_paths_must_be_relative_and_stay_in_manifest_directory(self):
        for value in (None, str(self.root / "human-1.txt"), "../escape.txt"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.text_path(self.root, value)

    def test_invalid_or_duplicate_windows_fail_before_loading_model(self):
        with patch.object(runner, "load_engine_module") as load:
            for windows in ([1], [769], [256, 256], [True]):
                with self.subTest(windows=windows), self.assertRaises(ValueError):
                    runner.run_comparison(self.manifest_path, self.engine_dir, self.root / "out", windows)
            load.assert_not_called()

    def test_runtime_records_actual_versions_and_rejects_lock_mismatch(self):
        with patch.object(runner, "package_version", return_value="1.2"):
            actual = runner.runtime_fingerprint(self.engine_dir)
        self.assertEqual(actual["runtime"]["packages"], {"fixture-package": "1.2"})
        self.assertEqual(actual["runtime_sha256"], runner.canonical_sha(actual["runtime"]))
        self.assertEqual(actual["requirements_lock_sha256"], runner.sha(self.engine_dir / "requirements.lock"))
        with patch.object(runner, "package_version", return_value="2.0"), self.assertRaisesRegex(ValueError, "Dependency"):
            runner.runtime_fingerprint(self.engine_dir)

    def test_model_manifest_profile_and_weight_pins_are_enforced(self):
        actual = runner.verify_model_pin(self.engine_dir, self.module)
        self.assertTrue(actual["model_pin_verified"])
        self.assertEqual(actual["model_files_verified_before_loading"], 2)
        bad_module = SimpleNamespace(MODEL_ID="different", REVISION="revision")
        with self.assertRaisesRegex(ValueError, "identity"):
            runner.verify_model_pin(self.engine_dir, bad_module)
        weights = self.engine_dir / "models/revision/model.safetensors"
        weights.write_text("changed")
        with self.assertRaisesRegex(ValueError, "tokenizer changed"):
            runner.verify_model_pin(self.engine_dir, self.module)
        self.profile["model_manifest_sha256"] = "0" * 64
        (self.engine_dir / "profile.json").write_text(json.dumps(self.profile))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            runner.verify_model_pin(self.engine_dir, self.module)

    def test_receipt_and_score_fingerprints_bind_runtime_runner_and_configs(self):
        receipt = self.run_fixture()
        self.assertEqual(receipt["dataset_summary"]["class_counts"]["mixed"], 1)
        self.assertFalse(receipt["promotable"])
        self.assertNotIn("no mixed", receipt["no_promotion"])
        self.assertNotIn("same AI task", receipt["no_promotion"])
        self.assertTrue(receipt["source_config_unchanged"])
        self.assertTrue(receipt["model_pin_verified"])
        self.assertEqual(receipt["runner_sha256"], runner.sha(runner.__file__))
        self.assertEqual(receipt["result_validator_sha256"], runner.sha(runner.evaluator.__file__))
        for size in (256, 768):
            records = [json.loads(line) for line in (self.root / f"out/scores-{size}.jsonl").read_text().splitlines()]
            fingerprints = [record["inference_fingerprint"] for record in records]
            self.assertTrue(all(value == fingerprints[0] for value in fingerprints))
            self.assertEqual(fingerprints[0]["runtime_sha256"], receipt["runtime_sha256"])
            self.assertEqual(fingerprints[0]["effective_config_sha256"], runner.sha(self.root / f"out/config-{size}.json"))
            effective = json.loads((self.root / f"out/config-{size}.json").read_text())
            self.assertEqual(effective, dict(self.config, window_tokens=size))
        self.assertEqual(json.loads((self.engine_dir / "config-four-class.json").read_text()), self.config)

    def test_failed_attempts_remain_in_output_and_receipt_denominator(self):
        def score(text):
            if text.startswith("mixed"):
                raise ValueError("fixture failure")
            return dict(self.valid_result, label="human")
        self.fake.score = score
        receipt = self.run_fixture([256])
        self.assertEqual(receipt["attempts_by_window"]["256"], {"complete": 2, "error": 1})
        rows = (self.root / "out/scores-256.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(json.loads(rows[-1])["status"], "error")

    def test_config_change_prevents_successful_receipt(self):
        def score(text):
            (self.engine_dir / "config-four-class.json").write_text("{}")
            return dict(self.valid_result, label="human")
        self.fake.score = score
        with self.assertRaisesRegex(RuntimeError, "Protected configuration"):
            self.run_fixture([256])
        self.assertFalse((self.root / "out/run-receipt.json").exists())

    def test_loaded_model_identity_must_match_verified_weights(self):
        self.fake.model_sha = "different"
        with self.assertRaisesRegex(ValueError, "Loaded model"):
            self.run_fixture()
        self.assertFalse((self.root / "out").exists())

    def save_receipt(self, receipt):
        path = self.root / "out/run-receipt.json"
        path.write_text(json.dumps(receipt))
        return path

    def test_future_receipt_binds_output_bytes_and_verifies_without_model(self):
        receipt = self.run_fixture()
        self.assertEqual(receipt["receipt_schema_version"], 2)
        for size in (256, 768):
            metadata = receipt["outputs_by_window"][str(size)]
            for kind, name in (("config", f"config-{size}.json"), ("scores", f"scores-{size}.jsonl")):
                path = self.root / "out" / name
                self.assertEqual(metadata[kind], {"sha256": runner.sha(path), "bytes": path.stat().st_size})
        with patch.object(runner, "load_engine_module") as load:
            verification = runner.verify_receipt(self.root / "out/run-receipt.json")
            load.assert_not_called()
        self.assertTrue(verification["verified"])
        self.assertEqual(verification["samples_per_window"], 3)
        self.assertEqual(verification["verified_output_files"], 4)
        self.assertEqual(verification["receipt_sha256"], runner.sha(self.root / "out/run-receipt.json"))

    def test_modified_config_scores_and_truncated_scores_fail_integrity_check(self):
        self.run_fixture([256])
        for name, transform in (("config-256.json", lambda raw: raw.replace(b"0.7", b"0.8")),
                                ("scores-256.jsonl", lambda raw: raw.replace(b"uncertain", b"human", 1)),
                                ("scores-256.jsonl", lambda raw: raw.splitlines(keepends=True)[0])):
            path = self.root / "out" / name
            original = path.read_bytes()
            changed = transform(original)
            self.assertNotEqual(changed, original)
            path.write_bytes(changed)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash or byte count mismatch"):
                runner.verify_receipt(self.root / "out/run-receipt.json")
            path.write_bytes(original)

    def test_missing_outputs_and_legacy_receipts_do_not_verify(self):
        receipt = self.run_fixture([256])
        legacy = copy.deepcopy(receipt)
        del legacy["receipt_schema_version"]
        with self.assertRaisesRegex(ValueError, "preserve historical"):
            runner.verify_receipt(self.save_receipt(legacy))
        self.save_receipt(receipt)
        (self.root / "out/scores-256.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            runner.verify_receipt(self.root / "out/run-receipt.json")

    def test_receipt_sample_and_status_counts_are_checked_against_outputs(self):
        receipt = self.run_fixture([256])
        for field in ("samples", "attempts_by_window"):
            altered = copy.deepcopy(receipt)
            if field == "samples":
                altered[field] = 4
            else:
                altered[field]["256"] = {"complete": 2, "error": 1}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "count mismatch"):
                runner.verify_receipt(self.save_receipt(altered))

    def test_rehashed_truncation_still_fails_record_count_check(self):
        receipt = self.run_fixture([256])
        path = self.root / "out/scores-256.jsonl"
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))
        receipt["outputs_by_window"]["256"]["scores"] = runner.output_metadata(path)
        with self.assertRaisesRegex(ValueError, "record or status count mismatch"):
            runner.verify_receipt(self.save_receipt(receipt))

    def test_rehashed_config_still_fails_score_fingerprint_check(self):
        receipt = self.run_fixture([256])
        path = self.root / "out/config-256.json"
        changed = json.loads(path.read_text())
        changed["minimum_score"] = .9
        path.write_text(json.dumps(changed))
        receipt["outputs_by_window"]["256"]["config"] = runner.output_metadata(path)
        with self.assertRaisesRegex(ValueError, "Score fingerprint mismatch"):
            runner.verify_receipt(self.save_receipt(receipt))

    def test_rehashed_duplicate_records_and_changed_fingerprints_are_rejected(self):
        receipt = self.run_fixture([256])
        path = self.root / "out/scores-256.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for change in ("duplicate", "fingerprint"):
            records = copy.deepcopy(original)
            if change == "duplicate":
                records[1]["id"] = records[0]["id"]
            else:
                records[1]["inference_fingerprint"]["engine_sha256"] = "0" * 64
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            receipt["outputs_by_window"]["256"]["scores"] = runner.output_metadata(path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.verify_receipt(self.save_receipt(receipt))

    def test_outputs_changed_during_later_window_prevent_completion_receipt(self):
        def score(text):
            if self.fake.config["window_tokens"] == 768:
                (self.root / "out/scores-256.jsonl").write_text("")
            return dict(self.valid_result, label="human")
        self.fake.score = score
        with self.assertRaisesRegex(ValueError, "hash or byte count mismatch"):
            self.run_fixture()
        self.assertFalse((self.root / "out/run-receipt.json").exists())

    def test_verification_cli_requires_no_engine_and_rejects_tampering(self):
        self.run_fixture([256])
        command = [sys.executable, runner.__file__, "--verify-receipt", str(self.root / "out/run-receipt.json")]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["verified"])
        (self.root / "out/scores-256.jsonl").write_text("")
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("hash or byte count mismatch", failed.stderr)

    def test_invalid_or_unserializable_engine_results_retain_error_attempts(self):
        cases = [
            ("nan", {"token_distributions": [[float("nan"), 0, 0, 1]]}, "ValueError"),
            ("infinity", {"token_distributions": [[float("inf"), 0, 0, 1]]}, "ValueError"),
            ("vector-width", {"token_distributions": [[1, 0, 0]]}, "ValueError"),
            ("vector-sum", {"token_distributions": [[.2, .2, .2, .2]]}, "ValueError"),
            ("vector-range", {"token_distributions": [[-.1, 1.1, 0, 0]]}, "ValueError"),
            ("coverage", {"tokens": 3, "windows": [{"token_start": 0, "token_end": 2,
                                                       "scores": [1, 0, 0, 0]}]}, "ValueError"),
            ("range", {"tokens": 2, "windows": [{"token_start": 0, "token_end": 3,
                                                    "scores": [1, 0, 0, 0]}]}, "ValueError"),
            ("dual-coverage", dict(self.valid_result, windows=[{"token_start": 0, "token_end": 1,
                                                                "scores": [.25, .25, .25, .25]}]), "ValueError"),
            ("dual-conflict", dict(self.valid_result, token_distributions=[[1, 0, 0, 0]] * 2,
                                  windows=[{"token_start": 0, "token_end": 2,
                                            "scores": [0, 1, 0, 0]}]), "ValueError"),
            ("dual-length", dict(self.valid_result, token_distributions=[[.25, .25, .25, .25]]), "ValueError"),
            ("dual-null-windows", dict(self.valid_result, windows=None), "ValueError"),
            ("dual-empty-windows", dict(self.valid_result, windows=[]), "ValueError"),
            ("dual-malformed-window", dict(self.valid_result, windows=[None]), "ValueError"),
            ("token-count", dict(self.valid_result, tokens=3), "ValueError"),
            ("unserializable", dict(self.valid_result, diagnostic={1, 2}), "TypeError"),
            ("nonfinite-extra", dict(self.valid_result, diagnostic=float("nan")), "ValueError"),
        ]
        for name, invalid, error_type in cases:
            with self.subTest(case=name):
                self.fake.score = lambda text: invalid if text.startswith("mixed") else copy.deepcopy(self.valid_result)
                receipt = self.run_fixture([256], out_name=name)
                self.assertEqual(receipt["attempts_by_window"]["256"], {"complete": 2, "error": 1})
                records = [json.loads(line) for line in (self.root / name / "scores-256.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 3)
                self.assertEqual([record["id"] for record in records], [row["id"] for row in self.manifest["samples"]])
                self.assertEqual(records[-1]["status"], "error")
                self.assertTrue(records[-1]["error"].startswith(error_type + ":"))
                self.assertNotIn("result", records[-1])
                self.assertEqual(records[0]["result"], self.valid_result)
                self.assertTrue(runner.verify_receipt(self.root / name / "run-receipt.json")["verified"])

    def test_valid_token_window_and_dual_results_preserve_exact_payload(self):
        overlap = dict(self.valid_result, tokens=3, token_distributions=[
            [.1, .2, .3, .4], [.3 + 2e-13, .3 - 2e-13, .2, .2], [.5, .4, .1, 0]], windows=[
                {"token_start": 0, "token_end": 2, "scores": [.1, .2, .3, .4]},
                {"token_start": 1, "token_end": 3, "scores": [.5, .4, .1, 0]}])
        for name, result in (("tokens", {key: value for key, value in self.valid_result.items() if key != "windows"}),
                             ("windows", {key: value for key, value in self.valid_result.items() if key != "token_distributions"}),
                             ("both", self.valid_result), ("overlap-roundoff", overlap)):
            with self.subTest(representation=name):
                self.fake.score = lambda text: copy.deepcopy(result)
                receipt = self.run_fixture([256], out_name=name)
                self.assertEqual(receipt["attempts_by_window"]["256"], {"complete": 3, "error": 0})
                records = [json.loads(line) for line in (self.root / name / "scores-256.jsonl").read_text().splitlines()]
                self.assertTrue(all(record["result"] == result for record in records))
                self.assertTrue(all(json.dumps(record["result"], sort_keys=True) == json.dumps(result, sort_keys=True)
                                    for record in records))
                self.assertTrue(all(record["result"]["label"] == "uncertain" for record in records))

    def test_output_write_and_flush_failures_abort_without_receipt(self):
        original_open = Path.open
        for operation in ("write", "flush"):
            class FailedOutput:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    self.stream.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def write(self, value):
                    if operation == "write":
                        raise OSError("injected disk write failure")
                    return self.stream.write(value)

                def flush(self):
                    raise OSError("injected disk flush failure")

            def open_with_failure(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                if path.name == "scores-256.jsonl" and args == ("w",):
                    return FailedOutput(stream)
                return stream

            with self.subTest(operation=operation), patch.object(Path, "open", open_with_failure):
                with self.assertRaisesRegex(OSError, "injected disk"):
                    self.run_fixture([256], out_name=operation)
            self.assertFalse((self.root / operation / "run-receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
