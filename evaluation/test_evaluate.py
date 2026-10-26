import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import evaluate


CONFIG = {"minimum_score": .7, "region_override": True, "region_confidence": .6, "mixed_min_fraction": .1}


def row(identifier, truth="human", split="development", prediction="human"):
    return {"id": identifier, "label": truth, "split": split, "prediction": prediction,
            "text_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            "groups": {field: field + identifier for field in evaluate.GROUP_FIELDS},
            "independence_group": identifier,
            "inference_fingerprint": {"synthetic_fixture": True}, "synthetic_fixture": True,
            "result": {"token_distributions": [[1, 0, 0, 0]]}}


class EvaluationTests(unittest.TestCase):
    def test_denominators_include_abstentions_and_errors(self):
        rows = [row("h1"), row("h2", prediction="ai"), row("h3", prediction="uncertain"),
                row("h4", prediction="error"), row("a1", "ai", prediction="ai"),
                row("m1", "mixed", prediction="mixed"), row("m2", "mixed", prediction="human")]
        result = evaluate.metrics(rows)
        self.assertEqual(result["overall_accuracy"]["rate"], 3 / 7)
        self.assertEqual(result["human_false_positive_rate"]["rate"], 1 / 4)
        self.assertEqual(result["mixed_recall"]["rate"], 1 / 2)
        self.assertEqual(result["coverage"]["rate"], 5 / 7)
        self.assertEqual(result["selective_accuracy"]["rate"], 3 / 5)
        self.assertEqual(result["abstentions"], 1)
        self.assertEqual(result["operational_failures"], 1)

    def test_undefined_class_rate_is_null_not_zero(self):
        result = evaluate.metrics([row("h1", prediction="uncertain")])
        self.assertIsNone(result["mixed_recall"]["rate"])
        self.assertIsNone(result["selective_accuracy"]["rate"])

    def test_every_declared_group_prevents_cross_split_reuse(self):
        for field in evaluate.GROUP_FIELDS:
            with self.subTest(field=field):
                a, b = row("a"), row("b", split="test")
                b["groups"][field] = a["groups"][field]
                self.assertTrue(evaluate.validate_records([a, b])["cross_split_overlaps"])

    def test_exact_duplicate_across_split_is_caught(self):
        a, b = row("a"), row("b", split="test")
        b["text_sha256"] = a["text_sha256"]
        self.assertEqual(evaluate.validate_records([a, b])["cross_split_overlaps"][0]["field"], "text_sha256")

    def test_unknown_group_is_distinct_from_not_applicable(self):
        a = row("a")
        a["groups"]["model"] = None
        self.assertTrue(evaluate.validate_records([a])["unknown_group_metadata"])
        a["not_applicable_groups"] = ["model"]
        self.assertFalse(evaluate.validate_records([a])["unknown_group_metadata"])

    def test_four_classes_and_spatial_mixture(self):
        for values, expected in [([1, 0, 0, 0], "human"), ([0, 1, 0, 0], "ai"),
                                 ([0, 0, 1, 0], "mixed"), ([0, 0, 0, 1], "ai")]:
            self.assertEqual(evaluate.classify({"token_distributions": [values]}, CONFIG), expected)
        result = {"token_distributions": [[1, 0, 0, 0], [0, 1, 0, 0]]}
        self.assertEqual(evaluate.classify(result, CONFIG), "mixed")
        self.assertEqual(evaluate.classify(result, dict(CONFIG, region_override=False)), "uncertain")

    def test_windows_are_overlap_averaged_and_require_full_coverage(self):
        result = {"tokens": 3, "windows": [
            {"token_start": 0, "token_end": 2, "scores": [1, 0, 0, 0]},
            {"token_start": 1, "token_end": 3, "scores": [0, 1, 0, 0]}]}
        self.assertEqual(evaluate.token_scores(result), [[1, 0, 0, 0], [.5, .5, 0, 0], [0, 1, 0, 0]])
        result["tokens"] = 4
        with self.assertRaises(ValueError):
            evaluate.token_scores(result)

    def test_dual_representations_cannot_disagree_even_when_each_is_valid(self):
        tokens = {"token_distributions": [[1, 0, 0, 0], [1, 0, 0, 0]]}
        windows = {"tokens": 2, "windows": [
            {"token_start": 0, "token_end": 2, "scores": [0, 1, 0, 0]}]}
        self.assertEqual(evaluate.classify(tokens, CONFIG), "human")
        self.assertEqual(evaluate.classify(windows, CONFIG), "ai")
        with self.assertRaisesRegex(ValueError, "disagree"):
            evaluate.classify({**tokens, **windows}, CONFIG)

        # Equal document means cannot hide a contradiction at individual tokens.
        permuted = {"tokens": 2, "token_distributions": [[0, 1, 0, 0], [1, 0, 0, 0]],
                    "windows": [{"token_start": 0, "token_end": 1, "scores": [1, 0, 0, 0]},
                                {"token_start": 1, "token_end": 2, "scores": [0, 1, 0, 0]}]}
        with self.assertRaisesRegex(ValueError, "disagree"):
            evaluate.token_scores(permuted)

    def test_dual_distribution_lengths_must_match(self):
        for length in (1, 3):
            result = {"tokens": 2, "token_distributions": [[1, 0, 0, 0]] * length,
                      "windows": [{"token_start": 0, "token_end": 2, "scores": [1, 0, 0, 0]}]}
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "lengths differ"):
                evaluate.token_scores(result)

    def test_valid_tokens_cannot_hide_invalid_present_windows(self):
        valid_window = {"token_start": 0, "token_end": 2, "scores": [1, 0, 0, 0]}
        for windows in (None, [], {}, "windows", [None], [{}],
                        [dict(valid_window, token_end=1)],  # Incomplete coverage.
                        [dict(valid_window, token_end=3)],
                        [dict(valid_window, token_start=False)],
                        [dict(valid_window, token_end=2.0)],
                        [dict(valid_window, scores=None)],
                        [dict(valid_window, scores={"human": 1})],
                        [dict(valid_window, scores=[True, 0, 0, 0])],
                        [dict(valid_window, scores=[float("inf"), 0, 0, 0])],
                        [dict(valid_window, scores=[float("nan"), 0, 0, 1])]):
            result = {"tokens": 2, "token_distributions": [[1, 0, 0, 0]] * 2, "windows": windows}
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                evaluate.token_scores(result)
        for size in (None, False, 0, -1, 2.0):
            result = {"tokens": size, "token_distributions": [[1, 0, 0, 0]] * 2,
                      "windows": [valid_window]}
            with self.subTest(size=size), self.assertRaises(ValueError):
                evaluate.token_scores(result)

    def test_dual_overlap_allows_only_tiny_roundoff_and_preserves_stored_values(self):
        # Decimal addition and direct decimal values need not have identical bits.
        result = {"tokens": 3, "windows": [
            {"token_start": 0, "token_end": 2, "scores": [.1, .2, .3, .4]},
            {"token_start": 1, "token_end": 3,
             "scores": dict(zip(evaluate.FOUR, [.5, .4, .1, 0]))}],
            "token_distributions": [[.1, .2, .3, .4], [.3 + 2e-13, .3 - 2e-13, .2, .2],
                                    [.5, .4, .1, 0]]}
        original = json.dumps(result, sort_keys=True)
        stored = result["token_distributions"]
        returned = evaluate.token_scores(result)
        self.assertEqual(returned, stored)
        self.assertTrue(all(a is b for a, b in zip(returned, stored)))
        self.assertEqual(json.dumps(result, sort_keys=True), original)
        reconstructed = evaluate.token_scores({"tokens": 3, "windows": result["windows"]})
        self.assertNotEqual(returned[1], reconstructed[1])

        for component in range(4):
            excessive = copy.deepcopy(result)
            excessive["token_distributions"][1] = list(reconstructed[1])
            excessive["token_distributions"][1][component] += 2e-12
            excessive["token_distributions"][1][(component + 1) % 4] -= 2e-12
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, "disagree"):
                evaluate.token_scores(excessive)

    def test_consistency_absolute_tolerance_boundary(self):
        tolerance = evaluate.TOKEN_WINDOW_ABS_TOLERANCE
        for delta, accepted in ((tolerance, True), (math.nextafter(tolerance, math.inf), False)):
            result = {"tokens": 1, "windows": [
                {"token_start": 0, "token_end": 1, "scores": [.5, 0, 0, .5]}],
                "token_distributions": [[.5 - delta, delta, 0, .5]]}
            with self.subTest(delta=delta):
                if accepted:
                    self.assertEqual(evaluate.token_scores(result), result["token_distributions"])
                else:
                    with self.assertRaisesRegex(ValueError, "disagree"):
                        evaluate.token_scores(result)

    def test_roundoff_does_not_replace_scores_at_classification_boundary(self):
        below = math.nextafter(.7, 0)
        result = {"tokens": 1, "windows": [
            {"token_start": 0, "token_end": 1, "scores": [.7, .1, .1, .1]}],
            "token_distributions": [[below, .1 + (.7 - below), .1, .1]]}
        config = dict(CONFIG, region_override=False)
        self.assertEqual(evaluate.classify({"tokens": 1, "windows": result["windows"]}, config), "human")
        self.assertEqual(evaluate.classify(result, config), "uncertain")
        self.assertEqual(evaluate.token_scores(result), result["token_distributions"])

    def test_bad_probability_vectors_are_rejected(self):
        for vector in ([float("nan"), 0, 0, 1], [1, 1, 0, 0], [-.1, 1.1, 0, 0]):
            with self.assertRaises(ValueError):
                evaluate.probability_vector(vector)

    def test_consumed_or_synthetic_data_cannot_be_heldout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            config.write_text(json.dumps(CONFIG))
            a = row("a", split="test")
            data = root / "data.jsonl"
            data.write_text(json.dumps(a) + "\n")
            with self.assertRaisesRegex(ValueError, "synthetic"):
                evaluate.evaluate(data, config, "test", "heldout", 20)
            a.pop("synthetic_fixture")
            a["consumed_for_development"] = True
            data.write_text(json.dumps(a) + "\n")
            with self.assertRaisesRegex(ValueError, "Consumed"):
                evaluate.evaluate(data, config, "test", "heldout", 20)

    def test_fingerprints_bind_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            config.write_text(json.dumps(CONFIG))
            data = root / "data.jsonl"
            data.write_text(json.dumps(row("a")) + "\n")
            result = evaluate.evaluate(data, config, bootstrap=20)
            self.assertEqual(result["data_sha256"], evaluate.sha(data))
            self.assertEqual(result["decision_config_sha256"], evaluate.sha(config))
            self.assertFalse(result["promotable"])


if __name__ == "__main__":
    unittest.main()
