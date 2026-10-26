#!/usr/bin/env python3
"""Evaluate labeled cached scores without loading a model or contacting a service."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

LABELS = ("human", "ai", "mixed")
PREDICTIONS = LABELS + ("uncertain", "error")
FOUR = ("human", "ai", "ai_edited", "humanized")
GROUP_FIELDS = ("source", "prompt", "writer", "model")
# Both representations use Python float overlap sums/division. Allow only tiny
# absolute roundoff; this never rounds or replaces stored token probabilities.
TOKEN_WINDOW_ABS_TOLERANCE = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def validate_records(rows):
    """Check labels, identity, exact duplicates and declared group separation."""
    seen_ids, texts, owners = set(), {}, defaultdict(dict)
    overlaps, unknowns = [], []
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ValueError("Every record needs a unique nonempty id")
        seen_ids.add(identifier)
        if row.get("label") not in LABELS:
            raise ValueError(f"{identifier}: unsupported truth label")
        split = row.get("split")
        if split not in ("train", "development", "calibration", "test"):
            raise ValueError(f"{identifier}: split must be explicit")
        digest = row.get("text_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{identifier}: exact extracted-text SHA-256 is required")
        if digest in texts:
            other = texts[digest]
            if other[0] != row["label"]:
                raise ValueError(f"Conflicting labels for identical text: {identifier} and {other[1]}")
            if other[2] != split:
                overlaps.append({"field": "text_sha256", "value": digest, "records": [other[1], identifier]})
        else:
            texts[digest] = (row["label"], identifier, split)
        groups = row.get("groups", {})
        for field in GROUP_FIELDS:
            value = groups.get(field)
            if value is None or value == []:
                if field not in row.get("not_applicable_groups", []):
                    unknowns.append({"record": identifier, "field": field, "split": split})
                continue
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{identifier}: invalid {field} group")
            for group in values:
                previous = owners[field].get(group)
                if previous and previous[0] != split:
                    overlaps.append({"field": field, "value": group, "records": [previous[1], identifier],
                                     "splits": [previous[0], split]})
                owners[field][group] = (split, identifier)
    return {"cross_split_overlaps": overlaps, "unknown_group_metadata": unknowns,
            "near_duplicate_check": "not_performed; hashes detect exact duplicates only"}


def probability_vector(values):
    if isinstance(values, dict):
        if any(k not in values for k in FOUR):
            raise ValueError("Invalid four-class probability vector")
        values = [values[k] for k in FOUR]
    if not isinstance(values, (list, tuple)) or len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-5:
        raise ValueError("Invalid four-class probability vector")
    return values


def _window_token_scores(result):
    windows = result["windows"]
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("Windows must be a nonempty list of complete distributions")
    size = result.get("tokens")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("Window records need an explicit total token count")
    sums = [[0.0] * 4 for _ in range(size)]
    visits = [0] * size
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("Every window must be an object")
        start, end = window.get("token_start"), window.get("token_end")
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in (start, end)) or not 0 <= start < end <= size:
            raise ValueError("Invalid window range")
        values = probability_vector(window.get("scores"))
        for i in range(start, end):
            visits[i] += 1
            for k in range(4):
                sums[i][k] += values[k]
    if not all(visits):
        raise ValueError("Cached windows do not cover the complete text")
    return [[v / visits[i] for v in sums[i]] for i in range(size)]


def token_scores(result):
    """Validate every present representation and retain consistent token values."""
    if not isinstance(result, dict):
        raise ValueError("Cached result must be an object")
    scores = None
    if "token_distributions" in result:
        tokens = result["token_distributions"]
        if not isinstance(tokens, (list, tuple)) or not tokens:
            raise ValueError("No valid token distributions")
        scores = [probability_vector(v) for v in tokens]
    if "windows" in result:
        reconstructed = _window_token_scores(result)
        if scores is None:
            return reconstructed
        if len(scores) != len(reconstructed):
            raise ValueError("Token and window distribution lengths differ")
        if any(abs(stored - rebuilt) > TOKEN_WINDOW_ABS_TOLERANCE
               for token, window in zip(scores, reconstructed)
               for stored, rebuilt in zip(token, window)):
            raise ValueError("Token and window distributions disagree")
    if scores is None:
        raise ValueError("Token or complete window distributions are required for region rules")
    return scores


def classify(result, config):
    """Reproduce V4 decision rules on cached scores; this performs no inference."""
    for key in ("minimum_score", "region_confidence", "mixed_min_fraction"):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), (int, float)) or not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if not isinstance(config.get("region_override"), bool):
        raise ValueError("region_override must be boolean")
    scores = token_scores(result)
    means = [sum(p[k] for p in scores) / len(scores) for k in range(4)]
    three = (means[0], means[1] + means[3], means[2])
    winner = max(range(3), key=lambda i: three[i])
    label = LABELS[winner] if three[winner] >= config["minimum_score"] else "uncertain"
    human_region = sum(p[0] >= config["region_confidence"] for p in scores) / len(scores)
    ai_region = sum(p[1] + p[3] >= config["region_confidence"] for p in scores) / len(scores)
    if config["region_override"] and min(human_region, ai_region) >= config["mixed_min_fraction"]:
        label = "mixed"
    return label


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, midpoint - half), min(1, midpoint + half)]


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
            "wilson_95": wilson(numerator, denominator)}


def metrics(rows):
    confusion = {truth: {prediction: 0 for prediction in PREDICTIONS} for truth in LABELS}
    for row in rows:
        if row["prediction"] not in PREDICTIONS:
            raise ValueError("Unsupported prediction")
        confusion[row["label"]][row["prediction"]] += 1
    total = len(rows)
    correct = sum(confusion[label][label] for label in LABELS)
    abstentions = sum(confusion[label]["uncertain"] for label in LABELS)
    errors = sum(confusion[label]["error"] for label in LABELS)
    classified = total - abstentions - errors
    human = sum(confusion["human"].values())
    mixed = sum(confusion["mixed"].values())
    nonhuman = total - human
    return {"total": total, "confusion": confusion,
            "overall_accuracy": fraction(correct, total),
            "human_false_positive_rate": fraction(confusion["human"]["ai"] + confusion["human"]["mixed"], human),
            "mixed_recall": fraction(confusion["mixed"]["mixed"], mixed),
            "nonhuman_missed_as_human": fraction(confusion["ai"]["human"] + confusion["mixed"]["human"], nonhuman),
            "coverage": fraction(classified, total), "selective_accuracy": fraction(correct, classified),
            "abstentions": abstentions, "operational_failures": errors,
            "per_class_recall": {label: fraction(confusion[label][label], sum(confusion[label].values())) for label in LABELS}}


def cluster_intervals(rows, samples=2000, seed=20261022):
    grouped = defaultdict(list)
    for row in rows:
        group = row.get("independence_group")
        if not isinstance(group, str) or not group:
            return {"status": "unavailable_missing_independence_group"}
        grouped[group].append(row)
    if len(grouped) < 2:
        return {"status": "unavailable_fewer_than_two_groups", "groups": len(grouped)}
    rng = random.Random(seed)
    keys = list(grouped)
    selected = ("overall_accuracy", "human_false_positive_rate", "mixed_recall", "coverage")
    values = {key: [] for key in selected}
    for _ in range(samples):
        draw = [row for group in rng.choices(keys, k=len(keys)) for row in grouped[group]]
        measured = metrics(draw)
        for key in selected:
            rate = measured[key]["rate"]
            if rate is not None:
                values[key].append(rate)
    intervals = {}
    for key, observations in values.items():
        observations.sort()
        intervals[key] = None if not observations else [observations[int(.025 * (len(observations) - 1))], observations[int(.975 * (len(observations) - 1))]]
    return {"status": "descriptive_cluster_bootstrap", "groups": len(keys), "resamples": samples,
            "seed": seed, "intervals_95": intervals,
            "limits": "Depends on correct declared independence groups; few groups and development selection limit inference."}


def evaluate(path, config_path, split="development", mode="development", bootstrap=2000):
    path, config_path = Path(path), Path(config_path)
    rows = load_jsonl(path)
    checks = validate_records(rows)
    if checks["cross_split_overlaps"]:
        raise ValueError("Cross-split group or exact-text overlap: " + json.dumps(checks["cross_split_overlaps"]))
    if mode == "heldout" and (split != "test" or checks["unknown_group_metadata"]):
        raise ValueError("Held-out reporting requires test split and complete declared group metadata")
    config = json.loads(config_path.read_text())
    selected, fingerprints = [], set()
    for original in rows:
        if original["split"] != split:
            continue
        row = dict(original)
        if mode == "heldout" and (row.get("consumed_for_development") or row.get("synthetic_fixture")):
            raise ValueError("Consumed development or synthetic fixture cannot become held-out evidence")
        if "inference_fingerprint" not in row:
            raise ValueError("Each score record needs inference_fingerprint")
        fingerprints.add(canonical_sha(row["inference_fingerprint"]))
        if row.get("status") == "error":
            row["prediction"] = "error"
        else:
            row["prediction"] = classify(row["result"], config)
        selected.append(row)
    if not selected:
        raise ValueError("Requested split is empty")
    if len(fingerprints) != 1:
        raise ValueError("Do not pool different inference systems/window configurations in one evaluation")
    return {"schema_version": 1, "mode": mode, "split": split,
            "data_sha256": sha(path), "decision_config_sha256": sha(config_path),
            "evaluator_sha256": sha(__file__), "inference_fingerprint_sha256": next(iter(fingerprints)),
            "inference_fingerprint": selected[0]["inference_fingerprint"], "split_checks": checks,
            "metrics": metrics(selected), "cluster_intervals": cluster_intervals(selected, bootstrap),
            "predictions": [{"id": row["id"], "label": row["label"], "prediction": row["prediction"],
                             "text_sha256": row["text_sha256"]} for row in selected],
            "promotable": False,
            "limits": ["Development results are not independent accuracy validation.",
                       "Heldout mode checks declared metadata, not undisclosed training overlap or prior access.",
                       "Wilson intervals assume independent trials; inspect cluster intervals and group design.",
                       "Unknown training overlap, near duplicates and label semantics require separate review."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="development", choices=("train", "development", "calibration", "test"))
    parser.add_argument("--mode", default="development", choices=("development", "heldout"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output path; preserve prior evaluations")
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    try:
        report = evaluate(args.records, args.config, args.split, args.mode, args.bootstrap)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"mode": report["mode"], "metrics": report["metrics"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
