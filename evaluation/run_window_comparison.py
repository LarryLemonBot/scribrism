#!/usr/bin/env python3
"""Development-only inference from an explicit sample manifest; no network."""
import argparse
from collections import Counter
import hashlib
import importlib.util
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import sys

if __package__:
    from . import evaluate as evaluator
else:
    import evaluate as evaluator

LABELS = ("human", "ai", "mixed")
GROUP_FIELDS = ("source", "prompt", "writer", "model")
DEVELOPMENT_CAVEAT = ("Consumed development data only: this comparison cannot establish held-out "
                      "accuracy or authorize promotion. Declared groups and labels are not independent verification.")
RECEIPT_SCHEMA_VERSION = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_path(base_dir, value):
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError("Sample text_path must be a nonempty relative path")
    base_dir = Path(base_dir).resolve()
    path = (base_dir / value).resolve()
    if not path.is_relative_to(base_dir):
        raise ValueError("Sample text_path escapes its manifest directory")
    return path


def audit_manifest(manifest, base_dir):
    """Validate reproducible development input and describe only declared facts."""
    if not isinstance(manifest, dict) or manifest.get("purpose") != "development_only":
        raise ValueError("This exploratory runner accepts development_only manifests")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest needs a nonempty samples list")
    limitations = manifest.get("limitations", [])
    if not isinstance(limitations, list) or any(not isinstance(x, str) for x in limitations):
        raise ValueError("Manifest limitations must be a list of strings")
    seen_ids, seen_texts = set(), {}
    labels = Counter()
    groups = {field: Counter() for field in GROUP_FIELDS}
    by_label = {label: {field: Counter() for field in GROUP_FIELDS} for label in LABELS}
    unknown, not_applicable, independent, missing_independence = [], [], Counter(), []
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Every sample must be an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen_ids:
            raise ValueError("Every sample needs a unique nonempty id")
        seen_ids.add(identifier)
        label = row.get("label")
        if label not in LABELS:
            raise ValueError(identifier + ": unsupported truth label")
        labels[label] += 1
        if row.get("split") != "development" or row.get("consumed_for_development") is not True:
            raise ValueError("Every sample must be explicitly consumed development data")
        digest = row.get("text_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(identifier + ": invalid text SHA-256")
        if digest in seen_texts:
            previous = seen_texts[digest]
            kind = "Conflicting labels for identical text" if previous[1] != label else "Duplicate exact text"
            raise ValueError(f"{kind}: {previous[0]} and {identifier}")
        seen_texts[digest] = (identifier, label)
        path = text_path(base_dir, row.get("text_path"))
        if sha(path) != digest:
            raise ValueError("Sample hash mismatch: " + identifier)
        declared = row.get("groups", {})
        na = row.get("not_applicable_groups", [])
        if not isinstance(declared, dict) or set(declared) - set(GROUP_FIELDS):
            raise ValueError(identifier + ": invalid group fields")
        if (not isinstance(na, list) or any(field not in GROUP_FIELDS for field in na)
                or len(set(na)) != len(na)):
            raise ValueError(identifier + ": invalid not_applicable_groups")
        for field in GROUP_FIELDS:
            value = declared.get(field)
            values = [] if value is None else value if isinstance(value, list) else [value]
            if (any(not isinstance(v, str) or not v.strip() for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(identifier + ": invalid " + field + " group")
            if field in na:
                if values:
                    raise ValueError(identifier + ": populated group also declared not applicable: " + field)
                not_applicable.append({"record": identifier, "field": field, "label": label})
            elif not values:
                unknown.append({"record": identifier, "field": field, "label": label})
            for value in values:
                groups[field][value] += 1
                by_label[label][field][value] += 1
        independence = row.get("independence_group")
        if independence is None:
            missing_independence.append(identifier)
        elif not isinstance(independence, str) or not independence.strip():
            raise ValueError(identifier + ": invalid independence_group")
        else:
            independent[independence] += 1
    return {
        "samples": len(samples), "class_counts": {label: labels[label] for label in LABELS},
        "unique_texts": len(seen_texts), "exact_duplicate_check": "passed; duplicate texts rejected",
        "near_duplicate_check": "not_performed; hashes detect exact duplicates only",
        "declared_group_counts": {field: len(values) for field, values in groups.items()},
        "declared_group_counts_by_class": {label: {field: len(values) for field, values in fields.items()}
                                           for label, fields in by_label.items()},
        "repeated_declared_groups": {field: {value: count for value, count in sorted(values.items()) if count > 1}
                                     for field, values in groups.items()},
        "unknown_group_metadata": unknown, "not_applicable_group_metadata": not_applicable,
        "unknown_group_counts": {field: sum(x["field"] == field for x in unknown) for field in GROUP_FIELDS},
        "not_applicable_group_counts": {field: sum(x["field"] == field for x in not_applicable) for field in GROUP_FIELDS},
        "declared_independence_groups": len(independent), "missing_independence_groups": missing_independence,
        "repeated_independence_groups": {value: count for value, count in sorted(independent.items()) if count > 1},
        "manifest_limitations": limitations,
    }


def validate_windows(windows):
    if not windows or any(isinstance(size, bool) or not isinstance(size, int) or not 2 <= size <= 768 for size in windows):
        raise ValueError("This diagnostic allows only 2–768 token windows")
    if len(set(windows)) != len(windows):
        raise ValueError("Window sizes must be unique to avoid overwriting results")


def runtime_fingerprint(engine_dir):
    lock = Path(engine_dir) / "requirements.lock"
    packages = {}
    for line in lock.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = package_version(name)
        cpu_build = name == 'torch' and expected == '2.8.0' and actual == '2.8.0+cpu'
        if actual != expected and not cpu_build:
            raise ValueError("Dependency version differs from lock: " + name)
        packages[name] = actual
    if not packages:
        raise ValueError("Dependency lock is empty")
    runtime = {"python": platform.python_version(), "implementation": platform.python_implementation(),
               "platform": platform.platform(), "machine": platform.machine(),
               "optimize": sys.flags.optimize, "packages": packages}
    return {"runtime": runtime, "runtime_sha256": canonical_sha(runtime),
            "requirements_lock_sha256": sha(lock)}


def verify_model_pin(engine_dir, module):
    """Explicitly check the local profile pin and all model/tokenizer file bytes."""
    engine_dir = Path(engine_dir)
    profile = json.loads((engine_dir / "profile.json").read_text())
    if (profile.get("backend") != "four_class" or profile.get("config_file") != "config-four-class.json"
            or profile.get("model_id") != module.MODEL_ID or profile.get("model_revision") != module.REVISION):
        raise ValueError("Engine and pinned profile identity differ")
    cache = Path(os.environ.get('SCRIBRISM_MODEL_CACHE', engine_dir / 'models')).expanduser()
    folder = cache / module.REVISION
    manifest_path = folder / "manifest.local.json"
    digest = sha(manifest_path)
    if digest != profile.get("model_manifest_sha256"):
        raise ValueError("Pinned model manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_id") != module.MODEL_ID or manifest.get("revision") != module.REVISION:
        raise ValueError("Pinned model identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise ValueError("Model manifest has no weight file")
    for name, record in files.items():
        if sha(text_path(folder, name)) != record.get("sha256"):
            raise ValueError("Pinned model or tokenizer changed: " + name)
    return {"model_id": module.MODEL_ID, "model_revision": module.REVISION,
            "model_manifest_sha256": digest, "model_sha256": files["model.safetensors"]["sha256"],
            "model_pin_verified": True, "model_files_verified_before_loading": len(files)}


def load_engine_module(engine_dir):
    sys.path.insert(0, str(engine_dir))
    spec = importlib.util.spec_from_file_location("comparison_engine", engine_dir / "four_class.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_metadata_from_bytes(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def output_metadata(path):
    return output_metadata_from_bytes(Path(path).read_bytes())


def verify_receipt_outputs(receipt, directory):
    """Check a trusted receipt's output bytes and counts; this is not authentication."""
    if not isinstance(receipt, dict) or receipt.get("receipt_schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("Receipt has no supported output integrity schema; preserve historical receipts unchanged")
    windows = receipt.get("windows")
    if not isinstance(windows, list):
        raise ValueError("Receipt windows must be a list")
    validate_windows(windows)
    samples = receipt.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Receipt sample count must be a positive integer")
    keys = {str(size) for size in windows}
    outputs, attempts = receipt.get("outputs_by_window"), receipt.get("attempts_by_window")
    if not isinstance(outputs, dict) or set(outputs) != keys or not isinstance(attempts, dict) or set(attempts) != keys:
        raise ValueError("Receipt windows, outputs and attempt counts differ")
    first_ids = None
    for size in windows:
        metadata = outputs[str(size)]
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid output metadata for window {size}")
        payloads = {}
        for kind, name in (("config", f"config-{size}.json"), ("scores", f"scores-{size}.jsonl")):
            raw = (Path(directory) / name).read_bytes()
            expected = metadata.get(kind)
            if not isinstance(expected, dict) or output_metadata_from_bytes(raw) != expected:
                raise ValueError("Output hash or byte count mismatch: " + name)
            payloads[kind] = raw
        config = json.loads(payloads["config"])
        if not isinstance(config, dict) or config.get("window_tokens") != size:
            raise ValueError(f"Effective config window mismatch: {size}")
        expected_counts = attempts[str(size)]
        if (not isinstance(expected_counts, dict) or set(expected_counts) != {"complete", "error"}
                or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                       for count in expected_counts.values()) or sum(expected_counts.values()) != samples):
            raise ValueError(f"Receipt attempt count mismatch: {size}")
        counts, ids = Counter(), []
        for line in payloads["scores"].splitlines():
            record = json.loads(line)
            if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                    or not record["id"].strip() or record["id"] in ids
                    or record.get("status") not in ("complete", "error")):
                raise ValueError(f"Invalid or duplicate score record: {size}")
            fingerprint = record.get("inference_fingerprint")
            if (not isinstance(fingerprint, dict)
                    or canonical_sha(fingerprint) != metadata.get("inference_fingerprint_sha256")
                    or fingerprint.get("window_tokens") != size or fingerprint.get("stride_tokens") != size // 2
                    or fingerprint.get("effective_config_sha256") != metadata["config"]["sha256"]
                    or any(receipt.get(key) != value for key, value in fingerprint.items()
                           if key not in ("window_tokens", "stride_tokens", "effective_config_sha256"))):
                raise ValueError(f"Score fingerprint mismatch: {size}")
            counts[record["status"]] += 1
            ids.append(record["id"])
        if len(ids) != samples or {status: counts[status] for status in ("complete", "error")} != expected_counts:
            raise ValueError(f"Score record or status count mismatch: {size}")
        if first_ids is not None and ids != first_ids:
            raise ValueError(f"Sample order or identities differ across windows: {size}")
        first_ids = ids
    return {"verified": True, "windows": windows, "samples_per_window": samples,
            "verified_output_files": 2 * len(windows)}


def verify_receipt(receipt_path):
    """Verify saved outputs without importing the inference engine or loading weights."""
    receipt_path = Path(receipt_path).resolve()
    raw = receipt_path.read_bytes()
    result = verify_receipt_outputs(json.loads(raw), receipt_path.parent)
    return {**result, "receipt_sha256": hashlib.sha256(raw).hexdigest()}


def validate_result(result):
    """Use the cached-score contract without changing any engine probabilities."""
    if not isinstance(result, dict):
        raise ValueError("Engine result must be an object")
    scores = evaluator.token_scores(result)
    if "tokens" in result:
        size = result["tokens"]
        if isinstance(size, bool) or not isinstance(size, int) or size != len(scores):
            raise ValueError("Engine token count differs from complete token distributions")


def run_comparison(manifest_path, engine_dir, out, windows):
    manifest_path, engine_dir, out = (Path(path).resolve() for path in (manifest_path, engine_dir, out))
    if out.exists():
        raise ValueError("Use a new output directory")
    validate_windows(windows)
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    summary = audit_manifest(manifest, manifest_path.parent)
    samples = manifest["samples"]
    # Freeze exactly the bytes checked by the audit; later path changes cannot alter predictions.
    texts = {}
    for row in samples:
        raw = text_path(manifest_path.parent, row["text_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["text_sha256"]:
            raise ValueError("Sample changed after manifest audit: " + row["id"])
        texts[row["id"]] = raw.decode("utf-8")
    config_path = engine_dir / "config-four-class.json"
    validator_path = Path(evaluator.__file__).resolve()
    protected = [config_path, engine_dir / "four_class.py", engine_dir / "review.py",
                 engine_dir / "profile.json", engine_dir / "requirements.lock", Path(__file__).resolve(),
                 validator_path, manifest_path]
    if (engine_dir / 'model_store.py').is_file():
        protected.append(engine_dir / 'model_store.py')
    protected_hashes = {path: sha(path) for path in protected}
    if protected_hashes[manifest_path] != manifest_sha:
        raise ValueError("Sample manifest changed during audit")
    config = json.loads(config_path.read_text())
    runtime = runtime_fingerprint(engine_dir)
    module = load_engine_module(engine_dir)
    model_pin = verify_model_pin(engine_dir, module)
    engine = module.Engine(dict(config))
    if engine.model_sha != model_pin["model_sha256"]:
        raise ValueError("Loaded model differs from verified pin")
    common = {**model_pin, **runtime, "runner_sha256": protected_hashes[Path(__file__).resolve()],
              "engine_sha256": protected_hashes[engine_dir / "four_class.py"],
              "extractor_sha256": protected_hashes[engine_dir / "review.py"],
              "model_store_sha256": protected_hashes.get(engine_dir / 'model_store.py'),
              "result_validator_sha256": protected_hashes[validator_path],
              "source_config_sha256": protected_hashes[config_path],
              "profile_sha256": protected_hashes[engine_dir / "profile.json"],
              "sample_manifest_sha256": manifest_sha,
              "input_contract": "exact UTF-8 text bytes from manifest; no document extraction",
              "score_contract": "complete fixed-token overlapping windows; per-token overlap averages"}
    out.mkdir(parents=True)
    statuses, outputs = {}, {}
    for size in windows:
        effective = dict(config, window_tokens=size)
        engine.config = effective
        effective_path = out / f"config-{size}.json"
        effective_path.write_text(json.dumps(effective, indent=2) + "\n")
        fingerprint = {**common, "window_tokens": size, "stride_tokens": size // 2,
                       "effective_config_sha256": sha(effective_path)}
        counts = Counter()
        with (out / f"scores-{size}.jsonl").open("w") as output:
            for row in samples:
                record = {key: value for key, value in row.items() if key != "text_path"}
                record["inference_fingerprint"] = fingerprint
                try:
                    result = engine.score(texts[row["id"]])
                    validate_result(result)
                    complete = dict(record, status="complete", result=result)
                    serialized = json.dumps(complete, allow_nan=False) + "\n"
                    record = complete
                except Exception as error:
                    record.pop("result", None)
                    record.update(status="error", error=f"{type(error).__name__}: {error}")
                    serialized = json.dumps(record, allow_nan=False) + "\n"
                # Output I/O must remain fatal: an unwritten attempt cannot be counted as saved.
                output.write(serialized)
                output.flush()
                counts[record["status"]] += 1
                print(json.dumps({"window": size, "id": row["id"], "status": record["status"],
                                  "prediction": record.get("result", {}).get("label")}), flush=True)
        statuses[str(size)] = {status: counts[status] for status in ("complete", "error")}
        outputs[str(size)] = {"scores": output_metadata(out / f"scores-{size}.jsonl"),
                              "config": output_metadata(effective_path),
                              "inference_fingerprint_sha256": canonical_sha(fingerprint)}
    for path, expected in protected_hashes.items():
        if sha(path) != expected:
            raise RuntimeError("Protected configuration, code or manifest changed: " + str(path))
    model_manifest = Path(os.environ.get('SCRIBRISM_MODEL_CACHE', engine_dir / 'models')).expanduser() / module.REVISION / 'manifest.local.json'
    if sha(model_manifest) != model_pin["model_manifest_sha256"]:
        raise RuntimeError("Model manifest changed during comparison")
    receipt = {"receipt_schema_version": RECEIPT_SCHEMA_VERSION,
               "purpose": "development_only", **common, "source_config_unchanged": True,
               "protected_code_and_manifests_unchanged": True, "model_manifest_unchanged": True,
               "windows": windows, "samples": len(samples), "dataset_summary": summary,
               "attempts_by_window": statuses, "outputs_by_window": outputs,
               "promotable": False, "no_promotion": DEVELOPMENT_CAVEAT}
    verify_receipt_outputs(receipt, out)
    (out / "run-receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--windows", type=int, nargs="+")
    parser.add_argument("--verify-receipt", type=Path, help="Verify saved output hashes/counts without loading a model")
    args = parser.parse_args()
    try:
        if args.verify_receipt:
            if args.manifest or args.engine_dir or args.out or args.windows:
                parser.error("--verify-receipt cannot be combined with inference arguments")
            print(json.dumps(verify_receipt(args.verify_receipt), sort_keys=True))
        else:
            if not args.manifest or not args.engine_dir or not args.out:
                parser.error("inference requires manifest, --engine-dir and --out")
            run_comparison(args.manifest, args.engine_dir, args.out, args.windows or [256, 768])
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
