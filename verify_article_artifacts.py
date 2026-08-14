#!/usr/bin/env python3
"""Verify the fixed files and headline metrics used in the article."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "article_artifacts.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for tied scores."""
    labels = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j

    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        fail("AUC requires both positive and negative observations")
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    n_classes = probabilities.shape[1]
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(n_classes, dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(n_classes, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(n_classes, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    auc = [
        binary_auc((y_true == class_index).astype(np.int8), probabilities[:, class_index])
        for class_index in range(n_classes)
    ]
    return {
        "support": support.tolist(),
        "accuracy": float(true_positive.sum() / support.sum()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "auc": auc,
    }


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        fail(f"{label}: expected {expected:.15g}, found {actual:.15g}")


def main() -> None:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    for relative_path, expected_hash in manifest["artifact_checksums"].items():
        path = ROOT / relative_path
        if not path.is_file():
            fail(f"missing artifact: {relative_path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            fail(f"checksum mismatch: {relative_path}")

    canonical_split = None
    for relative_path in manifest["split_files"]:
        with (ROOT / relative_path).open(encoding="utf-8") as stream:
            split = json.load(stream)
        if split["final_test_records"] != manifest["expected_test_records"]:
            fail(f"unexpected held-out records in {relative_path}")
        if canonical_split is None:
            canonical_split = split
        elif split != canonical_split:
            fail(f"data split differs across architectures: {relative_path}")

    canonical_labels = None
    for model_name, expected in manifest["models"].items():
        with np.load(ROOT / expected["prediction_file"], allow_pickle=False) as arrays:
            required = {"y_true", "y_pred_probs", "y_pred_classes"}
            if set(arrays.files) != required:
                fail(f"{model_name}: unexpected prediction-array keys")
            y_true = arrays["y_true"].astype(np.int64, copy=False)
            probabilities = arrays["y_pred_probs"].astype(np.float64)
            y_pred = arrays["y_pred_classes"].astype(np.int64, copy=False)

        if probabilities.shape != (y_true.size, len(manifest["class_names"])):
            fail(f"{model_name}: unexpected probability-array shape")
        if not np.array_equal(y_pred, probabilities.argmax(axis=1)):
            fail(f"{model_name}: stored classes do not match probability argmax")
        if canonical_labels is None:
            canonical_labels = y_true.copy()
        elif not np.array_equal(y_true, canonical_labels):
            fail(f"{model_name}: test labels differ from the other architectures")

        observed = metrics(y_true, y_pred, probabilities)
        if observed["support"] != manifest["expected_support"]:
            fail(f"{model_name}: unexpected class support {observed['support']}")
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            assert_close(observed[key], expected[key], f"{model_name} {key}")
        for class_name, actual_auc, expected_auc in zip(
            manifest["class_names"], observed["auc"], expected["auc"]
        ):
            assert_close(actual_auc, expected_auc, f"{model_name} {class_name} AUC")

        print(
            f"{model_name:15s} "
            f"accuracy={observed['accuracy']:.6f} "
            f"macro-F1={observed['macro_f1']:.6f} "
            f"weighted-F1={observed['weighted_f1']:.6f}"
        )

    kfold_spec = manifest["main_model_kfold"]
    with (ROOT / kfold_spec["file"]).open(encoding="utf-8") as stream:
        kfold = json.load(stream)
    assert_close(
        kfold["mean_accuracy"],
        kfold_spec["mean_accuracy"],
        "CNN-Conformer k-fold mean accuracy",
    )
    assert_close(
        kfold["std_accuracy"],
        kfold_spec["std_accuracy"],
        "CNN-Conformer k-fold accuracy SD",
    )

    print(
        f"PASS: {len(manifest['artifact_checksums'])} checksums, "
        "four common test arrays, four metric sets, and the main-model "
        "cross-validation summary were verified."
    )


if __name__ == "__main__":
    main()
