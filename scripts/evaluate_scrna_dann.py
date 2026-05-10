"""Evaluate saved scRNA DANN classifiers on human young/old RNA matrices."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def load_keras3_model(model_path: Path):
    """Load a Keras v3 .keras model while keeping global TF/Keras unchanged."""
    vendor = Path("vendor/keras3")
    if vendor.exists():
        sys.path.insert(0, str(vendor))
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import keras  # noqa: PLC0415

    # Some saved models include this key, while the local Keras reader rejects it.
    original_dense_init = keras.layers.Dense.__init__

    def patched_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return original_dense_init(self, *args, **kwargs)

    keras.layers.Dense.__init__ = patched_dense_init
    try:
        import keras.src.layers.core.dense as dense_mod  # noqa: PLC0415

        dense_mod.Dense.__init__ = patched_dense_init
    except Exception:
        pass

    return keras.saving.load_model(model_path, compile=False)


def read_features(path: Path) -> list[str]:
    features = pd.read_csv(path, sep="\t", index_col=0).index
    return [str(item).upper() for item in features]


def read_class_order(path: Path) -> list[str]:
    mapping: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            idx, label = line.strip().split(":", maxsplit=1)
            mapping[int(idx)] = label
    return [mapping[i] for i in sorted(mapping)]


def resolve_input(data_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.exists() or path.is_absolute():
        return path
    return data_dir / path


def load_gene_matrix(csv_path: Path, features: list[str]) -> np.ndarray:
    print(f"Loading {csv_path.name}")
    feature_to_col = {gene: idx for idx, gene in enumerate(features)}

    found: set[str] = set()
    with csv_path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
        n_cells = max(0, header.count(b","))
        x = np.zeros((n_cells, len(features)), dtype=np.float32)

        for line_no, line in enumerate(handle, start=2):
            comma = line.find(b",")
            if comma < 0:
                continue
            gene = line[:comma].strip().strip(b'"').decode("utf-8", errors="ignore").upper()
            col_idx = feature_to_col.get(gene)
            if col_idx is None:
                continue
            values = np.fromstring(line[comma + 1 :], sep=",", dtype=np.float32)
            if values.size != n_cells:
                padded = np.zeros(n_cells, dtype=np.float32)
                padded[: min(values.size, n_cells)] = values[:n_cells]
                values = padded
            x[:, col_idx] = values
            found.add(gene)
            if len(found) % 5000 == 0:
                print(f"  found_genes={len(found)} at line={line_no}")

    print(f"  cells={x.shape[0]} features={x.shape[1]} found_genes={len(found)}")
    return x


def scale_like_scanpy(x: np.ndarray, max_value: float = 6.0) -> np.ndarray:
    mean = np.nanmean(x, axis=0, dtype=np.float64)
    std = np.nanstd(x, axis=0, ddof=1, dtype=np.float64)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    x = (x - mean.astype(np.float32)) / std.astype(np.float32)
    x = np.clip(x, -max_value, max_value)
    return x.astype(np.float32, copy=False)


def scale_with_reference(x: np.ndarray, feature_path: Path, max_value: float = 6.0) -> np.ndarray:
    stats = pd.read_csv(feature_path, sep="\t", index_col=0)
    mean = stats["mean"].to_numpy(dtype=np.float32)
    std = stats["std"].to_numpy(dtype=np.float32)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    x = (x - mean) / std
    x = np.clip(x, -max_value, max_value)
    return x.astype(np.float32, copy=False)


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    data_dir = Path(args.data_dir)
    model_path = resolve_input(data_dir, args.model)
    feature_path = resolve_input(data_dir, args.features)
    class_order_path = resolve_input(data_dir, args.class_order)

    model = load_keras3_model(model_path)
    expected_dim = int(model.input_shape[-1])
    print(f"Model input dimension: {expected_dim}")

    features = read_features(feature_path)
    print(f"Feature file rows: {len(features)}")
    if len(features) != expected_dim:
        raise ValueError(
            f"Feature count {len(features)} does not match model input dimension {expected_dim}."
        )

    class_order = read_class_order(class_order_path)
    print(f"Class order: {class_order}")

    x_young = load_gene_matrix(data_dir / args.human_young, features)
    x_old = load_gene_matrix(data_dir / args.human_old, features)
    x = np.vstack([x_young, x_old])
    y_true = np.array(["young"] * len(x_young) + ["old"] * len(x_old))

    print(f"Scaling combined human test matrix: {args.scaling}")
    if args.scaling == "test":
        x = scale_like_scanpy(x)
    elif args.scaling == "reference":
        x = scale_with_reference(x, feature_path)
    elif args.scaling == "none":
        x = x.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown scaling mode: {args.scaling}")

    print("Predicting")
    y_prob = model.predict(x, batch_size=args.batch_size, verbose=0)
    y_pred = np.array(class_order)[np.argmax(y_prob, axis=1)]

    positive_label = args.positive_label
    pos_idx = class_order.index(positive_label)
    y_true_bin = (y_true == positive_label).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=["old", "young"])
    out = {
        "analysis": args.analysis,
        "model": args.model,
        "human_young": args.human_young,
        "human_old": args.human_old,
        "n_young": int(len(x_young)),
        "n_old": int(len(x_old)),
        "n_features": int(expected_dim),
        "positive_label": positive_label,
        "scaling": args.scaling,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true_bin, y_prob[:, pos_idx])),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label="old")),
        "specificity": float(recall_score(y_true, y_pred, pos_label="young")),
        "tn_young": int(cm[1, 1]),
        "fp_young_as_old": int(cm[1, 0]),
        "fn_old_as_young": int(cm[0, 1]),
        "tp_old": int(cm[0, 0]),
    }
    return out


def write_outputs(result: dict[str, float | int | str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / "scrna_dann_evaluation_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    metrics = outdir / "scrna_dann_reproduced_metrics.tsv"
    rows = [
        ("HSC_scRNA_DANN", "human_test_AUROC", result["auroc"]),
        ("HSC_scRNA_DANN", "balanced_accuracy", result["balanced_accuracy"]),
        ("HSC_scRNA_DANN", "F1", result["f1"]),
        ("HSC_scRNA_DANN", "sensitivity", result["sensitivity"]),
        ("HSC_scRNA_DANN", "specificity", result["specificity"]),
    ]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["analysis", "metric", "observed", "run_id", "notes"])
        for analysis, metric, observed in rows:
            writer.writerow([analysis, metric, observed, result["model"], "evaluation_only_saved_model"])

    print(f"Wrote {summary}")
    print(f"Wrote {metrics}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("AGING_DATA_DIR", "data/raw"))
    parser.add_argument("--analysis", default="HSC_scRNA_DANN")
    parser.add_argument("--model", default="models/train_clfrna_model_newest12.21.keras")
    parser.add_argument("--features", default="featurestemprna.txt")
    parser.add_argument("--class-order", default="onehot_encoder.txt")
    parser.add_argument("--human-young", default="humanyoung_RNA_dataset.csv")
    parser.add_argument("--human-old", default="humanold_RNA_dataset.csv")
    parser.add_argument("--positive-label", default="old")
    parser.add_argument("--scaling", choices=["test", "reference", "none"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    result = evaluate(args)
    for key, value in result.items():
        print(f"{key}: {value}")
    write_outputs(result, Path(args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
