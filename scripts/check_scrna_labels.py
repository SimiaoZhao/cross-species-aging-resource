"""Sanity-check HSC scRNA young/old labels before DANN training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_scrna_dann import load_gene_matrix, read_features, scale_like_scanpy


def find_aging_dir() -> Path:
    for path in Path("G:/").glob("*/Aging"):
        if (path / "8wk_RNA_dataset.csv").exists():
            return path
    return Path("data/raw")


def load_domain(data_dir: Path, features: list[str], young_file: str, old_file: str):
    x_young = load_gene_matrix(data_dir / young_file, features)
    x_old = load_gene_matrix(data_dir / old_file, features)
    x = np.vstack([x_young, x_old])
    y = np.array(["young"] * len(x_young) + ["old"] * len(x_old))
    return scale_like_scanpy(x), y


def load_or_cache_domain(
    data_dir: Path,
    features: list[str],
    young_file: str,
    old_file: str,
    cache_path: Path,
):
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        return loaded["x"], loaded["y"].astype(str)
    x, y = load_domain(data_dir, features, young_file, old_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y)
    return x, y


def linear_model(seed: int):
    return make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            penalty="elasticnet",
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
            n_jobs=-1,
        ),
    )


def metrics(y_true: np.ndarray, prob_old: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = np.where(prob_old >= threshold, "old", "young")
    return {
        "auroc_old_positive": float(roc_auc_score((y_true == "old").astype(int), prob_old)),
        "auroc_if_flipped": float(roc_auc_score((y_true == "young").astype(int), prob_old)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--features", default="results/featurestemprna_full_16264.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results/scrna_label_sanity")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else find_aging_dir()
    feature_path = Path(args.features)
    features = read_features(feature_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x_mouse, y_mouse = load_or_cache_domain(
        data_dir,
        features,
        "8wk_RNA_dataset.csv",
        "24mon_RNA_dataset.csv",
        outdir / "mouse_scrna_16264_scaled.npz",
    )
    x_human, y_human = load_or_cache_domain(
        data_dir,
        features,
        "humanyoung_RNA_dataset.csv",
        "humanold_RNA_dataset.csv",
        outdir / "human_scrna_16264_scaled.npz",
    )

    rows = []
    for name, x, y in [("mouse_within_cv", x_mouse, y_mouse), ("human_within_cv", x_human, y_human)]:
        clf = linear_model(args.seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        prob = cross_val_predict(clf, x, y, cv=cv, method="predict_proba", n_jobs=1)
        classes = sorted(np.unique(y).astype(str).tolist())
        old_idx = classes.index("old")
        row = metrics(y, prob[:, old_idx])
        row.update({"analysis": name, "n": len(y), "n_old": int((y == "old").sum()), "n_young": int((y == "young").sum())})
        rows.append(row)

    clf = linear_model(args.seed)
    clf.fit(x_mouse, y_mouse)
    old_idx = clf.classes_.astype(str).tolist().index("old")
    row = metrics(y_human, clf.predict_proba(x_human)[:, old_idx])
    row.update({"analysis": "train_mouse_predict_human", "n": len(y_human), "n_old": int((y_human == "old").sum()), "n_young": int((y_human == "young").sum())})
    rows.append(row)

    clf = linear_model(args.seed)
    clf.fit(x_human, y_human)
    old_idx = clf.classes_.astype(str).tolist().index("old")
    row = metrics(y_mouse, clf.predict_proba(x_mouse)[:, old_idx])
    row.update({"analysis": "train_human_predict_mouse", "n": len(y_mouse), "n_old": int((y_mouse == "old").sum()), "n_young": int((y_mouse == "young").sum())})
    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "scrna_label_sanity_metrics.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
