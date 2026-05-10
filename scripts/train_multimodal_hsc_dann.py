"""Train and evaluate a multimodal HSC RNA+ATAC DANN model.

The script uses the prepared multimodal arrays generated from the source HSC
RNA and ATAC preprocessing workflow. Features are split by the `RNA_` and `ATAC_` prefixes,
encoded through separate branches, fused, and trained with an age classifier
plus a gradient-reversal domain classifier.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
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
from sklearn.preprocessing import LabelEncoder


def default_drive_aging_dir() -> Path:
    return Path("data/raw")


def set_reproducible(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf  # noqa: PLC0415

    tf.random.set_seed(seed)


def resolve(data_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return data_dir / path


def read_feature_names(path: Path) -> list[str]:
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].astype(str).tolist()


def normalized_feature_key(feature: str) -> str:
    if "_" not in feature:
        return feature.upper()
    modality, gene = feature.split("_", maxsplit=1)
    return f"{modality.upper()}_{gene.upper()}"


def feature_masks(features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    names = np.array(features, dtype=object)
    atac = np.char.startswith(names.astype(str), "ATAC_")
    rna = np.char.startswith(names.astype(str), "RNA_")
    if not atac.any() or not rna.any():
        raise ValueError(
            f"Expected RNA_ and ATAC_ feature prefixes, found {rna.sum()} RNA and {atac.sum()} ATAC."
        )
    return rna, atac


def subset_rows(
    x: np.ndarray,
    meta: pd.DataFrame,
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if max_rows is None or len(meta) <= max_rows:
        return x, meta
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(meta), size=max_rows, replace=False))
    return x[keep], meta.iloc[keep].reset_index(drop=True)


def load_arrays(args: argparse.Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    x_source = np.load(resolve(data_dir, args.train_matrix), mmap_mode="r")
    x_target = np.load(resolve(data_dir, args.test_matrix), mmap_mode="r")
    train_meta = pd.read_csv(resolve(data_dir, args.train_meta))
    test_meta = pd.read_csv(resolve(data_dir, args.test_meta))
    train_features = read_feature_names(resolve(data_dir, args.train_features))
    test_features = read_feature_names(resolve(data_dir, args.test_features))

    if x_source.shape[1] != len(train_features):
        raise ValueError("train feature count does not match train matrix width")
    if x_target.shape[1] != len(test_features):
        raise ValueError("test feature count does not match test matrix width")

    train_keys = [normalized_feature_key(feature) for feature in train_features]
    test_keys = [normalized_feature_key(feature) for feature in test_features]

    if train_keys == test_keys:
        common_features = train_features
        train_idx = np.arange(len(train_features))
        test_idx = np.arange(len(test_features))
    else:
        train_lookup = {}
        for idx, key in enumerate(train_keys):
            train_lookup.setdefault(key, idx)
        test_lookup = {}
        for idx, key in enumerate(test_keys):
            test_lookup.setdefault(key, idx)
        common_keys = [key for key in train_keys if key in test_lookup]
        common_keys = list(dict.fromkeys(common_keys))
        common_features = common_keys
        train_idx = np.array([train_lookup[key] for key in common_keys], dtype=np.int64)
        test_idx = np.array([test_lookup[key] for key in common_keys], dtype=np.int64)
        print(
            "Aligning multimodal features by intersection: "
            f"train={len(train_features)} test={len(test_features)} common={len(common_features)}"
        )
        if not common_features:
            raise ValueError("train/test multimodal feature lists have no overlap")

    x_source, train_meta = subset_rows(x_source, train_meta, args.max_source, args.seed)
    x_target, test_meta = subset_rows(x_target, test_meta, args.max_target, args.seed + 1)
    x_source = np.asarray(x_source[:, train_idx], dtype=np.float32)
    x_target = np.asarray(x_target[:, test_idx], dtype=np.float32)
    rna_mask, atac_mask = feature_masks(common_features)

    return {
        "x_source": x_source,
        "x_target": x_target,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "features": common_features,
        "rna_mask": rna_mask,
        "atac_mask": atac_mask,
    }


def build_model(n_rna: int, n_atac: int, n_classes: int, lambda_grl: float):
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow import keras  # noqa: PLC0415
    from tensorflow.keras import layers  # noqa: PLC0415

    class GradientReversal(layers.Layer):
        def __init__(self, strength: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.strength = strength

        def call(self, x):
            strength = self.strength

            @tf.custom_gradient
            def reverse(y):
                def grad(dy):
                    return -strength * dy

                return y, grad

            return reverse(x)

        def get_config(self):
            config = super().get_config()
            config["strength"] = self.strength
            return config

    def branch(inp, width: int, prefix: str):
        x = layers.Dense(width, name=f"{prefix}_dense1")(inp)
        x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
        x = layers.Activation("elu", name=f"{prefix}_elu1")(x)
        x = layers.Dropout(0.25, name=f"{prefix}_drop1")(x)
        y = layers.Dense(width, name=f"{prefix}_res_dense1")(x)
        y = layers.BatchNormalization(name=f"{prefix}_res_bn1")(y)
        y = layers.Activation("elu", name=f"{prefix}_res_elu1")(y)
        y = layers.Dense(width, name=f"{prefix}_res_dense2")(y)
        y = layers.BatchNormalization(name=f"{prefix}_res_bn2")(y)
        x = layers.Add(name=f"{prefix}_res_add")([x, y])
        return layers.Activation("elu", name=f"{prefix}_res_out")(x)

    rna_input = keras.Input(shape=(n_rna,), name="rna")
    atac_input = keras.Input(shape=(n_atac,), name="atac")
    rna = branch(rna_input, 256, "rna")
    atac = branch(atac_input, 256, "atac")
    fused = layers.Concatenate(name="fusion")([rna, atac])
    fused = layers.Dense(256, activation="elu", name="embedding_dense")(fused)
    fused = layers.BatchNormalization(name="embedding_bn")(fused)
    embedding = layers.Dropout(0.35, name="embedding_dropout")(fused)
    age = layers.Dense(n_classes, activation="softmax", name="age")(embedding)
    domain = GradientReversal(lambda_grl, name="gradient_reversal")(embedding)
    domain = layers.Dense(128, activation="elu", name="domain_dense")(domain)
    domain = layers.Dropout(0.25, name="domain_dropout")(domain)
    domain = layers.Dense(2, activation="softmax", name="domain")(domain)

    model = keras.Model([rna_input, atac_input], {"age": age, "domain": domain}, name="hsc_multimodal_dann")
    embedder = keras.Model([rna_input, atac_input], embedding, name="hsc_multimodal_embedder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"age": "sparse_categorical_crossentropy", "domain": "sparse_categorical_crossentropy"},
        loss_weights={"age": 1.0, "domain": args_domain_weight()},
        metrics={"age": "accuracy", "domain": "accuracy"},
    )
    return model, embedder


def args_domain_weight() -> float:
    return float(os.environ.get("DANN_DOMAIN_WEIGHT", "0.2"))


def split_modalities(x: np.ndarray, rna_mask: np.ndarray, atac_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {"rna": x[:, rna_mask].astype(np.float32), "atac": x[:, atac_mask].astype(np.float32)}


def evaluate(model, x_target: dict[str, np.ndarray], y_true: np.ndarray, class_names: list[str]) -> dict[str, float | int]:
    probs = model.predict(x_target, batch_size=256, verbose=0)["age"]
    pred_idx = np.argmax(probs, axis=1)
    y_pred = np.array(class_names)[pred_idx]
    positive = "old" if "old" in class_names else class_names[0]
    pos_idx = class_names.index(positive)
    y_true_bin = (y_true == positive).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=["old", "young"])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true_bin, probs[:, pos_idx])),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label="old")),
        "specificity": float(recall_score(y_true, y_pred, pos_label="young")),
        "tn_young": int(cm[1, 1]),
        "fp_young_as_old": int(cm[1, 0]),
        "fn_old_as_young": int(cm[0, 1]),
        "tp_old": int(cm[0, 0]),
    }


def write_umap(embedder, x_source, x_target, train_meta, test_meta, out_prefix: Path) -> None:
    import umap  # noqa: PLC0415

    z_source = embedder.predict(x_source, batch_size=256, verbose=0)
    z_target = embedder.predict(x_target, batch_size=256, verbose=0)
    z = np.vstack([z_source, z_target])
    labels = pd.concat([train_meta, test_meta], ignore_index=True)
    labels["group"] = labels["species"].str.capitalize() + "_" + labels["age_group"].str.capitalize()
    coords = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.25).fit_transform(z)
    df = pd.DataFrame({"UMAP1": coords[:, 0], "UMAP2": coords[:, 1], "group": labels["group"]})

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)

    colors = {
        "Mouse_Young": "#b2182b",
        "Mouse_Old": "#ef8a62",
        "Human_Young": "#2166ac",
        "Human_Old": "#67a9cf",
    }
    markers = {"Mouse_Young": "o", "Mouse_Old": "D", "Human_Young": "s", "Human_Old": "^"}
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for group in ["Mouse_Young", "Mouse_Old", "Human_Young", "Human_Old"]:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=12,
            c=colors[group],
            marker=markers[group],
            alpha=0.78,
            linewidths=0.15,
            edgecolors="white",
            label=group.replace("_", " "),
        )
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("HSC multimodal DANN latent UMAP")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("AGING_DATA_DIR", str(default_drive_aging_dir())))
    parser.add_argument("--train-matrix", default="X_train_multimodal.npy")
    parser.add_argument("--test-matrix", default="X_test_multimodal.npy")
    parser.add_argument("--train-meta", default="train_meta_multimodal.csv")
    parser.add_argument("--test-meta", default="test_meta_multimodal.csv")
    parser.add_argument("--train-features", default="train_feature_names_multimodal.csv")
    parser.add_argument("--test-features", default="test_feature_names_multimodal.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--domain-weight", type=float, default=0.2)
    parser.add_argument("--lambda-grl", type=float, default=1.0)
    parser.add_argument("--max-source", type=int, default=None)
    parser.add_argument("--max-target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results/multimodal_hsc_dann")
    args = parser.parse_args()

    os.environ["DANN_DOMAIN_WEIGHT"] = str(args.domain_weight)
    set_reproducible(args.seed)
    arrays = load_arrays(args)

    label_encoder = LabelEncoder()
    y_source = label_encoder.fit_transform(arrays["train_meta"]["age_group"].astype(str))
    y_target = arrays["test_meta"]["age_group"].astype(str).to_numpy()
    class_names = label_encoder.classes_.astype(str).tolist()
    x_source = split_modalities(arrays["x_source"], arrays["rna_mask"], arrays["atac_mask"])
    x_target = split_modalities(arrays["x_target"], arrays["rna_mask"], arrays["atac_mask"])

    x_all = {"rna": np.vstack([x_source["rna"], x_target["rna"]]), "atac": np.vstack([x_source["atac"], x_target["atac"]])}
    y_age = np.concatenate([y_source, np.zeros(len(y_target), dtype=np.int64)])
    y_domain = np.concatenate([np.zeros(len(y_source), dtype=np.int64), np.ones(len(y_target), dtype=np.int64)])
    age_weight = np.concatenate([np.ones(len(y_source), dtype=np.float32), np.zeros(len(y_target), dtype=np.float32)])
    domain_weight = np.ones(len(y_domain), dtype=np.float32)

    model, embedder = build_model(x_source["rna"].shape[1], x_source["atac"].shape[1], len(class_names), args.lambda_grl)
    history = model.fit(
        x_all,
        {"age": y_age, "domain": y_domain},
        sample_weight={"age": age_weight, "domain": domain_weight},
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_target, {"age": label_encoder.transform(y_target), "domain": np.ones(len(y_target), dtype=np.int64)}),
        verbose=2,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = evaluate(model, x_target, y_target, class_names)
    result.update(
        {
            "n_source": int(len(y_source)),
            "n_target": int(len(y_target)),
            "n_rna_features": int(x_source["rna"].shape[1]),
            "n_atac_features": int(x_source["atac"].shape[1]),
            "classes": "|".join(class_names),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
        }
    )
    pd.DataFrame(history.history).to_csv(outdir / "training_history.csv", index=False)
    with (outdir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    model.save(outdir / "hsc_multimodal_dann.keras")
    embedder.save(outdir / "hsc_multimodal_embedder.keras")
    write_umap(embedder, x_source, x_target, arrays["train_meta"], arrays["test_meta"], outdir / "hsc_multimodal_dann_umap")

    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
