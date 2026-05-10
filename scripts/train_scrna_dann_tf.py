"""Train HSC scRNA DANN from raw cached matrices with recorded seeds/metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


CLASS_TO_ID = {"old": 0, "young": 1}
ID_TO_CLASS = np.array(["old", "young"])


def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf  # noqa: PLC0415

    tf.random.set_seed(seed)


def configure_gpu() -> None:
    import tensorflow as tf  # noqa: PLC0415

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def load_cache(cache_dir: Path):
    mouse = np.load(cache_dir / "mouse_scrna_16264_scaled.npz", allow_pickle=True)
    human = np.load(cache_dir / "human_scrna_16264_scaled.npz", allow_pickle=True)
    x_source = mouse["x"].astype(np.float32)
    y_source = np.array([CLASS_TO_ID[str(item)] for item in mouse["y"].astype(str)], dtype=np.int64)
    x_target = human["x"].astype(np.float32)
    y_target = np.array([CLASS_TO_ID[str(item)] for item in human["y"].astype(str)], dtype=np.int64)
    return x_source, y_source, x_target, y_target


def make_model(input_dim: int, width: int, embedding_dim: int, dropout: float, use_grl: bool):
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow import keras  # noqa: PLC0415
    from tensorflow.keras import layers  # noqa: PLC0415

    class GradientReversal(layers.Layer):
        def __init__(self, strength=1.0, **kwargs):
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

    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(width, name="input_dense")(inp)
    x = layers.BatchNormalization(name="input_bn")(x)
    x = layers.Activation("elu", name="input_elu")(x)
    x = layers.Dropout(dropout, name="input_dropout")(x)
    for block in range(2):
        y = layers.Dense(width, name=f"res{block}_dense1")(x)
        y = layers.BatchNormalization(name=f"res{block}_bn1")(y)
        y = layers.Activation("elu", name=f"res{block}_elu1")(y)
        y = layers.Dense(width, name=f"res{block}_dense2")(y)
        y = layers.BatchNormalization(name=f"res{block}_bn2")(y)
        x = layers.Add(name=f"res{block}_add")([x, y])
        x = layers.Activation("elu", name=f"res{block}_out")(x)
        x = layers.Dropout(dropout, name=f"res{block}_dropout")(x)
    emb = layers.Dense(embedding_dim, activation="elu", name="embedding")(x)
    age = layers.Dense(2, activation="softmax", name="age")(emb)
    dom = GradientReversal(1.0, name="grl")(emb) if use_grl else emb
    dom = layers.Dense(width // 2, activation="elu", name="domain_dense")(dom)
    dom = layers.Dropout(dropout, name="domain_dropout")(dom)
    dom = layers.Dense(2, activation="softmax", name="domain")(dom)
    model = keras.Model(inp, {"age": age, "domain": dom})
    embedder = keras.Model(inp, emb)
    domain_model = keras.Model(inp, dom)
    return model, embedder, domain_model


def get_domain_weights(model):
    return [layer.get_weights() for layer in model.layers if layer.name.startswith("domain")]


def set_domain_weights(model, weights):
    for layer, saved in zip([layer for layer in model.layers if layer.name.startswith("domain")], weights):
        layer.set_weights(saved)


def get_non_domain_weights(model):
    return [layer.get_weights() for layer in model.layers if not layer.name.startswith("domain")]


def set_non_domain_weights(model, weights):
    for layer, saved in zip([layer for layer in model.layers if not layer.name.startswith("domain")], weights):
        layer.set_weights(saved)


def evaluate(model, x: np.ndarray, y: np.ndarray, batch_size: int) -> dict[str, float]:
    chunks = []
    for start in range(0, len(x), batch_size):
        out = model(x[start : start + batch_size], training=False)
        chunks.append(out["age"].numpy())
    probs = np.vstack(chunks)
    prob_old = probs[:, CLASS_TO_ID["old"]]
    pred = np.argmax(probs, axis=1)
    y_str = ID_TO_CLASS[y]
    pred_str = ID_TO_CLASS[pred]
    return {
        "auroc": float(roc_auc_score((y == CLASS_TO_ID["old"]).astype(int), prob_old)),
        "accuracy": float(accuracy_score(y_str, pred_str)),
        "balanced_accuracy": float(balanced_accuracy_score(y_str, pred_str)),
        "f1": float(f1_score(y_str, pred_str, pos_label="old")),
        "sensitivity": float(recall_score(y_str, pred_str, pos_label="old")),
        "specificity": float(recall_score(y_str, pred_str, pos_label="young")),
    }


def save_umap(embedder, x_source, y_source, x_target, y_target, outdir: Path, epoch: int, seed: int, max_points: int = 2500):
    import umap  # noqa: PLC0415

    rng = np.random.default_rng(seed + epoch)
    src_idx = rng.choice(len(x_source), size=min(max_points, len(x_source)), replace=False)
    tgt_idx = rng.choice(len(x_target), size=min(max_points, len(x_target)), replace=False)
    z_source = np.vstack([embedder(x_source[src_idx][i : i + 128], training=False).numpy() for i in range(0, len(src_idx), 128)])
    z_target = np.vstack([embedder(x_target[tgt_idx][i : i + 128], training=False).numpy() for i in range(0, len(tgt_idx), 128)])
    z = np.vstack([z_source, z_target])
    labels = (
        [f"Mouse_{ID_TO_CLASS[item].capitalize()}" for item in y_source[src_idx]]
        + [f"Human_{ID_TO_CLASS[item].capitalize()}" for item in y_target[tgt_idx]]
    )
    coords = umap.UMAP(random_state=seed, n_neighbors=20, min_dist=0.25).fit_transform(z)
    colors = {"Mouse_Young": "#b2182b", "Mouse_Old": "#ef8a62", "Human_Young": "#2166ac", "Human_Old": "#67a9cf"}
    markers = {"Mouse_Young": "o", "Mouse_Old": "D", "Human_Young": "s", "Human_Old": "^"}
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    labels_arr = np.array(labels)
    for group in ["Mouse_Young", "Mouse_Old", "Human_Young", "Human_Old"]:
        mask = labels_arr == group
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, c=colors[group], marker=markers[group], alpha=0.75, label=group)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"HSC scRNA DANN epoch {epoch}")
    fig.tight_layout()
    fig.savefig(outdir / f"umap_epoch_{epoch:04d}.png", dpi=250)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="results/scrna_label_sanity")
    parser.add_argument("--outdir", default="results/scrna_dann_training_runs/run_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--domain-weight", type=float, default=0.2)
    parser.add_argument("--mode", choices=["grl", "two_phase", "fast_two_phase"], default="fast_two_phase")
    parser.add_argument("--umap-every", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    set_seed(args.seed)
    configure_gpu()
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow import keras  # noqa: PLC0415

    x_source, y_source, x_target, y_target = load_cache(Path(args.cache_dir))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "params.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model, embedder, domain_model = make_model(
        x_source.shape[1], args.width, args.embedding_dim, args.dropout, use_grl=args.mode == "grl"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={"age": "sparse_categorical_crossentropy", "domain": "sparse_categorical_crossentropy"},
        loss_weights={"age": 1.0, "domain": args.domain_weight},
        metrics={"age": "accuracy", "domain": "accuracy"},
    )
    domain_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    age_loss_fn = keras.losses.SparseCategoricalCrossentropy()
    domain_loss_fn = keras.losses.SparseCategoricalCrossentropy()
    combined_opt = keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    domain_opt = keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)

    def split_vars():
        domain_vars = []
        non_domain_vars = []
        for layer in model.layers:
            target = domain_vars if layer.name.startswith("domain") else non_domain_vars
            target.extend(layer.trainable_variables)
        return non_domain_vars, domain_vars

    non_domain_vars, domain_vars = split_vars()

    @tf.function
    def fast_two_phase_step(xb, y_age_b, y_domain_true_b, y_domain_confuse_b, age_w_b, domain_w_b):
        with tf.GradientTape() as tape:
            out = model(xb, training=True)
            per_age = keras.losses.sparse_categorical_crossentropy(y_age_b, out["age"])
            per_domain = keras.losses.sparse_categorical_crossentropy(y_domain_confuse_b, out["domain"])
            age_loss = tf.reduce_sum(per_age * age_w_b) / (tf.reduce_sum(age_w_b) + 1e-7)
            domain_loss = tf.reduce_mean(per_domain * domain_w_b)
            total_loss = age_loss + domain_loss
        grads = tape.gradient(total_loss, non_domain_vars)
        combined_opt.apply_gradients([(g, v) for g, v in zip(grads, non_domain_vars) if g is not None])

        with tf.GradientTape() as tape2:
            out2 = model(xb, training=True)
            d_loss = domain_loss_fn(y_domain_true_b, out2["domain"])
        grads2 = tape2.gradient(d_loss, domain_vars)
        domain_opt.apply_gradients([(g, v) for g, v in zip(grads2, domain_vars) if g is not None])
        age_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out["age"], axis=1, output_type=tf.int64), y_age_b), tf.float32) * age_w_b)
        domain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out2["domain"], axis=1, output_type=tf.int64), y_domain_true_b), tf.float32))
        return {
            "loss": total_loss,
            "age_loss": age_loss,
            "domain_loss": domain_loss,
            "age_accuracy": age_acc,
            "domain_accuracy": domain_acc,
        }

    rows = []
    best_score = -np.inf
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = max(1, min(len(x_source), len(x_target)) // args.batch_size)
    for epoch in range(1, args.epochs + 1):
        batch_stats = []
        source_perm = rng.permutation(len(x_source))
        target_perm = rng.permutation(len(x_target))
        for step in range(steps_per_epoch):
            src = source_perm[(step * args.batch_size) : ((step + 1) * args.batch_size)]
            tgt = target_perm[(step * args.batch_size) : ((step + 1) * args.batch_size)]
            xb = np.vstack([x_source[src], x_target[tgt]]).astype(np.float32)
            y_age = np.concatenate([y_source[src], np.zeros(len(tgt), dtype=np.int64)])
            y_domain_true = np.concatenate([np.zeros(len(src), dtype=np.int64), np.ones(len(tgt), dtype=np.int64)])
            y_domain_confuse = np.concatenate([np.ones(len(src), dtype=np.int64), np.zeros(len(tgt), dtype=np.int64)])
            age_weights = np.concatenate([np.ones(len(src), dtype=np.float32), np.zeros(len(tgt), dtype=np.float32)])
            progress = ((epoch - 1) * steps_per_epoch + step) / max(1, args.epochs * steps_per_epoch)
            lambda_s = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
            lambda_p = float(lambda_s * (1.0 - progress) ** 2)
            domain_weights = np.ones(len(y_domain_true), dtype=np.float32) * (args.domain_weight * lambda_p)
            if args.mode == "fast_two_phase":
                stats_tf = fast_two_phase_step(
                    tf.convert_to_tensor(xb),
                    tf.convert_to_tensor(y_age, dtype=tf.int64),
                    tf.convert_to_tensor(y_domain_true, dtype=tf.int64),
                    tf.convert_to_tensor(y_domain_confuse, dtype=tf.int64),
                    tf.convert_to_tensor(age_weights, dtype=tf.float32),
                    tf.convert_to_tensor(domain_weights, dtype=tf.float32),
                )
                stats = {key: float(value.numpy()) for key, value in stats_tf.items()}
            elif args.mode == "two_phase":
                domain_weights_saved = get_domain_weights(model)
                stats = model.train_on_batch(
                    xb,
                    {"age": y_age, "domain": y_domain_confuse},
                    sample_weight={"age": age_weights, "domain": domain_weights},
                    return_dict=True,
                )
                set_domain_weights(model, domain_weights_saved)
                non_domain_weights_saved = get_non_domain_weights(model)
                domain_model.train_on_batch(xb, y_domain_true, return_dict=True)
                set_non_domain_weights(model, non_domain_weights_saved)
            else:
                stats = model.train_on_batch(
                    xb,
                    {"age": y_age, "domain": y_domain_true},
                    sample_weight={"age": age_weights, "domain": domain_weights},
                    return_dict=True,
                )
            batch_stats.append(stats)
        hist = {key: [float(np.mean([item[key] for item in batch_stats]))] for key in batch_stats[0]}
        human = evaluate(model, x_target, y_target, args.batch_size)
        mouse = evaluate(model, x_source, y_source, args.batch_size)
        row = {"epoch": epoch, **{f"train_{k}": float(v[0]) for k, v in hist.items()}}
        row.update({f"human_{k}": v for k, v in human.items()})
        row.update({f"mouse_{k}": v for k, v in mouse.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / "metrics_by_epoch.csv", index=False)
        print(
            f"epoch={epoch} human_auroc={human['auroc']:.4f} "
            f"human_bacc={human['balanced_accuracy']:.4f} mouse_auroc={mouse['auroc']:.4f}"
        )
        score = human["auroc"] + human["balanced_accuracy"]
        if score > best_score:
            best_score = score
            model.save_weights(outdir / "best_model_weights.h5")
            embedder.save_weights(outdir / "best_embedder_weights.h5")
        if args.umap_every and (epoch == 1 or epoch % args.umap_every == 0 or epoch == args.epochs):
            save_umap(embedder, x_source, y_source, x_target, y_target, outdir, epoch, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
