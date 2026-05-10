# Reviewer Quickstart

This checklist downloads the released artifacts and reproduces the main HSC
scRNA DANN checks.

## 1. Install

```bash
conda env create -f environment.yml
conda activate cross-species-aging
python -m pip install -e .
```

## 2. Download Artifacts

```bash
python scripts/download_hf_artifacts.py \
  --data-repo simiaoAA/cross-species-aging-data \
  --model-repo simiaoAA/cross-species-aging-models \
  --data-dir data/raw \
  --model-dir models
```

Then confirm that all required files are available:

```bash
python scripts/check_inputs.py --manifest config/data_manifest.tsv --root .
```

## 3. Label Sanity Check

```bash
python scripts/check_scrna_labels.py \
  --data-dir data/raw \
  --features data/raw/featurestemprna.txt \
  --outdir results/scrna_label_sanity
```

## 4. Saved-Model Evaluation

```bash
python scripts/evaluate_scrna_dann.py \
  --data-dir data/raw \
  --model models/train_clfrna_model_newest12.21.keras \
  --features data/raw/featurestemprna.txt \
  --class-order data/raw/onehot_encoder.txt \
  --human-young humanyoung_RNA_dataset.csv \
  --human-old humanold_RNA_dataset.csv \
  --outdir results/scrna_eval
```

## 5. From-Scratch Training

```bash
python scripts/train_scrna_dann_tf.py \
  --mode fast_two_phase \
  --epochs 50 \
  --batch-size 64 \
  --width 256 \
  --embedding-dim 64 \
  --dropout 0.3 \
  --lr 0.0001 \
  --domain-weight 1.0 \
  --umap-every 10 \
  --seed 42 \
  --outdir results/scrna_dann_training_runs/fast_twophase_seed42
```

Outputs include `params.json`, `metrics_by_epoch.csv`, model checkpoints, and
epoch UMAPs in the selected output directory.
