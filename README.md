# Cross-Species Aging Resource

This repository provides a compact reviewer-facing package for reproducing the
cross-species aging-state annotation analyses described in:

**Deep learning enables cross-species annotation and attribution of ageing states
in haematopoietic stem and immune cells**

The repository contains code, environment files, input manifests, and concise
reproduction instructions. Large data matrices and trained model weights are
hosted separately on Hugging Face Hub.

## Data And Models

- Data: `https://huggingface.co/datasets/simiaoAA/cross-species-aging-data`
- Models: `https://huggingface.co/simiaoAA/cross-species-aging-models`

During peer review these repositories may require reviewer access. After access
is granted, download all artifacts with:

```bash
python scripts/download_hf_artifacts.py \
  --data-repo simiaoAA/cross-species-aging-data \
  --model-repo simiaoAA/cross-species-aging-models \
  --data-dir data/raw \
  --model-dir models
```

## Install

```bash
conda env create -f environment.yml
conda activate cross-species-aging
python -m pip install -e .
```

TensorFlow GPU availability differs across operating systems. Full training runs
are recommended on Linux or Colab GPU; saved-model evaluation can also be run on
CPU.

## Reproduce Key Checks

Verify that required files are present:

```bash
python scripts/check_inputs.py --manifest config/data_manifest.tsv --root .
```

Sanity-check HSC scRNA labels:

```bash
python scripts/check_scrna_labels.py \
  --data-dir data/raw \
  --features data/raw/featurestemprna.txt \
  --outdir results/scrna_label_sanity
```

Evaluate the released HSC scRNA DANN model:

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

Run from-scratch HSC scRNA DANN training:

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

See `docs/REVIEWER_QUICKSTART.md` for the same workflow in checklist form and
`docs/REPRODUCTION_STATUS.md` for recorded metrics.

## Repository Contents

```text
config/      Input manifests and expected/observed metric tables
docs/        Reviewer quickstart and reproduction status
scripts/     Download, input-check, evaluation, and training scripts
data/        Local data download directory; large files are not tracked
models/      Local model download directory; large files are not tracked
```

Only code, manifests, and concise reproduction documentation are tracked in
this repository. Large artifacts are distributed through the Hugging Face
repositories above.
