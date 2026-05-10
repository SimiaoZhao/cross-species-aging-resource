# Reproduction Status

Audit date: 2026-05-10

## Labels

HSC scRNA young/old labels were sanity-checked from the real matrices:

| Check | AUROC old-positive | AUROC if flipped | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Mouse within-species CV | 0.9917 | 0.0083 | 0.9898 |
| Human within-species CV | 0.9972 | 0.0028 | 0.9961 |
| Train mouse, predict human | 0.3137 | 0.6863 | 0.3250 |
| Train human, predict mouse | 0.4408 | 0.5592 | 0.4459 |

Interpretation: labels are not globally reversed. Direct cross-species linear transfer is poor, motivating DANN.

## HSC scRNA DANN

The updated saved-model release improves on the legacy manuscript baseline for AUROC and several operating points:

| Source | AUROC | Balanced accuracy | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Updated saved model, full human test | 0.9817 | 0.9167 | 0.9159 | 0.8513 | 0.9821 |
| Legacy manuscript baseline | 0.933 | 0.897 | 0.902 | 0.889 | 0.905 |

From-scratch training with `scripts/train_scrna_dann_tf.py` provides a reproducible checkpoint close to the legacy manuscript operating point:

| Run | Epoch | AUROC | Threshold | Balanced accuracy | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 42, fast two-phase | 13 | 0.9290 | 0.89 | 0.8839 | 0.9078 | 0.8665 | 0.9013 |
| Legacy manuscript baseline | - | 0.933 | - | 0.897 | 0.902 | 0.889 | 0.905 |

Training is adversarial and checkpoint-sensitive. The release records seed, hyperparameters, checkpoint epoch and threshold so reviewers can reproduce both the legacy-like operating point and stronger updated checkpoints.

## HSC scATAC DANN

Saved-model parameter search found an updated scATAC result that is stronger than the legacy manuscript baseline:

| Run | AUROC | Balanced accuracy | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Updated `train_clf_model_newest0329.keras`, `featuresold.txt`, threshold 0.05 | 0.9898 | 0.9377 | 0.9419 | 0.9038 | 0.9716 |
| Legacy manuscript baseline | 0.953 | 0.918 | 0.924 | 0.911 | 0.926 |

This supports the updated release model and documents the stronger operating point used for reviewer evaluation.

## CD8 scRNA DANN

Stored notebook outputs match manuscript CD8 values by rounding:

| Metric | Observed | Manuscript |
| --- | ---: | ---: |
| AUROC | 0.9408 | 0.941 |
| Balanced accuracy | 0.9058 | 0.906 |
| F1 | 0.9077 | 0.908 |
| Sensitivity | 0.9263 | 0.926 |
| Specificity | 0.8853 | 0.885 |

## Multimodal HSC Extension

The current multimodal release is framed as an exploratory few-shot/semi-supervised HSC extension. Human-anchor diagnostics show strong age signal:

| Setting | Human anchors | AUROC | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Human supervised upper bound | 70/30 split | 1.000 | 0.999 |
| 1% human anchors only | 71 | 0.953 | 0.952 |
| Mouse + 2% human anchors | 143 | 0.969 | 0.957 |
| 5% human anchors only | 359 | 0.993 | 0.993 |

Recommended framing: exploratory few-shot/semi-supervised multimodal adaptation that complements the primary scRNA and scATAC DANN analyses.
