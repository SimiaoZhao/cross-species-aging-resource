# Model Directory

Large trained model files are not stored in Git. Download the Hugging Face model
repository into this directory with:

```bash
python scripts/download_hf_artifacts.py \
  --data-repo simiaoAA/cross-species-aging-data \
  --model-repo simiaoAA/cross-species-aging-models \
  --data-dir data/raw \
  --model-dir models
```
