"""Download data/model artifacts from Hugging Face repositories."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-repo", required=True, help="Hugging Face dataset repo, e.g. org/name.")
    parser.add_argument("--model-repo", required=True, help="Hugging Face model repo, e.g. org/name.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: python -m pip install huggingface_hub") from exc

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.data_repo,
        repo_type="dataset",
        local_dir=data_dir,
        revision=args.revision,
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id=args.model_repo,
        repo_type="model",
        local_dir=model_dir,
        revision=args.revision,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded data to {data_dir}")
    print(f"Downloaded models to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

