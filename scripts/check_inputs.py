"""Check whether required data/model inputs are present before rerunning notebooks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SEARCH_DIRS = (Path("data/raw"), Path("models"), Path("."), Path("output"))


def iter_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def find_candidate(root: Path, filename: str) -> Path | None:
    normalized = filename.replace("\\", "/")
    if normalized.endswith("/"):
        target = root / normalized
        return target if target.exists() else None
    for base in SEARCH_DIRS:
        candidate = root / base / normalized
        if candidate.exists():
            return candidate
        if "/" in normalized:
            basename_candidate = root / base / Path(normalized).name
            if basename_candidate.exists():
                return basename_candidate
    candidate = root / normalized
    return candidate if candidate.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="config/data_manifest.tsv")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        local_manifest = Path.cwd() / manifest
        root_manifest = root / manifest
        manifest = local_manifest if local_manifest.exists() else root_manifest

    missing = []
    print(f"Checking inputs from {manifest}")
    for row in iter_manifest(manifest):
        name = row["filename"]
        found = find_candidate(root, name)
        if found:
            size = found.stat().st_size if found.is_file() else 0
            print(f"OK\t{name}\t{found}\t{size}")
        else:
            missing.append(row)
            print(f"MISSING\t{name}\t{row['required_for']}")

    print()
    print(f"Missing inputs: {len(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
