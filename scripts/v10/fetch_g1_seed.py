"""Download the published BONES-SEED G1 retarget archive for a V10 fast-path pilot."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
REPO = "bones-studio/seed"
FILENAME = "g1.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/v10/downloads")
    parser.add_argument("--minimum-free-gib-after", type=float, default=30.0)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO, files_metadata=True)
    entry = next((file for file in info.siblings if file.rfilename == FILENAME), None)
    if entry is None or entry.size is None:
        raise SystemExit("G1 archive size is unavailable from Hugging Face")
    required = entry.size + args.minimum_free_gib_after * 1024**3
    if shutil.disk_usage(output).free < required:
        raise SystemExit("Insufficient disk space for G1 archive and reserve")

    path = Path(hf_hub_download(
        repo_id=REPO, filename=FILENAME, repo_type="dataset",
        revision=info.sha, local_dir=output,
    ))
    if path.stat().st_size != entry.size:
        raise RuntimeError("Downloaded G1 archive has an unexpected size")
    manifest = {
        "dataset": REPO,
        "revision": info.sha,
        "filename": FILENAME,
        "bytes": entry.size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "g1_download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {path}: {entry.size / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
