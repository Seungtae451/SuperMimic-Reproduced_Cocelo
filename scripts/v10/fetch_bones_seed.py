"""Fetch BONES-SEED metadata and, when storage is ready, SOMA Uniform BVH.

Requires the user's Hugging Face account to have accepted the dataset terms.
The script never prints an access token and does not fetch G1 retargeted CSVs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ID = 'bones-studio/seed'
METADATA_FILES = (
    'metadata/seed_metadata_v004.parquet',
    'soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh',
)
UNIFORM_ARCHIVE = 'soma_uniform.tar.gz'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--output-dir', type=Path, default=root / 'data/v10/downloads')
    parser.add_argument('--uniform-archive', action='store_true',
                        help='Also fetch the 45.2 GB SOMA Uniform archive; requires free space.')
    parser.add_argument('--minimum-free-gib', type=float, default=100.0,
                        help='Conservative minimum before downloading the archive (default: 100 GiB).')
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    except ImportError as exc:
        raise SystemExit('Install this repository dependencies or huggingface_hub first.') from exc

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.uniform_archive:
        free_gib = shutil.disk_usage(output).free / 1024**3
        if free_gib < args.minimum_free_gib:
            raise SystemExit(
                f'SOMA download paused: {free_gib:.1f} GiB free; '
                f'{args.minimum_free_gib:.1f} GiB required before starting.'
            )

    try:
        revision = HfApi().dataset_info(REPO_ID).sha
        files = (*METADATA_FILES, UNIFORM_ARCHIVE) if args.uniform_archive else METADATA_FILES
        downloaded = {}
        for filename in files:
            path = Path(hf_hub_download(
                repo_id=REPO_ID, filename=filename, repo_type='dataset',
                revision=revision, local_dir=output,
            ))
            downloaded[filename] = {'path': str(path.relative_to(output)), 'bytes': path.stat().st_size}
            print(f'Downloaded {filename}: {path.stat().st_size / 1024**2:.1f} MiB')
    except GatedRepoError as exc:
        raise SystemExit(
            'BONES-SEED access is gated. Accept its terms on Hugging Face and log in '
            'on this machine, then rerun this script.'
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f'Hugging Face request failed (HTTP {exc.response.status_code}).') from exc

    manifest = {
        'dataset': REPO_ID,
        'revision': revision,
        'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': downloaded,
    }
    (output / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved manifest: {output / "download_manifest.json"}')


if __name__ == '__main__':
    main()
