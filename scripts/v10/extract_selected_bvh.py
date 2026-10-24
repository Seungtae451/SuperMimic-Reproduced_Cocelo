"""Extract a bounded sample of prefiltered SOMA BVHs without unpacking the full archive.

The 45.2 GB tar.gz must still be downloaded in full. Its gzip stream has no
per-motion random access; selected entries are extracted while scanning it.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tarfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path,
                        default=root/'data/v10/downloads/soma_uniform.tar.gz')
    parser.add_argument('--manifest', type=Path,
                        default=root/'data/v10/manifests/bones_seed_prefilter.csv')
    parser.add_argument('--output-dir', type=Path, default=root/'data/v10/sample_bvh')
    parser.add_argument('--category', action='append', default=[],
                        help='Exact metadata category; may be specified more than once.')
    parser.add_argument('--one-per-category', action='store_true',
                        help='Extract at most one candidate from each selected category.')
    parser.add_argument('--name-contains', default='',
                        help='Optional case-insensitive substring in the motion name.')
    parser.add_argument('--limit', type=int, default=1,
                        help='Maximum selected BVHs to extract; explicit positive value required.')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    archive = args.archive.resolve()
    if not archive.is_file():
        parser.error(f'Archive not found: {archive}')
    with args.manifest.open(newline='') as f:
        selected = {row['soma_uniform_path']: row for row in csv.DictReader(f)
                    if row['status'] == 'candidate'
                    and args.name_contains.lower() in row['move_name'].lower()
                    and (not args.category or row['category'] in args.category)}
    output = args.output_dir.resolve()
    count = 0
    used_categories = set()
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.removeprefix('./')
            if name not in selected:
                continue
            category = selected[name]['category']
            if args.one_per_category and category in used_categories:
                continue
            parts = Path(name).parts
            if not parts or parts[0] != 'soma_uniform' or '..' in parts or Path(name).is_absolute():
                raise ValueError(f'Unsafe archive member: {member.name}')
            target = output.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.size:
                    raise ValueError(f'Existing target has wrong size: {target}')
            else:
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f'Could not read archive member: {member.name}')
                partial = target.with_name(target.name+'.part')
                try:
                    with partial.open('wb') as destination:
                        shutil.copyfileobj(source, destination)
                    if partial.stat().st_size != member.size:
                        raise ValueError(f'Incomplete extraction: {member.name}')
                    os.replace(partial, target)
                finally:
                    partial.unlink(missing_ok=True)
            count += 1
            used_categories.add(category)
            print(f'{category}: {target}')
            if count >= args.limit:
                break
    if count != args.limit:
        raise SystemExit(f'Only {count} selected BVHs found (requested {args.limit}).')
    print(f'Extracted {count} selected BVHs; excluded archive members were not written.')


if __name__ == '__main__':
    main()
