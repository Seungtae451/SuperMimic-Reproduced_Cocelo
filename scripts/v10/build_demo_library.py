"""Select a small, auditable V10 demo library from the accepted compact library.

No motion is downloaded, retargeted, or simulated here. The output contains full
reference clips, plus a JSON record of exact source names and substitutions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def select(spec: list[dict], index: dict) -> list[dict]:
    names = index["names"]
    lengths = index["lengths"]
    used: set[int] = set()
    selected = []
    for item in spec:
        needle = item["source_contains"].lower()
        candidates = [i for i, name in enumerate(names)
                      if needle in name.lower() and i not in used and lengths[i] >= 60]
        if not candidates:
            raise ValueError(f"No accepted V10 motion matches {item['label']!r}: {needle!r}")
        # Prefer a compact complete clip near four seconds; penalize long clips
        # instead of truncating them, so every demo motion keeps its real end.
        chosen = min(candidates, key=lambda i: (abs(lengths[i] - 200),
                                                 names[i].endswith("_M"), names[i]))
        used.add(chosen)
        selected.append({"category": item["category"], "label": item["label"],
                         "source_name": names[chosen], "source_index": chosen,
                         "source_start": int(index["starts"][chosen]),
                         "frames": int(lengths[chosen])})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/v10/library/accepted_compact.npy")
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/robots/humanoid_v10/demo_motions.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/v10/demo/v10_demo.npy")
    args = parser.parse_args()
    index = json.loads(args.source.with_suffix(".json").read_text())
    spec = json.loads(args.spec.read_text())
    selected = select(spec, index)
    source = np.load(args.source, mmap_mode="r", allow_pickle=False)
    if source.shape[0] != index["frames"] or source.shape[1] != index["reference_dimensions"]:
        raise ValueError("Source library and index disagree")
    total = sum(row["frames"] for row in selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ref = np.lib.format.open_memmap(args.out, mode="w+", dtype=source.dtype,
                                    shape=(total, source.shape[1]))
    starts = []
    offset = 0
    for row in selected:
        starts.append(offset)
        length = row["frames"]
        start = row["source_start"]
        ref[offset:offset + length] = source[start:start + length]
        offset += length
    ref.flush()
    del ref
    metadata = {"robot": "v10", "root_body": index["root_body"],
                "fps": index["fps"], "names": [r["source_name"] for r in selected],
                "starts": starts, "lengths": [r["frames"] for r in selected],
                "frames": total, "reference_dimensions": int(source.shape[1]),
                "demo_motions": selected,
                "note": "Substitutes are V10 BONES-SEED motions, not the official G1 clips."}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(selected)} motions, {total} frames: {args.out}")
    for row in selected:
        print(f"{row['category']:22s} {row['label']:48s} <- {row['source_name']}")


if __name__ == "__main__":
    main()
