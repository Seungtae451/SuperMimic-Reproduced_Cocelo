"""Import CMU/BVH/human NPZ datasets and retarget them to F26."""

import argparse
from pathlib import Path
from .retarget import retarget_external_reference


def main():
    parser = argparse.ArgumentParser(description='Human mocap (CMU AMC/BVH/NPZ) → PyRoki → F26')
    parser.add_argument('input', type=Path)
    parser.add_argument('--skeleton', type=Path, help='CMU ASF for the AMC; auto-detected when unambiguous')
    parser.add_argument('--source-fps', type=float, default=120., help='AMC fps (BVH/NPZ read their own fps)')
    parser.add_argument('--meters-per-unit', type=float, help='BVH meters/unit; default .01. CMU: .0254 / ASF length unit')
    parser.add_argument('--up-axis', choices=['Y', 'Z'], default='Y')
    parser.add_argument('--mapping', type=Path, help='JSON mapping canonical human landmarks to dataset joint names')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--prepend-stand', action='store_true')
    parser.add_argument('--append-stand', action='store_true')
    args = parser.parse_args()
    result = retarget_external_reference(args.input, args.output_dir, skeleton_path=args.skeleton,
        source_fps=args.source_fps, length_scale=args.meters_per_unit, up_axis=args.up_axis,
        mapping_path=args.mapping, config_path=args.config, max_frames=args.max_frames,
        prepend_safe_stand=args.prepend_stand, append_safe_stand=args.append_stand,
        progress=lambda message: print(message, flush=True))
    print(f'NPZ: {result.npz}\nMetadata: {result.metadata}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
