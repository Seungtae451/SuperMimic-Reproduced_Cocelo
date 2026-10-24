"""External human mocap → shared human/F26 PyRoki → nine-key reference NPZ."""

from pathlib import Path
import hashlib
import json
import shutil
import time

import mujoco

from apps.paths import CONFIG_ROOT, WORKSPACE_ROOT
from apps.humanoid_retargeter.motion_io import save_visualization_motion
from apps.humanoid_retargeter.pyroki_retargeter import retarget_motion_clip
from apps.humanoid_retargeter.simulation_export import update_archive_metadata
from .motion import load_external_motion

DEFAULT_CONFIG = CONFIG_ROOT / 'external_pyroki_retargeter.json'


def load_model(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    config = json.loads(path.read_text())
    model_path = (WORKSPACE_ROOT / config['model_path']).resolve()
    return mujoco.MjModel.from_xml_path(str(model_path)), model_path, config


def retarget_external_reference(source_path, output_dir, *, skeleton_path=None, source_fps=120.,
                                length_scale=None, up_axis='Y', mapping_path=None, config_path=None,
                                max_frames=None, prepend_safe_stand=None, append_safe_stand=None,
                                initial_pose_mode='upright', progress=None):
    """Use the same human body-length/heading/contact adaptation as GEM-X/Kimodo.

    No F26 source-joint prior, source-robot FK or benchmark is used: the input
    is an articulated human dataset, with a different skeleton and body size.
    """
    started = time.perf_counter()
    emit = progress or (lambda message: None)
    source = load_external_motion(source_path, skeleton_path=skeleton_path, source_fps=source_fps,
        length_scale=length_scale, up_axis=up_axis, mapping_path=mapping_path)
    output_dir = Path(output_dir).expanduser().resolve()
    # Normalized human snapshots and input provenance stay under source/;
    # the top-level external_50hz_retargeted.npz is the simulation contract.
    source_dir = output_dir / 'source'
    protected = {output_dir/'external_50hz_retargeted.npz', source_dir/'human_motion.npz',
                 output_dir/'solver_solution.npz'}
    if any(p in protected or source_dir in p.parents for p in source.source_files):
        raise ValueError('Choose an output directory that preserves the human input files.')
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.source_files}
    source_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(source.source_files):
        shutil.copyfile(path, source_dir/f'{i:02d}_{path.name}')
    normalized = save_visualization_motion(source.clip, source_dir/'human_motion.npz')
    (source_dir/'import.json').write_text(json.dumps(source.metadata, indent=2, ensure_ascii=False)+'\n')
    emit(f"사람 데이터셋 준비 완료: {source.metadata['format']} · {source.frames} 프레임 · {source.fps:g} Hz")
    result = retarget_motion_clip(source.clip, output_dir, source_motion_path=normalized,
        source_kind='External Dataset', config_path=config_path or DEFAULT_CONFIG,
        max_frames=max_frames, output_fps=50., prepend_safe_stand=prepend_safe_stand,
        append_safe_stand=append_safe_stand, initial_pose_mode=initial_pose_mode, progress=emit)
    metadata = json.loads(result.metadata.read_text())
    metadata['external_dataset'] = source.metadata
    metadata['original_source_motion'] = str(source.path)
    metadata['elapsed_s'] = time.perf_counter() - started
    metadata['provenance']['source_adapter_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__), Path(__file__).with_name('formats.py'), Path(__file__).with_name('motion.py'))}
    update_archive_metadata(result.npz, metadata)
    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Human source changed during retargeting: {path}')
    emit(f'외부 사람 모션 → F26 저장 완료: {result.frames} 프레임 · 50 Hz · 9개 키')
    return result
