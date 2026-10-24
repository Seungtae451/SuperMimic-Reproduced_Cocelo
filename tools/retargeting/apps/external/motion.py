"""External human datasets → canonical articulated human MotionClip for F26.

ASF/AMC names denote bone endpoints; BVH/NPZ names denote joint origins.
Mapping happens before the shared human-to-robot body-length adaptation.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from apps.humanoid_retargeter.motion_clip import MotionClip
from .formats import load_asf_amc, load_bvh, load_human_npz, validate_clip


# Ordered, anatomical human markers consumed by the shared F26 objective.
HUMAN_PARENTS = {
    'Hips': None, 'Spine1': 'Hips', 'Chest': 'Spine1', 'Neck1': 'Chest', 'Head': 'Neck1',
    'LeftShoulder': 'Chest', 'LeftArm': 'LeftShoulder', 'LeftForeArm': 'LeftArm', 'LeftHand': 'LeftForeArm',
    'RightShoulder': 'Chest', 'RightArm': 'RightShoulder', 'RightForeArm': 'RightArm', 'RightHand': 'RightForeArm',
    'LeftLeg': 'Hips', 'LeftShin': 'LeftLeg', 'LeftFoot': 'LeftShin', 'LeftToeEnd': 'LeftFoot',
    'RightLeg': 'Hips', 'RightShin': 'RightLeg', 'RightFoot': 'RightShin', 'RightToeEnd': 'RightFoot',
}
CMU_ENDPOINTS = {
    'Hips': 'root', 'Spine1': 'lowerback', 'Chest': 'thorax', 'Neck1': 'lowerneck', 'Head': 'head',
    'LeftShoulder': 'lclavicle', 'LeftArm': 'lclavicle', 'LeftForeArm': 'lhumerus', 'LeftHand': 'lradius',
    'RightShoulder': 'rclavicle', 'RightArm': 'rclavicle', 'RightForeArm': 'rhumerus', 'RightHand': 'rradius',
    'LeftLeg': 'lhipjoint', 'LeftShin': 'lfemur', 'LeftFoot': 'ltibia', 'LeftToeEnd': 'ltoes',
    'RightLeg': 'rhipjoint', 'RightShin': 'rfemur', 'RightFoot': 'rtibia', 'RightToeEnd': 'rtoes',
}


def _normalized(name):
    return re.sub('[^a-z0-9]', '', name.split(':')[-1].lower())


def landmark_mapping(names, *, cmu=False, overrides=None):
    overrides = dict(overrides or {})
    if set(overrides) - set(HUMAN_PARENTS):
        raise ValueError(f'Unknown canonical mapping keys: {sorted(set(overrides) - set(HUMAN_PARENTS))}.')
    lookup = {}
    for name in names:
        lookup.setdefault(_normalized(name), []).append(name)
    result = {}
    aliases = {
        'Hips': ['Hips', 'Pelvis', 'Root'], 'Spine1': ['Spine1', 'Spine', 'LowerBack'],
        'Chest': ['Chest', 'Spine2', 'Spine1', 'Spine'], 'Neck1': ['Neck1', 'Neck'], 'Head': ['Head'],
    }
    for side in ('Left', 'Right'):
        aliases.update({
            side+'Shoulder': [side+'Shoulder', side+'Collar', side+'Arm', side+'UpperArm'],
            side+'Arm': [side+'Arm', side+'UpperArm', side+'Shoulder'],
            side+'ForeArm': [side+'ForeArm', side+'LowerArm', side+'Elbow'],
            side+'Hand': [side+'Hand', side+'Wrist'],
            side+'Leg': [side+'UpLeg', side+'Thigh', side+'Leg', side+'Hip'],
            side+'Shin': [side+'Shin', side+'LowerLeg', side+'Leg', side+'Knee'],
            side+'Foot': [side+'Foot', side+'Ankle'],
            side+'ToeEnd': [side+'ToeEnd', side+'ToeBase_EndSite', side+'Toes_EndSite', side+'ToeBase', side+'Toes'],
        })
    for canonical in HUMAN_PARENTS:
        if canonical in overrides:
            selected = overrides[canonical]
            if selected not in names:
                raise ValueError(f'Mapping {canonical} refers to missing joint {selected}.')
            result[canonical] = selected
            continue
        candidates = [CMU_ENDPOINTS[canonical]] if cmu else aliases[canonical]
        for candidate in candidates:
            matches = lookup.get(_normalized(candidate), [])
            if len(matches) > 1:
                raise ValueError(f'Ambiguous joint {candidate}; use an explicit name mapping JSON.')
            if matches:
                result[canonical] = matches[0]; break
    # Compact human skeletons can use Head as the neck landmark. This does
    # not substitute any of the 16 required leg/arm/foot solver markers.
    if 'Neck1' not in result and 'Head' in result: result['Neck1'] = result['Head']
    missing = set(HUMAN_PARENTS) - set(result)
    if missing:
        raise ValueError('사람 스켈레톤의 F26 랜드마크 매핑이 필요합니다: '+', '.join(sorted(missing))+'. 이름 매핑 JSON을 지정하세요.')
    for side in ('Left', 'Right'):
        if result[side+'Leg'] == result[side+'Shin']:
            raise ValueError(f'{side} hip and knee resolve to the same joint; supply a name mapping.')
    return result


def canonical_clip(clip, mapping, *, up_axis='Y'):
    """Re-express mapped human joints with an anatomical hierarchy, preserving FK.

    Global marker poses are unchanged, apart from the explicit up-axis basis
    conversion. This makes heterogeneous source hierarchies share one adapter.
    """
    validate_clip(clip)
    if up_axis not in {'Y', 'Z'}:
        raise ValueError('Source up-axis must be Y or Z.')
    positions, quats = clip.global_transforms()
    names = list(HUMAN_PARENTS)
    ids = [clip.names.index(mapping[name]) for name in names]
    positions, quats = positions[:, ids], quats[:, ids]
    if up_axis == 'Z':
        basis = Rotation.from_euler('x', -90., degrees=True)
        positions = basis.apply(positions.reshape(-1, 3)).reshape(positions.shape)
        quats = (basis * Rotation.from_quat(quats.reshape(-1, 4)) * basis.inv()).as_quat().reshape(quats.shape)
    parents = np.asarray([-1 if HUMAN_PARENTS[name] is None else names.index(HUMAN_PARENTS[name]) for name in names], np.int32)
    local_pos, local_quat = positions.copy(), quats.copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            rp = Rotation.from_quat(quats[:, parent])
            local_pos[:, j] = rp.inv().apply(positions[:, j] - positions[:, parent])
            local_quat[:, j] = (rp.inv() * Rotation.from_quat(quats[:, j])).as_quat()
    return validate_clip(MotionClip(names, parents, local_pos[0], [[] for _ in names], local_pos, local_quat, clip.fps))


@dataclass(frozen=True)
class ExternalHumanMotion:
    path: Path
    clip: MotionClip
    metadata: dict
    source_files: tuple[Path, ...]

    @property
    def frames(self): return self.clip.frame_count

    @property
    def fps(self): return self.clip.fps


def load_external_motion(path, *, skeleton_path=None, source_fps=120., length_scale=None,
                         up_axis='Y', mapping_path=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f'Human motion was not found: {path}')
    suffix = path.suffix.lower()
    files, cmu = [path], suffix in {'.amc', '.asf'}
    if cmu:
        if suffix == '.asf':
            raise ValueError('모션 AMC를 선택하고 스켈레톤 ASF를 함께 지정하세요.')
        if skeleton_path:
            asf = Path(skeleton_path).expanduser().resolve()
        else:
            candidates = [path.with_name(path.stem.split('_')[0]+'.asf'), path.with_suffix('.asf')]
            candidates = list(dict.fromkeys(p for p in candidates if p.is_file()))
            if not candidates:
                candidates = list(path.parent.glob('*.asf'))
            if len(candidates) != 1:
                raise ValueError('CMU AMC에는 해당 피험자의 ASF 스켈레톤이 필요합니다.')
            asf = candidates[0]
        raw, scale = load_asf_amc(asf, path, fps=source_fps, length_scale=length_scale)
        files.append(asf)
    elif suffix == '.bvh':
        scale = .01 if length_scale is None else length_scale
        raw = load_bvh(path, length_scale=scale)
    elif suffix == '.npz':
        if length_scale not in (None, 1.):
            raise ValueError('Human NPZ fields ending in _m already use meters; choose automatic units.')
        scale, raw = 1., load_human_npz(path)
    else:
        raise ValueError('지원 형식: CMU ASF/AMC, BVH, 이름·계층을 가진 사람 모션 NPZ.')
    overrides = json.loads(Path(mapping_path).expanduser().read_text()) if mapping_path else {}
    if mapping_path: files.append(Path(mapping_path).expanduser().resolve())
    mapping = landmark_mapping(raw.names, cmu=cmu, overrides=overrides)
    clip = canonical_clip(raw, mapping, up_axis=up_axis)
    metadata = dict(format='CMU ASF/AMC' if cmu else suffix[1:].upper(),
                    source_joint_names=raw.names, landmark_mapping=mapping, meters_per_source_unit=scale,
                    input_up_axis=up_axis, normalized_up_axis='Y', normalized_units='meters',
                    source_fps=raw.fps, source_frames=raw.frame_count,
                    source_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    return ExternalHumanMotion(path, clip, metadata, tuple(files))
