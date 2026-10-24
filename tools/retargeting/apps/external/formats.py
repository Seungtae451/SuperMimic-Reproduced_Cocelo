"""Human mocap readers. All clips use meters and XYZW local rotations.

CMU ASF/AMC: https://mocap.cs.cmu.edu/faqs.php (inches / ASF length unit).
ASF transforms: https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
The ASF clip represents bone endpoints, so its joint labels need endpoint-aware
semantic mapping (e.g. femur endpoint is the knee, not the hip).
"""

from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from apps.humanoid_retargeter.motion_clip import MotionClip
from apps.humanoid_retargeter.motion_io import load_visualization_motion


def validate_clip(clip):
    j = len(clip.names)
    if clip.local_positions_m.ndim != 3 or clip.local_rotations.ndim != 3:
        raise ValueError('Human local positions and quaternions must be rank-three arrays.')
    t = clip.frame_count
    if j < 1 or t < 2 or len(set(clip.names)) != j:
        raise ValueError('Human motion requires unique joint names and at least two frames.')
    if np.shape(clip.parents) != (j,) or sum(clip.parents < 0) != 1:
        raise ValueError('Expected a connected human skeleton with one root.')
    for i, parent in enumerate(clip.parents):
        if parent < -1 or parent >= i:
            raise ValueError('Skeleton parents must precede children, with -1 for the root.')
    if clip.local_positions_m.shape != (t, j, 3) or clip.local_rotations.shape != (t, j, 4):
        raise ValueError('Human positions/quaternions must have shape (T,J,3)/(T,J,4).')
    if not np.isfinite(clip.fps) or clip.fps <= 0 or not all(np.isfinite(a).all() for a in (clip.local_positions_m, clip.local_rotations)):
        raise ValueError('Human motion contains invalid fps, NaN or Inf.')
    if np.any(np.linalg.norm(clip.local_rotations, axis=-1) < 1e-8):
        raise ValueError('Human motion contains zero quaternions.')
    return clip


def load_bvh(path, *, length_scale=.01):
    """Read arbitrary BVH hierarchy/channel orders; units are explicit meters/unit."""
    tokens = re.findall(r'[{}]|[^\s{}]+', Path(path).read_text(encoding='utf-8-sig'))
    cursor = 0
    def pop(expected=None):
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError('Truncated BVH file.')
        token = tokens[cursor]; cursor += 1
        if expected is not None and token.lower() != expected.lower():
            raise ValueError(f'Expected BVH {expected}, got {token}.')
        return token
    names, parents, offsets, channels = [], [], [], []
    def node(parent, end_site=False):
        name = names[parent] + '_EndSite' if end_site else pop()
        if end_site and name in names:
            name += f'_{len(names)}'
        index = len(names)
        names.append(name); parents.append(parent); offsets.append(None); channels.append([])
        pop('{')
        while True:
            token = pop()
            if token == '}':
                break
            if token.upper() == 'OFFSET':
                offsets[index] = [float(pop()) for _ in range(3)]
            elif token.upper() == 'CHANNELS':
                channels[index] = [pop() for _ in range(int(pop()))]
            elif token.upper() == 'JOINT':
                node(index)
            elif token.upper() == 'END':
                pop('Site'); node(index, True)
            else:
                raise ValueError(f'Unsupported BVH hierarchy token: {token}.')
        if offsets[index] is None:
            raise ValueError(f'Missing BVH offset for {name}.')
    pop('HIERARCHY'); pop('ROOT'); node(-1)
    pop('MOTION'); pop('Frames:'); count = int(pop())
    pop('Frame'); pop('Time:'); dt = float(pop())
    width = sum(map(len, channels))
    data = np.asarray([float(v) for v in tokens[cursor:]], dtype=np.float64)
    if count < 2 or len(data) != count * width or dt <= 0 or not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError('Invalid BVH frame/channel count, frame time or length scale.')
    data = data.reshape(count, width)
    offsets = np.asarray(offsets, dtype=np.float64) * length_scale
    positions = np.tile(offsets, (count, 1, 1))
    rotations = np.tile([0., 0., 0., 1.], (count, len(names), 1))
    cursor = 0
    for j, channel_names in enumerate(channels):
        axes, angles = [], []
        for channel in channel_names:
            axis = channel[0].upper()
            if axis not in 'XYZ':
                raise ValueError(f'Invalid BVH channel {channel}.')
            if channel.lower().endswith('position'):
                positions[:, j, 'XYZ'.index(axis)] += data[:, cursor] * length_scale
            elif channel.lower().endswith('rotation'):
                axes.append(axis); angles.append(data[:, cursor])
            else:
                raise ValueError(f'Unsupported BVH channel {channel}.')
            cursor += 1
        if axes:
            if len(set(axes)) != len(axes):
                raise ValueError('Repeated BVH rotation axes are unsupported.')
            # BVH Euler channels rotate about the evolving local axes.
            rotations[:, j] = Rotation.from_euler(''.join(axes), np.stack(angles, axis=1), degrees=True).as_quat()
    return validate_clip(MotionClip(names, np.asarray(parents, np.int32), offsets, channels, positions, rotations, 1 / dt))


def _lines(path):
    return [line.split('#', 1)[0].strip() for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.split('#', 1)[0].strip()]


def load_asf_amc(asf, amc, *, fps=120., length_scale=None):
    sections, section = {}, None
    for line in _lines(asf):
        if line.startswith(':'):
            section = line.split()[0].lower(); sections[section] = []
        elif section:
            sections[section].append(line)
    def entries(lines):
        return {parts[0].lower(): parts[1:] for line in lines if (parts := line.split())}
    units, root = entries(sections.get(':units', [])), entries(sections.get(':root', []))
    asf_length_unit = float(units.get('length', ['1'])[0])
    if not np.isfinite(asf_length_unit) or asf_length_unit <= 0:
        raise ValueError('ASF length unit must be positive and finite.')
    scale = .0254 / asf_length_unit if length_scale is None else float(length_scale)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('CMU fps and meters/unit must be positive and finite.')
    degrees = units.get('angle', ['deg'])[0].lower() in {'deg', 'degrees'}
    bones, block = {}, None
    for line in sections.get(':bonedata', []):
        if line.lower() == 'begin':
            block = []
        elif line.lower() == 'end':
            item = entries(block or [])
            name = item['name'][0]
            if name in bones or name == 'root':
                raise ValueError(f'Duplicate ASF bone {name}.')
            bones[name] = item; block = None
        elif block is not None:
            block.append(line)
    parent_names = {}
    for line in sections.get(':hierarchy', []):
        if line.lower() in {'begin', 'end'}:
            continue
        parent, *children = line.split()
        for child in children:
            if child in parent_names:
                raise ValueError(f'Multiple ASF parents for {child}.')
            parent_names[child] = parent
    names = ['root']
    while len(names) <= len(bones):
        more = [name for name in bones if name not in names and parent_names.get(name) in names]
        if not more:
            if len(names) == len(bones) + 1:
                break
            raise ValueError('ASF hierarchy is disconnected or cyclic.')
        names.extend(more)
    if set(parent_names) != set(bones):
        raise ValueError('ASF hierarchy must reference every bone exactly once.')
    parents = np.asarray([-1] + [names.index(parent_names[name]) for name in names[1:]], np.int32)
    dofs = [root.get('order', ['TX', 'TY', 'TZ', 'RX', 'RY', 'RZ'])] + [bones[name].get('dof', []) for name in names[1:]]
    dofs = [[d.lower() for d in row] for row in dofs]
    if any(d not in {'tx', 'ty', 'tz', 'rx', 'ry', 'rz'} or (j and d.startswith('t'))
           for j, row in enumerate(dofs) for d in row):
        raise ValueError('Unsupported ASF DOF; expected root translation and Euler rotations.')
    frames, frame, frame_ids, amc_degrees = [], None, [], degrees
    for line in _lines(amc):
        if line.startswith(':'):
            if line.upper() == ':DEGREES': amc_degrees = True
            elif line.upper() == ':RADIANS': amc_degrees = False
            continue
        if line.isdigit():
            if frame is not None: frames.append(frame)
            frame_ids.append(int(line)); frame = {}
        else:
            if frame is None: raise ValueError('AMC channel data precedes its frame number.')
            name, *values = line.split()
            if name not in names or name in frame:
                raise ValueError(f'Unknown or duplicate AMC bone {name}.')
            frame[name] = [float(v) for v in values]
    if frame is not None: frames.append(frame)
    if len(frames) < 2 or np.any(np.diff(frame_ids) != 1):
        raise ValueError('AMC requires at least two consecutive frames; fps is supplied separately.')
    n, j = len(frames), len(names)
    pos = np.zeros((n, j, 3)); quat = np.tile([0., 0., 0., 1.], (n, j, 1)); offsets = np.zeros((j, 3))
    for index, (name, channels) in enumerate(zip(names, dofs)):
        for f in frames:
            if len(f.get(name, [])) != len(channels):
                raise ValueError(f'AMC frame has missing/wrong channels for {name}.')
        values = np.asarray([f.get(name, []) for f in frames]).reshape(n, len(channels))
        if index == 0:
            orientation, axis = root.get('orientation', ['0', '0', '0']), root.get('axis', ['XYZ'])[0]
            offsets[index] = np.asarray(root.get('position', [0, 0, 0]), dtype=float) * scale
        else:
            bone = bones[name]
            orientation, axis = bone.get('axis', ['0', '0', '0', 'XYZ'])[:3], bone.get('axis', ['0', '0', '0', 'XYZ'])[3]
            offsets[index] = np.asarray(bone['direction'], dtype=float) * float(bone['length'][0]) * scale
        basis = Rotation.from_euler(axis.lower(), np.asarray(orientation, dtype=float), degrees=degrees)
        axes = ''.join(ch[1] for ch in channels if ch.startswith('r'))
        angles = values[:, [i for i, ch in enumerate(channels) if ch.startswith('r')]]
        motion = Rotation.from_euler(axes, angles, degrees=amc_degrees) if axes else Rotation.identity(n)
        rotation = basis * motion * basis.inv()
        quat[:, index] = rotation.as_quat()
        # A bone endpoint moves by its own rotation about its proximal joint.
        pos[:, index] = rotation.apply(offsets[index]) if index else offsets[index]
        if not index:
            for i, ch in enumerate(channels):
                if ch.startswith('t'): pos[:, 0, 'xyz'.index(ch[1])] += values[:, i] * scale
    clip = validate_clip(MotionClip(names, parents, offsets, dofs, pos, quat, float(fps)))
    return clip, scale


def load_human_npz(path):
    with np.load(path, allow_pickle=False) as a:
        if 'parents' in a and (a['parents'].ndim != 1 or a['parents'].dtype.kind not in 'iu'):
            raise ValueError('Human skeleton parents must be an integer vector.')
        if {'names', 'parents', 'local_positions_m', 'local_rotations_xyzw', 'offsets_m', 'fps'}.issubset(a.files):
            return validate_clip(load_visualization_motion(path))
        required = {'joint_names', 'parents', 'joint_positions_m', 'fps'}
        if not required.issubset(a.files):
            raise ValueError('사람 NPZ에는 names/parents/local_positions_m/local_rotations_xyzw/fps 또는 joint_names/parents/joint_positions_m/fps가 필요합니다. F26 관절 궤적과 SMPL pose 파라미터는 사람 스켈레톤 입력이 아닙니다.')
        names, parents = a['joint_names'].tolist(), np.asarray(a['parents'], dtype=np.int32)
        positions = np.asarray(a['joint_positions_m'], dtype=np.float64)
        rotations = (np.asarray(a['joint_rotations_xyzw'], dtype=np.float64) if 'joint_rotations_xyzw' in a else
                     np.tile([0., 0., 0., 1.], (*positions.shape[:2], 1)))
        fps = float(a['fps'].item())
    if positions.ndim != 3 or positions.shape[1:] != (len(names), 3) or rotations.shape != (*positions.shape[:2], 4):
        raise ValueError('Expected human world joint positions (T,J,3) and optional world XYZW quaternions (T,J,4).')
    validate_clip(MotionClip(names, parents, np.zeros((len(names), 3)), [[] for _ in names], positions, rotations, fps))
    local_pos, local_quat = positions.copy(), rotations.copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            rp = Rotation.from_quat(rotations[:, parent])
            local_pos[:, j] = rp.inv().apply(positions[:, j] - positions[:, parent])
            local_quat[:, j] = (rp.inv() * Rotation.from_quat(rotations[:, j])).as_quat()
    return validate_clip(MotionClip(names, parents, local_pos[0], [[] for _ in names], local_pos, local_quat, fps))
