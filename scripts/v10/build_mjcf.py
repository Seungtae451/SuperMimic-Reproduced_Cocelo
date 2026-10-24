"""Compile the V10 URDF into a floating-base MuJoCo model for offline retargeting.

The robot's unmodified URDF and STL files remain the source of truth. MuJoCo
limits an STL to 200,000 facets, so existing *_modified.STL counterparts are
selected only for links whose original mesh exceeds that limit.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


def _facet_count(path: Path) -> int:
    with path.open('rb') as stream:
        header = stream.read(84)
    if len(header) != 84:
        raise ValueError(f'Invalid binary STL: {path}')
    return struct.unpack('<I', header[80:84])[0]


def build(urdf: Path, meshes: Path, control: Path, output: Path) -> dict:
    urdf = urdf.resolve()
    meshes = meshes.resolve()
    output = output.resolve()
    profile = json.loads(control.read_text())
    robot = ET.parse(urdf).getroot()
    assert robot.tag == 'robot' and robot.get('name') == 'humanoid_v10'

    substitutions: dict[str, str] = {}
    imported_text = urdf.read_text()
    for geom in robot.findall('.//geometry/mesh'):
        ref = geom.attrib['filename']
        prefix = 'package://humanoid_v10/meshes/'
        if not ref.startswith(prefix):
            raise ValueError(f'Unexpected mesh reference: {ref}')
        name = ref.removeprefix(prefix)
        original = meshes / name
        selected = original
        if _facet_count(original) > 200_000:
            selected = meshes / name.replace('.STL', '_modified.STL')
            if not selected.is_file() or _facet_count(selected) > 200_000:
                raise ValueError(f'No MuJoCo-compatible mesh for {name}')
        substitutions[name] = selected.name
        imported_text = imported_text.replace(ref, str(selected))

    with tempfile.TemporaryDirectory(prefix='v10-mjcf-') as scratch:
        imported = Path(scratch) / 'v10_import.urdf'
        raw_mjcf = Path(scratch) / 'v10_compiled.xml'
        imported.write_text(imported_text)
        compiled = mujoco.MjModel.from_xml_path(str(imported))
        mujoco.mj_saveLastXML(str(raw_mjcf), compiled)
        mjcf = ET.parse(raw_mjcf).getroot()

    for mesh in mjcf.findall('./asset/mesh'):
        mesh.set('file', os.path.relpath(mesh.attrib['file'], output.parent))

    world = mjcf.find('worldbody')
    if world is None:
        raise ValueError('MuJoCo conversion had no worldbody')
    original_children = list(world)
    if not original_children or original_children[0].tag != 'geom':
        raise ValueError('Unexpected converted pelvis layout')
    for child in original_children:
        world.remove(child)
    ET.SubElement(world, 'geom', {
        'name': 'ground', 'type': 'plane', 'size': '0 0 0.1',
        'friction': '1 0.005 0.0001', 'rgba': '0.85 0.85 0.85 1',
    })
    pelvis = ET.SubElement(world, 'body', {'name': 'pelvis_link', 'pos': '0 0 0.75'})
    ET.SubElement(pelvis, 'freejoint', {'name': 'root'})
    source_pelvis = robot.find("./link[@name='pelvis_link']/inertial")
    if source_pelvis is None:
        raise ValueError('Missing pelvis inertia')
    inertial_origin = source_pelvis.find('origin')
    tensor = source_pelvis.find('inertia')
    mass = source_pelvis.find('mass')
    if inertial_origin is None or tensor is None or mass is None:
        raise ValueError('Incomplete pelvis inertia')
    if inertial_origin.get('rpy', '0 0 0') != '0 0 0':
        raise ValueError('Pelvis inertia orientation needs explicit conversion')
    ET.SubElement(pelvis, 'inertial', {
        'pos': inertial_origin.attrib['xyz'], 'mass': mass.attrib['value'],
        'fullinertia': ' '.join(tensor.attrib[key] for key in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')),
    })
    for child in original_children:
        pelvis.append(child)

    for body in pelvis.iter('body'):
        geoms = body.findall('geom')
        if len(geoms) != 1:
            raise ValueError(f'{body.get("name")}: expected exactly one collision geom')
        geoms[0].set('name', f'{body.attrib["name"]}_collision')
        geoms[0].set('contype', '1')
        geoms[0].set('conaffinity', '1')
    pelvis_geom = pelvis.find('geom')
    if pelvis_geom is None:
        raise ValueError('Missing pelvis geom')
    pelvis_geom.set('name', 'pelvis_link_collision')
    pelvis_geom.set('contype', '1')
    pelvis_geom.set('conaffinity', '1')

    mapped = set()
    for joint in mjcf.findall('.//joint'):
        name = joint.get('name')
        if name not in profile['joints']:
            raise ValueError(f'Unmapped joint {name}')
        effort = float(profile['joints'][name]['effort_limit_nm'])
        joint.set('actuatorfrcrange', f'{-effort:g} {effort:g}')
        mapped.add(name)
    if mapped != set(profile['joints']):
        raise ValueError(f'Joint mismatch: missing {set(profile["joints"]) - mapped}')

    # Stabilize the zero pose with both soles near the ground. This is only
    # a nominal model pose; each retargeted clip carries its own root state.
    draft = ET.tostring(mjcf, encoding='unicode')
    # Recompile from the output location so relative mesh references resolve.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(draft)
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ground = model.geom('ground').id
    distances = [float(mujoco.mj_geomDistance(
        model, data, model.geom(f'{side}_ankle_roll_link_collision').id,
        ground, 2.0, np.zeros(6))) for side in ('left', 'right')]
    pelvis.set('pos', f'0 0 {0.75 - min(distances) + 0.005:.7f}')
    output.write_text(ET.tostring(mjcf, encoding='unicode'))

    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    neutral_overlaps: set[tuple[str, str]] = set()
    for contact in data.contact:
        if contact.dist >= -1e-4 or ground in (contact.geom1, contact.geom2):
            continue
        first = model.body(int(model.geom_bodyid[contact.geom1])).name
        second = model.body(int(model.geom_bodyid[contact.geom2])).name
        if first != second:
            neutral_overlaps.add(tuple(sorted((first, second))))
    if neutral_overlaps:
        excludes = ET.SubElement(mjcf, 'contact')
        for first, second in sorted(neutral_overlaps):
            ET.SubElement(excludes, 'exclude', {'body1': first, 'body2': second})
        output.write_text(ET.tostring(mjcf, encoding='unicode'))
        model = mujoco.MjModel.from_xml_path(str(output))
    if model.njnt != 27 or model.nq != 33 or model.nv != 32:
        raise ValueError(f'Unexpected model dimensions: {model.njnt}, {model.nq}, {model.nv}')
    return {
        'output': str(output), 'nbody': int(model.nbody), 'njnt': int(model.njnt),
        'nq': int(model.nq), 'nv': int(model.nv), 'ngeom': int(model.ngeom),
        'pelvis_mass_kg': float(model.body('pelvis_link').mass[0]),
        'mesh_substitutions': {k: v for k, v in substitutions.items() if k != v},
        'neutral_collision_exclusions': sorted(neutral_overlaps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path.cwd()
    parser.add_argument('--urdf', type=Path, default=root / 'assets/robots/humanoid_v10/urdf/humanoid_v10.urdf')
    parser.add_argument('--meshes', type=Path, default=root / 'assets/robots/humanoid_v10/meshes')
    parser.add_argument('--control', type=Path, default=root / 'configs/robots/humanoid_v10/control_profile.json')
    parser.add_argument('--output', type=Path, default=root / 'assets/robots/humanoid_v10/mjcf/humanoid_v10.xml')
    args = parser.parse_args()
    print(json.dumps(build(args.urdf, args.meshes, args.control, args.output), indent=2))


if __name__ == '__main__':
    main()
