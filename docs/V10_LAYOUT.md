# SuperMimic V10 workspace layout

This repository contains both the original G1 path and the V10 path selected
with `--robot v10`. The imported Video2Robot code is an offline source adapter;
the trainer consumes only accepted V10 reference NPZ files or their compact library.

| Path | State | Role |
| --- | --- | --- |
| `assets/robots/humanoid_v10/urdf/` | source copied | Original V10 URDF (`humanoid_v9.urdf` in the source folder, robot name V10). |
| `assets/robots/humanoid_v10/meshes/` | local copy, Git-ignored | Meshes referenced by the V10 URDF. |
| `assets/robots/humanoid_v10/mjcf/` | generated, static compile checked | Floating-base MuJoCo robot with ground/contact geoms. |
| `tools/retargeting/apps/` | V10 sample retarget verified | Human BVH import, PyRoki, collision/contact checks, NPZ export. |
| `configs/robots/humanoid_v10/` | control profile, PyRoki config, V10 output schema ready | V9 actuator gains/limits mapped to all V10 joints. |
| `configs/sources/bones_seed/` | metadata filter documented; direct SOMA-to-V10 mapping verified | SOMA BVH source, paper-name approximation, hand/object exclusions. |
| `src/supermimic/robots/v10/` | asset integrated | V10 body mapping and V9-derived joint actuator configuration. |
| `data/v10/` | metadata, full SOMA archive, prefilter manifest, accepted clips, and compact pilot library | Large files are Git-ignored; see `docs/V10_RETARGETING.md`. |

## Remaining work

1. Run resumable V10 PyRoki conversion on the full metadata candidate set,
   constrained by processing time and disk capacity.
2. Pack the final accepted set into the compact library and train long enough
   to measure policy coverage; the brief integration smoke did not learn the clips.
3. Evaluate every **training** clip from its first frame with V10 physics and
   record completion, return, and error. A held-out set is optional for the
   user's stated goal of reproducing the motions used for training.
