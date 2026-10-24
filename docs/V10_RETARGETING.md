# BONES-SEED → Humanoid V10 status

The SOMA BVH is retargeted **directly** to the V10 MuJoCo model by the
Video2Robot PyRoki trajectory solver. G1 motion files are not an intermediate
representation. The metadata prefilter uses the reproduction's approximate
paper-name exclusions plus V10's no-hand/no-object rule. It does not reproduce
the paper's unpublished list of 153 G1 feasibility failures.

The full `soma_uniform.tar.gz` is already present under `data/v10/downloads`.
It is one gzip stream, so the entire archive had to be downloaded. The
extractor scans the archive and writes only selected candidate BVHs; it does
not unpack all motions.

## Reproduce the pilot

Create a Python environment with `scripts/v10/requirements-retarget.txt`,
then run from the repository root:

```bash
python scripts/v10/extract_selected_bvh.py --name-contains walk --limit 1
python scripts/v10/retarget_bones_seed.py \
  data/v10/sample_bvh/soma_uniform/bvh/221118/injured_R_leg_walk_ff_start_180_002__A070.bvh
python scripts/v10/audit_retargeted.py --link-accepted
```

To sample additional categories without unpacking the archive, add repeated
`--category` filters and `--one-per-category` to the extraction command.

`--max-source-frames 120` runs a short prefix in a separate `__pilot_120`
output directory. Omit it for a full clip. The retargeter emits a 50 Hz,
nine-key named NPZ plus CSV and JSON diagnostics. The audit rejects prefixes,
schema mismatches, joint-limit violations, nonfinite values, geometry failures,
position RMS above 0.10 m, or joint speed above 20 rad/s. These thresholds
are pilot screening settings, not proven dynamic limits. Passing clips get
links in `data/v10/accepted`, suitable as a flat motion input directory once
the V10 training environment is ported.

## Verified on the current pilot

Four complete clips from basic locomotion, gestures, and sports produced
112–314 output frames each at 50 Hz. The V10 output has 26 controlled joints
and 27 robot bodies. Their position RMS is 0.029–0.043 m; no joint-limit,
ground-penetration, or non-adjacent self-collision violation was detected.
One deliberately shortened pilot was rejected as `partial_clip`. Peak joint
speeds are 3.7–11.9 rad/s, but peak accelerations reach 312 rad/s² in one
clip, so actuation feasibility needs a dynamic rollout. This is **kinematic**
validation. Balance under actuators, contact dynamics, tracking-policy success,
and generalization to the remaining dataset remain untested.

The PyRoki URDF kinematic tree was checked against the compiled V10 MuJoCo
model at ten random joint configurations: maximum link-origin difference was
2.1e-7 m and maximum orientation difference was 4.6e-7 rad. The old F26 foot
marker offsets align with the V10 foot collision mesh: its forward toe region
is roughly 0.096 m from the ankle-roll origin in the local marker axis.


## Side-by-side video preview

Run `MUJOCO_GL=egl python scripts/v10/render_retarget_preview.py` from the
repository root. The script uses the accepted complete clips by default and
writes `data/v10/videos/bones_seed_v10_retarget_preview.mp4` plus a PNG
thumbnail. The current preview has four clips, 428 frames at 25 fps
(17.12 seconds, 1280×720 H.264). The left figure is the SOMA BVH skeleton,
scaled and root-centered for visual pose comparison; the right is the V10
retargeted reference played through MuJoCo forward kinematics. This video
is **not** a simulated motor-control rollout or a trained policy demonstration.

## V10 training and scaling status

The V10 robot asset and V9-derived PD actuators are now selectable with
`--robot v10`. The environment, recorder, world model, policy, and benchmark
use 26 joints and a 65-dimensional floating-base state. A 15-iteration
four-clip smoke test completed all training phases and wrote a checkpoint.
The checkpoint failed all four training clips in the physics-based benchmark;
the run establishes integration, not learned motion coverage.

`scripts/v10/check_dynamic_tracking.py` replays reference joint targets with
the V9 gains, without a learned balance policy. All four initial clips fell
after 44–59 frames; this is a baseline, not a claim that the motions are
unlearnable. `scripts/v10/process_bones_seed.py` now streams the archive,
retargets one clip at a time, records accepted/rejected/error statuses in
SQLite, and resumes after interruption. The initial 12-clip processing batch
accepted 12 entries, including one previously converted clip, leaving 15
distinct accepted references across six metadata categories.

`scripts/v10/build_compact_library.py` packs accepted references into an fp16
memory-mapped 65-column library. V10 full and compact references were checked
against each other on CPU, and the compact library was replayed in two GPU
environments. The full 106,924-candidate conversion has **not** run yet;
the current per-clip retarget cost is tens of seconds and the internal disk
has limited remaining capacity. See `scripts/v10/README.md` for the commands.
