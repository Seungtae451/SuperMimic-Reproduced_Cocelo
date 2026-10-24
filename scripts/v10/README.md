# V10 pipeline entry points

The V10 path uses BONES-SEED G1 CSV -> calibrated V10 mapping for the fast
library build; direct SOMA BVH -> V10 PyRoki remains the fallback for motions
that the fast path rejects. Both feed the same compact V10 library and the same
SuperMimic world-model and policy training algorithm with V10 dimensions and
actuators. The G1 path remains selectable with the default `--robot g1`.
V10 retains the V9-derived PD gains and effort limits. Its policy action is
bounded to `[-1, 1]`, scaled to cover each joint's full URDF range, and
clamped to the corresponding joint limit before execution. A linear plus
cubic map preserves fine control near zero while still reaching the full
joint range at the extremes. The G1 action scale alone would restrict
several V10 arm joints too much.
The V10 reward also omits the soft joint-limit penalty because retargeted
references may legitimately reach the hard limit (already enforced by the
action clamp), and uses a smaller action-rate weight for normalized actions.
The simulator and learned-model reward use the same V10 weights.
The V10 policy starts from the current reference joint angles and learns a
bounded residual for balance. It analytically inverts the action map, so this
prior is exactly the retargeted pose at initialization. The G1 policy remains
the original direct-action decoder.
See `docs/V10_TRAINING_STATUS.md` for the current physics gate and measured
policy results. The experimental `foot_tilt=20` retarget config and its three
complete NPZ/JSON outputs are preserved under
`configs/robots/humanoid_v10/bones_seed_pyroki_foot_tilt20.json` and
`data/v10/contact_trial_ft20/`; the default 16 accepted motions are unchanged.

`build_mjcf.py` generates the V10 MuJoCo model. `fetch_bones_seed.py` fetches
gated metadata and the SOMA Uniform archive. `prefilter_bones_seed.py` writes
the metadata candidate manifest, excluding any motion with an identified
person-object interaction, regardless of which body part contacts the object.
It checks annotated props, categories, motion names, and descriptions because
some object motions have `content_props=0`. `extract_selected_bvh.py` is
only for individual previews. The active resumable fast route is:

```bash
# In data/v10/fast_g1/venv, after G1 archive download and calibration:
data/v10/fast_g1/venv/bin/python -B scripts/v10/process_fast_g1_to_v10.py --workers 8
# In the SuperMimic project environment:
uv run python scripts/v10/build_compact_library.py
uv run sm-train --robot v10 \
  --data.library data/v10/library/accepted_compact.npy \
  --run-name v10_bones_seed
uv run python scripts/eval_benchmark.py \
  --ckpt logs/v10_bones_seed/ckpt_latest.pt \
  --library data/v10/library/accepted_compact.npy
```

`extract_g1_calibration_pairs.py` extracts 162 G1 CSVs matching existing
direct V10 results; `calibrate_fast_g1_to_v10.py` fits a regularized per-joint
linear mapping from 152 exact-length pairs and reports held-out error.
`process_fast_g1_to_v10.py` streams eligible G1 CSVs once, uses eight bounded
CPU workers, checks V10 limits, ground/self collision and velocity, and tries
bounded collision projection for small penetrations. Accepted motions append
to the same compact stream and main SQLite index as the direct route. Fast
rejects are recorded separately in `data/v10/fast_g1/process.sqlite`, so they
can be retried by direct SOMA retargeting. The active user service is
`v10-fast-g1-convert-8-v2.service`, limited to 24 GB memory; inspect with
`systemctl --user status v10-fast-g1-convert-8-v2` and
`journalctl --user -u v10-fast-g1-convert-8-v2 -f`.
The 68-motion stratified benchmark accepted 43 after calibration and collision
projection in 5.3 seconds with four CPU workers. This geometric check does
not verify dynamic balance or policy tracking.

The older two-GPU converter streams the SOMA archive once and assigns three PyRoki workers
to each GPU. A seven-motion comparison took 100 seconds with one worker per
GPU, 80 seconds with two, 71 seconds with three, and 66 seconds with four.
The eight-worker sample fit in RAM, but sustained retargeting exhausted the
system's 2 GB swap; the six-worker direct run was also paused after the PC
became unresponsive. It records
each clip's quality decision in
`data/v10/manifests/v10_process.sqlite` and appends accepted frames to
`data/v10/library/accepted_stream.f16`; temporary BVH and NPZ files are
removed after each clip. It resumes after interruption and stops below 10 GB
free space. Every 1,000 newly accepted clips it refreshes the trainable
`accepted_compact.npy` and JSON index, and publishes one final snapshot when
the archive scan completes. Keep the source
archive until all desired clips have been processed and the final compact
library has been verified. `process_bones_seed.py` remains available for
single-GPU debugging and named NPZ exports.

The full direct conversion remains paused after a workstation freeze and
force-restart. Its resumable user service is `v10-bones-seed-convert.service`.
Check it with
`systemctl --user status v10-bones-seed-convert` and follow
`logs/v10_bones_seed_parallel/convert.log`. If fallback is needed, rebuild its
PyRoki environment and use a smaller worker count with a memory limit; the
SQLite index prevents accepted motions from being converted twice.

For the separate G1 fast-path feasibility pilot, `fetch_g1_seed.py` downloads
the released G1 archive and `pilot_fast_g1_to_v10.py` measures CPU mapping,
joint-limit distortion, exact MuJoCo ground/self-collision checks, and pose
error on any named direct-V10 counterparts. It does not export motions to the
training library. Simple angle copying passed only 4/68 preliminary strict
cases; the calibrated production converter above is a separate, improved
method. Reports and selected archive paths are under `data/v10/fast_g1/`.

`check_dynamic_tracking.py` replays references with the V9-derived PD
controller as a physics preflight. It is not policy evaluation: the first 16
accepted clips all lost balance in this baseline. Use `eval_benchmark.py` on a
trained checkpoint to measure whether a learned policy follows each clip from
its first frame without a height or tilt termination.
`audit_initial_support.py` writes `data/v10/manifests/v10_initial_support.csv`
with the starting center-of-mass margin to the near-ground foot mesh and a
wider, potentially reachable sole area. This is a static diagnostic, not a
filter for dynamic motions. In the first 16 clips, 15 start with the center of
mass outside the near-ground contact polygon; the original idle/step clip has
a -0.081 m margin. A trial increasing the PyRoki `foot_tilt` weight from 1 to
20 brought that idle/step starting margin to +0.071 m while retaining the
geometry and motion-quality pass, although PD replay still fell after 68/254
frames. The retargeted reference must therefore be tested by a learned policy
before expanding the dataset.

The 15-iteration, four-clip V10 training run was an integration smoke test:
the training and benchmark paths execute, but its four motions all failed the
physics-based evaluation. Kinematic preview quality is not policy success.
A later 1,000-iteration 15-clip pilot also failed all 15 clips; it used the
former, too-small V10 action scale. Check the bounded, nonlinear full-range
action in a separate pilot before committing to a long, full-library run. On
systems where initial PyTorch compilation is slow, pass
`--wm-trainer.compile None --policy-trainer.compile None` for pilot runs.
The first 1,000-iteration V10 run with corrected reward reached 76/254 frames
on an idle/step clip in deterministic physics. The reference-prior policy's
best deterministic benchmark reached 123/254 frames at 3,000 training
iterations. Extending its imagined policy horizon from 8 to 32 reached
105/254 frames at iteration 4,000. Neither completed the motion. The first
full-clip policy result came from a corrected 96-frame shoulder gesture:
`apply_root_pitch_trial.py` pitches and grounds the reference, and a fresh
0.05-residual policy completed 96/96 frames. `search_root_pitch.py` records
bounded root-pitch PD replay sweeps; a fine sweep also produced a 112-frame
`just_realised` reference that passes PD replay. These are trial references,
not archive-wide retargeting rules. The short policy runs are integration
pilots; incomplete motions in these runs do not establish the limits of a
12-hour run on the two RTX 5080 GPUs. Next, measure retargeting throughput and acceptance on a
category-balanced slice, build the larger V10 library, and evaluate a longer
single-policy run on both trained and held-out motions. A short pilot does not
have to complete every clip before archive conversion starts.
The target long run is 12 hours on two RTX 5080 GPUs. A two-rank torchrun
smoke test has completed distributed updates and saved a checkpoint. Before
the long run, benchmark one rank's memory with the larger compact library:
each GPU stores its own library copy, and the cards have 16 GB each.
