# Humanoid V10 BONES-SEED training status

The objective is to replay BONES-SEED training motions on Humanoid V10 with
one SuperMimic policy. These are V10 engineering pilots using
SuperMimic-Reproduced; their observations and conditions must not be treated
as paper-equivalent G1 results.

Final-data selection follows the user's robot constraints, not a fixed
category quota: exclude every motion that includes an object interacting with
the person, whether touched by a hand or another body part, as well as terrain
absent from the flat-ground V10 simulator and invalid V10
retargetings (joint, geometry, or non-finite failures). Retain every other
candidate, including references that fall under the untrained PD preflight.
Do not apply the experimental root-pitch/foot-grounding correction to the
main training library. A category-balanced slice is only for measuring
conversion throughput and quality before processing all eligible clips.

## Data and integration

- The SOMA archive is present locally. The metadata prefilter currently
  keeps 100,330 candidates and excludes annotated props and explicit
  person-object interactions before retargeting. It also checks motion names
  and descriptions because some object interactions have `content_props=0`.
  These are metadata decisions, not a guarantee that every unlabeled prop has
  been discovered.
- Direct SOMA BVH to V10 PyRoki retargeting, V10 MJCF/URDF and meshes,
  V9-derived PD gains, 26-joint policy/world-model support, and compact
  library packing are implemented.
- The original pilot contained 16 complete uncorrected motions and 5,531
  frames. The current `accepted_compact.npy` snapshot grows during the full
  conversion; read its JSON index for the current count. These motions are
  **not** dynamically certified.
- The full 100,330-candidate direct conversion is **paused** after the
  workstation became unresponsive and was force-restarted on 2026-10-24.
  Its resumable user service is `v10-bones-seed-convert.service`, configured
  with three PyRoki workers per RTX 5080 when running.
  On a seven-motion sample, one/two/three/four workers per GPU took about
  100/80/71/66 seconds respectively. Eight workers exhausted swap during
  sustained conversion, so the faster short sample is not the sustained choice.
  Passing frames are appended directly to `accepted_stream.f16`; failed and
  temporary motions do not consume permanent data space. The service refreshes
  the trainable compact library every 1,000 new accepted clips and at normal
  completion. Its decision database is resumable.
- The published BONES-SEED `g1.tar.gz` is also downloaded locally. A separate
  G1-to-V10 angle-mapping pilot with neutral-axis sign correction, joint-limit
  clamping, root-height alignment, and a vertical ground lift takes roughly
  0.05–0.75 seconds per tested clip on one CPU worker. This is **not** a
  production converter: among 30 eligible test motions, only 2 passed a
  conservative preliminary gate (no self-collision, speed <=20 rad/s,
  joint-limit excess <=0.05 rad, ground lift <=0.05 m). In 15 motions paired
  with direct V10 references, 4 passed that gate; the median pelvis-local
  body-position RMS difference was 0.055 m. The pilot does not yet check
  stance support or dynamic balance. Reports are under `data/v10/fast_g1/`.
  A broader 68-motion sample with four metadata candidates per category took
  160 seconds including a near-full 23.5 GB archive scan, with a median
  0.182 seconds of mapping and preliminary geometry checking per clip. Only
  4/68 met the same preliminary gate; 65/68 needed joint-limit clipping and
  24/68 had self-collision. Thus this simple mapping cannot supply the desired
  broad V10 library in a 30-minute preparation window even though the angle
  calculation itself is fast. The direct SQLite index and compact stream
  remained consistent after the force-restart (160 accepted database entries,
  45,256 committed stream frames); both background services are stopped.
- `scripts/v10/audit_initial_support.py` measures the first-pose COM margin
  to near-ground foot meshes. Fifteen of the 16 accepted motions start with
  a negative near-contact margin. This diagnostic must not be used as a hard
  filter for moving motions.

### Active fast conversion (2026-10-24)

The separate calibrated G1 CSV -> V10 route is now running as the user service
`v10-fast-g1-convert-8-v2.service` with eight CPU workers, a 24 GB cgroup memory
limit, and a 15 GB free-disk floor. The direct PyRoki service remains stopped.
The first 4-worker service was stopped after 3,572 decisions and resumed from
the same committed stream and SQLite index. A separate 1,000-motion test
finished in 29.4 seconds with eight workers using about 18 GB of memory.
After a secondary audit found 20 stored clips above the 20 rad/s adjacent-frame
speed gate, those clips were removed and the compact stream reindexed; the
current service checks this criterion before and after fp16 quantization.
The calibration uses 152 exact-length pairs from the existing direct V10
results. On 31 held-out pairs, joint RMS fell from 0.519 rad with direct
angle copying to 0.231 rad after calibration and V10 joint clipping. The
68-motion stratified production-method benchmark accepted 43, rejected 10 for
joint distortion, 10 for unresolved collision, 4 for excessive ground lift,
and 1 for speed; four workers finished it in 5.3 seconds. This is an
engineering acceptance rate on a small sample, not a full-dataset yield
estimate. Small self-collisions may be corrected by a bounded MuJoCo mesh
projection; larger or failed corrections are rejected.

The fast process streams eligible G1 CSVs from the local archive into the
same SQLite-indexed fp16 V10 library. It writes fast rejections to a separate
SQLite file so they remain available for later direct SOMA fallback. It does
not certify stance support, dynamic balance, or learned policy performance.
The compact `accepted_compact.npy` snapshot is refreshed every 10,000 newly
accepted clips and at normal completion. Check the live service and its logs
before assuming the full library is ready.

## Physics and policy gate

Every one of the first 16 accepted motions falls during V9-gain PD reference
replay, after roughly 42–84 frames. A PyRoki pilot increasing `foot_tilt`
from 1 to 20 flattened the contact feet on three sample clips and passed the
same geometric and speed checks. For `idle_to_idle_right`, its starting COM
margin improved from -0.081 m to +0.071 m, but PD replay still fell after
68/254 frames. Its PD joint RMS error was about 0.062 rad and requested
torque stayed below the configured limits; the floating base moved forward
and fell. This points to dynamic balance of the reference/controller as the
next engineering problem, not merely joint-limit filtering.
The exact `foot_tilt=20` config and three trial NPZ/JSON pairs are preserved in
`configs/robots/humanoid_v10/bones_seed_pyroki_foot_tilt20.json` and
`data/v10/contact_trial_ft20/` (with SHA-256 hashes in its manifest). The
original 16 accepted motions are unchanged.
For the short `shoulder_clap` motion, a separate trial rotated the floating
base by -0.2 rad about world Y and grounded the foot meshes frame by frame.
`scripts/v10/apply_root_pitch_trial.py` recomputes the full body reference
through MuJoCo FK and link-origin velocities. This corrected reference is in
`data/v10/balance_trial/`; PD replay completed 96/96 frames, and the existing
V10 shoulder policy completed 96/96 frames with a 0.856 normalized return.
Its mean body-position change from the original retargeting is about 0.075 m
and the body-orientation change is 0.2 rad, so the success is for the corrected
motion rather than exact tracking of the original retargeted pose.
Training a fresh policy directly on this corrected shoulder reference, with
the V10 joint-reference residual scale reduced from 0.25 to 0.05, completed
96/96 frames at its 500-iteration checkpoint (normalized return 0.871).
The reduced residual keeps the policy from destabilizing a reference that is
already dynamically viable; it is an experimental setting, not a general
solution for all BONES-SEED motions.

A fine pitch sweep of `just_realised` found a second PD-complete reference at
-0.143 rad, again with per-frame foot grounding. It completed 112/112 PD
frames, with 0.040 m RMS floating-base position tracking error. The corrected
NPZ and provenance JSON are in `data/v10/balance_candidates/`, and the sweep
is recorded in `data/v10/manifests/v10_just_realised_pitch_fine.csv`. Against
the original retargeted reference, its mean body-position shift is 0.053 m
and body-orientation shift is 0.143 rad.
The shoulder-only policy still followed only 97/112 frames of this second
clip, so PD completion alone is not policy completion. A two-motion short
pilot was trained on both corrected clips and completed the shoulder motion.

Deterministic first-frame policy benchmarks on the full clip:

| Experiment | Frames followed | Full clip |
| --- | ---: | ---: |
| Grounded/pitch-corrected shoulder gesture, existing policy | **96** | **96** |
| Grounded/pitch-corrected shoulder gesture, freshly trained 0.05-residual policy at 500 iterations | **96** | **96** |
| Grounded/pitch-corrected `just_realised`, shoulder-only 0.05-residual policy | 97 | 112 |
| Two-motion 0.05-residual policy at 500 iterations: shoulder / `just_realised` | **96** / 100 | 96 / 112 |
| Original accepted idle/step, reference-prior policy, best checkpoint | 123 | 254 |
| Contact-corrected idle/step, root-weighted policy, best checkpoint | **129** | 254 |
| Same checkpoint with residual scale 0.26 instead of trained 0.25 (diagnostic sweep on this clip) | 145 | 254 |
| Contact-corrected idle/step, larger world model and policy at 3,000 iterations | 73 | 254 |
| Contact-corrected short gesture, single-motion policy at 2,000 iterations | 46 | 112 |
| Three-motion shared policy, larger model at 2,000 iterations: exercise / idle-step / gesture | 67 / 71 / 50 | 262 / 254 / 112 |

All policy rows except the corrected shoulder gesture terminated on root height
before the clip ended. For the two-motion run, the second clip followed 98/112
frames at both 1,000 and 1,500 iterations, so the 500-iteration checkpoint
is best on that clip. This short-run result is not a learning-limit estimate
for the planned 12-hour training run.
The 0.26 residual sweep is an in-sample diagnostic, not a validated policy
setting. The best standard checkpoint is
`logs/v10_idle_flatfoot_root2/ckpt_0002000.pt`; it expects the contact-corrected
idle NPZ from `data/v10/contact_trial_ft20/`, not the original accepted NPZ.

## Training observations

- `PolicyTrainer.set_lr` previously replaced the configured critic learning
  rate with the actor learning rate at every schedule step. It now preserves
  the configured critic-to-actor ratio, with a regression test. The fresh V10
  pilot using actor `5e-5` and critic `2e-4` reached only 56/254 frames at
  2,000 iterations, so V10 pilots should explicitly use `critic_lr=5e-5`
  until a stronger setting is validated.
- Checkpoints intentionally omit the replay buffer. Resuming the best policy
  with only 20 warmup steps degraded the benchmark. A 500-step replay refill
  preserved 129/254 before learning resumed, but 500 further updates reduced
  it to 58/254. Longer training needs checkpoint selection and a validated
  resume procedure, rather than assuming the latest checkpoint is best.
- Increasing the policy imagination horizon to 16 or 32, training from only
  the first frame, and increasing model size did not complete the tested
  motion in the short pilots. Increasing parallel environments to 512 and
  2,048 also did not complete that motion in the same short-run budget.

The one-motion policy-completion gate is now met. Short runs of 500–2,000
iterations are integration pilots and cannot establish that V10 cannot learn
motions under a 12-hour, two-RTX-5080 training run. After conversion, train
one policy on all accepted data and evaluate trained and held-out motions.
The conversion gate verifies NPZ schema/finite values and V10 joint and
geometry limits. Do not require every short-run policy or PD replay to
complete before expanding the dataset.

`scripts/v10/search_root_pitch.py` searched
117 coarse variants of 13 short accepted clips in ±0.2 rad; only the shoulder
clip completed under PD replay. Its CSV audit is in
`data/v10/manifests/v10_root_pitch_search.csv`. A further 25-angle fine
search around -0.143 rad found a PD-complete `just_realised` reference. The
pitch search is an experimental correction, not a general replacement for
contact/dynamics-aware retargeting. The full conversion uses
`process_bones_seed_parallel.py` and keeps only accepted frames in its compact
stream; the source archive remains available until conversion is complete.
The full-data training target should be judged after
the planned two-GPU long run rather than extrapolated from these short pilots.
The run must first benchmark throughput and per-GPU memory with the compact
V10 library: each `torchrun` rank keeps its own simulator and a full library
copy, so two 16 GB cards do not combine into one 32 GB memory pool.

A 60-iteration `torchrun --nproc_per_node=2` V10 smoke test used one RTX 5080
per rank, completed eight distributed gradient updates, exited cleanly, and
saved `logs/v10_two_gpu_smoke/ckpt_0000060.pt`. Its first policy update spent
most of the short test on compilation; the measured 60-iteration rate is not
a useful 12-hour throughput estimate. This smoke used two NPZ clips, not the
future full compact library, so its memory use also cannot certify that the
full dataset will fit in 16 GB per card.

On 2026-10-24, `v10-supermimic-train-after-conversion.service` was armed. It
waits for the fast converter to exit successfully, checks the final V10 stream
and packed library with `scripts/v10/audit_fast_g1_library.py`, and starts a
12.5-hour, two-rank `torchrun` training run on the complete accepted snapshot.
The launch script is `scripts/v10/train_after_fast_conversion.sh`; its status
is under `logs/v10_seed_fast_full_2x5080_20261024/pipeline.log`. It starts with
1,024 environments per GPU and retries with smaller batches on a startup CUDA
out-of-memory error. Each rank stores the full compact library on its own GPU.
If disk space drops below 25 GiB before training, the script removes only the
older `soma_uniform.tar.gz` direct human-to-V10 source, as authorized; it
retains the G1 source, accepted V10 library, and checkpoints.

The fast converter finished successfully at 07:15:59 KST with 93,335 newly
processed candidates, 61,159 newly accepted motions, and zero conversion
errors. The final trainable snapshot has **64,991 motions, 21,208,574 frames,
and 2.757 GB**. The final audit checked the compact index, all finite values,
all fast-motion joint speeds, and 100 sampled geometry clips; it passed.
Dynamic balance and learned tracking performance remain unverified. The
two-rank training process started at **07:16:53 KST** on both RTX 5080s with
1,024 environments, a 2,048 policy batch, and a 1,024 world-model batch per
rank. Its configured stop time is 12.5 hours after the training loop starts,
with checkpoints every 1,000 iterations. By iteration 150, both ranks had
completed updates at roughly 2.9 iterations/s after initial compilation;
the cards used about 12.8/13.5 GiB with no OOM. These first training returns
do not measure eventual motion completion. The older SOMA source was retained
because 68 GiB remained free at training start.
At 07:25 KST, the run reached iteration 1,000, completed 384 updates, and
saved both `ckpt_0001000.pt` and `ckpt_latest.pt` (205 MB each). The user
service remained active with both GPUs training. The 12.5-hour duration has
been configured but has not yet elapsed.
