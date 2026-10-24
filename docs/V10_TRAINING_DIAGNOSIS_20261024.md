# V10 full-library training diagnosis (2026-10-24, 13:29 KST)

The two-RTX-5080 training service remains active. This analysis only read logs
and checkpoints and ran low-priority CPU evaluations. It did not alter the
running model, sampler, dataset, or training duration. The diagnosis concerns
the `v10_seed_fast_full_2x5080_20261024` run with 64,991 accepted V10 motions.

## Fixed-motion evidence of regression

Twenty-four fast-G1-to-V10 clips were sampled with NumPy seed `20261024` from
the final compact library, excluding the 15 pilot NPZ clips and limiting clip
lengths to 100–600 frames. All 24 selected clips have the
`fast_g1_calibrated` provenance. Every checkpoint was evaluated from each
clip's first frame on the same CPU MuJoCo Warp simulator, with deterministic
policy, no observation/action noise, no reset noise, and no domain
randomization. This is a small **training-set** sample, not a held-out or
full-library success estimate. The exact library subset and per-clip results
are stored in `data/v10/diagnostics/20261024_fixed_eval/`.

| Checkpoint | Full clips | Mean normalized return | Newly failed vs 30k |
| --- | ---: | ---: | ---: |
| 30,000 | 24/24 | 0.7188 | — |
| 50,000 | 22/24 | 0.6227 | 2 |
| 65,000 | 14/24 | 0.4869 | 10 |

All 24 clips have a lower per-clip normalized return at 65k than at 30k.
Eight of the ten new 65k failures ended at the root-height threshold and two
at root tilt. This is real degradation on the fixed sample, not merely a
change in the training sampler's choice of motions. The 30k checkpoint also
shows that this run learned to complete at least some uncorrected motions.

## Confirmed sampling pathology

`MultiClipMotionCommand._finalize_episodes` records discounted, **unnormalized**
return-to-go per 50-frame bin. Near a clip's end, even good tracking has few
rewarding steps left, so its return-to-go is intrinsically small. The reset
sampler applies `softmax(-25 * bin_return / 10)` and mixes it with 50% uniform,
40% hard, and 10% memory sampling. Its hard branch therefore interprets a
short terminal tail as a difficult motion segment.

Checkpoint inspection of the exact stored `bin_return` and `bin_memory` shows:

| Checkpoint | Largest single-bin reset probability | Reset mass on bins with ≤10 frames left | Example selected tail |
| --- | ---: | ---: | --- |
| 30,000 | 49.98% | 1.46% | 24 frames left in an injured-torso walk |
| 50,000 | 31.81% | 33.29% | 6 frames left in a hurry-idle motion |
| 65,000 | 8.34% | 39.95% | 2 frames left in an injured-torso idle loop |

At 65k the top eight bins together receive roughly half of all reset
probability. This concentration conflicts with the objective of learning
broad motion coverage. In addition, the 32-step imagined policy rollout clamps
the reference to its final frame after clip end but does not terminate the
rollout there. Very short terminal segments therefore produce imagined training
against a repeated final pose. The exact fraction of policy updates affected
cannot be inferred from reset probabilities alone because short episodes
contribute fewer replay transitions.

## Training dynamics and secondary factors

Across 25k–35k versus 60k–66k iteration windows, mean training return fell
from 0.499 to 0.446, mean episode length from 81.6 to 48.3 frames, and online
8-step world-model NSE rose from 0.097 to 0.189 (lower is better). Mean
imagined return fell from 1.907 to 1.211. The logged **average pre-clipping**
policy gradient norm exceeded the configured norm limit of 1.0 in 3.3% of
log windows during 10k–30k, versus 84.1% during 50k–67k. This supports
late-run instability, but does not by itself identify the first cause.

The launch used `total_iterations=1,000,000` to keep the process alive for the
requested 12.5 hours. The code also derives cosine LR decay length from that
same total, so the effective two-rank LR was still approximately `2.78e-4`
at 65k, only 1.6% below its starting value. Under the default 320k-step
schedule it would be about `2.40e-4` at that point. This is a plausible
contributor to instability and should be decoupled from the wall-clock stop
in a future run; it is not proven to be the primary cause.

An independent V9-gain joint-PD preflight on 80 random V10 references found
64 root-height failures within the first 150 frames. The initial root linear
and angular velocities were taken from the references, with angular velocity
transformed into the MuJoCo free-joint frame. This demonstrates that geometric
retargeting acceptance does not guarantee simple PD replay. It is **not** a
trained-policy failure rate: the fixed 30k policy evaluation above performs
substantially better on its separate 24-clip sample.

## Next controlled test after the current run

Preserve this run and compare its 30k, 50k, 65k, and final checkpoints on a
larger fixed set that includes trained and held-out motions. For a new run,
normalize or horizon-match the return-to-go priority, prevent short terminal
bins from dominating reset selection, and stop imagined rollouts at clip end.
Keep a separate LR schedule with a realistic decay length. Change one factor
at a time to establish causality. The current training process has not been
modified or stopped.
