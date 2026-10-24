# Humanoid V10 browser demo

This local browser demo replays a trained V10 policy in **MuJoCo physics** with a
translucent reference robot. It is inspired by the [SuperMimic browser demo](https://supermimic.github.io/demo/),
but does not run that site's G1 checkpoint or learned world model in the browser.

The curated library contains 35 complete V10 BONES-SEED motions. The catalogue
starts with the official demo's motion groups. Exact motions were kept where
available; labels say `substitute` where the accepted V10 library only has an
analogous motion. The five object manipulation motions were omitted because
the V10 training library excludes human-object interaction. LAFAN1 examples
were replaced by BONES-SEED analogues; they are **not** held-out motions. The
exact source name of every selected clip is visible under the motion picker
and saved in `data/v10/demo/v10_demo.json`.

To browse every accepted training motion, run the same viewer with
`--library data/v10/library/accepted_compact.npy`. The full catalogue contains
64,991 clips and is searched in pages of 60, so the browser never renders the
entire picker at once. The existing 35 curated entries appear first as
shortcuts. On this host, the CPU viewer used about 5.2 GiB of RAM with the full
library; it does not reserve either training GPU. No duplicate motion file is
created. For a separate full-library viewer alongside the curated demo, use
`--port 8084 --demo-port 8085` and open `http://127.0.0.1:8085/`.

Build or refresh the small library after changing the catalogue:

```bash
.venv/bin/python scripts/v10/build_demo_library.py
```

Run the browser viewer on CPU while both RTX 5080 cards are training:

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/mpl-v10-demo \
  WARP_CACHE_PATH=/tmp/warp-v10-demo OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 10 .venv/bin/python scripts/play.py \
  --ckpt logs/v10_seed_fast_full_2x5080_20261024/ckpt_0030000.pt \
  --library data/v10/demo/v10_demo.npy --num-envs 1 \
  --device cpu --viewer viser --demo-ui
```

Open [http://127.0.0.1:8081](http://127.0.0.1:8081) for the official-style
layout. The underlying Viser engineering controls remain available at
[http://127.0.0.1:8080](http://127.0.0.1:8080). Use **Motions** to search and
choose a clip; selection resets the robot to that clip's first frame. The
bottom dock controls playback, policy on/off, and policy action noise. The
solid V10 is the physical rollout; the magenta V10 at 42% opacity is the
retargeted reference. Use the **Goal** chip at top left to toggle that overlay.
This demo keeps simulating when the policy drifts from the reference; it resets
only when the selected clip ends or when you choose/reset a motion. Training
and benchmark environments retain their normal tracking-failure terminations.
The friction indicator is fixed at the evaluation setting of 1.0. The author's
browser-side world model and drag-to-perturb interaction are not implemented.
Both local servers bind to localhost by default. Pass `--host 0.0.0.0` only if
remote LAN access is intended.

For the final checkpoint, replace the `--ckpt` path after training ends. A
motion appearing in the demo is **not** a claim that the checkpoint can finish
it; use `scripts/eval_benchmark.py` to measure completion and tracking return.
