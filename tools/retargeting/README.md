# Offline retargeting source

The `apps/external` and `apps/humanoid_retargeter` modules were copied from
`/home/cocelo-server01/stkim_ws/Video2Robot_reference-main` on 2026-10-24.
They preserve the existing BVH reader, PyRoki F26 optimization, geometric
checks, and NPZ export. The original source has Humanoid Light v2 assumptions
in model, body, contact-geom, and export names. These modules are **not yet**
validated for V10 and must not be used to produce a training library until
the V10 MJCF and SOMA mapping are checked.

This tool lives in the same repository as training, but runs as an offline
Python environment. PyRoki/JAX are not imported by the SuperMimic trainer.
