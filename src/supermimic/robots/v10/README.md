# V10 training port

`asset.py` supplies the V10 MuJoCo asset, body map, and V9-derived per-joint
position actuators. `sm-train --robot v10` selects these and uses 26-joint
observations/actions and a 65-dimensional floating-base state. The default
`--robot g1` path remains available. The V10 policy path has passed a short
integration smoke, but a trained motion-tracking result is not yet established.
