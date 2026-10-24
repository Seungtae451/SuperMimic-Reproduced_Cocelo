# BONES-SEED source preparation

Use the dataset's SOMA Uniform BVH as human-motion input for V10. The source
metadata and base SOMA skeleton are in `data/v10/downloads/`; keep account
credentials out of this repository. `fetch_bones_seed.py --uniform-archive`
fetches the 45.2 GB compressed SOMA archive when at least 100 GiB is free.

The release stores all 142,220 SOMA Uniform BVHs in **one** `soma_uniform.tar.gz`.
The archive was downloaded at dataset revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`;
the metadata-only prefilter leaves 106,924 candidates (12,119 excluded by the
approximate paper-name screen, then 23,177 by the no-hand/object rule).
Metadata can be filtered before download, but Hugging Face cannot supply only
the selected BVHs from that gzip archive. Keep the archive compressed and
process selected members in a streaming pass; do not extract the entire set.

Run `python scripts/v10/prefilter_bones_seed.py` with `pyarrow` installed to
write `data/v10/manifests/bones_seed_prefilter.csv` and its JSON summary.
It applies two independent pre-retargeting screens:

- An approximation of SuperMimic Appendix E's name filter, using the keyword
  list in this reproduction's `scripts/data/filter_bones_seed.py`. That list
  rejects 12,119 metadata rows versus the paper's 12,553; the paper does not
  publish its exact list or the rejected motion IDs. Do not label this output
  as the exact paper subset.
- A V10-without-hands rule: reject motions tagged with a nonzero prop or
  classified as manipulation/consuming/household, plus explicit hand actions
  in the Object Interaction category. This conservative rule can reject
  non-grasp prop motions too. Every decision and reason is recorded per row.

The paper's 153 additional feasibility rejections require G1 trajectory
forward kinematics and cannot be identified from this metadata. V10's own
joint, contact, penetration, and support checks must run **after** direct
SOMA-to-V10 retargeting. G1 infeasibility is a useful conservative prior, not
proof of V10 infeasibility.
